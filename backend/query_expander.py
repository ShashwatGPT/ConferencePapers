"""
query_expander.py — RAG-Fusion query expansion for ConferencePapers.

Given a user query, generates N semantically diverse variants so that dense
retrieval covers a broader embedding neighbourhood.  Results are later merged
with Reciprocal Rank Fusion (RRF) by EmbeddingStore.search_rrf().

Strategy (in order):
  1. MODEL LLM (Azure OpenAI or GitHub Copilot SDK)  — instruction-tuned
  2. Rule-based heuristics                            — zero-cost fallback

Endpoint precedence (highest first):
  1. LLM_MODEL_ENDPOINT env var
  2. config.json  llm.model_endpoint
  3. Default: "ghcp" (GitHub Copilot SDK)

Model name precedence:
  1. LLM_MODEL_NAME env var
  2. config.json  llm.model_name
  3. Default: "gpt-4.1"

call_llm / call_llm_with_retry both return (response_text, (prompt_tok, compl_tok)).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Any

logger = logging.getLogger(__name__)

# ── Bootstrap MODEL path ──────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Load LLM config (config.json → env var override) ─────────────────────────
def _load_llm_cfg() -> Tuple[str, str | None]:
    """Return (model_name, model_endpoint) from config.json + env overrides."""
    model_name     = "gpt-4.1"
    model_endpoint: str | None = "ghcp"
    try:
        cfg_path = Path(__file__).parent.parent / "config.json"
        with open(cfg_path) as f:
            cfg = json.load(f)
        llm_cfg        = cfg.get("llm", {})
        model_name     = llm_cfg.get("model_name", model_name)
        model_endpoint = llm_cfg.get("model_endpoint", model_endpoint) or None
    except Exception:
        pass
    # Env vars always win
    model_name     = os.environ.get("LLM_MODEL_NAME", model_name)
    model_endpoint = os.environ.get("LLM_MODEL_ENDPOINT", model_endpoint or "") or model_endpoint
    return model_name, model_endpoint

_LLM_MODEL_NAME, _LLM_MODEL_ENDPOINT = _load_llm_cfg()

# ── LLM singleton (lazy) ──────────────────────────────────────────────────────
_LLM: Any = None

def _get_llm():
    global _LLM
    if _LLM is not None:
        return _LLM
    try:
        from MODEL.models import Model
        kwargs: dict = {}
        if _LLM_MODEL_ENDPOINT:
            kwargs["model_endpoint"] = _LLM_MODEL_ENDPOINT
        _LLM = Model(_LLM_MODEL_NAME, **kwargs)
        logger.info(f"[QueryExpander] LLM ready — {_LLM_MODEL_NAME} via {_LLM_MODEL_ENDPOINT or 'Azure'}")
    except Exception as exc:
        logger.warning(f"[QueryExpander] LLM unavailable ({exc}); using rule-based expansion")
        _LLM = False   # sentinel: tried and failed
    return _LLM


# ── Academic vocabulary for rule-based expansion ──────────────────────────────

_SYNONYMS: dict[str, list[str]] = {
    "agent":        ["autonomous agent", "LLM agent", "language agent", "AI agent"],
    "agents":       ["autonomous agents", "LLM-based agents", "interactive agents"],
    "task":         ["workflow", "subtask", "goal-directed task"],
    "planning":     ["task decomposition", "hierarchical planning", "goal planning"],
    "tool":         ["API calling", "function calling", "tool invocation"],
    "tool use":     ["function calling", "API calling", "external tool integration"],
    "code":         ["program synthesis", "code generation", "software code"],
    "web":          ["browser", "web navigation", "web interaction"],
    "memory":       ["retrieval-augmented", "episodic memory", "context retrieval"],
    "benchmark":    ["evaluation suite", "assessment framework", "test suite"],
    "reasoning":    ["chain-of-thought", "multi-step reasoning", "inference"],
    "multimodal":   ["vision-language", "image-language", "cross-modal"],
    "llm":          ["large language model", "foundation model", "language model"],
    "rl":           ["reinforcement learning", "reward-based learning", "policy learning"],
    "fine-tuning":  ["supervised fine-tuning", "instruction tuning", "RLHF"],
}

_ACADEMIC_PREFIXES = [
    "survey of",
    "theoretical analysis of",
    "benchmark for",
]

_FIELD_CONTEXTS = [
    "in natural language processing",
    "for large language models",
    "using transformer architectures",
]


def _rule_based_expand(query: str, n: int = 4) -> List[str]:
    """
    Generate query variants without an LLM.  Always returns len = n+1 (original + n).
    """
    q = query.strip()
    variants: list[str] = [q]   # always keep original

    q_lower = q.lower()

    # 1. Synonym substitution — swap the first matching term
    for term, syns in _SYNONYMS.items():
        if term in q_lower:
            for syn in syns[:2]:
                new_q = re.sub(re.escape(term), syn, q, flags=re.IGNORECASE, count=1)
                if new_q.lower() != q_lower and new_q not in variants:
                    variants.append(new_q)
                    if len(variants) >= n + 1:
                        break
            if len(variants) >= n + 1:
                break

    # 2. Academic prefix variants
    for prefix in _ACADEMIC_PREFIXES:
        if not q_lower.startswith(prefix):
            v = f"{prefix} {q}"
            if v not in variants:
                variants.append(v)
        if len(variants) >= n + 1:
            break

    # 3. Field context appended
    for ctx in _FIELD_CONTEXTS:
        if ctx not in q_lower:
            v = f"{q} {ctx}"
            if v not in variants:
                variants.append(v)
        if len(variants) >= n + 1:
            break

    # 4. Extract noun phrases (rough: 2-gram windows over words)
    words = [w for w in re.split(r'\W+', q) if len(w) > 3]
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        if bigram.lower() != q_lower:
            variants.append(bigram)
        if len(variants) >= n + 1:
            break

    # Pad to n+1 if needed
    while len(variants) < n + 1:
        variants.append(f"{q} research paper abstract")

    return variants[: n + 1]


_LLM_SYSTEM = """\
You are a research query expansion expert specialising in NLP and machine learning papers.
Your job is to help find academic papers by generating alternative phrasings of a search query.
"""

_LLM_USER_TMPL = """\
Original query: "{query}"

Generate exactly {n} alternative search queries that:
1. Preserve the core intent but use different academic vocabulary
2. Include one broader abstraction (more theoretical framing)
3. Include one HyDE variant — write 1 sentence as if it were from an abstract of a paper that answers the query
4. Include one keyword-dense variant (comma-separated key terms)
5. Include one narrower/specific sub-aspect

Return ONLY a numbered list, one query per line, no explanations.
"""


def expand_query(query: str, n: int = 4) -> List[str]:
    """
    Return [original_query] + n variant queries using MODEL LLM.

    call_llm / call_llm_with_retry return (response_text, tokens_tuple).
    This function always unpacks correctly and falls back to rule-based
    expansion if the LLM is unavailable or returns unusable output.

    Parameters
    ----------
    query : str  — The user's raw search query.
    n : int      — Number of *additional* queries (default 4 → 5 total).

    Returns
    -------
    List[str] of length n+1: original first, then variants.
    """
    query = query.strip()
    if not query:
        return [query]

    llm = _get_llm()
    if llm and llm is not False:
        try:
            prompt = _LLM_USER_TMPL.format(query=query, n=n)

            # call_llm_with_retry returns (response_text, tokens) — unpack correctly
            raw, _tokens = llm.call_llm_with_retry(
                user_message=prompt,
                system_message=_LLM_SYSTEM,
            )

            if raw is None:
                raise ValueError("LLM returned None")

            # Normalise: if a structured object was returned, coerce to str
            text: str = raw if isinstance(raw, str) else str(raw)

            # Parse numbered/bulleted list
            lines = [
                re.sub(r"^\s*[\d\-\*\.]+\s*", "", line).strip()
                for line in text.splitlines()
                if line.strip() and not line.strip().lower().startswith("here")
            ]
            lines = [l for l in lines if len(l) > 5][:n]

            if len(lines) >= 2:
                logger.info(
                    f"[QueryExpander] {_LLM_MODEL_NAME} generated {len(lines)} "
                    f"variants for '{query[:50]}'"
                )
                return [query] + lines
            logger.warning("[QueryExpander] LLM returned too few lines; falling back")
        except Exception as exc:
            logger.warning(f"[QueryExpander] LLM expansion failed ({exc}); falling back")

    return _rule_based_expand(query, n=n)


# ── Academic vocabulary for rule-based expansion ──────────────────────────────

_SYNONYMS: dict[str, list[str]] = {
    "agent":        ["autonomous agent", "LLM agent", "language agent", "AI agent"],
    "agents":       ["autonomous agents", "LLM-based agents", "interactive agents"],
    "task":         ["workflow", "subtask", "goal-directed task"],
    "planning":     ["task decomposition", "hierarchical planning", "goal planning"],
    "tool":         ["API calling", "function calling", "tool invocation"],
    "tool use":     ["function calling", "API calling", "external tool integration"],
    "code":         ["program synthesis", "code generation", "software code"],
    "web":          ["browser", "web navigation", "web interaction"],
    "memory":       ["retrieval-augmented", "episodic memory", "context retrieval"],
    "benchmark":    ["evaluation suite", "assessment framework", "test suite"],
    "reasoning":    ["chain-of-thought", "multi-step reasoning", "inference"],
    "multimodal":   ["vision-language", "image-language", "cross-modal"],
    "llm":          ["large language model", "foundation model", "language model"],
    "rl":           ["reinforcement learning", "reward-based learning", "policy learning"],
    "fine-tuning":  ["supervised fine-tuning", "instruction tuning", "RLHF"],
}

_ACADEMIC_PREFIXES = [
    "survey of",
    "theoretical analysis of",
    "benchmark for",
]

_FIELD_CONTEXTS = [
    "in natural language processing",
    "for large language models",
    "using transformer architectures",
]


def _rule_based_expand(query: str, n: int = 4) -> List[str]:
    """
    Generate query variants without an LLM.  Always returns len = n+1 (original + n).
    """
    q = query.strip()
    variants: list[str] = [q]   # always keep original

    q_lower = q.lower()

    # 1. Synonym substitution — swap the first matching term
    for term, syns in _SYNONYMS.items():
        if term in q_lower:
            for syn in syns[:2]:
                new_q = re.sub(re.escape(term), syn, q, flags=re.IGNORECASE, count=1)
                if new_q.lower() != q_lower and new_q not in variants:
                    variants.append(new_q)
                    if len(variants) >= n + 1:
                        break
            if len(variants) >= n + 1:
                break

    # 2. Academic prefix variants
    for prefix in _ACADEMIC_PREFIXES:
        if not q_lower.startswith(prefix):
            v = f"{prefix} {q}"
            if v not in variants:
                variants.append(v)
        if len(variants) >= n + 1:
            break

    # 3. Field context appended
    for ctx in _FIELD_CONTEXTS:
        if ctx not in q_lower:
            v = f"{q} {ctx}"
            if v not in variants:
                variants.append(v)
        if len(variants) >= n + 1:
            break

    # 4. Extract noun phrases (rough: 2-gram windows over words)
    words = [w for w in re.split(r'\W+', q) if len(w) > 3]
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        if bigram.lower() != q_lower:
            variants.append(bigram)
        if len(variants) >= n + 1:
            break

    # Ensure we have exactly n+1 variants (pad with HyDE-style rephrases)
    while len(variants) < n + 1:
        variants.append(f"{q} research paper abstract")

    return variants[: n + 1]


_LLM_SYSTEM = """\
You are a research query expansion expert specialising in NLP and machine learning papers.
Your job is to help find academic papers by generating alternative phrasings of a search query.
"""

_LLM_USER_TMPL = """\
Original query: "{query}"

Generate exactly {n} alternative search queries that:
1. Preserve the core intent but use different academic vocabulary
2. Include one broader abstraction (more theoretical framing)
3. Include one HyDE variant — write 1 sentence as if it were from an abstract of a paper that answers the query
4. Include one keyword-dense variant (comma-separated key terms)
5. Include one narrower/specific sub-aspect

Return ONLY a numbered list, one query per line, no explanations.
"""
