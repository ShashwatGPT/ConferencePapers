"""
limitation_analyser.py — Cluster-level limitation synthesis and solved-status checker.

Two-stage pipeline:

Stage 1 — synthesise_cluster_limitations()
  Given per-paper full-text results and a cluster label, asks the LLM to
  deduplicate and produce 3-7 *specific, quantified* genuine limitations for
  the whole cluster.

Stage 2 — check_solved_batch()
  For each genuine limitation, searches Semantic Scholar (and optionally
  Perplexica) for recent papers that address it, then asks LLM to judge:
    "fully_solved" | "partially_solved" | "ongoing" | "open"
  Returns enriched limitation dicts each containing a list of solving papers.

Output schema (list of LimitationEntry dicts)
──────────────────────────────────────────────
{
  "statement":       str,   # specific claim, with numbers if available
  "metric":          str | null,
  "category":        str,   # scalability | data | evaluation | compute |
                            #   generalization | theory | reproducibility | scope
  "papers_stating":  [str], # titles of papers that state this limitation
  "solved_status":   "fully_solved" | "partially_solved" | "ongoing" | "open",
  "solving_papers":  [{"title": str, "year": int, "url": str, "how": str}],
  "solved_summary":  str,   # 1-sentence synthesis of the solved-status
}
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_SYN_SYSTEM = (
    "You are a senior ML research analyst. Your job is to read limitations from "
    "multiple papers in a research cluster and synthesise the genuine, non-trivial "
    "ones. Be extremely specific — every limitation must name exact numbers, model "
    "sizes, dataset names, task types, or error modes that appear in the source text. "
    "Reject generic boilerplate (e.g. 'we leave this for future work' without detail)."
)

_SYN_PROMPT = """\
Research cluster: "{label}"

Below are limitations extracted from papers in this cluster.
Each entry is: [paper title] → limitation text.

{corpus}

Synthesise 3-7 GENUINE, SPECIFIC cluster-level limitations.
Rules:
- Merge near-duplicate limitations from different papers into one entry.
- Every statement must contain at least one concrete detail (number, model name, dataset, task type, failure mode).
  BAD: "The context window is limited."
  GOOD: "Evaluated only on context windows ≤32k tokens; performance degrades >18% on 3-hop tasks beyond 50k tokens (Table 3)."
- category is one of: scalability | data | evaluation | compute | generalization | theory | reproducibility | scope
- papers_stating: list the paper titles (shortened) that mention this

Output ONLY a JSON array:
[
  {{
    "statement": "<specific limitation>",
    "metric": "<exact figure or null>",
    "category": "<category>",
    "papers_stating": ["<title1>", "<title2>"]
  }},
  ...
]"""


_SOLVED_SYSTEM = (
    "You are a research analyst. Given a specific research limitation and a list of "
    "recent papers (2023-2026), judge whether those papers address the limitation "
    "and to what extent."
)

_SOLVED_PROMPT = """\
Limitation: "{limitation}"

Recent papers from Semantic Scholar that may address this:
{paper_list}

Classify the solved status as ONE of:
  fully_solved     — a paper definitively solves the limitation with evidence
  partially_solved — a paper makes clear progress but does not fully resolve it
  ongoing          — active area, multiple partial solutions, not converged
  open             — no paper addresses this substantively

For each relevant paper (max 4), note HOW it addresses (or doesn't) the limitation in ≤25 words.

Output ONLY JSON:
{{
  "solved_status": "<status>",
  "solving_papers": [
    {{"title": "<title>", "year": <year>, "url": "<url or empty>", "how": "<≤25 words>"}}
  ],
  "solved_summary": "<1 sentence summary>"
}}"""


# ---------------------------------------------------------------------------
# Stage 1 — Cluster synthesis
# ---------------------------------------------------------------------------

def _build_corpus(ft_map: dict[str, dict], cluster_papers: list[dict]) -> str:
    """
    Build a text corpus of limitations for the LLM prompt, from:
      1. llm_limitations list (preferred — already specific)
      2. raw limitations / future_work section text (fallback)
      3. abstract-derived sentences (last resort)
    """
    lines: list[str] = []
    for paper in cluster_papers:
        pid   = paper.get("id") or paper.get("semantic_scholar_id") or ""
        title = paper.get("title", "Unknown")[:80]
        ft    = ft_map.get(pid, {})

        # 1. LLM-extracted specific items
        llm_lims: list[dict] = ft.get("llm_limitations", [])
        if llm_lims:
            for item in llm_lims:
                stmt = item.get("statement", "").strip()
                met  = item.get("metric") or ""
                text = stmt + (f" [{met}]" if met else "")
                lines.append(f"  [{title}] → {text}")
            continue

        # 2. Raw PDF section text (truncated to first 1500 chars per paper)
        raw = (ft.get("limitations") or ft.get("future_work") or "").strip()
        if raw:
            lines.append(f"  [{title}] → {raw[:1500]}")
            continue

        # 3. Abstract-derived sentences (from paper dict)
        abstract = (paper.get("abstract") or "")[:600]
        if abstract:
            lines.append(f"  [{title}] (abstract-only) → {abstract}")

    return "\n".join(lines) if lines else "(no limitation text available)"


def synthesise_cluster_limitations(
    ft_map: dict[str, dict],
    cluster_papers: list[dict],
    cluster_label: str,
    llm,
) -> list[dict]:
    """
    Stage 1: Deduplicate and synthesise cluster-level limitations via LLM.

    Returns a list of LimitationEntry dicts (without solved_status yet).
    Falls back to [] on LLM failure.
    """
    corpus = _build_corpus(ft_map, cluster_papers)
    prompt = _SYN_PROMPT.format(label=cluster_label, corpus=corpus)

    try:
        raw, _ = llm.call_llm_with_retry(
            user_message=prompt,
            system_message=_SYN_SYSTEM,
        )
        if not raw:
            return []
        clean = re.sub(r"```(?:json)?\s*", "", str(raw)).strip().rstrip("`")
        parsed = json.loads(clean)
        if not isinstance(parsed, list):
            return []
        entries = []
        for item in parsed:
            if not isinstance(item, dict) or not item.get("statement"):
                continue
            entries.append({
                "statement":      str(item["statement"]).strip(),
                "metric":         item.get("metric") or None,
                "category":       str(item.get("category", "scope")).strip(),
                "papers_stating": [str(t) for t in item.get("papers_stating", [])],
                "solved_status":  "open",
                "solving_papers": [],
                "solved_summary": "",
            })
        return entries
    except Exception as exc:
        logger.warning(f"[LimitationAnalyser] Synthesis failed for {cluster_label!r}: {exc}")
        return []


# ---------------------------------------------------------------------------
# Stage 2 — Solved-status checker
# ---------------------------------------------------------------------------

async def _s2_search_for_limitation(
    client: httpx.AsyncClient,
    limitation: str,
    base_url: str,
    year_min: int = 2023,
    top_k: int = 6,
) -> list[dict]:
    """
    Search Semantic Scholar for papers that may address the given limitation.
    Uses the first 120 chars of the limitation statement as the query.
    Returns a list of {title, year, url, abstract} dicts.
    """
    # Shorten to key noun phrases for a better S2 keyword query
    query_text = re.sub(r"\[.*?\]", "", limitation[:120]).strip()
    fields = "title,year,url,abstract,openAccessPdf"
    params = {
        "query":  query_text,
        "fields": fields,
        "limit":  top_k,
    }
    if year_min:
        params["year"] = f"{year_min}-"

    try:
        r = await client.get(f"{base_url}/paper/search", params=params, timeout=15)
        if r.status_code == 200:
            hits = r.json().get("data", [])
            out = []
            for h in hits:
                pdf_url = ""
                oap = h.get("openAccessPdf") or {}
                if oap:
                    pdf_url = oap.get("url", "")
                out.append({
                    "title":    h.get("title", ""),
                    "year":     h.get("year") or 0,
                    "url":      h.get("url") or pdf_url or "",
                    "abstract": (h.get("abstract") or "")[:400],
                })
            return out
    except Exception as exc:
        logger.debug(f"[LimitationAnalyser] S2 search error: {exc}")
    return []


async def _perplexica_search(
    client: httpx.AsyncClient,
    query: str,
    base_url: str,
    top_k: int = 4,
) -> list[dict]:
    """
    Fallback web search via Perplexica API (if configured).
    Returns list of {title, year, url, abstract} dicts.
    """
    try:
        payload = {
            "query": query,
            "focusMode": "academicSearch",
            "optimizationMode": "balanced",
        }
        r = await client.post(f"{base_url}/api/search", json=payload, timeout=30)
        if r.status_code != 200:
            return []
        data = r.json()
        sources = data.get("sources", [])[:top_k]
        results = []
        for s in sources:
            results.append({
                "title":    s.get("metadata", {}).get("title", s.get("url", "")),
                "year":     0,
                "url":      s.get("url", ""),
                "abstract": s.get("pageContent", "")[:400],
            })
        return results
    except Exception as exc:
        logger.debug(f"[LimitationAnalyser] Perplexica search error: {exc}")
    return []


def _llm_judge_solved(
    limitation: str,
    candidate_papers: list[dict],
    llm,
) -> dict:
    """
    Ask LLM to judge how well the candidates address the limitation.
    Returns dict with solved_status, solving_papers, solved_summary.
    """
    if not candidate_papers:
        return {
            "solved_status":  "open",
            "solving_papers": [],
            "solved_summary": "No recent papers found that address this limitation.",
        }

    paper_lines = []
    for p in candidate_papers[:8]:
        abst = p.get("abstract", "No abstract.")[:300]
        paper_lines.append(
            f'- "{p["title"]}" ({p.get("year") or "?"})\n  {abst}'
        )
    paper_list = "\n".join(paper_lines)

    prompt = _SOLVED_PROMPT.format(limitation=limitation, paper_list=paper_list)
    try:
        raw, _ = llm.call_llm_with_retry(
            user_message=prompt,
            system_message=_SOLVED_SYSTEM,
        )
        if not raw:
            raise ValueError("empty LLM response")
        clean = re.sub(r"```(?:json)?\s*", "", str(raw)).strip().rstrip("`")
        result = json.loads(clean)
        # Normalise
        status = result.get("solved_status", "open")
        if status not in ("fully_solved", "partially_solved", "ongoing", "open"):
            status = "open"
        return {
            "solved_status":  status,
            "solving_papers": result.get("solving_papers", []),
            "solved_summary": str(result.get("solved_summary", "")).strip(),
        }
    except Exception as exc:
        logger.debug(f"[LimitationAnalyser] Solved judge failed: {exc}")
    return {
        "solved_status":  "open",
        "solving_papers": [],
        "solved_summary": "Could not determine solved status.",
    }


async def check_solved_batch(
    limitations: list[dict],
    s2_base_url: str,
    llm,
    perplexica_base_url: Optional[str] = None,
    concurrency: int = 3,
) -> list[dict]:
    """
    Stage 2: For each limitation, search for solving papers and judge solved status.

    Runs S2 searches concurrently (rate-limited by `concurrency`).
    LLM judge calls run in the default thread-pool executor (blocking).

    Returns the same list with solved_status / solving_papers / solved_summary filled in.
    """
    sem = asyncio.Semaphore(concurrency)
    loop = asyncio.get_event_loop()

    async def _process_one(lim: dict) -> dict:
        async with sem:
            statement = lim["statement"]

            async with httpx.AsyncClient(
                follow_redirects=True,
                headers={"User-Agent": "ConferencePapers-Researcher/1.0"},
            ) as client:
                # Primary: Semantic Scholar
                candidates = await _s2_search_for_limitation(
                    client, statement, s2_base_url, year_min=2023
                )

                # Supplement: Perplexica (if configured and S2 returned few results)
                if perplexica_base_url and len(candidates) < 4:
                    perp = await _perplexica_search(
                        client,
                        f"research paper that solves: {statement[:100]}",
                        perplexica_base_url,
                    )
                    # Merge deduped by title
                    existing_titles = {c["title"].lower() for c in candidates}
                    for p in perp:
                        if p["title"].lower() not in existing_titles:
                            candidates.append(p)
                            existing_titles.add(p["title"].lower())

            # LLM judge (blocking — run in executor)
            verdict = await loop.run_in_executor(
                None, _llm_judge_solved, statement, candidates, llm
            )
            return {**lim, **verdict}

    enriched = await asyncio.gather(
        *[_process_one(lim) for lim in limitations],
        return_exceptions=True,
    )
    out = []
    for lim, result in zip(limitations, enriched):
        if isinstance(result, Exception):
            logger.warning(f"[LimitationAnalyser] check_solved error: {result}")
            out.append(lim)
        else:
            out.append(result)
    return out


# ---------------------------------------------------------------------------
# Convenience: full two-stage pipeline for one cluster
# ---------------------------------------------------------------------------

async def analyse_cluster_limitations(
    ft_map: dict[str, dict],
    cluster_papers: list[dict],
    cluster_label: str,
    s2_base_url: str,
    llm,
    perplexica_base_url: Optional[str] = None,
) -> list[dict]:
    """
    Run both stages for a cluster.
    Returns a fully annotated list of LimitationEntry dicts.
    """
    loop = asyncio.get_event_loop()

    # Stage 1 — synthesise (blocking LLM call)
    limitations = await loop.run_in_executor(
        None,
        synthesise_cluster_limitations,
        ft_map, cluster_papers, cluster_label, llm,
    )
    if not limitations:
        return []

    # Stage 2 — check solved status (async S2 + blocking LLM)
    limitations = await check_solved_batch(
        limitations,
        s2_base_url=s2_base_url,
        llm=llm,
        perplexica_base_url=perplexica_base_url,
        concurrency=3,
    )
    return limitations
