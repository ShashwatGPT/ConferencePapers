"""
citation_crawler.py — PASA-style iterative citation-traversal Crawler for ConferencePapers.

Architecture
────────────
Two-agent loop (mirrors PASA):

  ┌─────────────────────────────────────────────────────────────────┐
  │  Crawler Agent  (this module)                                   │
  │  ─────────────────────────────────────────────────────────────  │
  │  Round 0: seed_papers (already fetched from OpenReview/arXiv)   │
  │  Each round:                                                     │
  │    1. Pick "expand candidates" — papers worth following refs for │
  │       Criteria: accepted at target conf OR high citation count   │
  │       OR Selector scored ≥ threshold                            │
  │    2. Fetch references via S2 /paper/{id}/references            │
  │    3. LLM Selector scores every new reference's title+abstract  │
  │       against the original query → keep if score ≥ threshold    │
  │    4. LLM Coverage Judge: "Is the corpus sufficient?"           │
  │       Stop if YES or max_rounds reached or no new papers added  │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  Selector Agent  (inline LLM scoring)                           │
  │  Reads (query, title, abstract) → score 0.0–1.0                │
  │  Falls back to embedding cosine similarity when LLM unavailable │
  └─────────────────────────────────────────────────────────────────┘

Config knobs (all in config.json under "crawler"):
  max_rounds          int   2       Max BFS hops from seed papers
  max_total_papers    int   300     Hard ceiling on corpus size
  expand_top_n        int   20      Papers per round to expand refs for
  selector_threshold  float 0.45   Min relevance score to keep a ref
  batch_llm_size      int   10     Papers per LLM Selector call (batched)
  min_citations       int   5      Refs with fewer citations are skipped early
  target_conferences  list  []     Only keep refs from these venues (empty=all)

The crawler is OPTIONAL — if the LLM is unavailable the whole step is skipped
gracefully and the caller gets back the original seed papers unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from pathlib import Path
from typing import List, Optional, Set

logger = logging.getLogger(__name__)

# ── Bootstrap MODEL path ──────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Target conference venue keywords (for filtering references) ───────────────
_CONF_VENUE_KEYWORDS = {
    "ICLR":    ["iclr", "learning representations"],
    "NEURIPS": ["neurips", "nips", "neural information processing"],
    "ICML":    ["icml", "machine learning"],
    "AAAI":    ["aaai"],
    "ACL":     ["acl", "computational linguistics"],
    "EMNLP":   ["emnlp"],
    "ICCV":    ["iccv", "international conference on computer vision"],
    "CVPR":    ["cvpr"],
    "ECCV":    ["eccv"],
}


def _paper_in_target_conf(paper: dict, target_confs: List[str]) -> bool:
    if not target_confs:
        return True
    venue = (paper.get("publication_venue") or paper.get("venue") or "").lower()
    for conf in target_confs:
        kws = _CONF_VENUE_KEYWORDS.get(conf.upper(), [conf.lower()])
        if any(kw in venue for kw in kws):
            return True
    return False


# ── Selector: LLM-based relevance scoring ────────────────────────────────────

_SELECTOR_BATCH_PROMPT = """\
You are an academic relevance judge. For each numbered paper below, score its \
relevance to the research query on a scale of 0.0 to 1.0, where:
  1.0 = directly answers the query
  0.7 = closely related, likely useful
  0.4 = tangentially related
  0.0 = unrelated

Research query: "{query}"

Papers:
{papers_block}

Respond with ONLY a JSON array of numbers in the same order as the papers.
Example: [0.9, 0.3, 0.7]
No explanation, no labels — just the JSON array."""


def _llm_selector_batch(query: str, papers: List[dict], llm) -> List[float]:
    """
    Score a batch of papers against `query` using the LLM.
    Returns a list of floats parallel to `papers`. Falls back to 0.5 on error.
    """
    if not papers or not llm:
        return [0.5] * len(papers)

    lines = []
    for i, p in enumerate(papers, 1):
        title    = (p.get("title") or "").strip()
        abstract = (p.get("abstract") or "").strip()
        # Summarise abstract if very long: first 600 chars covers the key claim
        if len(abstract) > 600:
            abstract = abstract[:600] + "…"
        lines.append(f"{i}. Title: {title}\n   Abstract: {abstract or '(not available)'}")

    papers_block = "\n\n".join(lines)
    prompt = _SELECTOR_BATCH_PROMPT.format(query=query, papers_block=papers_block)

    try:
        raw, _ = llm.call_llm_with_retry(
            user_message=prompt,
            system_message=(
                "You are a precise academic relevance scorer. "
                "Output ONLY a JSON array of floats — nothing else."
            ),
        )
        if raw is None:
            return [0.5] * len(papers)
        text = raw if isinstance(raw, str) else str(raw)
        # Extract JSON array robustly
        m = re.search(r"\[[\d\.,\s]+\]", text)
        if m:
            scores = [float(x) for x in re.findall(r"[\d\.]+", m.group())]
            # Pad or truncate to match input length
            scores = (scores + [0.5] * len(papers))[: len(papers)]
            return [max(0.0, min(1.0, s)) for s in scores]
    except Exception as exc:
        logger.debug(f"[Selector] LLM batch scoring failed: {exc}")
    return [0.5] * len(papers)


# ── Coverage Judge ────────────────────────────────────────────────────────────

_COVERAGE_PROMPT = """\
You are a research coverage judge.

Original query: "{query}"
Target conferences: {confs}

Current corpus: {n_papers} papers
  - Accepted at target venues: {n_accepted}
  - arXiv-only (not yet confirmed): {n_arxiv}
  - Papers added in this round: {n_new}
  - Rounds completed: {round_idx}

Sample of titles in corpus (most cited first):
{sample_titles}

Is the search coverage SUFFICIENT for a comprehensive literature review \
on the query? Consider: breadth of sub-topics, coverage of key methods, \
and diminishing returns (low n_new suggests saturation).

Answer with EXACTLY one word: YES or NO."""


def _llm_coverage_judge(
    query: str,
    confs: List[str],
    all_papers: List[dict],
    n_new: int,
    round_idx: int,
    llm,
) -> bool:
    """Returns True if LLM says coverage is sufficient, False to keep crawling."""
    if not llm:
        return False

    accepted = [p for p in all_papers if p.get("tier") in ("oral", "spotlight", "poster")]
    arxiv_only = [p for p in all_papers if p.get("tier") == "arxiv"]
    top_by_cit = sorted(all_papers, key=lambda p: p.get("citation_count", 0), reverse=True)[:15]
    sample_titles = "\n".join(
        f"  • {p.get('title','')[:80]} ({p.get('year','?')}, {p.get('tier','?')}, "
        f"{p.get('citation_count',0)} cit.)"
        for p in top_by_cit
    )
    prompt = _COVERAGE_PROMPT.format(
        query=query,
        confs=", ".join(confs) or "any",
        n_papers=len(all_papers),
        n_accepted=len(accepted),
        n_arxiv=len(arxiv_only),
        n_new=n_new,
        round_idx=round_idx,
        sample_titles=sample_titles,
    )
    try:
        raw, _ = llm.call_llm_with_retry(
            user_message=prompt,
            system_message="You are a coverage judge. Respond with ONLY 'YES' or 'NO'.",
        )
        if raw:
            text = raw if isinstance(raw, str) else str(raw)
            return text.strip().upper().startswith("YES")
    except Exception as exc:
        logger.debug(f"[Coverage Judge] {exc}")
    return False


# ── Main Crawler ──────────────────────────────────────────────────────────────

class CitationCrawler:
    """
    Iterative citation-traversal agent.

    Usage:
        crawler = CitationCrawler(config, s2_extractor)
        enriched = await crawler.crawl(seed_papers, query, conferences, year)
    """

    def __init__(self, config: dict, s2_extractor):
        crawler_cfg = config.get("crawler", {})
        self.max_rounds         = int(crawler_cfg.get("max_rounds", 2))
        self.max_total_papers   = int(crawler_cfg.get("max_total_papers", 300))
        self.expand_top_n       = int(crawler_cfg.get("expand_top_n", 20))
        self.selector_threshold = float(crawler_cfg.get("selector_threshold", 0.45))
        self.batch_llm_size     = int(crawler_cfg.get("batch_llm_size", 10))
        self.min_citations      = int(crawler_cfg.get("min_citations", 5))
        self.s2                 = s2_extractor
        self._llm               = None  # lazy

    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        try:
            from query_expander import _get_llm as _ql
            self._llm = _ql()
        except Exception:
            self._llm = False
        return self._llm

    async def crawl(
        self,
        seed_papers: List[dict],
        query: str,
        conferences: List[str],
        year: int,
        verbose: bool = True,
    ) -> List[dict]:
        """
        Run the iterative citation-traversal loop.
        Returns the seed corpus extended with high-relevance references.
        """
        llm = self._get_llm()
        loop = asyncio.get_event_loop()

        corpus: List[dict] = list(seed_papers)
        seen_ids: Set[str] = {
            p.get("id") or p.get("semantic_scholar_id") or ""
            for p in corpus
            if p.get("id") or p.get("semantic_scholar_id")
        }
        seen_ids.discard("")

        if verbose:
            logger.info(
                f"[Crawler] Starting — {len(corpus)} seed papers, "
                f"query='{query[:60]}', confs={conferences}, max_rounds={self.max_rounds}"
            )

        for round_idx in range(1, self.max_rounds + 1):
            if len(corpus) >= self.max_total_papers:
                logger.info(f"[Crawler] Round {round_idx}: corpus ceiling {self.max_total_papers} reached — stopping")
                break

            # ── Pick expand candidates ────────────────────────────────────────
            # Priority: accepted conf papers > high-citation arXiv > rest
            accepted = [p for p in corpus if p.get("tier") in ("oral", "spotlight", "poster")]
            high_cit = sorted(
                [p for p in corpus if p.get("tier") == "arxiv" and
                 (p.get("citation_count") or 0) >= 20],
                key=lambda p: p.get("citation_count", 0), reverse=True
            )
            candidates = (accepted + high_cit)[: self.expand_top_n]
            if not candidates:
                candidates = sorted(corpus, key=lambda p: p.get("citation_count", 0), reverse=True)[: self.expand_top_n]

            s2_ids = [
                p.get("semantic_scholar_id") or p.get("id", "")
                for p in candidates
                if (p.get("semantic_scholar_id") or p.get("id", "")).strip()
            ]
            s2_ids = [x for x in s2_ids if x]

            if not s2_ids:
                logger.info(f"[Crawler] Round {round_idx}: no S2 IDs available — stopping")
                break

            logger.info(f"[Crawler] Round {round_idx}: expanding refs for {len(s2_ids)} papers…")

            # ── Fetch references from S2 ──────────────────────────────────────
            refs: List[dict] = await self.s2.fetch_references(s2_ids)

            # ── Pre-filter: dedup, year, min citations, not already seen ─────
            novel_refs = []
            for ref in refs:
                rid = ref.get("id") or ref.get("semantic_scholar_id") or ""
                if rid and rid in seen_ids:
                    continue
                if not ref.get("title"):
                    continue
                if (ref.get("citation_count") or 0) < self.min_citations:
                    continue
                ref_year = ref.get("year") or 0
                if ref_year and (ref_year < year - 4 or ref_year > year + 1):
                    continue  # too old or future — skip
                novel_refs.append(ref)

            if not novel_refs:
                logger.info(f"[Crawler] Round {round_idx}: no novel references found — stopping")
                break

            # ── Selector: score all novel refs in batches ─────────────────────
            # Apply conference filter BEFORE scoring to skip obviously irrelevant refs
            if conferences:
                conf_refs = [r for r in novel_refs if _paper_in_target_conf(r, conferences)]
                other_refs = [r for r in novel_refs if not _paper_in_target_conf(r, conferences)]
            else:
                conf_refs = novel_refs
                other_refs = []

            # Score conference-confirmed refs (likely relevant — use selector)
            # Score other refs too but they need a higher threshold to pass
            to_score = conf_refs + other_refs
            scores: List[float] = []

            if llm and llm is not False:
                # Batch LLM scoring
                for i in range(0, len(to_score), self.batch_llm_size):
                    batch = to_score[i: i + self.batch_llm_size]
                    batch_scores = await loop.run_in_executor(
                        None, _llm_selector_batch, query, batch, llm
                    )
                    scores.extend(batch_scores)
            else:
                # Fallback: embedding cosine similarity
                scores = await self._embedding_scores(query, to_score, loop)

            # Apply threshold — stricter for non-conf papers
            threshold_conf  = self.selector_threshold
            threshold_other = min(0.85, self.selector_threshold + 0.25)
            n_conf = len(conf_refs)

            kept: List[dict] = []
            for i, (ref, score) in enumerate(zip(to_score, scores)):
                threshold = threshold_conf if i < n_conf else threshold_other
                if score >= threshold:
                    ref["selector_score"] = round(score, 3)
                    kept.append(ref)
                    rid = ref.get("id") or ref.get("semantic_scholar_id") or ""
                    if rid:
                        seen_ids.add(rid)

            logger.info(
                f"[Crawler] Round {round_idx}: {len(novel_refs)} refs → "
                f"{len(to_score)} unique → {len(kept)} kept (threshold={threshold_conf})"
            )

            corpus.extend(kept)
            n_new = len(kept)

            if n_new == 0:
                logger.info(f"[Crawler] Round {round_idx}: no new papers passed selector — stopping")
                break

            # ── Coverage judge: should we stop? ──────────────────────────────
            if llm and llm is not False and round_idx >= 1:
                sufficient = await loop.run_in_executor(
                    None, _llm_coverage_judge, query, conferences, corpus, n_new, round_idx, llm
                )
                if sufficient:
                    logger.info(f"[Crawler] Round {round_idx}: LLM Coverage Judge says SUFFICIENT — stopping")
                    break

        logger.info(
            f"[Crawler] Done — {len(seed_papers)} seed → {len(corpus)} total papers "
            f"({len(corpus) - len(seed_papers)} added via citation traversal)"
        )
        return corpus

    async def _embedding_scores(
        self, query: str, papers: List[dict], loop
    ) -> List[float]:
        """Fallback: cosine similarity between query embedding and title+abstract."""
        try:
            from embedding_store import _get_model
            model = _get_model()
            if model is None:
                return [0.5] * len(papers)

            def _compute():
                import numpy as np
                q_vec = model.embed_one(query)
                texts = [
                    (p.get("title", "") + ". " + (p.get("abstract") or "")).strip()
                    for p in papers
                ]
                vecs = model.embed(texts)   # (N, 1536) L2-normalised
                return (vecs @ q_vec).tolist()

            scores = await loop.run_in_executor(None, _compute)
            return [max(0.0, min(1.0, float(s))) for s in scores]
        except Exception as exc:
            logger.debug(f"[Crawler] Embedding fallback failed: {exc}")
            return [0.5] * len(papers)
