"""
paper_fetcher.py — Async full-text PDF fetcher and section extractor.

Downloads the PDF for a paper (via pdf_url → arXiv PDF → OpenReview PDF),
extracts Limitations / Future Work / Conclusion sections using PyMuPDF,
and caches the result in  cache/_fulltext/{paper_id}.json.

Result schema
──────────────
{
  "paper_id":         str,
  "title":            str,
  "limitations":      str,   # text under "Limitations" / "Limitations and Future Work"
  "future_work":      str,   # text under "Future Work" (if no combined section)
  "conclusion":       str,   # first ~1200 chars under "Conclusion"
  "llm_limitations":  list,  # [{"statement": str, "metric": str|null, "category": str}] — LLM-extracted specific limitations
  "source_url":       str,   # URL that yielded the PDF
  "fetched_at":       str,   # ISO-8601 UTC
}

All text fields are "" when the section was not found or the PDF could not
be downloaded.  An empty result is still cached so we don't re-try on every
request.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

_FULLTEXT_DIR  = "_fulltext"
_TIMEOUT       = httpx.Timeout(30.0, connect=10.0)
_MAX_PDF_BYTES = 25 * 1024 * 1024   # 25 MB hard cap – ML papers rarely exceed 10 MB

# ---------------------------------------------------------------------------
# Section-header regexes
# ---------------------------------------------------------------------------
# Match a line that IS a section header (possibly preceded by a decimal number)
_HDR = lambda title: re.compile(
    rf"^\s*(?:\d+[\.\d]*\s+)?{title}\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_PAT: dict[str, re.Pattern] = {
    "limitations": re.compile(
        r"^\s*(?:\d+[\.\d]*\s+)?(?:limitations?(?:\s+and\s+future\s+work)?|"
        r"future\s+work(?:\s+and\s+limitations?)?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "future_work": re.compile(
        r"^\s*(?:\d+[\.\d]*\s+)?(?:future\s+(?:work|directions?|research)|"
        r"open\s+problems?|discussion\s+and\s+future\s+work)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "conclusion": re.compile(
        r"^\s*(?:\d+[\.\d]*\s+)?(?:conclusions?|concluding\s+remarks?|"
        r"summary\s+and\s+conclusions?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "references": re.compile(
        r"^\s*(?:\d+[\.\d]*\s+)?references?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # Generic – any capitalised short line that looks like a section title
    "any_header": re.compile(
        r"^\s*(?:\d+[\.\d]*\s+)?[A-Z][A-Za-z &\-]{2,45}\s*$",
        re.MULTILINE,
    ),
}


def _extract_section(text: str, pat: re.Pattern, max_chars: int = 4000) -> str:
    """
    Find the first heading matched by `pat`, collect the body text up to
    the next section heading or References, and return it cleaned.
    """
    m = pat.search(text)
    if not m:
        return ""

    body = text[m.end():]

    # Earliest stop: References OR a new section header (at least 60 chars in)
    stop = len(body)
    for stopper in (_PAT["references"], _PAT["any_header"]):
        sm = stopper.search(body)
        if sm and sm.start() > 60:
            stop = min(stop, sm.start())

    extracted = body[:stop].strip()
    extracted = re.sub(r"\n{3,}", "\n\n", extracted)   # collapse blank lines
    return extracted[:max_chars]


# ---------------------------------------------------------------------------
# LLM-based limitation extraction
# ---------------------------------------------------------------------------
_LLM_EXTRACT_SYSTEM = (
    "You are a precise research analyst. Extract limitations from a research paper. "
    "Be specific — include exact numbers, thresholds, model names, and dataset names "
    "mentioned in the text. Never write generic statements like 'limited scalability' "
    "without the concrete numbers that justify the claim."
)

_LLM_EXTRACT_PROMPT = """\
Paper title: "{title}"

Below is text from this paper (limitations section, future work, conclusion, or final pages):

{text}

Extract 3-6 genuine, specific limitations from this text.
For each limitation:
- State the concrete constraint with exact numbers/metrics if mentioned (e.g. context window ≤48k tokens fails on 3-hop tasks; evaluated only on GSM8K and MATH, not code)
- Note the category: one of: scalability | data | evaluation | compute | generalization | theory | reproducibility | scope
- Note any metric that quantifies the limitation (e.g. "accuracy drops from 87% to 63% beyond 5 steps")

Respond ONLY with a JSON array, no other text:
[
  {{"statement": "<specific limitation>", "metric": "<exact figure or null>", "category": "<category>"}},
  ...
]"""


def _llm_extract_limitations(title: str, text: str, llm) -> list[dict]:
    """
    Use LLM to extract specific, quantified limitations from paper text.
    Returns a list of {statement, metric, category} dicts.
    Falls back to [] on any error.
    """
    if not text or not text.strip():
        return []
    prompt = _LLM_EXTRACT_PROMPT.format(title=title, text=text[:8000])
    try:
        raw, _ = llm.call_llm_with_retry(
            user_message=prompt,
            system_message=_LLM_EXTRACT_SYSTEM,
        )
        if not raw:
            return []
        # Strip markdown code fences if present
        clean = re.sub(r"```(?:json)?\s*", "", str(raw)).strip().rstrip("`")
        parsed = json.loads(clean)
        if isinstance(parsed, list):
            return [
                {
                    "statement": str(item.get("statement", "")).strip(),
                    "metric":    item.get("metric") or None,
                    "category":  str(item.get("category", "scope")).strip(),
                }
                for item in parsed
                if isinstance(item, dict) and item.get("statement")
            ]
    except Exception as exc:
        logger.debug(f"[PaperFetcher] LLM extract failed for {title[:50]!r}: {exc}")
    return []


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------
def _pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    """Convert raw PDF bytes → plain text via PyMuPDF."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return "\n".join(pages)
    except Exception as exc:
        logger.warning(f"[PaperFetcher] PyMuPDF parse error: {exc}")
        return ""


def _arxiv_pdf_url(paper: dict) -> str | None:
    """Construct the arXiv PDF URL from arxiv_id or arxiv_url."""
    arxiv_id = paper.get("arxiv_id")
    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}"
    arxiv_url = paper.get("arxiv_url", "")
    if arxiv_url:
        return arxiv_url.replace("/abs/", "/pdf/").rstrip("/")
    return None


def _candidate_pdf_urls(paper: dict) -> list[str]:
    """Return ordered list of PDF URL candidates for a paper."""
    seen: set[str] = set()
    candidates: list[str] = []

    def _add(url: str | None):
        if url and url not in seen:
            seen.add(url)
            candidates.append(url)

    _add(paper.get("pdf_url"))
    _add(_arxiv_pdf_url(paper))
    # OpenReview: forum URL → PDF URL
    or_url = paper.get("openreview_url", "")
    if or_url:
        _add(or_url.replace("/forum?id=", "/pdf?id=").replace("/forum/", "/pdf/"))
    return candidates


# ---------------------------------------------------------------------------
# PaperFetcher
# ---------------------------------------------------------------------------
class PaperFetcher:
    """
    Async HTTP PDF fetcher with per-paper disk cache.

    Cache layout:
        <cache_dir>/_fulltext/<safe_paper_id>.json
    """

    def __init__(self, cache_dir: Path):
        self._ft_dir = Path(cache_dir) / _FULLTEXT_DIR
        self._ft_dir.mkdir(parents=True, exist_ok=True)

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _cache_path(self, paper_id: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", paper_id)[:120]
        return self._ft_dir / f"{safe}.json"

    def _load(self, paper_id: str) -> dict | None:
        p = self._cache_path(paper_id)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _save(self, paper_id: str, result: dict):
        try:
            self._cache_path(paper_id).write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning(f"[PaperFetcher] Cache write failed ({paper_id}): {exc}")

    # ── Main fetch ────────────────────────────────────────────────────────────

    async def fetch(self, paper: dict) -> dict:
        """
        Download + parse the paper's PDF.
        Returns a dict with: paper_id, title, limitations, future_work,
        conclusion, source_url, fetched_at.
        All text fields are '' if the section was not found.
        """
        paper_id = (
            paper.get("id")
            or paper.get("semantic_scholar_id")
            or paper.get("arxiv_id")
            or ""
        )
        title = paper.get("title", "")

        # ── Cache hit ─────────────────────────────────────────────────────────
        if paper_id:
            cached = self._load(paper_id)
            if cached is not None:
                logger.debug(f"[PaperFetcher] Disk cache hit: {paper_id!r}")
                return cached

        result: dict = {
            "paper_id":        paper_id,
            "title":           title,
            "limitations":     "",
            "future_work":     "",
            "conclusion":      "",
            "llm_limitations": [],
            "source_url":      "",
            "fetched_at":      _now_iso(),
        }

        urls = _candidate_pdf_urls(paper)
        if not urls:
            logger.debug(f"[PaperFetcher] No URL candidates for {paper_id!r} / {title[:60]!r}")
            if paper_id:
                self._save(paper_id, result)
            return result

        # ── Download PDF ──────────────────────────────────────────────────────
        pdf_bytes: bytes | None = None
        used_url = ""

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=_TIMEOUT,
            headers={"User-Agent": "ConferencePapers-Researcher/1.0"},
        ) as client:
            for url in urls:
                try:
                    logger.debug(f"[PaperFetcher] Requesting {url}")
                    resp = await client.get(url)
                    if resp.status_code == 200 and resp.content[:4] == b"%PDF":
                        if len(resp.content) <= _MAX_PDF_BYTES:
                            pdf_bytes = resp.content
                            used_url  = url
                            break
                        else:
                            logger.debug(
                                f"[PaperFetcher] PDF too large "
                                f"({len(resp.content)/1e6:.1f} MB), skipping: {url}"
                            )
                except Exception as exc:
                    logger.debug(f"[PaperFetcher] HTTP error for {url}: {exc}")

        if not pdf_bytes:
            logger.info(
                f"[PaperFetcher] Could not download PDF "
                f"for {paper_id!r} / {title[:60]!r}"
            )
            if paper_id:
                self._save(paper_id, result)
            return result

        # ── Parse sections ────────────────────────────────────────────────────
        full_text = _pdf_bytes_to_text(pdf_bytes)
        if not full_text.strip():
            if paper_id:
                self._save(paper_id, result)
            return result

        # "Limitations and Future Work" combined section → goes into limitations
        limitations_text = _extract_section(full_text, _PAT["limitations"], max_chars=4000)
        # Standalone "Future Work" only if no combined section found
        future_work_text = (
            ""
            if limitations_text
            else _extract_section(full_text, _PAT["future_work"], max_chars=3000)
        )
        conclusion_text = _extract_section(full_text, _PAT["conclusion"], max_chars=1500)

        result.update({
            "limitations": limitations_text,
            "future_work": future_work_text,
            "conclusion":  conclusion_text,
            "source_url":  used_url,
        })

        # ── LLM extraction of specific, quantified limitations ────────────────
        if llm and llm is not False:
            # Prefer the explicit limitation section; fall back to conclusion + last pages
            lim_source = limitations_text or future_work_text
            if not lim_source:
                # Use conclusion + ~3k chars from end-of-document as proxy
                end_text = full_text[-4000:] if len(full_text) > 4000 else full_text
                lim_source = (conclusion_text + "\n" + end_text).strip()
            result["llm_limitations"] = _llm_extract_limitations(title, lim_source, llm)
            logger.debug(
                f"[PaperFetcher] LLM extracted {len(result['llm_limitations'])} "
                f"limitations for {paper_id!r}"
            )

        logger.info(
            f"[PaperFetcher] {paper_id!r} — "
            f"lim={len(limitations_text)}c  fw={len(future_work_text)}c  "
            f"conc={len(conclusion_text)}c  url={used_url[:60]}"
        )

        if paper_id:
            self._save(paper_id, result)
        return result

    # ── Batch ─────────────────────────────────────────────────────────────────

    async def fetch_many(
        self,
        papers: list[dict],
        concurrency: int = 5,
        llm=None,
    ) -> dict[str, dict]:
        """
        Fetch full text for multiple papers concurrently.
        Returns {paper_id: result_dict}.
        Pass llm to enable LLM-based extraction of specific limitations.
        Errors per-paper are logged and skipped (never raise).
        """
        sem = asyncio.Semaphore(concurrency)

        async def _bounded(p: dict) -> dict:
            async with sem:
                return await self.fetch(p, llm=llm)

        results = await asyncio.gather(
            *[_bounded(p) for p in papers],
            return_exceptions=True,
        )
        out: dict[str, dict] = {}
        for p, res in zip(papers, results):
            pid = (
                p.get("id")
                or p.get("semantic_scholar_id")
                or p.get("arxiv_id")
                or ""
            )
            if isinstance(res, Exception):
                logger.warning(f"[PaperFetcher] Unhandled error for {pid!r}: {res}")
            elif isinstance(res, dict) and pid:
                out[pid] = res
        return out

    # ── Cache inspection ──────────────────────────────────────────────────────

    def cached_ids(self) -> list[str]:
        return [p.stem for p in self._ft_dir.glob("*.json")]

    def delete_cached(self, paper_id: str) -> bool:
        p = self._cache_path(paper_id)
        if p.exists():
            p.unlink()
            return True
        return False


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
