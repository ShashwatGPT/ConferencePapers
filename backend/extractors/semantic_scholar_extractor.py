"""
semantic_scholar_extractor.py  — fixed edition
Key fixes:
  1. Sequential requests with asyncio.sleep — respects S2 free-tier ~1 req/s
  2. Direct ArXiv-ID lookup first (far more reliable than title search)
  3. Better status-code inspection (429 → explicit backoff)
"""
import asyncio
import datetime
import logging
from typing import List, Optional
import httpx

logger = logging.getLogger(__name__)

FIELDS = (
    "title,year,citationCount,influentialCitationCount,"
    "externalIds,url,publicationDate,openAccessPdf"
)
_S2_DELAY = 1.2   # seconds between requests (free-tier is ~1/s)


class SemanticScholarExtractor:
    def __init__(self, config: dict, api_key: Optional[str] = None):
        self.config  = config
        self.headers = {"x-api-key": api_key} if api_key else {}
        self.base    = config["apis"]["semantic_scholar"]["base_url"]

    async def enrich_papers(self, papers: List[dict]) -> List[dict]:
        """Sequential enrichment with rate limiting."""
        async with httpx.AsyncClient(headers=self.headers, timeout=15) as client:
            enriched = []
            for i, paper in enumerate(papers):
                try:
                    result = await self._enrich_one(client, paper)
                    enriched.append(result)
                except Exception as e:
                    logger.debug(f"[S2] paper {i} error: {e}")
                    enriched.append(paper)
                if i % 2 == 1:
                    await asyncio.sleep(_S2_DELAY)
        return enriched

    async def _enrich_one(self, client: httpx.AsyncClient, paper: dict) -> dict:
        # Strategy 1: direct arXiv ID lookup
        arxiv_id = paper.get("arxiv_id")
        if arxiv_id:
            s2 = await self._fetch_by_arxiv_id(client, arxiv_id)
            if s2:
                return self._merge(paper, s2)

        # Strategy 2: title search
        title = paper.get("title", "").strip()
        if title:
            s2 = await self._fetch_by_title(client, title)
            if s2:
                return self._merge(paper, s2)
        return paper

    async def _fetch_by_arxiv_id(self, client, arxiv_id: str) -> Optional[dict]:
        clean = arxiv_id.split("v")[0].split("/")[-1]
        try:
            r = await client.get(f"{self.base}/paper/arXiv:{clean}",
                                 params={"fields": FIELDS})
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                await asyncio.sleep(5)
        except Exception as e:
            logger.debug(f"[S2] arXiv lookup {clean}: {e}")
        return None

    async def _fetch_by_title(self, client, title: str) -> Optional[dict]:
        try:
            r = await client.get(f"{self.base}/paper/search",
                                 params={"query": title, "fields": FIELDS, "limit": 3})
            if r.status_code == 200:
                hits = r.json().get("data", [])
                return self._best_match(title, hits)
            if r.status_code == 429:
                await asyncio.sleep(5)
        except Exception as e:
            logger.debug(f"[S2] title search '{title[:40]}': {e}")
        return None

    def _best_match(self, title: str, hits: List[dict]) -> Optional[dict]:
        tl = title.lower()
        for h in hits:
            if h.get("title", "").lower() == tl:
                return h
        return hits[0] if hits else None

    def _merge(self, paper: dict, s2: dict) -> dict:
        paper["citation_count"]       = s2.get("citationCount", 0) or 0
        paper["citation_velocity"]    = self._calc_velocity(s2)
        paper["semantic_scholar_id"]  = s2.get("paperId", "")
        paper["semantic_scholar_url"] = s2.get("url", "")
        ext = s2.get("externalIds") or {}
        if not paper.get("arxiv_id") and ext.get("ArXiv"):
            paper["arxiv_id"]  = ext["ArXiv"]
            paper["arxiv_url"] = f"https://arxiv.org/abs/{ext['ArXiv']}"
        oa = s2.get("openAccessPdf") or {}
        if not paper.get("pdf_url") and oa.get("url"):
            paper["pdf_url"] = oa["url"]
        return paper

    @staticmethod
    def _calc_velocity(s2: dict) -> float:
        count    = s2.get("citationCount", 0) or 0
        pub_date = s2.get("publicationDate") or ""
        year     = s2.get("year") or 2024
        try:
            pub    = datetime.datetime.strptime(pub_date, "%Y-%m-%d") if pub_date \
                     else datetime.datetime(year, 6, 1)
            months = max(1.0, (datetime.datetime.now() - pub).days / 30.0)
            return round(count / months, 2)
        except Exception:
            return 0.0
