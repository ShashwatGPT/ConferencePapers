"""
arxiv_extractor.py
Searches arXiv for Agents-for-Productivity papers to supplement OpenReview data.
"""
import logging
from typing import List
import arxiv

logger = logging.getLogger(__name__)


class ArxivExtractor:
    def __init__(self, config: dict):
        self.config = config
        self.client = arxiv.Client()

    def search(self, query: str, max_results: int = 50) -> List[dict]:
        """Search arXiv and return structured paper dicts."""
        full_query = f"({query}) AND (cat:cs.AI OR cat:cs.LG OR cat:cs.CL)"
        search = arxiv.Search(
            query=full_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        papers = []
        try:
            for result in self.client.results(search):
                papers.append({
                    "id": result.entry_id.split("/")[-1],
                    "title": result.title,
                    "abstract": result.summary,
                    "authors": [str(a) for a in result.authors],
                    "year": result.published.year,
                    "arxiv_id": result.entry_id.split("/")[-1],
                    "arxiv_url": result.entry_id,
                    "pdf_url": result.pdf_url,
                    "keywords": result.categories,
                    "decision": "arXiv",
                    "tier": "arxiv",
                    "openreview_url": "",
                    "reviews": [],
                })
        except Exception as e:
            logger.warning(f"[arXiv] Search error: {e}")

        logger.info(f"[arXiv] Found {len(papers)} papers for query: {query}")
        return papers

    def enrich_paper(self, paper: dict) -> dict:
        """If an arxiv_id is known, fetch full metadata."""
        arxiv_id = paper.get("arxiv_id")
        if not arxiv_id:
            return paper
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            result = next(self.client.results(search))
            paper["pdf_url"] = result.pdf_url
            paper["abstract"] = result.summary or paper.get("abstract", "")
            if not paper.get("authors"):
                paper["authors"] = [str(a) for a in result.authors]
        except Exception as e:
            logger.debug(f"[arXiv] Enrich error for {arxiv_id}: {e}")
        return paper
