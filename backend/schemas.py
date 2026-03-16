"""
schemas.py – Pydantic data models for ConferencePapers.
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class ReviewerStats(BaseModel):
    avg_rating: float = 0.0
    avg_confidence: float = 0.0
    num_reviews: int = 0
    ratings: List[float] = []


class Paper(BaseModel):
    id: str
    title: str
    abstract: str = ""
    authors: List[str] = []
    year: int
    venue: str = "ICLR"
    tier: str = "poster"               # oral | spotlight | poster | rejected
    decision: str = "Accept (poster)"
    openreview_url: str = ""
    arxiv_id: Optional[str] = None
    arxiv_url: Optional[str] = None
    pdf_url: Optional[str] = None
    semantic_scholar_id: Optional[str] = None
    citation_count: int = 0
    citation_velocity: float = 0.0     # citations / months since published
    github_stars: int = 0
    github_url: Optional[str] = None
    impact_niche_score: float = 0.0    # (velocity * stars) / density
    iclr_flavor_score: float = 0.0    # 0-1, higher = better fit for ICLR
    keywords: List[str] = []
    relevance_score: float = 0.0       # keyword match score for focus query
    reviewer_stats: ReviewerStats = Field(default_factory=ReviewerStats)
    embedding_x: float = 0.0           # UMAP x for landscape map
    embedding_y: float = 0.0           # UMAP y for landscape map
    cluster_id: int = 0
    cluster_label: str = ""
    fatal_flaws: List[str] = []        # extracted from low-rating reviews
    pivot_suggestion: str = ""


class LandscapeData(BaseModel):
    papers: List[Paper]
    cluster_summary: dict = {}


class FunnelData(BaseModel):
    """
    Sankey nodes & links: Submission -> Tier / Rejection
    """
    nodes: List[dict]
    links: List[dict]
    year_breakdown: dict = {}


class PivotRequest(BaseModel):
    idea: str
    conference: str = "ICLR"
    year: int = 2025


class PivotResponse(BaseModel):
    original_idea: str
    iclr_flavor_score: float
    verdict: str
    fatal_flaw: str
    pivot_suggestion: str
    theoretical_framing: str
    similar_accepted_papers: List[Paper] = []


class SearchResponse(BaseModel):
    total: int
    papers: List[Paper]
    query: str
    years: List[int]
