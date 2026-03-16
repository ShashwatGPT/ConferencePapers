"""
main.py – ConferencePapers FastAPI backend
Serves ICLR 2024-2026 "Agents for Productivity" data + analysis.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ── Project imports ──────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from schemas import Paper, ReviewerStats, SearchResponse, FunnelData, LandscapeData, PivotRequest, PivotResponse
from scorer import (
    compute_relevance,
    compute_iclr_flavor,
    compute_impact_niche,
    compute_reviewer_stats,
    extract_fatal_flaws,
    compute_clusters_and_embeddings,
    detect_white_spaces,
    generate_pivot,
)
from embedding_store import EmbeddingStore
from query_expander import expand_query
from extractors.openreview_extractor import OpenReviewExtractor
from extractors.semantic_scholar_extractor import SemanticScholarExtractor
from extractors.arxiv_extractor import ArxivExtractor

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(name)s │ %(message)s")
logger = logging.getLogger("main")

# ── Load config ───────────────────────────────────────────────────────────────
CONFIG_PATH = ROOT / "config.json"
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

# ── Disk cache (Obsidian-style .md files) ─────────────────────────────────────
from cache_store import CacheStore
_cache_cfg  = CONFIG.get("cache", {})
_cache_dir  = (ROOT / _cache_cfg.get("dir", "../cache")).resolve()
_cache_ttl  = int(_cache_cfg.get("ttl_days", 7))
_cache_on   = bool(_cache_cfg.get("enabled", True))
CACHE       = CacheStore(_cache_dir, ttl_days=_cache_ttl)

from embedding_store import EmbeddingStore
EMBEDS      = EmbeddingStore(_cache_dir)

import asyncio
import re as _re

KEYWORDS      = CONFIG["focus"]["keywords"]
ANTI_KEYWORDS = CONFIG["focus"]["anti_keywords"]
YEARS         = list(range(2020, 2028))
DEFAULT_YEARS = [2024, 2025, 2026]
CONFERENCES   = ["ICLR", "ICML", "NeurIPS", "AAAI", "ACL", "EMNLP"]

# ── In-memory cache + per-key locks (prevents thundering herd) ──────────────
# _PAPER_CACHE: read/written only under the corresponding asyncio.Lock.
# Python dict reads/writes are GIL-atomic for simple assignments, but the
# check-then-fetch pattern is not — hence one Lock per cache key.
_PAPER_CACHE: dict           = {}   # cache_key → List[Paper]
_FETCH_LOCKS: dict           = {}   # cache_key → asyncio.Lock
_FETCH_LOCKS_GUARD           = asyncio.Lock()  # protects _FETCH_LOCKS creation

async def _get_or_create_lock(key: str) -> asyncio.Lock:
    """Return (creating if needed) the asyncio.Lock for a specific cache key."""
    # Fast path — no global lock needed after first creation
    lock = _FETCH_LOCKS.get(key)
    if lock:
        return lock
    async with _FETCH_LOCKS_GUARD:
        # Re-check under the guard (another coroutine may have created it)
        if key not in _FETCH_LOCKS:
            _FETCH_LOCKS[key] = asyncio.Lock()
        return _FETCH_LOCKS[key]

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="ConferencePapers – Agents for Productivity",
    description="Discovery & strategy engine for ICLR 2024-2026.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend ────────────────────────────────────────────────────────────
FRONTEND_DIR = ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def serve_frontend():
        html_path = FRONTEND_DIR / "index.html"
        return HTMLResponse(content=html_path.read_text(), status_code=200)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _raw_to_paper(raw: dict, accepted_density: float = 1.0) -> Paper:
    """Convert a raw dict (from extractor) to a typed Paper model."""
    reviews = raw.get("reviews", [])
    rstats  = compute_reviewer_stats(reviews)
    flaws   = extract_fatal_flaws(reviews)

    return Paper(
        id                  = raw.get("id", ""),
        title               = raw.get("title", ""),
        abstract            = raw.get("abstract", ""),
        authors             = raw.get("authors", [])[:6],
        year                = raw.get("year", 2024),
        venue               = raw.get("venue", "ICLR"),
        tier                = raw.get("tier", "poster"),
        decision            = raw.get("decision", "Accept (poster)"),
        openreview_url      = raw.get("openreview_url", ""),
        arxiv_id            = raw.get("arxiv_id"),
        arxiv_url           = raw.get("arxiv_url"),
        pdf_url             = raw.get("pdf_url"),
        semantic_scholar_id = raw.get("semantic_scholar_id"),
        citation_count      = raw.get("citation_count", 0),
        citation_velocity   = raw.get("citation_velocity", 0.0),
        github_stars        = raw.get("github_stars", 0),
        github_url          = raw.get("github_url"),
        impact_niche_score  = compute_impact_niche(raw, accepted_density),
        iclr_flavor_score   = compute_iclr_flavor(raw, CONFIG),
        keywords            = raw.get("keywords", []),
        relevance_score     = compute_relevance(raw, KEYWORDS, ANTI_KEYWORDS),
        reviewer_stats      = ReviewerStats(**rstats),
        embedding_x         = raw.get("embedding_x", 0.0),
        embedding_y         = raw.get("embedding_y", 0.0),
        cluster_id          = raw.get("cluster_id", 0),
        cluster_label       = raw.get("cluster_label", ""),
        fatal_flaws         = flaws,
    )


FOCUS_TOPIC = CONFIG["focus"]["topic"]


def _mem_key(conference: str, year: int, topic: str) -> str:
    slug = _re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")
    return f"{conference.upper()}_{year}_{slug}"


async def _fetch_year_papers(
    year: int,
    conference: str = "ICLR",
    topic: str | None = None,
    force_refresh: bool = False,
) -> List[Paper]:
    """
    Fetch pipeline with two-level caching:
      L1 - in-memory dict (_PAPER_CACHE), guarded per key by asyncio.Lock
      L2 - Obsidian .md files on disk (CacheStore) — read/written in executor
      L3 - live API calls (OpenReview → arXiv fallback → S2 enrichment)

    Concurrency safety:
      - One asyncio.Lock per (conf, year, topic) key prevents parallel
        coroutines from both entering L3 for the same query (thundering herd).
      - All blocking disk I/O (CACHE.get/put) and CPU-heavy work
        (compute_clusters_and_embeddings) run in the default thread-pool
        executor so the event loop is never blocked.
    """
    if topic is None:
        topic = FOCUS_TOPIC
    cache_key = _mem_key(conference, year, topic)
    loop = asyncio.get_event_loop()

    # ── L1: in-memory (check before acquiring lock — fast path) ────────────
    if not force_refresh and cache_key in _PAPER_CACHE:
        logger.debug(f"[L1 HIT] {conference} {year}")
        return _PAPER_CACHE[cache_key]

    # Acquire per-key lock — only ONE coroutine does L2/L3 at a time per key
    key_lock = await _get_or_create_lock(cache_key)
    async with key_lock:
        # Re-check L1 inside the lock (another coroutine may have populated it)
        if not force_refresh and cache_key in _PAPER_CACHE:
            logger.debug(f"[L1 HIT-after-lock] {conference} {year}")
            return _PAPER_CACHE[cache_key]

        # ── L2: disk cache (blocking read in executor) ───────────────────────
        if _cache_on and not force_refresh:
            cached_raw = await loop.run_in_executor(
                None, CACHE.get, conference, year, topic
            )
            if cached_raw:
                accepted_density = max(len(cached_raw), 1)
                papers = [_raw_to_paper(r, accepted_density) for r in cached_raw]
                papers.sort(key=lambda p: p.relevance_score, reverse=True)
                _PAPER_CACHE[cache_key] = papers   # atomic dict assignment (GIL)
                logger.info(f"[L2 HIT] {conference} {year} — {len(papers)} papers")
                if not EMBEDS.exists(cache_key):
                    asyncio.create_task(_build_embeddings_bg(cache_key, cached_raw))
                return papers

        # ── L3: live fetch ───────────────────────────────────────────────────
        logger.info(f"[L3 FETCH] {conference} {year} / {topic} — calling APIs ...")
        raw_papers: List[dict] = []

        # 3a. OpenReview (for conferences that use it)
        _OR_CONFERENCES = {"ICLR", "ICML", "NEURIPS", "NeurIPS"}
        if conference.upper() in {c.upper() for c in _OR_CONFERENCES}:
            or_extractor = OpenReviewExtractor(CONFIG, conference=conference)
            try:
                # OpenReview SDK is synchronous — run in executor
                raw_papers = await loop.run_in_executor(
                    None, or_extractor.fetch_accepted_papers, year
                )
            except KeyError as e:
                logger.info(f"[OpenReview] No venue config — {e}. Will use arXiv.")
            except Exception as e:
                logger.warning(f"[OpenReview] Failed for {conference} {year}: {e}")

        # 3b. arXiv fallback / supplement
        if not raw_papers:
            logger.info(f"[L3] Using arXiv for {conference} {year}")
            ax_extractor = ArxivExtractor(CONFIG)
            kw_parts = []
            for kw in KEYWORDS:
                clean = kw.strip()
                if not clean:
                    continue
                if " " in clean:
                    kw_parts.append(f'ti:"{clean}" OR abs:"{clean}"')
                else:
                    kw_parts.append(f'ti:{clean} OR abs:{clean}')
            focus_q = " OR ".join(kw_parts)
            year_q  = f"submittedDate:[{year}0101000000 TO {year}1231235959]"
            full_q  = f"({focus_q}) AND ({year_q})"
            max_res = CONFIG.get("apis", {}).get("arxiv", {}).get("max_results", 200)
            # arXiv client is synchronous — run in executor
            raw_papers = await loop.run_in_executor(
                None, ax_extractor.search, full_q, max_res
            )
            logger.info(f"[arXiv] Got {len(raw_papers)} papers before relevance filter")

        # 3c. Semantic Scholar enrichment (already async, uses httpx)
        s2 = SemanticScholarExtractor(CONFIG)
        try:
            raw_papers = await s2.enrich_papers(raw_papers)
        except Exception as e:
            logger.warning(f"[S2] Enrichment failed: {e}")

        # 3d. Filter + cluster (CPU-heavy — run in executor)
        def _cluster_and_filter(papers):
            relevant = [p for p in papers
                        if compute_relevance(p, KEYWORDS, ANTI_KEYWORDS) > 0.05]
            if not relevant:
                relevant = papers
            return compute_clusters_and_embeddings(
                relevant, n_clusters=min(8, max(2, len(relevant) // 3))
            )

        relevant = await loop.run_in_executor(None, _cluster_and_filter, raw_papers)
        accepted_density = max(len(relevant), 1)

        # ── Write to disk cache (L2) — blocking, run in executor ────────────
        if _cache_on and relevant:
            try:
                await loop.run_in_executor(None, CACHE.put, conference, year, topic, relevant)
            except Exception as e:
                logger.warning(f"[Cache WRITE ERR] {e}")

        # ── Build embeddings in background — non-blocking ───────────────────
        asyncio.create_task(_build_embeddings_bg(cache_key, relevant))

        # ── Populate L1 (atomic assignment) ──────────────────────────────────
        papers = [_raw_to_paper(r, accepted_density) for r in relevant]
        papers.sort(key=lambda p: p.relevance_score, reverse=True)
        _PAPER_CACHE[cache_key] = papers
        logger.info(f"[L3→Cache] {conference} {year} — {len(papers)} papers stored")
        return papers


async def _build_embeddings_bg(cache_key: str, raw_papers: List[dict]):
    """Background task: build embedding index in a thread so the event loop is free."""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, EMBEDS.build, cache_key, raw_papers)
        logger.debug(f"[Embeddings] Built index for {cache_key}")
    except Exception as e:
        logger.debug(f"[Embeddings] Background build failed for {cache_key}: {e}")


async def _all_papers(
    conferences: List[str],
    year_min: int,
    year_max: int,
    topic: str | None = None,
) -> List[Paper]:
    """
    Fetch all (conference × year) combos concurrently with asyncio.gather.
    Each combo is independently lock-guarded inside _fetch_year_papers, so
    there are no races between parallel fetches for different keys, and no
    duplicate L3 fetches for the same key.
    """
    topic = topic or FOCUS_TOPIC
    tasks = [
        _fetch_year_papers(year, conference=conf, topic=topic)
        for conf in conferences
        for year in range(year_min, year_max + 1)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    papers: List[Paper] = []
    for r in results:
        if isinstance(r, Exception):
            logger.warning(f"[_all_papers] task error: {r}")
        else:
            papers.extend(r)
    return papers


# ── API Routes ────────────────────────────────────────────────────────────────

@app.get("/api/config", tags=["Config"])
async def get_config():
    """Return available conferences, current topic, year range for the frontend UI."""
    return {
        "conferences": CONFERENCES,
        "default_conferences": ["ICLR"],
        "year_min": 2020,
        "year_max": 2027,
        "default_year_min": 2024,
        "default_year_max": 2026,
        "default_topic": FOCUS_TOPIC,
        "keywords": KEYWORDS[:20],
    }


@app.get("/api/papers", response_model=SearchResponse, tags=["Papers"])
async def get_papers(
    conferences: str  = Query("ICLR", description="Comma-separated conferences"),
    year_min:    int  = Query(2024, ge=2020, le=2027),
    year_max:    int  = Query(2026, ge=2020, le=2027),
    topic:       str  = Query("", description="Research area / topic (empty = use default)"),
    tier:        str  = Query("all", description="all|oral|spotlight|poster|oral+spotlight"),
    sort_by:     str  = Query("relevance"),
    min_relevance: float = Query(0.0, ge=0.0, le=1.0),
    limit:       int  = Query(200, ge=1, le=500),
):
    conf_list  = [c.strip() for c in conferences.split(",") if c.strip()]
    real_topic = topic.strip() or FOCUS_TOPIC
    all_papers = await _all_papers(conf_list, year_min, year_max, real_topic)

    # Tier filter — accepts comma-separated list e.g. "oral,spotlight"
    tier_lower = tier.lower().strip()
    if tier_lower not in ("all", ""):
        allowed = {t.strip() for t in tier_lower.replace("+", ",").split(",") if t.strip()}
        all_papers = [p for p in all_papers if p.tier in allowed]

    if min_relevance > 0:
        all_papers = [p for p in all_papers if p.relevance_score >= min_relevance]

    sort_map = {
        "relevance":   lambda p: p.relevance_score,
        "impact":      lambda p: p.impact_niche_score,
        "citation":    lambda p: p.citation_count,
        "iclr_flavor": lambda p: p.iclr_flavor_score,
        "year":        lambda p: p.year,
    }
    all_papers.sort(key=sort_map.get(sort_by, sort_map["relevance"]), reverse=True)

    return SearchResponse(
        total=len(all_papers),
        papers=all_papers[:limit],
        query=real_topic,
        years=list(range(year_min, year_max + 1)),
    )


@app.get("/api/papers/cluster/{cluster_id}", tags=["Papers"])
async def get_papers_by_cluster(
    cluster_id:  int,
    conferences: str = Query("ICLR"),
    year_min:    int = Query(2024),
    year_max:    int = Query(2026),
    topic:       str = Query(""),
):
    """Return all papers in a specific cluster (for landscape map click-through)."""
    conf_list  = [c.strip() for c in conferences.split(",") if c.strip()]
    real_topic = topic.strip() or FOCUS_TOPIC
    all_papers = await _all_papers(conf_list, year_min, year_max, real_topic)
    filtered   = [p for p in all_papers if p.cluster_id == cluster_id]
    filtered.sort(key=lambda p: p.impact_niche_score, reverse=True)
    label = filtered[0].cluster_label if filtered else f"Cluster {cluster_id}"
    return {"cluster_id": cluster_id, "label": label, "total": len(filtered), "papers": filtered}


@app.get("/api/paper/{paper_id}", response_model=Paper, tags=["Papers"])
async def get_paper(paper_id: str):
    for key, papers in _PAPER_CACHE.items():
        for p in papers:
            if p.id == paper_id:
                return p
    raise HTTPException(status_code=404, detail="Paper not found")


@app.get("/api/landscape", tags=["Visualisation"])
async def get_landscape(
    conferences: str  = Query("ICLR"),
    year_min:    int  = Query(2024),
    year_max:    int  = Query(2026),
    topic:       str  = Query(""),
    min_relevance: float = Query(0.0),
):
    conf_list  = [c.strip() for c in conferences.split(",") if c.strip()]
    real_topic = topic.strip() or FOCUS_TOPIC
    all_papers = await _all_papers(conf_list, year_min, year_max, real_topic)

    if min_relevance > 0:
        all_papers = [p for p in all_papers if p.relevance_score >= min_relevance]

    cluster_summary: dict = {}
    for p in all_papers:
        cid = p.cluster_id
        if cid not in cluster_summary:
            cluster_summary[cid] = {"label": p.cluster_label, "count": 0, "avg_citations": 0.0, "papers": []}
        cluster_summary[cid]["count"] += 1
        cluster_summary[cid]["avg_citations"] += p.citation_count
        cluster_summary[cid]["papers"].append(p.id)

    for cid, info in cluster_summary.items():
        n = max(info["count"], 1)
        info["avg_citations"] = round(info["avg_citations"] / n, 1)
        info["papers"] = info["papers"][:5]

    return {
        "points": [
            {
                "id":             p.id,
                "title":          p.title,
                "x":              p.embedding_x,
                "y":              p.embedding_y,
                "cluster_id":     p.cluster_id,
                "cluster_label":  p.cluster_label,
                "tier":           p.tier,
                "year":           p.year,
                "conference":     p.venue,
                "relevance":      p.relevance_score,
                "impact":         p.impact_niche_score,
                "iclr_flavor":    p.iclr_flavor_score,
                "citation_count": p.citation_count,
                "openreview_url": p.openreview_url,
                "arxiv_url":      p.arxiv_url or "",
            }
            for p in all_papers
        ],
        "cluster_summary": cluster_summary,
    }


@app.get("/api/funnel", tags=["Visualisation"])
async def get_funnel(
    years:       str = Query("2024,2025,2026"),
    conferences: str = Query("ICLR"),
    year_min:    int = Query(2024),
    year_max:    int = Query(2026),
    topic:       str = Query(""),
):
    """
    Returns Sankey diagram data (global ICLR estimates) PLUS
    per-cluster / per-area tier breakdown from the actual loaded papers.
    """
    year_list = [int(y.strip()) for y in years.split(",") if y.strip().isdigit()]
    if not year_list:
        year_list = list(range(year_min, year_max + 1))

    # ── Global ICLR estimates (for Sankey) ───────────────────────────────────
    ICLR_ESTIMATES = {
        2024: {"total": 2665, "oral": 83,  "spotlight": 156, "poster": 591, "reject": 1835},
        2025: {"total": 7993, "oral": 218, "spotlight": 447, "poster": 1095, "reject": 6233},
        2026: {"total": 9200, "oral": 250, "spotlight": 500, "poster": 1200, "reject": 7250},
    }

    nodes = [{"id": "Submissions", "label": "All Submissions"}]
    links = []
    year_breakdown = {}

    for year in year_list:
        if year not in ICLR_ESTIMATES:
            continue
        est = ICLR_ESTIMATES[year]
        year_node = f"ICLR {year}"
        nodes.append({"id": year_node, "label": year_node})
        links.append({"source": "Submissions", "target": year_node, "value": est["total"]})
        oral_node = f"{year} Oral"
        spot_node = f"{year} Spotlight"
        post_node = f"{year} Poster"
        rej_node  = f"{year} Rejected"
        nodes += [
            {"id": oral_node, "label": f"Oral ({est['oral']})"},
            {"id": spot_node, "label": f"Spotlight ({est['spotlight']})"},
            {"id": post_node, "label": f"Poster ({est['poster']})"},
            {"id": rej_node,  "label": f"Rejected ({est['reject']})"},
        ]
        links += [
            {"source": year_node, "target": oral_node, "value": est["oral"]},
            {"source": year_node, "target": spot_node, "value": est["spotlight"]},
            {"source": year_node, "target": post_node, "value": est["poster"]},
            {"source": year_node, "target": rej_node,  "value": est["reject"]},
        ]
        year_breakdown[year] = est

    # ── Per-cluster breakdown from actual fetched papers ─────────────────────
    conf_list  = [c.strip() for c in conferences.split(",") if c.strip()]
    real_topic = topic.strip() or FOCUS_TOPIC
    all_papers = await _all_papers(conf_list, year_min, year_max, real_topic)

    # Build per-cluster stats
    from collections import defaultdict
    cluster_stats: dict = defaultdict(lambda: {"oral": 0, "spotlight": 0, "poster": 0, "arxiv": 0, "total": 0})
    for p in all_papers:
        label = p.cluster_label or "General"
        cluster_stats[label][p.tier] = cluster_stats[label].get(p.tier, 0) + 1
        cluster_stats[label]["total"] += 1

    # Build per-cluster + per-year pivot
    year_cluster_stats: dict = defaultdict(lambda: defaultdict(lambda: {"oral": 0, "spotlight": 0, "poster": 0, "total": 0}))
    for p in all_papers:
        label = p.cluster_label or "General"
        yr    = p.year
        year_cluster_stats[yr][label][p.tier] = year_cluster_stats[yr][label].get(p.tier, 0) + 1
        year_cluster_stats[yr][label]["total"] += 1

    # Convert defaultdicts to plain dicts for JSON serialisation
    area_breakdown = {k: dict(v) for k, v in cluster_stats.items()}
    # Sort by total descending
    area_breakdown = dict(sorted(area_breakdown.items(), key=lambda x: x[1]["total"], reverse=True))

    year_area_breakdown = {
        yr: {label: dict(stats) for label, stats in clusters.items()}
        for yr, clusters in year_cluster_stats.items()
    }

    return {
        "nodes": nodes,
        "links": links,
        "year_breakdown": year_breakdown,
        "area_breakdown": area_breakdown,
        "year_area_breakdown": year_area_breakdown,
        "total_papers_fetched": len(all_papers),
    }


@app.get("/api/trends", tags=["Analysis"])
async def get_trends(
    conferences: str = Query("ICLR"),
    year_min:    int = Query(2024),
    year_max:    int = Query(2026),
    topic:       str = Query(""),
):
    conf_list  = [c.strip() for c in conferences.split(",") if c.strip()]
    real_topic = topic.strip() or FOCUS_TOPIC
    year_list  = list(range(year_min, year_max + 1))

    TREND_TOPICS = [
        "tool use", "multi-agent", "long-horizon planning",
        "code generation", "web agent", "benchmark",
        "memory", "grounding", "llm agent", "reasoning"
    ]

    trend_data: dict = {t: {} for t in TREND_TOPICS}
    for year in year_list:
        papers = await _all_papers(conf_list, year, year, real_topic)
        for topic_t in TREND_TOPICS:
            trend_data[topic_t][str(year)] = sum(
                1 for p in papers if topic_t.lower() in (p.title + " " + p.abstract).lower()
            )

    return {"topics": TREND_TOPICS, "data": trend_data, "years": [str(y) for y in year_list]}


@app.get("/api/white_spaces", tags=["Analysis"])
async def get_white_spaces(
    conferences: str = Query("ICLR"),
    year_min:    int = Query(2024),
    year_max:    int = Query(2026),
    topic:       str = Query(""),
):
    """
    Identify White Spaces: clusters where citation interest exists
    but academic paper density is low.
    """
    conf_list  = [c.strip() for c in conferences.split(",") if c.strip()]
    real_topic = topic.strip() or FOCUS_TOPIC
    all_papers = await _all_papers(conf_list, year_min, year_max, real_topic)

    if not all_papers:
        return {"white_spaces": [], "message": "No papers loaded yet"}

    raw_dicts  = [p.dict() for p in all_papers]
    white_spaces = detect_white_spaces(raw_dicts)
    return {"white_spaces": white_spaces, "total_papers": len(all_papers)}


@app.post("/api/pivot", response_model=PivotResponse, tags=["Strategy"])
async def pivot_idea(request: PivotRequest):
    result = generate_pivot(request.idea, CONFIG)

    # Semantic search over cached papers
    cache_keys = EMBEDS.list_keys()
    similar_ids = EMBEDS.search(cache_keys, request.idea, top_k=8) if cache_keys else []

    similar: List[Paper] = []
    for hit in similar_ids:
        for papers in _PAPER_CACHE.values():
            for p in papers:
                if p.id == hit["id"]:
                    similar.append(p)
                    break
        if len(similar) >= 5:
            break

    # Fallback: keyword overlap
    if not similar:
        all_papers: List[Paper] = []
        for papers in _PAPER_CACHE.values():
            all_papers.extend(papers)
        idea_words = set(request.idea.lower().split())
        all_papers.sort(
            key=lambda p: len(idea_words & set((p.title + " " + p.abstract).lower().split())),
            reverse=True
        )
        similar = all_papers[:5]

    return PivotResponse(
        original_idea=request.idea,
        iclr_flavor_score=result["iclr_flavor_score"],
        verdict=result["verdict"],
        fatal_flaw=result["fatal_flaw"],
        pivot_suggestion=result["pivot_suggestion"],
        theoretical_framing=result["theoretical_framing"],
        similar_accepted_papers=similar,
    )


@app.get("/api/semantic_search", tags=["Strategy"])
async def semantic_search(
    query:       str   = Query(..., description="Natural language idea or topic"),
    conferences: str   = Query("ICLR"),
    year_min:    int   = Query(2024),
    year_max:    int   = Query(2026),
    top_k:       int   = Query(10, ge=1, le=50),
    topic:       str   = Query(""),
    expand:      bool  = Query(True, description="Enable RAG-Fusion query expansion"),
):
    """RAG-Fusion semantic search: expands the query into variants, runs dense
    retrieval for each, then fuses rankings with Reciprocal Rank Fusion (RRF)."""
    conf_list  = [c.strip() for c in conferences.split(",") if c.strip()]
    real_topic = topic.strip() or FOCUS_TOPIC

    # Ensure papers are loaded so cache keys exist
    await _all_papers(conf_list, year_min, year_max, real_topic)

    # ── Query expansion (RAG-Fusion) ─────────────────────────────────────────
    loop = asyncio.get_event_loop()
    if expand:
        expanded_queries: list[str] = await loop.run_in_executor(
            None, expand_query, query, 4
        )
    else:
        expanded_queries = [query]

    # ── Dense retrieval with RRF fusion ──────────────────────────────────────
    keys = EMBEDS.list_keys()
    hits = await loop.run_in_executor(
        None, lambda: EMBEDS.search_rrf(keys, expanded_queries, top_k=top_k)
    )

    # ── Hydrate with full paper objects ──────────────────────────────────────
    id_map: dict = {}
    for papers in _PAPER_CACHE.values():
        for p in papers:
            id_map[p.id] = p

    results = []
    for h in hits:
        p = id_map.get(h["id"])
        if p:
            results.append({
                **p.dict(),
                "semantic_score": h.get("best_score", h.get("rrf_score", 0)),
                "rrf_score":      h.get("rrf_score", 0),
            })
        else:
            results.append({
                "id":             h["id"],
                "title":          h["title"],
                "semantic_score": h.get("best_score", 0),
                "rrf_score":      h.get("rrf_score", 0),
            })

    return {
        "query":            query,
        "expanded_queries": expanded_queries,
        "total":            len(results),
        "results":          results,
    }


@app.post("/api/refresh", tags=["Admin"])
async def refresh_cache(
    background_tasks: BackgroundTasks,
    disk: bool = True,
):
    """
    Clear both in-memory and (optionally) disk caches, then re-fetch.
    Pass ?disk=false to keep .md files and only reset the in-memory state.
    """
    global _PAPER_CACHE, _LANDSCAPE_CACHE, _FUNNEL_CACHE
    _PAPER_CACHE.clear()
    _LANDSCAPE_CACHE = None
    _FUNNEL_CACHE = None
    if disk and _cache_on:
        removed = CACHE.invalidate_all()
        logger.info(f"[Refresh] Invalidated {removed} disk cache files")
    for year in YEARS:
        background_tasks.add_task(_fetch_year_papers, year, True)
    return {"message": "Cache cleared. Re-fetching in background.", "disk_cleared": disk}


# ── Cache management routes ──────────────────────────────────────────────────

@app.get("/api/cache", tags=["Cache"])
async def list_cache():
    """
    List every cached query as metadata rows.
    Each entry maps to one Obsidian .md file on disk.
    """
    queries = CACHE.list_queries() if _cache_on else []
    return {
        "enabled":   _cache_on,
        "cache_dir": str(_cache_dir),
        "ttl_days":  _cache_ttl,
        "entries":   queries,
        "total":     len(queries),
    }


@app.get("/api/cache/{key}/raw", tags=["Cache"])
async def get_cache_raw(key: str):
    """Return the raw Obsidian .md file for a cache key (for debugging / Obsidian integration)."""
    if not _cache_on:
        raise HTTPException(status_code=503, detail="Cache is disabled")
    text = CACHE.read_raw(key)
    if text is None:
        raise HTTPException(status_code=404, detail=f"Cache key '{key}' not found")
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(text, media_type="text/markdown")


@app.delete("/api/cache/{conference}/{year}", tags=["Cache"])
async def invalidate_cache_entry(conference: str, year: int):
    """Invalidate the disk cache for a specific (conference, year, focus_topic) triple."""
    if not _cache_on:
        raise HTTPException(status_code=503, detail="Cache is disabled")
    deleted = CACHE.invalidate(conference.upper(), year, FOCUS_TOPIC)
    # also evict from memory
    _PAPER_CACHE.pop(str(year), None)
    if not deleted:
        raise HTTPException(status_code=404, detail="Cache entry not found")
    return {"message": f"Invalidated cache for {conference.upper()} {year}", "key": f"{conference.upper()}_{year}"}


@app.get("/api/stats", tags=["Admin"])
async def get_stats():
    disk_entries = CACHE.list_queries() if _cache_on else []
    return {
        "memory_cache": {
            "cached_years": list(_PAPER_CACHE.keys()),
            "paper_counts": {k: len(v) for k, v in _PAPER_CACHE.items()},
        },
        "disk_cache": {
            "enabled":  _cache_on,
            "dir":      str(_cache_dir),
            "ttl_days": _cache_ttl,
            "entries":  len(disk_entries),
            "files":    [e["file"] for e in disk_entries],
        },
        "focus_topic": CONFIG["focus"]["topic"],
        "keywords":    KEYWORDS,
    }


@app.on_event("startup")
async def startup_event():
    """Pre-warm cache on startup."""
    import asyncio
    logger.info("🚀 ConferencePapers backend starting...")
    # Fire and forget
    for year in YEARS:
        asyncio.create_task(_fetch_year_papers(year))


if __name__ == "__main__":
    import uvicorn
    port = CONFIG["app"]["port"]
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
