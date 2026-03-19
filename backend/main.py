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

from query_graph import QueryGraph
QUERY_GRAPH = QueryGraph(_cache_dir)

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


async def _fetch_single_topic(
    year: int,
    conference: str = "ICLR",
    topic: str | None = None,
    force_refresh: bool = False,
) -> List[Paper]:
    """
    Single-topic fetch pipeline (L1 → L2 → L3) for exactly ONE (conf, year, topic).
    Called by _fetch_year_papers for each query variant.

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
                # Re-resolve venue-based tiers (idempotent: skips non-arxiv papers
                # and conference papers already stored with correct tier in cache).
                cached_raw = _resolve_tiers_from_venue(list(cached_raw), conference)
                accepted_density = max(len(cached_raw), 1)
                papers = [_raw_to_paper(r, accepted_density) for r in cached_raw]
                _assign_proxy_tiers(papers)   # fallback proxy for remaining "arxiv"
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

        # 3c-ii. Use S2 venue to identify papers actually published at `conference`
        #        and assign oral/spotlight/poster by citation percentile.
        #        Must run BEFORE clustering so tier is stored in cache.
        raw_papers = _resolve_tiers_from_venue(raw_papers, conference)

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
        _assign_proxy_tiers(papers)   # fallback proxy only for remaining "arxiv" papers
        papers.sort(key=lambda p: p.relevance_score, reverse=True)
        _PAPER_CACHE[cache_key] = papers
        logger.info(f"[L3→Cache] {conference} {year} — {len(papers)} papers stored")
        return papers


async def _fetch_year_papers(
    year: int,
    conference: str = "ICLR",
    topic: str | None = None,
    force_refresh: bool = False,
) -> List[Paper]:
    """
    Graph-aware RAG-Fusion fetch.

    On every call:
      1. Expand `topic` into N variants using expand_query().
      2. Fetch each variant through _fetch_single_topic() in parallel
         (each variant has its own L1/L2/L3 pipeline + per-key lock).
      3. Load L1-cached papers from any graph-related keys (siblings/parent
         from prior searches) — ZERO new API calls for those.
      4. Register all expansion edges in QueryGraph for future reuse.
      5. Merge + dedup across all sources; store merged result under the
         root key in L1 so subsequent calls get the fast path.

    This means:
      - First search for Q1  → fetches Q1, Q1a, Q1b, Q1c, Q1d from API
      - Later search for Q1b → Q1b is L2 hit; sibling Q1, Q1a, Q1c, Q1d
        are loaded from L1 for free; Q1b is expanded into NEW Q1b-v1, Q1b-v2
        which are L3 fetched; RRF runs across all keys.
    """
    if topic is None:
        topic = FOCUS_TOPIC
    root_key = _mem_key(conference, year, topic)
    loop = asyncio.get_event_loop()

    # ── L1 fast path for the merged root result ──────────────────────────────
    if not force_refresh and root_key in _PAPER_CACHE:
        logger.debug(f"[L1 HIT root] {conference} {year} '{topic}'")
        return _PAPER_CACHE[root_key]

    # ── Expand topic into variants ────────────────────────────────────────────
    variants: List[str] = await loop.run_in_executor(None, expand_query, topic, 4)
    # variants[0] == topic (original always first)
    child_variants = variants[1:]  # the generated expansions
    child_keys     = [_mem_key(conference, year, v) for v in child_variants]

    # ── Fetch all variants concurrently (L1/L2/L3 each, independent locks) ───
    all_variant_tasks = [_fetch_single_topic(year, conference, v, force_refresh) for v in variants]
    variant_results = await asyncio.gather(*all_variant_tasks, return_exceptions=True)

    # ── Load graph-related keys already in L1 (no API calls) ─────────────────
    related_graph_keys = await loop.run_in_executor(None, QUERY_GRAPH.get_related_keys, root_key)
    all_variant_key_set = {_mem_key(conference, year, v) for v in variants}
    graph_extra: List[Paper] = []
    for gk in related_graph_keys:
        if gk not in all_variant_key_set and gk in _PAPER_CACHE:
            graph_extra.extend(_PAPER_CACHE[gk])
            logger.debug(f"[QueryGraph] Loaded {len(_PAPER_CACHE[gk])} papers from graph neighbor {gk!r}")

    # ── Register expansion edges in graph ────────────────────────────────────
    if child_keys:
        await loop.run_in_executor(
            None,
            QUERY_GRAPH.register_root,
            root_key, topic, conference, year, child_keys, child_variants,
        )

    # ── Merge + dedup across variants + graph neighbors ──────────────────────
    all_flat: List[Paper] = list(graph_extra)
    for r in variant_results:
        if isinstance(r, list):
            all_flat.extend(r)
        elif isinstance(r, Exception):
            logger.warning(f"[_fetch_year_papers] variant fetch error: {r}")

    seen_ids: set = set()
    merged: List[Paper] = []
    for p in all_flat:
        pid = getattr(p, "id", None)
        if pid and pid not in seen_ids:
            seen_ids.add(pid)
            merged.append(p)
        elif not pid:
            merged.append(p)  # keep if no id (shouldn't happen, but safe)

    merged.sort(key=lambda p: p.relevance_score, reverse=True)
    _PAPER_CACHE[root_key] = merged
    logger.info(
        f"[RAG-Fusion] {conference} {year} '{topic}' "
        f"→ {len(variants)} variants + {len(graph_extra)} graph-neighbor papers "
        f"→ {len(merged)} unique merged"
    )
    return merged


async def _build_embeddings_bg(cache_key: str, raw_papers: List[dict]):
    """Background task: build embedding index in a thread so the event loop is free."""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, EMBEDS.build, cache_key, raw_papers)
        logger.debug(f"[Embeddings] Built index for {cache_key}")
    except Exception as e:
        logger.debug(f"[Embeddings] Background build failed for {cache_key}: {e}")


def _assign_proxy_tiers(papers: List) -> None:
    """
    FALLBACK: For papers that still have tier='arxiv' after venue-based tier
    resolution, apply a citation-percentile proxy within each (venue, year) group.
    Papers confirmed via S2 venue as conference papers already have
    oral/spotlight/poster set — those are NOT touched here.

    Percentile thresholds (applied only within the arxiv-only group):
      top  5%  → 'oral'        (very high citation traction)
      next 15% → 'spotlight'
      next 40% → 'poster'
      bottom 40% → 'arxiv'    (kept to preserve "preprint / unconfirmed" signal)
    """
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for p in papers:
        if getattr(p, "tier", "arxiv") == "arxiv":
            groups[(getattr(p, "venue", ""), getattr(p, "year", 0))].append(p)

    for group in groups.values():
        group.sort(key=lambda p: getattr(p, "citation_count", 0) or 0, reverse=True)
        n = len(group)
        oral_n      = max(1, round(n * 0.05))
        spotlight_n = max(1, round(n * 0.15))
        poster_n    = max(1, round(n * 0.40))
        for i, p in enumerate(group):
            if i < oral_n:
                p.tier = "oral"
            elif i < oral_n + spotlight_n:
                p.tier = "spotlight"
            elif i < oral_n + spotlight_n + poster_n:
                p.tier = "poster"
            # else: keep "arxiv"


# Semantic Scholar venue strings for each conference (lowercase substrings)
_CONF_VENUE_KEYWORDS: dict = {
    "ICLR":    ["iclr", "international conference on learning representations"],
    "NEURIPS": ["neurips", "nips", "neural information processing systems"],
    "ICML":    ["icml", "international conference on machine learning"],
    "AAAI":    ["aaai", "association for the advancement of artificial intelligence"],
    "ACL":     ["association for computational linguistics", " acl "],
    "EMNLP":   ["emnlp", "empirical methods in natural language processing"],
}


def _resolve_tiers_from_venue(raw_papers: List[dict], conference: str) -> List[dict]:
    """
    Use the Semantic Scholar `publication_venue` field (set by S2 enrichment)
    to distinguish papers *actually published* at the target conference from
    pure arXiv preprints.

    Logic:
      1. If a paper has tier != 'arxiv' already (e.g. from OpenReview), skip it.
      2. If S2 venue string contains a known keyword for `conference`:
           → mark as venue-confirmed accepted.
      3. Among venue-confirmed papers, assign oral/spotlight/poster by
         citation-count percentile (best proxy when we lack OpenReview decisions).
      4. Everything else stays 'arxiv' (preprint / rejected / unknown).

    This runs on *raw dicts* right after S2 enrichment, before clustering and
    before conversion to Paper objects. Results are stored in cache.
    """
    kws = _CONF_VENUE_KEYWORDS.get(conference.upper(), [])
    if not kws:
        return raw_papers

    confirmed: List[dict] = []
    for p in raw_papers:
        if p.get("tier", "arxiv") != "arxiv":
            continue  # Already has a real tier from OpenReview — don't override
        venue = (p.get("publication_venue") or "").lower().strip()
        if venue and any(kw in venue for kw in kws):
            p["tier"] = "_venue_confirmed"   # temp marker, replaced below
            confirmed.append(p)

    if not confirmed:
        return raw_papers

    # Citation-percentile tiers within venue-confirmed papers
    confirmed.sort(key=lambda p: p.get("citation_count", 0) or 0, reverse=True)
    n = len(confirmed)
    for i, p in enumerate(confirmed):
        frac = i / n
        if frac < 0.05:
            p["tier"] = "oral"
        elif frac < 0.20:
            p["tier"] = "spotlight"
        else:
            p["tier"] = "poster"

    logger.info(
        f"[TierResolve] {conference}: {len(confirmed)}/{len(raw_papers)} "
        f"papers confirmed via S2 venue → oral/spotlight/poster assigned"
    )
    return raw_papers


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
    tier:        str  = Query("all"),
    min_relevance: float = Query(0.0),
):
    conf_list  = [c.strip() for c in conferences.split(",") if c.strip()]
    real_topic = topic.strip() or FOCUS_TOPIC
    all_papers = await _all_papers(conf_list, year_min, year_max, real_topic)

    tier_lower = tier.lower().strip()
    if tier_lower not in ("all", ""):
        allowed = {t.strip() for t in tier_lower.replace("+", ",").split(",") if t.strip()}
        all_papers = [p for p in all_papers if p.tier in allowed]

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
    tier:        str = Query("all"),
):
    """
    Returns Sankey diagram data driven by the ACTUAL query papers (not global
    conference totals), plus per-cluster / per-area tier breakdown.
    Global ICLR acceptance-rate percentages are supplied separately as
    `year_breakdown` for the rate-chart context cards.
    """
    year_list = [int(y.strip()) for y in years.split(",") if y.strip().isdigit()]
    if not year_list:
        year_list = list(range(year_min, year_max + 1))

    # ── Global ICLR estimates (acceptance-rate context only) ─────────────────
    ICLR_ESTIMATES = {
        2020: {"total": 2594,  "oral": 48,  "spotlight": 108, "poster": 531,  "reject": 1907},
        2021: {"total": 2997,  "oral": 53,  "spotlight": 114, "poster": 693,  "reject": 2137},
        2022: {"total": 3391,  "oral": 54,  "spotlight": 176, "poster": 865,  "reject": 2296},
        2023: {"total": 4966,  "oral": 81,  "spotlight": 155, "poster": 1338, "reject": 3392},
        2024: {"total": 7262,  "oral": 85,  "spotlight": 182, "poster": 1991, "reject": 5004},
        2025: {"total": 9500,  "oral": 218, "spotlight": 447, "poster": 1950, "reject": 6885},
        2026: {"total": 11000, "oral": 270, "spotlight": 550, "poster": 2400, "reject": 7780},
    }

    def _pct(n: int, total: int) -> float:
        return round(n / total * 100, 1) if total else 0.0

    year_breakdown = {}
    for year in year_list:
        if year not in ICLR_ESTIMATES:
            continue
        est    = ICLR_ESTIMATES[year]
        total  = est["total"]
        accept = est["oral"] + est["spotlight"] + est["poster"]
        year_breakdown[year] = {
            **est,
            "pct_oral":      _pct(est["oral"],     total),
            "pct_spotlight": _pct(est["spotlight"], total),
            "pct_poster":    _pct(est["poster"],    total),
            "pct_reject":    _pct(est["reject"],    total),
            "pct_accept":    _pct(accept,           total),
        }

    # ── Load actual query papers ──────────────────────────────────────────────
    conf_list  = [c.strip() for c in conferences.split(",") if c.strip()]
    real_topic = topic.strip() or FOCUS_TOPIC
    all_papers = await _all_papers(conf_list, year_min, year_max, real_topic)

    # Tier filter  
    tier_lower = tier.lower().strip()
    if tier_lower not in ("all", ""):
        allowed = {t.strip() for t in tier_lower.replace("+", ",").split(",") if t.strip()}
        all_papers = [p for p in all_papers if p.tier in allowed]

    # ── Build query-driven Sankey ─────────────────────────────────────────────
    # Nodes: root → year → tier
    nodes = [{"id": "Query Papers", "label": f"Query: {real_topic[:40]}"}]
    links = []

    for year in year_list:
        yr_papers = [p for p in all_papers if p.year == year]
        if not yr_papers:
            continue
        yr_total = len(yr_papers)
        yr_oral      = sum(1 for p in yr_papers if p.tier == "oral")
        yr_spotlight = sum(1 for p in yr_papers if p.tier == "spotlight")
        yr_poster    = sum(1 for p in yr_papers if p.tier == "poster")
        yr_arxiv     = sum(1 for p in yr_papers if p.tier == "arxiv")

        # Global acceptance rate for the label (context)
        global_rate = ""
        if year in year_breakdown:
            global_rate = f" · {year_breakdown[year]['pct_accept']}% global acc."

        yr_node = f"Query {year}"
        nodes.append({"id": yr_node, "label": f"{year} ({yr_total} papers{global_rate})"})
        links.append({"source": "Query Papers", "target": yr_node, "value": yr_total})

        for tier_name, count, label_suffix in [
            ("oral",      yr_oral,      f"Oral ★"),
            ("spotlight", yr_spotlight, f"Spotlight"),
            ("poster",    yr_poster,    f"Poster"),
            ("arxiv",     yr_arxiv,     f"arXiv-only"),
        ]:
            if count == 0:
                continue
            t_node = f"{year} {tier_name.capitalize()}"
            nodes.append({"id": t_node, "label": f"{label_suffix} — {count}"})
            links.append({"source": yr_node, "target": t_node, "value": count})

    # ── Per-cluster breakdown from query papers ───────────────────────────────
    from collections import defaultdict
    cluster_stats: dict = defaultdict(lambda: {"oral": 0, "spotlight": 0, "poster": 0, "arxiv": 0, "total": 0})
    for p in all_papers:
        label = p.cluster_label or "General"
        cluster_stats[label][p.tier] = cluster_stats[label].get(p.tier, 0) + 1
        cluster_stats[label]["total"] += 1

    year_cluster_stats: dict = defaultdict(lambda: defaultdict(lambda: {"oral": 0, "spotlight": 0, "poster": 0, "total": 0}))
    for p in all_papers:
        label = p.cluster_label or "General"
        yr    = p.year
        year_cluster_stats[yr][label][p.tier] = year_cluster_stats[yr][label].get(p.tier, 0) + 1
        year_cluster_stats[yr][label]["total"] += 1

    area_breakdown = dict(sorted(
        {k: dict(v) for k, v in cluster_stats.items()}.items(),
        key=lambda x: x[1]["total"], reverse=True
    ))
    year_area_breakdown = {
        yr: {label: dict(stats) for label, stats in clusters.items()}
        for yr, clusters in year_cluster_stats.items()
    }

    return {
        "nodes":               nodes,
        "links":               links,
        "year_breakdown":      year_breakdown,  # global ICLR rates (context)
        "area_breakdown":      area_breakdown,
        "year_area_breakdown": year_area_breakdown,
        "total_papers_fetched": len(all_papers),
        "query_topic":         real_topic,
    }


@app.get("/api/trends", tags=["Analysis"])
async def get_trends(
    conferences: str = Query("ICLR"),
    year_min:    int = Query(2024),
    year_max:    int = Query(2026),
    topic:       str = Query(""),
    tier:        str = Query("all"),
):
    conf_list  = [c.strip() for c in conferences.split(",") if c.strip()]
    real_topic = topic.strip() or FOCUS_TOPIC
    year_list  = list(range(year_min, year_max + 1))

    tier_lower = tier.lower().strip()
    allowed_tiers = None
    if tier_lower not in ("all", ""):
        allowed_tiers = {t.strip() for t in tier_lower.replace("+", ",").split(",") if t.strip()}

    TREND_TOPICS = [
        "tool use", "multi-agent", "long-horizon planning",
        "code generation", "web agent", "benchmark",
        "memory", "grounding", "llm agent", "reasoning"
    ]

    trend_data: dict = {t: {} for t in TREND_TOPICS}
    for year in year_list:
        papers = await _all_papers(conf_list, year, year, real_topic)
        if allowed_tiers:
            papers = [p for p in papers if p.tier in allowed_tiers]
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
    tier:        str = Query("all"),
):
    """
    Identify White Spaces: structured per-cluster analysis showing
    Demand / Exists (Y) / Tried-not-accepted (Z') / Gap (Z).
    Top results are enriched with an LLM narrative.
    """
    conf_list  = [c.strip() for c in conferences.split(",") if c.strip()]
    real_topic = topic.strip() or FOCUS_TOPIC
    all_papers = await _all_papers(conf_list, year_min, year_max, real_topic)

    tier_lower = tier.lower().strip()
    if tier_lower not in ("all", ""):
        allowed = {t.strip() for t in tier_lower.replace("+", ",").split(",") if t.strip()}
        all_papers = [p for p in all_papers if p.tier in allowed]

    if not all_papers:
        return {"white_spaces": [], "message": "No papers loaded yet"}

    raw_dicts    = [p.dict() for p in all_papers]
    white_spaces = detect_white_spaces(raw_dicts)

    # ── LLM narrative enrichment (top 6 results, parallel) ──────────────────
    loop = asyncio.get_event_loop()

    def _llm_narrative(ws: dict) -> str | None:
        """
        Generate a 3-section analysis grounded in actual abstracts and
        author-identified limitations.  Returns text with labelled sections:
          DONE: ...
          CHALLENGES: ...
          OPPORTUNITY: ...
        """
        try:
            from query_expander import _get_llm
            llm = _get_llm()
            if not llm or llm is False:
                return None

            # ── Format accepted papers WITH abstract snapshots ───────────────
            exists_blocks = []
            for p in ws.get("what_exists", []):
                abst_snippet = ""
                for a in ws.get("abstracts_sample", []):
                    if a["title"] == p["title"] and a.get("abstract"):
                        abst_snippet = a["abstract"][:500]
                        break
                block = (
                    f'  • "{p["title"][:90]}" ({p.get("year","?")}, {p.get("tier","?")}, '
                    f'{p.get("citations",0)} cit.)'
                )
                if abst_snippet:
                    block += f'\n    ↳ Abstract: {abst_snippet}'
                exists_blocks.append(block)
            exists_text = "\n".join(exists_blocks) or "  • (no accepted papers in this cluster)"

            # ── arXiv attempts ───────────────────────────────────────────────
            tried_text = "\n".join(
                f'  • "{p["title"][:90]}" ({p.get("year","?")}, arXiv, {p.get("citations",0)} cit.)'
                for p in ws.get("attempted", [])
            ) or "  • (no arXiv attempts found)"

            # ── Limitations extracted from abstracts ─────────────────────────
            limit_lines = []
            for item in ws.get("limitations_corpus", []):
                for sent in item.get("sentences", []):
                    limit_lines.append(f'  • [{item["title"][:60]}]: "{sent.strip()}"')
            limitations_text = (
                "\n".join(limit_lines[:6])
                or "  • (no explicit limitation sentences detected in abstracts)"
            )

            # ── GitHub + year trend ──────────────────────────────────────────
            stars = ws.get("total_stars", 0)
            github_str = f"{stars} total GitHub ⭐ across cluster" if stars > 0 else "no GitHub star data"
            year_trend = ws.get("year_trend", {})
            if len(year_trend) >= 2:
                vals = list(year_trend.values())
                direction = "rising" if vals[-1] > vals[0] else "declining"
                trend_str = (
                    f"publications {direction} "
                    f"({', '.join(f'{y}: {n}' for y, n in year_trend.items())})"
                )
            else:
                trend_str = "limited year-over-year data"

            prompt = f"""\
You are a research strategy analyst specialising in venue strategy for ICLR, NeurIPS, and ICML.

Research cluster: "{ws['cluster_label']}"
Stats: {ws['paper_count']} papers total | {ws['accepted_count']} accepted at top venues | {ws['arxiv_count']} arXiv-only
Avg citations: {ws['avg_citations']} | Citation velocity: {ws['avg_velocity']} cit/month
GitHub: {github_str} | Trend: {trend_str}
Gap score: {ws['gap_score']}× above average sparsity (higher = more room for new work)
Demand signals: {', '.join(ws.get('demand_signals', []))}

══ ACCEPTED PAPERS — what has been published and accepted ══
{exists_text}

══ ARXIV-ONLY — attempted but not accepted at top venues ══
{tried_text}

══ AUTHOR-IDENTIFIED LIMITATIONS & FUTURE WORK (extracted directly from paper abstracts) ══
{limitations_text}

Write a structured analysis with EXACTLY these three labelled sections.
Be concrete — cite specific paper titles, quote specific limitations, and name specific methods.

DONE: What the accepted papers actually accomplished. Name specific papers, their methods, and the sub-problems they solved. What benchmarks or results did they demonstrate? 2–3 sentences.

CHALLENGES: The specific unsolved challenges. Draw directly from the limitations above — quote or closely paraphrase author-stated problems. Explain why the arXiv attempts likely failed to get accepted (missing theory, narrow scope, no rigorous ablations, etc.). 2–3 sentences with concrete specifics.

OPPORTUNITY: A concrete, actionable research opportunity a researcher could pursue tomorrow. State the exact gap, the specific approach or method that would address it, why it would satisfy reviewers at ICLR/NeurIPS/ICML (theoretical depth, generalization, novel benchmark, etc.), and the expected contribution type (new framework / theorem / benchmark / empirical study). 3–4 sentences."""

            raw, _ = llm.call_llm_with_retry(
                user_message=prompt,
                system_message=(
                    "You are a precise research strategy analyst. "
                    "Output ONLY the three labelled sections: DONE:, CHALLENGES:, OPPORTUNITY: "
                    "— no preamble, no bullet points inside sections, no extra commentary."
                ),
            )
            if raw is None:
                return None
            return raw.strip() if isinstance(raw, str) else str(raw).strip()
        except Exception as exc:
            logger.warning(f"[WhiteSpace] LLM narrative failed: {exc}")
            return None

    async def _enrich(ws: dict) -> dict:
        narrative = await loop.run_in_executor(None, _llm_narrative, ws)
        if narrative:
            ws["opportunity"] = narrative
            ws["llm_narrative"] = True
        else:
            ws["llm_narrative"] = False
        return ws

    # Enrich top 6 in parallel; leave the rest with template narratives
    top_n = min(6, len(white_spaces))
    if top_n:
        enriched = await asyncio.gather(*[_enrich(ws) for ws in white_spaces[:top_n]])
        white_spaces[:top_n] = list(enriched)

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
        QUERY_GRAPH.clear()  # graph nodes are invalid once cache files are gone
        logger.info(f"[Refresh] Invalidated {removed} disk cache files + cleared query graph")
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
    """
    Startup: reconcile query graph with existing cache files, then log readiness.
    No pre-warming — all fetches are demand-driven once the user submits a search.

    Parallelism model (already in place):
      • asyncio.gather()      — concurrent (conference × year) tasks
      • loop.run_in_executor() — thread-pool for blocking arXiv / OpenReview SDKs
      • Per-key asyncio.Lock  — prevents thundering-herd duplicate L3 fetches
      • CPU-heavy work (clustering, UMAP, embeddings) also runs in executor
      • QueryGraph            — cross-search graph-neighbor reuse for RAG-Fusion
    """
    # Reconcile query graph: drop nodes whose .md files were manually deleted
    existing_keys = {p.stem for p in _cache_dir.glob("*.md") if not p.name.startswith("_")}
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, QUERY_GRAPH.reconcile, existing_keys)
    logger.info(f"🚀 ConferencePapers backend ready — {len(QUERY_GRAPH)} query graph nodes loaded.")


if __name__ == "__main__":
    import uvicorn
    port = CONFIG["app"]["port"]
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
