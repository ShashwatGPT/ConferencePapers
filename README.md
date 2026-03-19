# ConferencePapers

A full-stack research discovery and strategy engine for analysing accepted (and rejected) papers across **ICLR, ICML, NeurIPS, AAAI, ACL, and EMNLP** in the **"Agents for Productivity"** space. Built on FastAPI + Vanilla JS with a three-level cache, RAG-Fusion semantic search, and an LLM-powered Pivot Engine.

---

## Quick Start

### 1 — Create & activate the environment

**conda (recommended)**
```bash
conda create -n conferencepapers python=3.11 -y
conda activate conferencepapers
```

**or pip venv**
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 2 — Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 3 — Start the backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

**First-boot behaviour:**
1. Server checks `cache/` for existing `.md` files — instant load if present.
2. On a cold cache miss it fetches from OpenReview → arXiv (fallback) → Semantic Scholar enrichment (~30–90 s per conference-year pair).
3. Results are persisted as Obsidian-style `.md` files; subsequent boots load in milliseconds.

A **green dot** in the top-right of the UI confirms data is ready.

### 4 — Open the dashboard

```
http://localhost:8000
```

No build step — the frontend is a single static HTML file served directly by FastAPI.

---

## Features

| Module | What it does |
|---|---|
| 📊 **Overview** | Stats, top papers by Impact-Niche Score, tier breakdown, ICLR Flavor scatter |
| 🗺 **Landscape Map** | 2D UMAP cluster map of paper abstracts (click nodes to deep-read) |
| 🔻 **Prestige Funnel** | Sankey diagram — Submission → Oral / Spotlight / Poster / Rejected |
| 📄 **Papers** | Searchable, sortable card grid with relevance, ICLR Flavor, and Impact-Niche scores |
| 📈 **Trends** | Topic frequency across years (tool use, multi-agent, planning, etc.) |
| ⬜ **White Spaces** | Clusters where GitHub traction > academic paper density |
| 🔄 **Pivot Engine** | Enter any applied idea → get ICLR fit score + theoretical reframing |
| 💾 **Cache** | Inspect, view raw, and invalidate Obsidian-style `.md` query caches |

---

## Smart Caching — Obsidian-style `.md` files

API calls to OpenReview, arXiv, and Semantic Scholar are expensive and slow. The engine uses a **three-level cache** so every unique query is fetched *at most once* per TTL window:

```
L1  In-memory Python dict        — microseconds, lost on restart
L2  Disk .md files  (cache/)     — milliseconds, survives restarts
L3  Live APIs                    — seconds/minutes, only on cold miss
```

### L2 Cache file anatomy

Every `(conference, year, topic)` triple maps to one Obsidian-compatible Markdown file:

```
cache/
  _index.md                              ← master query log (wiki-linked)
  ICLR_2024_agents_for_productivity.md
  ICLR_2025_agents_for_productivity.md
  ICLR_2026_agents_for_productivity.md
```

Each file has YAML frontmatter (query metadata) + one `## [[Paper Title]]` section per result:

```markdown
---
conference: ICLR
year: 2024
topic: Agents for Productivity
fetched_at: 2026-03-16T00:12:34
ttl_days: 7
total_papers: 42
---

# ICLR 2024 — Agents for Productivity

## [[Latent Space Planning for LLM Agents]]
- id: `openreview_abc123`
- tier: `poster`
- relevance_score: `0.82`
- iclr_flavor_score: `0.71`
- citation_count: `103`
- cluster_label: `Long-Horizon Planning`
- authors: `Alice Smith, Bob Jones`
> We propose a theoretical framework for latent goal representations…
```

**Opening `cache/` in Obsidian** gives you a free knowledge graph — papers wiki-link naturally via their titles.

### Cache management API

| Endpoint | What it does |
|---|---|
| `GET /api/cache` | List all cached queries with metadata |
| `GET /api/cache/{key}/raw` | Return the raw `.md` file |
| `DELETE /api/cache/{conference}/{year}` | Invalidate one entry |
| `POST /api/refresh?disk=true` | Clear all caches and re-fetch from live APIs |

TTL is set in `config.json → cache.ttl_days` (default **7 days**). Set to `0` to never expire.

---

## Architecture

### Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.11 / FastAPI + uvicorn |
| **Data sources** | `openreview-py`, `arxiv`, Semantic Scholar REST API |
| **Scoring & clustering** | scikit-learn (TF-IDF, K-Means), UMAP, numpy |
| **Semantic search** | Azure OpenAI `text-embedding-3-small` (1536-d) via `EmbeddingStore` |
| **LLM reasoning** | `gpt-4.1` via GitHub Copilot SDK or Azure OpenAI (`query_expander.py`, Pivot Engine) |
| **Frontend** | Vanilla HTML5 / Tailwind CSS CDN / Plotly.js |
| **Cache** | `CacheStore` — Obsidian `.md` + `EmbeddingStore` — `.npz` vector files |

### Fetch pipeline (L1 → L2 → L3)

Every paper request goes through a three-level cache guarded by per-key `asyncio.Lock` objects (thundering-herd prevention):

```
L1  In-memory Python dict          — microseconds, lost on restart
L2  Disk .md + .npz files (cache/) — milliseconds, survives restarts
L3  Live APIs                      — seconds/minutes, only on cold miss
```

On an L3 fetch the pipeline runs:
1. **OpenReview extractor** — decisions, tiers (oral / spotlight / poster), reviewer ratings and confidence scores.
2. **Semantic Scholar enrichment** — citation count, citation velocity, influential citations.
3. **arXiv extractor** — arXiv ID, abstract, PDF URL (used as fallback when OpenReview has no abstract).
4. **Scoring** — relevance, ICLR Flavor, Impact-Niche scores computed in-process.
5. **Clustering** — TF-IDF vectorisation → K-Means → UMAP 2D projection (dense embeddings preferred when `EmbeddingModel` is available).
6. **Persist** — write `.md` frontmatter file + `.npz` vector file to `cache/`.

### RAG-Fusion semantic search (`/api/semantic_search`)

Queries are expanded into N semantically diverse variants by the LLM (`query_expander.py`). Each variant is fetched or loaded from L1/L2. Dense embeddings for all papers are stored in `.npz` files and searched via **Reciprocal Rank Fusion (RRF)**:

$$RRF(p) = \sum_i \frac{1}{k + rank_i(p)}$$

The `QueryGraph` (`query_graph.py`) persists parent → child query relationships in `cache/_query_graph.json` so future searches for related topics can load sibling-query papers from disk without new API calls.

### Scoring formulas

**Impact-Niche Score**

$$Score = \log\!\left(1 + \frac{Cit.Velocity \times \max(GitHub.Stars,\ Cit.Count / 10)}{Accepted.Paper.Density}\right)$$

Falls back to `citation_count / 10` as a GitHub-star proxy for papers without a linked repository.

**ICLR Flavor Score**

Weighted phrase matching over title + abstract, sigmoid-normalised to [0, 1]:

- Weight +3: `proof`, `theorem`, `convergence`, `lower bound`, `upper bound`, `expressivity`
- Weight +2: `latent space`, `representation`, `scaling law`, `emergent`, `world model`, `symmetry`
- Weight +1: `generalization`, `alignment`, `interpretability`, `in-context learning`, `architecture`
- Weight −2: `user study`, `deployed system`, `engineering system`
- Weight −3: `api wrapper`, `middleware`, `scraping`

≥ 0.6 → good ICLR fit · 0.3–0.6 → borderline · < 0.3 → likely rejected

**Relevance Score**

Keyword hit-rate over `focus.keywords` minus an anti-keyword penalty (`focus.anti_keywords`), capped at 1.0.

### Pivot Engine (`POST /api/pivot`)

Accepts any applied idea and returns:
- ICLR fit score with explanation
- Identified fatal flaws based on historical reviewer patterns
- Concrete theoretical reframing suggestions (e.g. "rename to *Multi-Step Latent Planning for Hierarchical Task Orchestration*")

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Single-page dashboard |
| `GET` | `/api/config` | Active configuration |
| `GET` | `/api/papers` | Filtered, sorted paper list |
| `GET` | `/api/papers/cluster/{id}` | Papers in a specific cluster |
| `GET` | `/api/paper/{id}` | Single paper by ID |
| `GET` | `/api/landscape` | UMAP 2D data for cluster map |
| `GET` | `/api/funnel` | Sankey data (submission → tier) |
| `GET` | `/api/trends` | Topic frequency across years |
| `GET` | `/api/white_spaces` | Clusters with high GitHub traction but low paper density |
| `GET` | `/api/semantic_search` | RAG-Fusion dense retrieval |
| `POST` | `/api/pivot` | Pivot Engine — reframe idea for a target conference |
| `POST` | `/api/refresh` | Invalidate caches and re-fetch |
| `GET` | `/api/cache` | List L2 cache entries |
| `GET` | `/api/cache/{key}/raw` | Raw `.md` cache file |
| `DELETE` | `/api/cache/{conference}/{year}` | Invalidate one cache entry |
| `GET` | `/api/stats` | Server stats (paper counts, cache size) |

Interactive docs available at `http://localhost:8000/docs` (Swagger UI).

---

## Configuration

Edit [`config.json`](config.json) at the project root — no hardcoding anywhere in the codebase.

| Key | Purpose |
|---|---|
| `conferences.{CONF}.{year}.venue_id` | OpenReview venue string |
| `conferences.{CONF}.{year}.acceptance_tiers` | Decision label → tier mapping |
| `focus.keywords` | Topics counted as "Agents for Productivity" |
| `focus.anti_keywords` | Topics that lower the relevance score |
| `iclr_flavor_heuristics` | Config-level good/bad signal overrides |
| `apis.semantic_scholar.fields` | Fields requested from Semantic Scholar |
| `apis.arxiv.max_results` | arXiv results per query |
| `llm.model_name` | LLM used by the Pivot Engine and Query Expander |
| `llm.model_endpoint` | `"ghcp"` for GitHub Copilot SDK, or an Azure URL |
| `cache.ttl_days` | Days before a cache entry goes stale (0 = never) |
| `cache.dir` | Path to the `.md`/`.npz` cache directory |

Environment variable overrides: `LLM_MODEL_NAME`, `LLM_MODEL_ENDPOINT`, `EMBEDDING_DEPLOYMENT`.

---

## Data Sources

| Source | Data retrieved |
|---|---|
| [OpenReview](https://openreview.net) | Decisions, acceptance tiers, reviewer ratings (1–10), confidence scores |
| [Semantic Scholar](https://www.semanticscholar.org) | Citation count, citation velocity, influential citations |
| [arXiv](https://arxiv.org) | Abstract, PDF URL, arXiv ID (fallback when OpenReview data is sparse) |

---

## Project Structure

```
ConferencePapers/
├── config.json                          # All runtime configuration
├── agents.md                            # Product spec
├── cache/                               # L2 cache (auto-created)
│   ├── _index.md                        # Master query log (wiki-links)
│   ├── _query_graph.json                # Query expansion relationship graph
│   ├── ICLR_2024_agents_for_productivity.md   # Per-query Obsidian .md file
│   └── ICLR_2024_agents_for_productivity.npz  # Per-query embedding vectors
├── backend/
│   ├── main.py                          # FastAPI app, all endpoints, fetch pipeline
│   ├── schemas.py                       # Pydantic data models (Paper, PivotRequest, …)
│   ├── scorer.py                        # Relevance, ICLR Flavor, Impact-Niche, clustering
│   ├── cache_store.py                   # L2 disk cache — Obsidian .md read/write
│   ├── embedding_store.py               # .npz vector store — build / search / RRF
│   ├── query_expander.py                # LLM + rule-based query expansion
│   ├── query_graph.py                   # Persistent query relationship graph
│   ├── requirements.txt
│   └── extractors/
│       ├── __init__.py
│       ├── openreview_extractor.py      # OpenReview API v2 client
│       ├── semantic_scholar_extractor.py
│       └── arxiv_extractor.py
└── frontend/
    └── index.html                       # Single-page dashboard (Tailwind + Plotly.js)
```
