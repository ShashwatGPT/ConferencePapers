# ConferencePapers — Agents for Productivity @ ICLR

A full-stack discovery and strategy engine for analysing **ICLR 2024–2026** accepted papers in the **"Agents for Productivity"** space.

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

### Cache management

| Endpoint | What it does |
|---|---|
| `GET /api/cache` | List all cached queries with metadata |
| `GET /api/cache/{key}/raw` | Return the raw `.md` file |
| `DELETE /api/cache/{conf}/{year}` | Invalidate one entry |
| `POST /api/refresh?disk=true` | Clear all caches and re-fetch |

TTL is set in `config.json → cache.ttl_days` (default **7 days**). Set to `0` to never expire.

---

## Stack

- **Backend**: Python / FastAPI + `openreview-py` + `arxiv` + Semantic Scholar API
- **Frontend**: Vanilla HTML5 / Tailwind CSS CDN / Plotly.js
- **Cache**: `CacheStore` — Obsidian-style `.md` files, YAML frontmatter, wiki-links
- **Scoring**: TF-IDF + K-Means clustering, UMAP 2D, Impact-Niche Score formula

$$Score = \frac{Cit.Velocity \times GitHub.Stars}{Accepted.Paper.Density}$$

---

## Quick Start

### 1 — Create & activate the conda environment

```bash
conda create -n conferencepapers python=3.11 -y
conda activate conferencepapers
```

### 2 — Install dependencies

```bash
pip install -r backend/requirements.txt
```

> All packages are pinned in `backend/requirements.txt`.

### 3 — Start the backend

```bash
conda activate conferencepapers
cd backend
uvicorn main:app --reload --port 8000
```

**First boot behaviour:**
1. Server checks `cache/` for existing `.md` files — instant load if present.
2. On cache miss, fetches from OpenReview → arXiv fallback → Semantic Scholar enrichment (~30–90 s per year).
3. Results are persisted as `.md` files; subsequent boots load in milliseconds.

A **green dot** in the top-right of the UI confirms data is ready.

### 4 — Open the dashboard

```
http://localhost:8000
```

No build step — the frontend is a single static HTML file served by FastAPI.

---

### Running without conda (pip venv)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
cd backend && uvicorn main:app --reload --port 8000
```

---

## Configuration

Edit [`config.json`](config.json) at the project root:

| Key | Purpose |
|---|---|
| `conferences.ICLR.*` | Add/change venue IDs and acceptance tiers |
| `focus.keywords` | Topics that count as "Agents for Productivity" |
| `iclr_flavor_heuristics` | Signals that boost/penalise ICLR flavor score |
| `apis.*` | Base URLs and API keys |
| `cache.ttl_days` | How many days before a cache entry goes stale |
| `cache.dir` | Path to the `.md` cache directory (relative to project root) |

---

## Data Sources

| Source | Data |
|---|---|
| [OpenReview](https://openreview.net) | Decisions, reviewer ratings, confidence scores |
| [Semantic Scholar](https://www.semanticscholar.org) | Citation counts, citation velocity |
| [arXiv](https://arxiv.org) | Full-text search, PDF links (fallback) |

---

## Project Structure

```
ConferencePapers/
├── config.json                          # All configuration (no hardcoding)
├── agents.md                            # Product spec
├── cache/                               # Obsidian-style .md query cache
│   ├── _index.md                        # Master query log (auto-maintained)
│   └── ICLR_2024_agents_for_productivity.md
├── backend/
│   ├── main.py                          # FastAPI app + all API endpoints
│   ├── models.py                        # Pydantic data models
│   ├── scorer.py                        # Impact-Niche, ICLR Flavor, clustering
│   ├── cache_store.py                   # L2 disk cache (Obsidian .md format)
│   ├── requirements.txt
│   └── extractors/
│       ├── openreview_extractor.py
│       ├── semantic_scholar_extractor.py
│       └── arxiv_extractor.py
└── frontend/
    └── index.html                       # Single-page dashboard
```

## Quick Start

### 1 — Create & activate the conda environment

```bash
conda create -n conferencepapers python=3.11 -y
conda activate conferencepapers
```

### 2 — Install dependencies

```bash
pip install -r backend/requirements.txt
```

> All packages (FastAPI, openreview-py, arxiv, scikit-learn, umap-learn, sentence-transformers, Plotly, etc.) are pinned in `backend/requirements.txt`.

### 3 — Start the backend

```bash
conda activate conferencepapers
cd backend
uvicorn main:app --reload --port 8000
```

On first boot the server pre-warms its cache — it fetches ICLR 2024–2026 accepted papers from OpenReview, enriches them via Semantic Scholar, runs TF-IDF clustering and UMAP embedding. This takes **~30–90 s** depending on API latency. A green dot in the top-right of the UI confirms data is ready.

### 4 — Open the dashboard

```
http://localhost:8000
```

No extra build step — the frontend is a single static HTML file served directly by FastAPI.

---

### Running without conda (pip venv)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
cd backend && uvicorn main:app --reload --port 8000
```

## Configuration

Edit [`config.json`](config.json) at the project root to:
- Add/change conference years (`conferences.ICLR.*`)
- Adjust focus keywords (`focus.keywords`)
- Tune ICLR flavor signals (`iclr_flavor_heuristics`)
- Switch API keys / base URLs (`apis.*`)

## Data Sources

| Source | Data |
|---|---|
| [OpenReview](https://openreview.net) | Decisions, reviewer ratings, confidence scores |
| [Semantic Scholar](https://www.semanticscholar.org) | Citation counts, citation velocity |
| [arXiv](https://arxiv.org) | Full-text search, PDF links (fallback) |

## Project Structure

```
ConferencePapers/
├── config.json                    # All configuration (no hardcoding)
├── agents.md                      # Product spec
├── backend/
│   ├── main.py                    # FastAPI app + all API endpoints
│   ├── models.py                  # Pydantic data models
│   ├── scorer.py                  # Impact-Niche, ICLR Flavor, clustering
│   ├── requirements.txt
│   └── extractors/
│       ├── openreview_extractor.py
│       ├── semantic_scholar_extractor.py
│       └── arxiv_extractor.py
└── frontend/
    └── index.html                 # Single-page dashboard
```
# ConferencePapers
