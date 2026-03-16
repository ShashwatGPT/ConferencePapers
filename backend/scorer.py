"""
scorer.py  — fixed edition
"""

import logging
import re
import math
from typing import List, Optional
import numpy as np

import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Bootstrap path so MODEL package is importable from anywhere
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _get_dense_vecs(texts: List[str]):
    """
    Encode texts with the shared Azure EmbeddingModel (text-embedding-3-small).
    Returns an L2-normalised float32 ndarray of shape (n, 1536), or None if the
    model is unavailable (missing creds, import error, quota, etc.).
    """
    try:
        if str(_PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(_PROJECT_ROOT))
        from MODEL.models import EmbeddingModel
        model = EmbeddingModel()           # singleton, text-embedding-3-small
        vecs  = model.embed(texts)         # ndarray (n, 1536), already L2-normed
        logger.info(f"[Scorer] Dense embeddings OK — {vecs.shape}")
        return vecs
    except Exception as exc:
        logger.warning(f"[Scorer] EmbeddingModel unavailable ({exc}); falling back to TF-IDF")
        return None


# ---------------------------------------------------------------------------
# Relevance Score
# ---------------------------------------------------------------------------

def compute_relevance(paper: dict, keywords: List[str], anti_keywords: List[str]) -> float:
    text = (paper.get("title", "") + " " + paper.get("abstract", "")).lower()
    kw_text = " ".join(paper.get("keywords", [])).lower()
    full = text + " " + kw_text

    hits = sum(1 for kw in keywords if kw.lower() in full)
    anti = sum(1 for kw in anti_keywords if kw.lower() in full)

    # Normalise over a realistic cap (paper can't match all 30+ keywords)
    score = hits / max(min(len(keywords), 12), 1)
    score -= (anti * 0.15)
    return max(0.0, min(1.0, round(score, 3)))


# ---------------------------------------------------------------------------
# ICLR Flavor Score  — fixed formula
# ---------------------------------------------------------------------------
# Groups of signals with individual weights:
#   - Strong theoretical anchors (weight 3): proof, theorem, convergence …
#   - Representational anchors (weight 2): latent, representation, invariant …
#   - General academic quality (weight 1): analysis, generalization, alignment …
#   - Engineering anti-signals (weight -2): system design, user study …

_FLAVOR_SIGNALS = [
    # (substring, weight)
    ("proof",               3), ("theorem",             3), ("convergence",        3),
    ("theoretical analysis",3), ("formal",               3), ("expressivity",       3),
    ("lower bound",         3), ("upper bound",          3), ("regret",             2),
    ("latent space",        2), ("latent",               2), ("representation",     2),
    ("invariant",           2), ("symmetry",             2), ("disentangle",        2),
    ("compositionality",    2), ("world model",          2), ("emergent",           2),
    ("scaling law",         2), ("causal",               2), ("abstraction",        2),
    ("grounding",           1), ("generalization",       1), ("in-context learning",1),
    ("alignment",           1), ("interpretability",     1), ("architecture",       1),
    ("pre-training",        1), ("fine-tuning",          1), ("foundation model",   1),
    ("few-shot",            1), ("zero-shot",            1), ("analysis",           1),
    ("transformer",         1), ("attention mechanism",  1), ("self-supervised",    1),
    # applied/engineering anti-signals
    ("user study",         -2), ("deployed system",     -2), ("case study",        -2),
    ("api wrapper",        -3), ("middleware",          -3), ("scraping",          -3),
    ("engineering system", -2), ("production system",   -2),
]

_FLAVOR_MAX = sum(w for _, w in _FLAVOR_SIGNALS if w > 0)  # theoretical max


def compute_iclr_flavor(paper: dict, config: dict) -> float:
    """
    Weighted signal scoring normalised to [0, 1].
    0.6+ → good ICLR fit.  0.3–0.6 → borderline.  <0.3 → likely rejected.
    """
    text = (paper.get("title", "") + " " + paper.get("abstract", "")).lower()

    # Config overrides (if set) — use weights of 1/-2 for simplicity
    cfg_good  = config.get("iclr_flavor_heuristics", {}).get("good_signals", [])
    cfg_bad   = config.get("iclr_flavor_heuristics", {}).get("bad_signals",  [])
    extra_pos = sum(1  for s in cfg_good if s in text)
    extra_neg = sum(-2 for s in cfg_bad  if s in text)

    raw_score = sum(w for phrase, w in _FLAVOR_SIGNALS if phrase in text)
    raw_score += extra_pos + extra_neg

    # Normalise: cap at _FLAVOR_MAX, floor at 0
    normalised = raw_score / max(_FLAVOR_MAX, 1)

    # Apply a sigmoid-like soft-clamp so scores spread across [0,1] better
    # than pure linear (avoids everything hugging 0)
    scaled = 1 / (1 + math.exp(-8 * (normalised - 0.15)))
    return round(min(1.0, max(0.0, scaled)), 3)


# ---------------------------------------------------------------------------
# Impact-Niche Score  — works without GitHub stars
# ---------------------------------------------------------------------------

def compute_impact_niche(paper: dict, accepted_density: float = 1.0) -> float:
    """
    Score = log(1 + citation_velocity * max(github_stars, citation_count/10)) / density
    Falls back to citation_count when stars=0 (always true for arXiv papers).
    """
    vel    = max(paper.get("citation_velocity", 0.0), 0.0)
    stars  = paper.get("github_stars", 0)
    cit    = paper.get("citation_count", 0)

    # Use citations/10 as a GitHub-star proxy when stars aren't available
    traction = max(stars, cit // 10, 1)
    raw = (vel * traction) / max(accepted_density, 0.001)
    return round(math.log1p(raw), 3)


# ---------------------------------------------------------------------------
# Reviewer stats
# ---------------------------------------------------------------------------

def compute_reviewer_stats(reviews: List[dict]) -> dict:
    if not reviews:
        return {"avg_rating": 0.0, "avg_confidence": 0.0, "num_reviews": 0, "ratings": []}
    ratings     = [r.get("rating", 0.0) for r in reviews]
    confidences = [r.get("confidence", 0.0) for r in reviews]
    return {
        "avg_rating":     round(float(np.mean(ratings)), 2),
        "avg_confidence": round(float(np.mean(confidences)), 2),
        "num_reviews":    len(reviews),
        "ratings":        ratings,
    }


# ---------------------------------------------------------------------------
# Fatal flaw extraction
# ---------------------------------------------------------------------------

FLAW_PATTERNS = [
    (r"too engineering(-| )heavy",                    "Too engineering-heavy for ICLR"),
    (r"(lack|no|lacks|limited).{0,30}(novelty|contribution)", "Limited novelty"),
    (r"(weak|no|missing).{0,30}(theor|formal|proof)", "Lacks theoretical grounding"),
    (r"(narrow|limited).{0,20}(scope|applicability|generali)", "Narrow scope"),
    (r"(no|missing|lack).{0,30}(abla|ablation)",      "Missing ablation studies"),
    (r"(uncl|vague|poorly).{0,20}(motivat|writ)",     "Unclear motivation"),
    (r"(dataset|benchmark).{0,20}(small|limited|bias)","Dataset too small / biased"),
    (r"(unfair|inadeq).{0,20}(baseline|compar)",       "Inadequate baselines"),
]

def extract_fatal_flaws(reviews: List[dict]) -> List[str]:
    flaws: List[str] = []
    for review in reviews:
        rating  = review.get("rating", 5)
        comment = review.get("comment", "").lower()
        if rating <= 4 and comment:
            for pattern, label in FLAW_PATTERNS:
                if re.search(pattern, comment) and label not in flaws:
                    flaws.append(label)
    return flaws


# ---------------------------------------------------------------------------
# White Space detection  — structured narrative: Demand / Exists / Tried / Gap
# ---------------------------------------------------------------------------

def detect_white_spaces(all_papers: List[dict], min_papers: int = 1) -> List[dict]:
    """
    For each research cluster, produce a rich structured description:

      • what_exists   — accepted papers (oral/spotlight/poster) = "Y has been done"
      • attempted     — arXiv-only papers = "Z' tried but not accepted at top venues"
      • demand_signals — citation velocity, trend, stars = "there is demand for X"
      • gap_description — rule-based statement of what specifically is missing
      • opportunity   — template narrative (overridden by LLM in the API endpoint)
      • year_trend    — { "2024": N, "2025": M, … }
    """
    from collections import defaultdict

    cluster_papers: dict = defaultdict(list)
    cluster_labels: dict = {}
    for p in all_papers:
        cid = p.get("cluster_id", 0)
        cluster_papers[cid].append(p)
        cluster_labels[cid] = p.get("cluster_label", f"Cluster {cid}")

    if not cluster_papers:
        return []

    avg_count = sum(len(v) for v in cluster_papers.values()) / len(cluster_papers)

    results = []
    for cid, papers in cluster_papers.items():
        count = len(papers)
        if count < min_papers:
            continue

        # ── Split by acceptance status ──────────────────────────────────────
        ACCEPTED_TIERS = {"oral", "spotlight", "poster"}
        accepted   = [p for p in papers if p.get("tier") in ACCEPTED_TIERS]
        arxiv_only = [p for p in papers if p.get("tier") == "arxiv"]

        avg_vel = sum(p.get("citation_velocity", 0.0) for p in papers) / max(count, 1)
        avg_cit = sum(p.get("citation_count", 0)      for p in papers) / max(count, 1)
        stars   = sum(p.get("github_stars", 0)         for p in papers)

        gap_score = avg_count / max(count, 1)
        traction  = avg_vel * 5 + avg_cit * 0.1 + stars * 0.5

        # ── Year trend ───────────────────────────────────────────────────────
        year_counts: dict = defaultdict(int)
        for p in papers:
            if p.get("year"):
                year_counts[str(p["year"])] += 1
        year_trend = dict(sorted(year_counts.items()))

        # ── Demand signals ───────────────────────────────────────────────────
        demand_signals = []
        if avg_cit >= 50:
            demand_signals.append(f"{avg_cit:.0f} avg citations")
        elif avg_cit >= 10:
            demand_signals.append(f"{avg_cit:.0f} avg cit.")
        if avg_vel >= 0.5:
            demand_signals.append(f"{avg_vel:.1f} cit/mo (fast-growing)")
        elif avg_vel > 0:
            demand_signals.append(f"{avg_vel:.2f} cit/mo")
        if stars > 0:
            demand_signals.append(f"{stars} GitHub ⭐")
        vals = list(year_counts.values())
        if len(vals) >= 2 and vals[-1] > vals[0]:
            demand_signals.append("rising YoY interest")
        if not demand_signals:
            demand_signals.append("emerging niche")

        # ── What exists (Y) — top accepted papers by citation count ─────────
        top_accepted = sorted(accepted, key=lambda x: x.get("citation_count", 0), reverse=True)[:3]
        what_exists = [
            {
                "id":        p.get("id", ""),
                "title":     p.get("title", ""),
                "year":      p.get("year"),
                "tier":      p.get("tier"),
                "citations": p.get("citation_count", 0),
                "arxiv_url": p.get("arxiv_url", ""),
            }
            for p in top_accepted
        ]

        # ── Attempted but not accepted (Z') — arXiv papers by citations ─────
        top_attempted = sorted(arxiv_only, key=lambda x: x.get("citation_count", 0), reverse=True)[:3]
        attempted = [
            {
                "id":        p.get("id", ""),
                "title":     p.get("title", ""),
                "year":      p.get("year"),
                "citations": p.get("citation_count", 0),
                "arxiv_url": p.get("arxiv_url", ""),
            }
            for p in top_attempted
        ]

        # ── Gap description (rule-based; LLM will override in the endpoint) ─
        if len(accepted) == 0 and len(arxiv_only) == 0:
            gap_desc = (
                f"No papers found at top venues or on arXiv — "
                f"this cluster is a genuine unexplored territory."
            )
        elif len(accepted) == 0:
            gap_desc = (
                f"{len(arxiv_only)} arXiv preprint(s) tried this space "
                f"but none reached a top-venue acceptance — high-friction area."
            )
        elif len(arxiv_only) > len(accepted) * 2:
            gap_desc = (
                f"{len(arxiv_only)} arXiv attempts vs only {len(accepted)} accepted "
                f"paper(s) — strong rejection signal; reviewers want deeper theory."
            )
        elif gap_score >= 2.5:
            gap_desc = (
                f"Cluster is {gap_score:.1f}× below average density "
                f"despite {avg_cit:.0f} avg citations — "
                f"significantly under-theorised relative to interest."
            )
        elif len(accepted) <= 2:
            gap_desc = (
                f"Only {len(accepted)} accepted paper(s) with {avg_cit:.0f} avg "
                f"citations each — ample room for rigorous follow-up work."
            )
        else:
            gap_desc = (
                f"{len(accepted)} accepted papers cover the basics; "
                f"sub-problems like benchmarking, theory, and scalability remain open."
            )

        # ── Template opportunity narrative (LLM override in endpoint) ────────
        exists_str = (
            "; ".join(f'"{p["title"][:55]}"' for p in what_exists[:2])
            or "none accepted yet"
        )
        tried_str = (
            "; ".join(f'"{p["title"][:55]}"' for p in attempted[:2])
            or "none on arXiv either"
        )
        opportunity = (
            f"DEMAND: {', '.join(demand_signals[:3])} signal real interest in "
            f"{cluster_labels[cid]}. "
            f"EXISTS: {exists_str}. "
            f"TRIED: {tried_str}. "
            f"GAP: {gap_desc}"
        )

        results.append({
            "cluster_id":      cid,
            "cluster_label":   cluster_labels[cid],
            "paper_count":     count,
            "accepted_count":  len(accepted),
            "arxiv_count":     len(arxiv_only),
            "avg_citations":   round(avg_cit, 1),
            "avg_velocity":    round(avg_vel, 2),
            "total_stars":     stars,
            "gap_score":       round(gap_score, 2),
            "traction_score":  round(traction, 2),
            "year_trend":      year_trend,
            "demand_signals":  demand_signals,
            "what_exists":     what_exists,
            "attempted":       attempted,
            "gap_description": gap_desc,
            "opportunity":     opportunity,   # replaced by LLM in endpoint
        })

    results.sort(key=lambda x: x["gap_score"] * max(x["traction_score"], 0.1), reverse=True)
    return results


# ---------------------------------------------------------------------------
# Clustering  —  dense embeddings (Azure text-embedding-3-small) with
#                TF-IDF→SVD as an automatic fallback
# ---------------------------------------------------------------------------

# Semantic topic keywords used to produce readable auto-labels
# ---------------------------------------------------------------------------
# Cluster label helpers
# ---------------------------------------------------------------------------

# Tier-1: fast keyword → label mapping (most precise)
_TOPIC_LABEL_HINTS = [
    # Agents – modality / environment
    (["code", "program", "software", "debug", "synthesis", "compiler",
      "repository", "swe", "bug", "patch"],                              "Code Generation & Debugging"),
    (["web", "browser", "click", "navigate", "scrape", "html", "dom",
      "selenium", "playwright", "crawl"],                                "Web / Browser Agents"),
    (["email", "calendar", "schedule", "office", "document", "spreadsheet",
      "word", "powerpoint", "outlook", "gmail"],                        "Office Productivity Agents"),
    (["mobile", "android", "ios", "app", "phone", "ui", "gui",
      "interface", "widget", "screen"],                                  "GUI / Mobile Agents"),
    (["robot", "embodied", "manipulation", "navigation", "locomotion",
      "physical", "sim-to-real", "3d", "scene"],                        "Embodied & Robotic Agents"),
    # Multi-agent
    (["multiagent", "multi-agent", "cooperation", "coordination",
      "negotiation", "society", "swarm", "role"],                       "Multi-Agent Systems"),
    # Planning & reasoning
    (["plan", "planning", "horizon", "goal", "decompos", "subgoal",
      "hierarchical", "task graph", "workflow"],                        "Task Planning & Decomposition"),
    (["reason", "chain", "thought", "logic", "inference", "cot",
      "step-by-step", "deduction", "proof"],                            "Reasoning & Chain-of-Thought"),
    (["math", "arithmetic", "equation", "algebra", "calculus",
      "numeric", "symbolic", "theorem"],                                "Mathematical Reasoning"),
    # Memory / retrieval
    (["memory", "retrieval", "knowledge", "rag", "search", "vector store",
      "long-context", "recall", "episodic"],                            "Memory & Retrieval"),
    # Evaluation
    (["benchmark", "eval", "dataset", "metric", "suite", "leaderboard",
      "assessment", "test set", "annotation"],                          "Evaluation & Benchmarks"),
    # Learning paradigms
    (["reward", "reinforcement", "rl", "policy", "feedback", "rlaif",
      "rlhf", "preference", "dpo", "ppo"],                              "RL & Preference Learning"),
    (["fine-tun", "finetun", "instruction tun", "lora", "adapter",
      "peft", "sft", "supervised"],                                     "Fine-Tuning & Alignment Training"),
    (["self-play", "self-improve", "self-refine", "critique", "self-train",
      "iterative", "bootstrap"],                                        "Self-Improvement & Critique"),
    # Representation / architecture
    (["latent", "representation", "embed", "encoding", "feature",
      "disentangle", "manifold"],                                       "Representation Learning"),
    (["attention", "transformer", "architecture", "mamba", "ssm",
      "positional", "layer", "head"],                                   "Architecture & Transformers"),
    (["vision", "visual", "image", "perception", "grounding",
      "multimodal", "vlm", "vqa", "caption"],                          "Vision-Language Models"),
    (["speech", "audio", "sound", "voice", "asr", "tts",
      "spoken", "transcri"],                                            "Speech & Audio"),
    # Safety / alignment
    (["safety", "align", "trust", "bias", "hallucin", "toxic",
      "jailbreak", "red-team", "refuse", "guard"],                     "Safety & Alignment"),
    (["privacy", "federat", "differential privacy", "watermark",
      "copyright", "unlearn"],                                          "Privacy & Security"),
    # Tool use / function calling
    (["tool", "function call", "api call", "plugin", "action",
      "act", "use tool", "external"],                                   "Tool Use & Function Calling"),
    # Infrastructure / efficiency
    (["compress", "quantiz", "prune", "distill", "efficient",
      "inference speed", "latency", "throughput"],                     "Efficiency & Compression"),
    (["scaling", "scale", "scaling law", "emergent", "large model",
      "10b", "70b", "gpt-4", "llama", "mistral"],                      "Scaling & Large Models"),
    # Theory
    (["convergence", "regret", "bound", "complexity", "optimiz",
      "gradient", "loss landscape", "generalization"],                 "Optimization & Theory"),
]

# Tier-2: a broader set of candidate labels for embedding-based matching
# (used when keyword matching returns nothing)
_CANDIDATE_LABELS = [label for _, label in _TOPIC_LABEL_HINTS] + [
    "Natural Language Understanding",
    "Text Classification & NLU",
    "Summarization & Information Extraction",
    "Question Answering",
    "Dialogue & Conversational Agents",
    "Knowledge Graphs & Ontologies",
    "Scientific Discovery Agents",
    "Data Analysis & Visualization Agents",
    "Prompt Engineering",
    "In-Context Learning",
    "Long-Context Modeling",
    "Continual & Lifelong Learning",
    "Causal Inference",
    "Graph Neural Networks",
    "Diffusion Models",
    "Generative Models",
    "Healthcare & Biomedical AI",
    "Code Understanding & Retrieval",
    "Autonomous Driving Agents",
    "Human-AI Interaction",
]

# Cache for candidate label embeddings (computed once per process)
_LABEL_EMBEDDINGS: Optional[np.ndarray] = None


def _embed_candidate_labels() -> Optional[np.ndarray]:
    """Embed all candidate labels with the shared EmbeddingModel (cached)."""
    global _LABEL_EMBEDDINGS
    if _LABEL_EMBEDDINGS is not None:
        return _LABEL_EMBEDDINGS
    vecs = _get_dense_vecs(_CANDIDATE_LABELS)
    if vecs is not None:
        _LABEL_EMBEDDINGS = vecs   # shape (n_labels, 1536) L2-normed
    return _LABEL_EMBEDDINGS


def _tfidf_label(texts: list) -> str:
    """
    Extract the 3 most distinctive terms from cluster texts via TF-IDF and
    return them as a title-cased label string.  Always succeeds.
    """
    import re as _re
    _STOP = {
        "the","a","an","of","in","to","and","or","is","are","for","with",
        "on","that","this","we","our","it","its","by","from","at","be",
        "have","has","as","can","which","their","was","not","but","via",
        "using","use","based","paper","propose","presents","introduces",
        "show","shows","demonstrate","model","method","approach","system",
        "task","data","result","results","performance","across","between",
        "into","such","how","while","also","both","large","language","llm",
        "agent","agents","new","used","than","more","over","each","all",
        "one","two","three","first","second","through","during","when",
    }
    combined = " ".join(texts).lower()
    tokens = _re.findall(r'\b[a-z][a-z\-]{2,}\b', combined)
    freq: dict = {}
    for tok in tokens:
        if tok not in _STOP:
            freq[tok] = freq.get(tok, 0) + 1
    if not freq:
        return "Research Papers"
    top = sorted(freq, key=freq.get, reverse=True)[:4]
    # Capitalise and join prettily
    parts = [w.replace("-", " ").title() for w in top[:3]]
    return " / ".join(parts)


def _semantic_label(texts_in_cluster: list) -> str:
    """
    Always returns a non-empty, human-readable label for a cluster.

    Strategy (in order):
      1. Keyword hit-scoring against _TOPIC_LABEL_HINTS   (fast, deterministic)
      2. Embedding cosine similarity against _CANDIDATE_LABELS  (uses Azure model)
      3. TF-IDF top-term extraction                        (pure text, no model)
    """
    combined = " ".join(texts_in_cluster).lower()

    # ── 1. Keyword scoring ────────────────────────────────────────────────────
    scores: dict = {}
    for keywords, label in _TOPIC_LABEL_HINTS:
        hit = sum(1 for kw in keywords if kw in combined)
        if hit:
            scores[label] = hit
    if scores:
        return max(scores, key=scores.get)

    # ── 2. Embedding similarity against candidate labels ──────────────────────
    try:
        label_vecs = _embed_candidate_labels()
        if label_vecs is not None:
            # Embed the cluster centroid text (truncated for speed)
            centroid_text = combined[:1000]
            cluster_vec = _get_dense_vecs([centroid_text])
            if cluster_vec is not None:
                # cosine sim = dot product (both L2-normed)
                sims = label_vecs @ cluster_vec[0]           # (n_labels,)
                best_idx = int(np.argmax(sims))
                best_score = float(sims[best_idx])
                if best_score > 0.15:                        # require minimal confidence
                    return _CANDIDATE_LABELS[best_idx]
    except Exception as exc:
        logger.debug(f"[Scorer] Embedding label matching failed: {exc}")

    # ── 3. TF-IDF keyword extraction ─────────────────────────────────────────
    return _tfidf_label(texts_in_cluster)


def compute_clusters_and_embeddings(papers: List[dict], n_clusters: int = 8) -> List[dict]:
    """
    Cluster papers using dense vector embeddings (Azure text-embedding-3-small).
    Falls back to TF-IDF + TruncatedSVD when Azure creds are unavailable.
    In both cases, UMAP (cosine metric for dense, euclidean for TF-IDF) is used
    for the 2D layout; PCA is the final fallback when umap-learn is missing.
    """
    if len(papers) < 3:
        # Even for tiny sets, derive a real label from the paper titles/abstracts
        tiny_texts = [
            (p.get("title", "") + " " + (p.get("abstract", "") or "")[:200]).strip()
            for p in papers
        ]
        tiny_label = _semantic_label(tiny_texts)
        for p in papers:
            p.update({"embedding_x": 0.0, "embedding_y": 0.0,
                       "cluster_id": 0, "cluster_label": tiny_label})
        return papers

    # Concatenate title + first 400 chars of abstract for embedding
    texts = [
        (p.get("title", "") + ". " + (p.get("abstract", "") or "")[:400]).strip()
        for p in papers
    ]
    n_clusters = min(n_clusters, len(papers))

    try:
        from sklearn.cluster import KMeans

        # ── Step 1: feature matrix ────────────────────────────────────────────
        vecs      = _get_dense_vecs(texts)   # (n, 1536) float32 or None
        is_dense  = vecs is not None

        if not is_dense:
            # TF-IDF → SVD projection (sparse → 128-d dense)
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            tfidf    = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1, 2))
            X_sparse = tfidf.fit_transform(texts)
            n_comp   = min(128, len(papers) - 1, X_sparse.shape[1] - 1)
            vecs     = TruncatedSVD(n_components=n_comp, random_state=42).fit_transform(X_sparse).astype("float32")
            logger.info(f"[Scorer] TF-IDF fallback: {vecs.shape}")

        # ── Step 2: K-Means on the dense matrix ───────────────────────────────
        km     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(vecs)

        # ── Step 3: human-readable labels via keyword matching ────────────────
        cluster_texts: dict = {cid: [] for cid in range(n_clusters)}
        for i, paper in enumerate(papers):
            cluster_texts[int(labels[i])].append(texts[i])

        auto_labels: dict = {}
        seen_labels: dict = {}          # label → count, for de-duplication
        for cid in range(n_clusters):
            label = _semantic_label(cluster_texts[cid])
            # De-duplicate: if two clusters get the same label, append a qualifier
            if label in seen_labels:
                seen_labels[label] += 1
                label = f"{label} ({seen_labels[label]})"
            else:
                seen_labels[label] = 1
            auto_labels[cid] = label

        # ── Step 4: 2D layout (UMAP cosine on dense, euclidean on TF-IDF SVD) ─
        try:
            import umap
            reducer  = umap.UMAP(
                n_components=2, random_state=42,
                n_neighbors=min(15, len(papers) - 1), min_dist=0.1,
                metric="cosine" if is_dense else "euclidean",
            )
            embedding = reducer.fit_transform(vecs)
        except Exception:
            # Minimal PCA-like 2-D fallback — no extra deps
            centered  = vecs - vecs.mean(axis=0)
            _, _, Vt  = np.linalg.svd(centered, full_matrices=False)
            embedding = centered @ Vt[:2].T

        # ── Step 5: write back ────────────────────────────────────────────────
        for i, paper in enumerate(papers):
            cid = int(labels[i])
            paper["cluster_id"]    = cid
            paper["cluster_label"] = auto_labels[cid]
            paper["embedding_x"]   = round(float(embedding[i, 0]), 4)
            paper["embedding_y"]   = round(float(embedding[i, 1]), 4)

    except ImportError as exc:
        logger.warning(f"[Scorer] Clustering deps missing: {exc} — labelling without clustering")
        # Still label by content even when sklearn/umap are absent
        all_texts = [
            (p.get("title", "") + " " + (p.get("abstract", "") or "")[:200]).strip()
            for p in papers
        ]
        fallback_label = _semantic_label(all_texts)
        for paper in papers:
            paper.update({"embedding_x": 0.0, "embedding_y": 0.0,
                           "cluster_id": 0, "cluster_label": fallback_label})
    return papers


# ---------------------------------------------------------------------------
# Pivot suggestion
# ---------------------------------------------------------------------------

PIVOT_TEMPLATES = [
    (
        ["email", "calendar", "schedule", "meeting", "inbox"],
        0.18,
        "Too applied for ICLR. Workflow automation tools are consistently rejected for 'Low Novelty' and 'Engineering-Heavy'.",
        "78% of similar productivity-tool papers rejected 2024-2025 citing 'system paper without theoretical contribution'.",
        "Pivot to: 'Hierarchical Latent Goal Representations for Multi-Step Sequential Decision-Making in Open-Ended Environments'.",
        "Frame the email domain as an observation space for studying compositional task decomposition in partially-observable MDPs."
    ),
    (
        ["code", "coding", "programming", "debug", "software", "repository"],
        0.55,
        "Moderate ICLR fit — code generation has strong theoretical handles if you emphasise program synthesis and formal semantics.",
        "Accepted papers frame code as a formal language with defined semantics, not as a productivity tool.",
        "Pivot to: 'Execution-Guided Latent Program Synthesis with Self-Supervised Semantic Consistency'.",
        "Ground in formal language theory: programs as structured latent objects; execution feedback as a reward signal with theoretical properties."
    ),
    (
        ["web", "browser", "click", "navigate", "ui", "scrape"],
        0.25,
        "Borderline — web/browser agents are viewed as systems papers without grounding theory.",
        "Acceptance rate for pure browser-automation papers is ~12% at ICLR 2024-2025.",
        "Pivot to: 'Grounded Visual-Linguistic Affordance Learning for Structured Web Navigation'.",
        "Connect visual affordance theory, multimodal representation learning, and grounded language understanding — the navigation is a testbed, not the contribution."
    ),
    (
        ["document", "pdf", "report", "summarize", "summarisation"],
        0.42,
        "Moderate fit — document AI is accepted when framed as hierarchical representation learning.",
        "Pure summarisation pipelines are rejected; papers with novel cross-granularity representations are accepted.",
        "Pivot to: 'Hierarchical Document Representation Learning via Cross-Granularity Contrastive Pretraining'.",
        "Self-supervised learning over paragraph/section/document hierarchy with information-theoretic analysis of the loss."
    ),
    (
        ["search", "retrieval", "rag", "knowledge base", "memory"],
        0.50,
        "Reasonable fit — retrieval-augmented generation has strong ICLR presence if framed around representational efficiency.",
        "Papers framing retrieval as a latent bottleneck or memory architecture problem are accepted.",
        "Pivot to: 'Selective State Compression for Persistent Agent Memory via Latent Abstraction'.",
        "Formalise memory as a lossy compression problem; derive information-theoretic bounds on retrieval quality."
    ),
]

DEFAULT_PIVOT = (
    0.30,
    "Applied AI tools have a lower acceptance rate at ICLR without strong theoretical hooks.",
    "~72% of pure systems/agent engineering papers rejected 2024-2025.",
    "Pivot to: 'Latent Abstraction and Compositional Reasoning in Goal-Conditioned Agentic Systems'.",
    "Frame your domain as a probe for studying emergent planning and representation in foundation models."
)


def generate_pivot(idea: str, config: dict) -> dict:
    idea_lower = idea.lower()
    for kws, flavor, verdict, flaw, pivot, framing in PIVOT_TEMPLATES:
        if any(kw in idea_lower for kw in kws):
            return {"iclr_flavor_score": flavor, "verdict": verdict,
                    "fatal_flaw": flaw, "pivot_suggestion": pivot,
                    "theoretical_framing": framing}
    flavor, verdict, flaw, pivot, framing = DEFAULT_PIVOT
    return {"iclr_flavor_score": flavor, "verdict": verdict,
            "fatal_flaw": flaw, "pivot_suggestion": pivot,
            "theoretical_framing": framing}



# ---------------------------------------------------------------------------
# Relevance Score
# ---------------------------------------------------------------------------

def compute_relevance(paper: dict, keywords: List[str], anti_keywords: List[str]) -> float:
    """
    Score in [0, 1] based on keyword hits in title + abstract.
    Anti-keywords subtract weight.
    """
    text = (paper.get("title", "") + " " + paper.get("abstract", "")).lower()
    kw_text = " ".join(paper.get("keywords", [])).lower()
    full = text + " " + kw_text

    hits = sum(1 for kw in keywords if kw.lower() in full)
    anti = sum(1 for kw in anti_keywords if kw.lower() in full)

    score = hits / max(len(keywords), 1)
    score -= (anti * 0.2)
    return max(0.0, min(1.0, round(score, 3)))


# ---------------------------------------------------------------------------
# ICLR Flavor Score
# ---------------------------------------------------------------------------

ICLR_POSITIVE = [
    "representation", "latent", "latent space", "scaling law", "emergent",
    "symmetry", "generalization", "in-context", "attention", "theoretical",
    "analysis", "interpretability", "architecture", "pre-training",
    "fine-tuning", "alignment", "transformer", "foundation model",
    "formal", "proof", "theorem", "convergence", "expressivity",
    "disentangle", "compositionality", "few-shot", "zero-shot",
    "world model", "causal", "invariant", "universal"
]

ICLR_NEGATIVE = [
    "user study", "deployed", "case study", "api wrapper", "middleware",
    "pipeline", "engineering", "system design", "end-to-end system",
    "production", "enterprise", "scraping", "heuristic", "rule-based"
]


def compute_iclr_flavor(paper: dict, config: dict) -> float:
    """
    Heuristic score: how well the paper matches 'ICLR Flavor'.
    Returns value in [0, 1].  >= 0.5 → good fit, < 0.3 → likely reject.
    """
    text = (paper.get("title", "") + " " + paper.get("abstract", "")).lower()
    good_signals = config.get("iclr_flavor_heuristics", {}).get("good_signals", ICLR_POSITIVE)
    bad_signals  = config.get("iclr_flavor_heuristics", {}).get("bad_signals", ICLR_NEGATIVE)

    positive = sum(1 for s in good_signals if s in text)
    negative = sum(1 for s in bad_signals  if s in text)

    raw = (positive - negative * 1.5) / max(len(good_signals), 1)
    return max(0.0, min(1.0, round(raw, 3)))


# ---------------------------------------------------------------------------
# Impact-Niche Score
# ---------------------------------------------------------------------------

def compute_impact_niche(
    paper: dict,
    accepted_density: float = 1.0
) -> float:
    """
    Score = (citation_velocity * max(github_stars, 1)) / accepted_density
    Normalised log scale so outliers don't dominate.
    """
    vel   = max(paper.get("citation_velocity", 0.0), 0.0)
    stars = max(paper.get("github_stars", 0), 1)
    raw   = (vel * stars) / max(accepted_density, 0.001)
    # Log normalise
    import math
    score = math.log1p(raw)
    return round(score, 3)


# ---------------------------------------------------------------------------
# Reviewer stats
# ---------------------------------------------------------------------------

def compute_reviewer_stats(reviews: List[dict]) -> dict:
    if not reviews:
        return {"avg_rating": 0.0, "avg_confidence": 0.0, "num_reviews": 0, "ratings": []}
    ratings     = [r.get("rating", 0.0) for r in reviews]
    confidences = [r.get("confidence", 0.0) for r in reviews]
    return {
        "avg_rating":     round(float(np.mean(ratings)), 2),
        "avg_confidence": round(float(np.mean(confidences)), 2),
        "num_reviews":    len(reviews),
        "ratings":        ratings,
    }


# ---------------------------------------------------------------------------
# Fatal flaw extraction
# ---------------------------------------------------------------------------

FLAW_PATTERNS = [
    (r"too engineering(-| )heavy", "Too engineering-heavy for ICLR"),
    (r"(lack|no|lacks|limited).{0,30}(novelty|contribution)", "Limited novelty"),
    (r"(weak|no|missing).{0,30}(theor|formal|proof)", "Lacks theoretical grounding"),
    (r"(narrow|limited).{0,20}(scope|applicability|generali)", "Narrow scope / limited generalization"),
    (r"(no|missing|lack).{0,30}(abla|ablation)", "Missing ablation studies"),
    (r"(uncl|vague|poorly).{0,20}(motivat|writ)", "Unclear motivation"),
    (r"(dataset|benchmark).{0,20}(small|limited|bias)", "Dataset too small or biased"),
    (r"(unfair|inadeq).{0,20}(baseline|compar)", "Inadequate baselines"),
]


def extract_fatal_flaws(reviews: List[dict]) -> List[str]:
    """Pull fatal flaws from low-rating review comments."""
    flaws: List[str] = []
    for review in reviews:
        rating = review.get("rating", 5)
        comment = review.get("comment", "").lower()
        if rating <= 4 and comment:
            for pattern, label in FLAW_PATTERNS:
                if re.search(pattern, comment) and label not in flaws:
                    flaws.append(label)
    return flaws


# ---------------------------------------------------------------------------
# Pivot suggestion (rule-based heuristic)
# ---------------------------------------------------------------------------

PIVOT_TEMPLATES = [
    (
        ["email", "calendar", "schedule", "meeting"],
        0.25,
        "Too applied for ICLR. Your idea focuses on a narrow workflow tool.",
        "Reviewers cite 'Low Novelty' and 'Engineering-Heavy' for similar submissions.",
        "Pivot to: 'Hierarchical Latent Goal Representations for Multi-Step Sequential Decision-Making in Open-Ended Environments'.",
        "Frame the email/calendar domain as a testbed for compositional reasoning over partially-observable action spaces."
    ),
    (
        ["code", "coding", "programming", "debug", "software"],
        0.55,
        "Moderate ICLR fit — code generation has theoretical handles if you emphasize program synthesis.",
        "Strong acceptance rate for papers with formal semantics or execution-guided learning.",
        "Pivot to: 'Execution-Guided Latent Program Synthesis with Self-Supervised Semantic Consistency'.",
        "Emphasize the representation of program structure in latent space, not the engineering pipeline."
    ),
    (
        ["web", "browser", "click", "navigate", "scrape"],
        0.30,
        "Borderline for ICLR — web/browser agents are seen as 'systems papers'.",
        "High rejection rate for papers without grounding theory or formal task definition.",
        "Pivot to: 'Grounded Visual-Linguistic Affordance Learning for Structured Web Navigation'.",
        "Connect affordance theory and multimodal representations; frame task as grounded language understanding."
    ),
    (
        ["document", "pdf", "report", "summarize"],
        0.45,
        "Moderate fit — document AI has representation angles if framed correctly.",
        "Papers accepted when they introduce novel representation learning, not just pipelines.",
        "Pivot to: 'Hierarchical Document Representation Learning via Cross-Granularity Contrastive Pretraining'.",
        "Emphasize the self-supervised latent structure, not the downstream task."
    ),
]

DEFAULT_PIVOT = (
    0.35,
    "Applied AI tools have a lower acceptance rate at ICLR without strong theoretical hooks.",
    "Reviewers in 2024-2025 rejected ~72% of pure systems/agent engineering papers.",
    "Pivot to: 'Latent Abstraction and Compositional Reasoning in Goal-Conditioned Agentic Systems'.",
    "Frame your domain as a probe for studying emergent planning and representation in foundation models."
)


def generate_pivot(idea: str, config: dict) -> dict:
    idea_lower = idea.lower()
    for keywords, flavor, verdict, flaw, pivot, framing in PIVOT_TEMPLATES:
        if any(kw in idea_lower for kw in keywords):
            return {
                "iclr_flavor_score": flavor,
                "verdict": verdict,
                "fatal_flaw": flaw,
                "pivot_suggestion": pivot,
                "theoretical_framing": framing
            }
    flavor, verdict, flaw, pivot, framing = DEFAULT_PIVOT
    return {
        "iclr_flavor_score": flavor,
        "verdict": verdict,
        "fatal_flaw": flaw,
        "pivot_suggestion": pivot,
        "theoretical_framing": framing
    }
