"""
embedding_store.py
Stores paper embeddings alongside the Obsidian-style .md cache files.

Uses Azure OpenAI EmbeddingModel from /Project/MODEL/models.py
(singleton, backed by DefaultAzureCredential).

Default deployment: text-embedding-3-small  (1536-d, controlled by
EMBEDDING_DEPLOYMENT env-var or the model_name parameter).

Files:
  cache/ICLR_2024_agents_for_productivity.npz   ← embeddings + paper IDs
  (one .npz per cache entry, keyed by same slug as the .md file)

Usage:
    store = EmbeddingStore(cache_dir)
    store.build(key, papers)
    results = store.search([key1, key2], query_text, top_k=10)
    # → [{"id": ..., "title": ..., "score": 0.93, "key": ...}, ...]
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap MODEL package path (works regardless of working directory)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

_EMBED_MODEL = None   # lazy singleton


def _get_model(deployment: str | None = None) -> "EmbeddingModel | None":  # noqa: F821
    """Return (and cache) the EmbeddingModel singleton."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        name = deployment or os.environ.get("EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
        try:
            from MODEL.models import EmbeddingModel
            logger.info(f"[Embeddings] Initialising EmbeddingModel({name!r}) …")
            _EMBED_MODEL = EmbeddingModel(name)
            logger.info(f"[Embeddings] Ready — {_EMBED_MODEL}")
        except Exception as exc:
            logger.warning(f"[Embeddings] Could not initialise EmbeddingModel: {exc}")
            _EMBED_MODEL = None
    return _EMBED_MODEL


class EmbeddingStore:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── File path ──────────────────────────────────────────────────────────────

    def _npz_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.npz"

    # ── Build (encode and save) ────────────────────────────────────────────────

    def build(self, key: str, papers: List[dict], deployment: str | None = None) -> bool:
        """Encode title+abstract of every paper and save as .npz. Returns True on success."""
        model = _get_model(deployment)
        if model is None:
            return False

        texts  = [
            (p.get("title", "") + ". " + (p.get("abstract", "") or "")).strip()
            for p in papers
        ]
        ids    = [p.get("id", str(i)) for i, p in enumerate(papers)]
        titles = [p.get("title", "") for p in papers]

        try:
            vecs = model.embed(texts)   # L2-normalised float32 ndarray
            npz  = self._npz_path(key)
            np.savez_compressed(
                npz,
                embeddings=vecs,        # already float32
                ids=np.array(ids, dtype=object),
                titles=np.array(titles, dtype=object),
            )
            logger.info(f"[Embeddings] Built {npz.name} — {len(papers)} vectors")
            return True
        except Exception as e:
            logger.warning(f"[Embeddings] Build failed for {key}: {e}")
            return False

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        keys: List[str],
        query: str,
        top_k: int = 10,
        deployment: str | None = None,
    ) -> List[dict]:
        """
        Semantic search across one or more cache keys.
        Returns top_k results ordered by cosine similarity.
        """
        model = _get_model(deployment)
        if model is None:
            return []

        try:
            q_vec = model.embed_one(query)  # shape (dim,), L2-normalised
        except Exception as e:
            logger.warning(f"[Embeddings] Query encode failed: {e}")
            return []

        all_results = []
        for key in keys:
            npz_path = self._npz_path(key)
            if not npz_path.exists():
                logger.debug(f"[Embeddings] Missing {npz_path.name}")
                continue
            try:
                data   = np.load(npz_path, allow_pickle=True)
                vecs   = data["embeddings"]    # (N, dim) float32
                ids    = data["ids"].tolist()
                titles = data["titles"].tolist()

                # Cosine similarity (vectors are already L2-normalised)
                scores = (vecs @ q_vec).tolist()

                for pid, title, score in zip(ids, titles, scores):
                    all_results.append({
                        "id":    pid,
                        "title": title,
                        "score": round(float(score), 4),
                        "key":   key,
                    })
            except Exception as e:
                logger.warning(f"[Embeddings] Search error on {key}: {e}")

        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    # ── Multi-query RRF search ─────────────────────────────────────────────────

    def search_rrf(
        self,
        keys: List[str],
        queries: List[str],
        top_k: int = 10,
        rrf_k: int = 60,
        deployment: str | None = None,
    ) -> List[dict]:
        """
        RAG-Fusion: dense retrieval for *each* query then Reciprocal Rank Fusion.

        RRF score for paper p = Σ_i  1 / (rrf_k + rank_i(p))

        All queries are embedded in a single batch for efficiency.
        Returns top_k results ordered by fused score, each entry includes:
          id, title, key, rrf_score, best_score (highest single-query cosine)
        """
        if not queries:
            return []
        model = _get_model(deployment)
        if model is None:
            # Graceful degradation: fall back to single-query search
            return self.search(keys, queries[0], top_k, deployment)

        try:
            q_vecs = model.embed(queries)   # (n_queries, dim) L2-normalised
        except Exception as e:
            logger.warning(f"[Embeddings] RRF query encode failed: {e}")
            return []

        # rrf_acc maps paper_id → {"rrf": float, "best_score": float, "title": str, "key": str}
        rrf_acc: dict[str, dict] = {}

        for key in keys:
            npz_path = self._npz_path(key)
            if not npz_path.exists():
                continue
            try:
                data   = np.load(npz_path, allow_pickle=True)
                vecs   = data["embeddings"]              # (N, dim)
                ids    = data["ids"].tolist()
                titles = data["titles"].tolist()

                # score_matrix: (N, n_queries) — cosine sims (L2-normalised)
                score_matrix = vecs @ q_vecs.T           # fast matmul

                # For each query column, calculate 1/(rrf_k + rank) contribution
                # ranks are 1-indexed; np.argsort descending
                n_papers, n_queries = score_matrix.shape
                rrf_contributions = np.zeros(n_papers, dtype=np.float64)
                best_scores = np.max(score_matrix, axis=1)

                for qi in range(n_queries):
                    col = score_matrix[:, qi]
                    # argsort ascending → reverse for descending rank
                    order = np.argsort(col)[::-1]   # indices from best to worst
                    ranks = np.empty_like(order)
                    ranks[order] = np.arange(1, n_papers + 1)
                    rrf_contributions += 1.0 / (rrf_k + ranks)

                for idx, (pid, title) in enumerate(zip(ids, titles)):
                    rrf_score = float(rrf_contributions[idx])
                    best      = float(best_scores[idx])
                    if pid not in rrf_acc or rrf_score > rrf_acc[pid]["rrf_score"]:
                        rrf_acc[pid] = {
                            "id":         pid,
                            "title":      title,
                            "key":        key,
                            "rrf_score":  rrf_score,
                            "best_score": round(best, 4),
                        }
            except Exception as e:
                logger.warning(f"[Embeddings] RRF search error on {key}: {e}")

        results = sorted(rrf_acc.values(), key=lambda x: x["rrf_score"], reverse=True)
        # Normalise rrf_score to [0,1] for display purposes
        if results:
            max_rrf = results[0]["rrf_score"]
            if max_rrf > 0:
                for r in results:
                    r["rrf_score"] = round(r["rrf_score"] / max_rrf, 4)
        return results[:top_k]

    # ── Utilities ──────────────────────────────────────────────────────────────

    def exists(self, key: str) -> bool:
        return self._npz_path(key).exists()

    def delete(self, key: str) -> bool:
        npz = self._npz_path(key)
        if npz.exists():
            npz.unlink()
            return True
        return False

    def list_keys(self) -> List[str]:
        return [p.stem for p in self.cache_dir.glob("*.npz")]
