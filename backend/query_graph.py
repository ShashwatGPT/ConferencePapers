"""
query_graph.py — Persistent query relationship graph for RAG-Fusion.

Tracks which cache keys are semantically "related" (co-expanded from the same
root query) so that future searches for similar topics can load previously-
fetched papers from disk without hitting external APIs again.

Relationship model
──────────────────
When the user first searches for "Agents for Productivity":

  expand_query("Agents for Productivity")
  → ["Agents for Productivity",             ← root (key: ICLR_2024_agents_for_productivity)
     "LLM-based agents for task automation", ← child 1
     "autonomous agent workflow",             ← child 2
     "survey of agents for productivity",     ← child 3
     "agent tool use in NLP"]                 ← child 4

Each child gets its own cache key and its own disk .md + .npz files.
The graph records root → [child1, child2, child3, child4].

When the user later searches for "autonomous agent workflow":
  get_related_keys("ICLR_2024_autonomous_agent_workflow")
  → ["ICLR_2024_autonomous_agent_workflow",   ← self
     "ICLR_2024_agents_for_productivity",      ← root (parent)
     "ICLR_2024_llm_based_agents_...",         ← sibling
     "ICLR_2024_survey_of_agents_...",         ← sibling
     "ICLR_2024_agent_tool_use_..."]           ← sibling

So the _fetch_year_papers call for "autonomous agent workflow" can:
 1. Load siblings from L1/L2 (cache hit — no API call)
 2. Expand "autonomous agent workflow" into its own NEW variants → L3 fetch
 3. Run RRF across ALL keys: siblings + its own new expansion variants

Storage: cache/_query_graph.json
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_GRAPH_FILE = "_query_graph.json"


class QueryGraph:
    """Thread-safe, file-backed directed graph of query expansions."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self._path = self.cache_dir / _GRAPH_FILE
        self._graph: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self):
        if self._path.exists():
            try:
                self._graph = json.loads(self._path.read_text(encoding="utf-8"))
                logger.info(f"[QueryGraph] Loaded {len(self._graph)} nodes from {_GRAPH_FILE}")
            except Exception as exc:
                logger.warning(f"[QueryGraph] Load failed ({exc}); starting fresh")
                self._graph = {}

    def _save(self):
        try:
            self._path.write_text(
                json.dumps(self._graph, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning(f"[QueryGraph] Save failed: {exc}")

    # ── Mutations ──────────────────────────────────────────────────────────────

    def register_root(
        self,
        root_key: str,
        root_query: str,
        conference: str,
        year: int,
        child_keys: List[str],
        child_queries: List[str],
    ):
        """
        Record that root_query was expanded into child_queries.
        Idempotent: re-registering with the same keys only adds new children.
        """
        with self._lock:
            # Upsert root node — merge any existing expansion_keys
            root_node = self._graph.get(root_key, {})
            existing_children = set(root_node.get("expansion_keys", []))
            merged_children = list(existing_children | set(child_keys))
            root_node.update(
                {
                    "query": root_query,
                    "conference": conference,
                    "year": year,
                    "timestamp": _now_iso(),
                    "root_key": None,
                    "expansion_keys": merged_children,
                }
            )
            self._graph[root_key] = root_node

            # Upsert each child node
            for key, query in zip(child_keys, child_queries):
                child = self._graph.get(key, {})
                child.update(
                    {
                        "query": query,
                        "conference": conference,
                        "year": year,
                        "timestamp": _now_iso(),
                        "root_key": root_key,
                        # preserve any children this node already has from its own expansion
                        "expansion_keys": child.get("expansion_keys", []),
                    }
                )
                self._graph[key] = child

            self._save()
            logger.debug(
                f"[QueryGraph] Registered root={root_key!r} "
                f"→ {len(child_keys)} children"
            )

    def remove_key(self, key: str):
        """Remove a node and clean up all back-references to it."""
        with self._lock:
            if key not in self._graph:
                return
            root_key = self._graph[key].get("root_key")
            if root_key and root_key in self._graph:
                exps = self._graph[root_key].get("expansion_keys", [])
                self._graph[root_key]["expansion_keys"] = [k for k in exps if k != key]
            del self._graph[key]
            self._save()
            logger.debug(f"[QueryGraph] Removed node {key!r}")

    def reconcile(self, existing_cache_keys: Set[str]):
        """
        Drop graph nodes whose .md cache file no longer exists on disk.
        Call this on startup after CacheStore._reconcile_index().
        """
        with self._lock:
            stale = [k for k in list(self._graph) if k not in existing_cache_keys]
            for k in stale:
                root_key = self._graph[k].get("root_key")
                if root_key and root_key in self._graph:
                    exps = self._graph[root_key].get("expansion_keys", [])
                    self._graph[root_key]["expansion_keys"] = [e for e in exps if e != k]
                del self._graph[k]
            if stale:
                logger.info(f"[QueryGraph] Reconciled — removed {len(stale)} stale nodes")
                self._save()

    def clear(self):
        """Remove all nodes (called when the whole disk cache is wiped)."""
        with self._lock:
            self._graph.clear()
            self._save()
        logger.info("[QueryGraph] Cleared all nodes")

    # ── Queries ────────────────────────────────────────────────────────────────

    def get_related_keys(self, key: str) -> List[str]:
        """
        Return all cache keys related to `key`:
          - key itself
          - its root parent (if key is a child)
          - all siblings (other children of the same root)
          - key's own expansion children (if key is itself a root)

        Only returns keys that exist in the graph — callers decide whether
        those keys are actually populated in L1/L2.
        """
        with self._lock:
            if key not in self._graph:
                return [key]

            related: Set[str] = {key}
            node = self._graph[key]

            root_key = node.get("root_key")
            if root_key and root_key in self._graph:
                related.add(root_key)
                # All siblings (children of the same root, including self)
                for sibling in self._graph[root_key].get("expansion_keys", []):
                    related.add(sibling)

            # Key's own expansion children
            for child in node.get("expansion_keys", []):
                related.add(child)

        return list(related)

    def __len__(self) -> int:
        with self._lock:
            return len(self._graph)

    def __repr__(self) -> str:
        return f"QueryGraph({len(self)} nodes, path={self._path})"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
