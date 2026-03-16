"""
cache_store.py — Obsidian-style Markdown disk cache for ConferencePapers.

Every unique (conference, year, topic) query is stored as a single .md file:

    cache/ICLR_2024_agents_for_productivity.md

File anatomy:
  ─────────────────────────────────────────────────
  ---                                 ← YAML front-matter (query metadata)
  conference: ICLR
  year: 2024
  topic: Agents for Productivity
  topic_slug: agents_for_productivity
  fetched_at: 2026-03-16T00:12:34
  ttl_days: 7
  total_papers: 42
  ---

  # ICLR 2024 — Agents for Productivity

  ## [[Latent Space Planning for LLM Agents]]   ← wiki-link title
  - id: `openreview_abc123`
  - tier: `poster`
  - year: `2024`
  - relevance_score: `0.82`
  - iclr_flavor_score: `0.71`
  ...
  > Abstract text here

  ## [[Next Paper Title]]
  ...
  ─────────────────────────────────────────────────

A master `cache/_index.md` records every query ever made (wiki-linked).

Usage:
    store = CacheStore(cache_dir, ttl_days=7)
    papers = store.get("ICLR", 2024, "Agents for Productivity")  # None on miss
    store.put("ICLR", 2024, "Agents for Productivity", papers_list)
    store.list_queries()  # → [{"key": ..., "conference": ..., ...}]
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # PyYAML — already a transitive dep; fallback below

logger = logging.getLogger(__name__)

# ── YAML shim (PyYAML may not be installed) ──────────────────────────────────
try:
    import yaml as _yaml

    def _yaml_load(s: str) -> dict:
        return _yaml.safe_load(s) or {}

    def _yaml_dump(d: dict) -> str:
        return _yaml.dump(d, default_flow_style=False, allow_unicode=True).strip()

except ImportError:
    # Minimal YAML for our flat key: value frontmatter
    def _yaml_load(s: str) -> dict:  # type: ignore[misc]
        out: dict = {}
        for line in s.splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                out[k.strip()] = v.strip()
        return out

    def _yaml_dump(d: dict) -> str:  # type: ignore[misc]
        return "\n".join(f"{k}: {v}" for k, v in d.items())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _cache_key(conference: str, year: int, topic: str) -> str:
    return f"{conference.upper()}_{year}_{_slugify(topic)}"


# ── Paper → Markdown section ─────────────────────────────────────────────────

_PAPER_FIELDS = [
    "id", "tier", "year", "decision",
    "relevance_score", "iclr_flavor_score", "impact_niche_score",
    "citation_count", "citation_velocity", "github_stars",
    "cluster_id", "cluster_label",
    "arxiv_id", "arxiv_url", "openreview_url", "pdf_url",
    "semantic_scholar_id",
    "embedding_x", "embedding_y",
]

_LIST_FIELDS = ["authors", "keywords", "fatal_flaws"]


def _paper_to_md(paper: dict) -> str:
    title = paper.get("title", "(untitled)").replace("|", "｜")
    lines = [f"## [[{title}]]"]
    for field in _PAPER_FIELDS:
        val = paper.get(field)
        if val is not None and val != "" and val != 0.0:
            lines.append(f"- {field}: `{val}`")
    for field in _LIST_FIELDS:
        vals = paper.get(field, [])
        if vals:
            lines.append(f"- {field}: `{', '.join(str(v) for v in vals)}`")
    # Reviewer stats inline
    rs = paper.get("reviewer_stats") or {}
    if isinstance(rs, dict) and rs.get("num_reviews"):
        lines.append(f"- reviewer_avg_rating: `{rs.get('avg_rating', 0)}`")
        lines.append(f"- reviewer_avg_confidence: `{rs.get('avg_confidence', 0)}`")
        lines.append(f"- reviewer_num_reviews: `{rs.get('num_reviews', 0)}`")
    # Abstract as blockquote
    abstract = (paper.get("abstract") or "").strip()
    if abstract:
        wrapped = textwrap.fill(abstract, width=120)
        lines.append("> " + wrapped.replace("\n", "\n> "))
    return "\n".join(lines)


def _md_to_paper(section: str) -> dict:
    """Parse a single ## [[Title]] section back into a dict."""
    paper: dict = {}

    # Title
    title_match = re.match(r"##\s+\[\[(.+?)\]\]", section.strip())
    if title_match:
        paper["title"] = title_match.group(1)

    # Scalar fields
    for m in re.finditer(r"-\s+(\w+):\s+`([^`]*)`", section):
        key, val = m.group(1), m.group(2)
        if key in _LIST_FIELDS:
            paper[key] = [v.strip() for v in val.split(",") if v.strip()]
        else:
            # Type coerce
            try:
                if "." in val:
                    paper[key] = float(val)
                else:
                    paper[key] = int(val)
            except (ValueError, TypeError):
                paper[key] = val

    # Reviewer stats reconstruction
    if "reviewer_avg_rating" in paper:
        paper["reviewer_stats"] = {
            "avg_rating":     paper.pop("reviewer_avg_rating", 0.0),
            "avg_confidence": paper.pop("reviewer_avg_confidence", 0.0),
            "num_reviews":    int(paper.pop("reviewer_num_reviews", 0)),
            "ratings":        [],
        }

    # Abstract (blockquote)
    abstract_match = re.search(r"^>(.+)", section, re.MULTILINE | re.DOTALL)
    if abstract_match:
        raw = abstract_match.group(1)
        paper["abstract"] = re.sub(r"\n>\s*", " ", raw).strip()

    return paper


# ── CacheStore ────────────────────────────────────────────────────────────────

class CacheStore:
    """
    Disk-backed, Obsidian-style Markdown cache for query results.

    Directory layout::

        cache/
          _index.md                                 ← master query log
          ICLR_2024_agents_for_productivity.md
          ICLR_2025_agents_for_productivity.md
          ICLR_2026_agents_for_productivity.md
    """

    INDEX_FILE = "_index.md"

    def __init__(self, cache_dir: Path, ttl_days: int = 7):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_days = ttl_days
        logger.info(f"[Cache] dir={self.cache_dir}  ttl={ttl_days}d")

    # ── Public API ────────────────────────────────────────────────────────────

    def get(
        self,
        conference: str,
        year: int,
        topic: str,
        ignore_ttl: bool = False,
    ) -> Optional[List[dict]]:
        """
        Return cached papers or None on cache-miss / stale.
        """
        path = self._path(conference, year, topic)
        if not path.exists():
            logger.debug(f"[Cache MISS] {path.name}")
            return None

        raw = path.read_text(encoding="utf-8")
        fm, body = self._split_frontmatter(raw)
        if not fm:
            return None

        # TTL check
        if not ignore_ttl:
            fetched_at_str = fm.get("fetched_at", "")
            if fetched_at_str:
                try:
                    fetched_at = datetime.fromisoformat(fetched_at_str).replace(tzinfo=timezone.utc)
                    age_days = (datetime.now(timezone.utc) - fetched_at).days
                    ttl = int(fm.get("ttl_days", self.ttl_days))
                    if age_days > ttl:
                        logger.info(f"[Cache STALE] {path.name} is {age_days}d old (ttl={ttl}d)")
                        return None
                except ValueError:
                    pass

        papers = self._parse_papers(body)
        logger.info(f"[Cache HIT] {path.name} → {len(papers)} papers")
        return papers if papers else None

    def put(
        self,
        conference: str,
        year: int,
        topic: str,
        papers: List[dict],
    ) -> None:
        """Write/overwrite a cache entry."""
        key  = _cache_key(conference, year, topic)
        path = self.cache_dir / f"{key}.md"

        fm_data = {
            "conference": conference,
            "year":        year,
            "topic":       topic,
            "topic_slug":  _slugify(topic),
            "fetched_at":  _now_iso(),
            "ttl_days":    self.ttl_days,
            "total_papers": len(papers),
        }

        header = (
            f"---\n{_yaml_dump(fm_data)}\n---\n\n"
            f"# {conference} {year} — {topic}\n\n"
            f"_Generated by ConferencePapers · {_now_iso()} UTC_\n\n"
            f"**{len(papers)} papers matched**\n\n"
            "---\n\n"
        )

        sections = "\n\n".join(_paper_to_md(p) for p in papers)
        path.write_text(header + sections, encoding="utf-8")
        logger.info(f"[Cache WRITE] {path.name} ({len(papers)} papers)")
        self._update_index(conference, year, topic, key, len(papers))

    def list_queries(self) -> List[dict]:
        """Return metadata for every cached query file."""
        results = []
        for md_file in sorted(self.cache_dir.glob("*.md")):
            if md_file.name.startswith("_"):
                continue
            raw = md_file.read_text(encoding="utf-8")
            fm, _ = self._split_frontmatter(raw)
            if fm:
                results.append(
                    {
                        "key":           md_file.stem,
                        "conference":    fm.get("conference", ""),
                        "year":          fm.get("year", ""),
                        "topic":         fm.get("topic", ""),
                        "fetched_at":    fm.get("fetched_at", ""),
                        "total_papers":  fm.get("total_papers", 0),
                        "ttl_days":      fm.get("ttl_days", self.ttl_days),
                        "file":          md_file.name,
                        "size_kb":       round(md_file.stat().st_size / 1024, 1),
                    }
                )
        return results

    def invalidate(self, conference: str, year: int, topic: str) -> bool:
        """Delete a specific cache entry.  Returns True if file existed."""
        path = self._path(conference, year, topic)
        if path.exists():
            path.unlink()
            logger.info(f"[Cache INVALIDATE] {path.name}")
            self._update_index_remove(_cache_key(conference, year, topic))
            return True
        return False

    def invalidate_all(self) -> int:
        """Delete all cache files (not _index). Returns count deleted."""
        count = 0
        for md_file in self.cache_dir.glob("*.md"):
            if not md_file.name.startswith("_"):
                md_file.unlink()
                count += 1
        (self.cache_dir / self.INDEX_FILE).write_text(self._empty_index(), encoding="utf-8")
        logger.info(f"[Cache INVALIDATE_ALL] {count} files removed")
        return count

    def read_raw(self, key: str) -> Optional[str]:
        """Return raw markdown for a cache key (for API passthrough)."""
        path = self.cache_dir / f"{key}.md"
        return path.read_text(encoding="utf-8") if path.exists() else None

    # ── Internals ─────────────────────────────────────────────────────────────

    def _path(self, conference: str, year: int, topic: str) -> Path:
        return self.cache_dir / f"{_cache_key(conference, year, topic)}.md"

    @staticmethod
    def _split_frontmatter(text: str):
        """Return (frontmatter_dict, body_str). Both empty on failure."""
        text = text.lstrip()
        if not text.startswith("---"):
            return {}, text
        end = text.find("\n---", 3)
        if end == -1:
            return {}, text
        fm_str = text[3:end].strip()
        body   = text[end + 4:].strip()
        try:
            fm = _yaml_load(fm_str)
        except Exception:
            fm = {}
        return fm, body

    @staticmethod
    def _parse_papers(body: str) -> List[dict]:
        """Split body on ## [[...]] headings and parse each section."""
        sections = re.split(r"(?=^## \[\[)", body, flags=re.MULTILINE)
        papers = []
        for sec in sections:
            if sec.startswith("## [["):
                p = _md_to_paper(sec)
                if p.get("title"):
                    papers.append(p)
        return papers

    # ── Index maintenance ─────────────────────────────────────────────────────

    def _update_index(self, conference: str, year: int, topic: str, key: str, count: int):
        index_path = self.cache_dir / self.INDEX_FILE
        existing = index_path.read_text(encoding="utf-8") if index_path.exists() else self._empty_index()

        # Remove existing row if present
        existing = re.sub(rf"^\|.*`{re.escape(key)}`.*\n", "", existing, flags=re.MULTILINE)

        # Append new row before the last line
        new_row = (
            f"| {conference} | {year} | {topic} | {_now_iso()} | {count} "
            f"| [[{key}]] |\n"
        )
        # Insert after table header separator
        if "| --- |" in existing:
            existing = re.sub(
                r"(\| --- \| --- \| --- \| --- \| --- \| --- \|\n)",
                r"\1" + new_row,
                existing,
            )
        else:
            existing += new_row

        # Update header timestamp
        existing = re.sub(
            r"last_updated:.*",
            f"last_updated: {_now_iso()}",
            existing,
        )
        existing = re.sub(
            r"total_queries:.*",
            f"total_queries: {self._count_rows(existing)}",
            existing,
        )
        index_path.write_text(existing, encoding="utf-8")

    def _update_index_remove(self, key: str):
        index_path = self.cache_dir / self.INDEX_FILE
        if not index_path.exists():
            return
        text = index_path.read_text(encoding="utf-8")
        text = re.sub(rf"^\|.*`{re.escape(key)}`.*\n", "", text, flags=re.MULTILINE)
        index_path.write_text(text, encoding="utf-8")

    @staticmethod
    def _count_rows(text: str) -> int:
        return len(re.findall(r"^\| [A-Z]", text, re.MULTILINE))

    @staticmethod
    def _empty_index() -> str:
        return textwrap.dedent(f"""\
            ---
            last_updated: {_now_iso()}
            total_queries: 0
            ---

            # ConferencePapers — Query Index

            _This file is auto-maintained by the cache engine._
            _Each row links to a cached result file (Obsidian wiki-link)._

            | Conference | Year | Topic | Fetched At | Papers | File |
            | --- | --- | --- | --- | --- | --- |
            """)
