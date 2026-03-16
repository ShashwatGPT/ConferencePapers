"""
openreview_extractor.py
Pulls accepted (and rejected) papers from OpenReview for a given ICLR venue.
Uses openreview-py SDK (API v2).
"""
import logging
from typing import List, Optional
import openreview

logger = logging.getLogger(__name__)

# Map tier strings to canonical names
TIER_MAP = {
    "oral": "oral",
    "spotlight": "spotlight",
    "poster": "poster",
    "reject": "rejected",
    "none": "rejected",
}


def _normalize_tier(decision_str: str) -> str:
    s = decision_str.lower()
    for key, val in TIER_MAP.items():
        if key in s:
            return val
    return "rejected"


class OpenReviewExtractor:
    def __init__(self, config: dict, conference: str = "ICLR"):
        self.config = config
        self.conference = conference.upper()
        # Use guest (unauthenticated) client for public data
        self.client = openreview.api.OpenReviewClient(
            baseurl=config["apis"]["openreview"]["base_url"]
        )

    def _get_venue_cfg(self, year: int) -> dict:
        """Look up venue config; raises KeyError if not configured for this conference+year."""
        conf_cfg = self.config["conferences"].get(self.conference, {})
        year_cfg = conf_cfg.get(str(year))
        if not year_cfg:
            raise KeyError(f"No OpenReview config for {self.conference} {year}")
        return year_cfg

    def fetch_accepted_papers(self, year: int) -> List[dict]:
        """
        Returns a list of raw paper dicts for accepted papers at ICLR {year}.
        Each dict has: id, title, abstract, authors, decision, tier,
                        openreview_url, reviews (list of {rating, confidence})
        """
        vcfg = self._get_venue_cfg(year)
        venue_id = vcfg["venue_id"]
        tiers_accepted = set(vcfg["acceptance_tiers"].values())

        logger.info(f"[OpenReview] Fetching {venue_id} ...")

        try:
            # For ICLR 2024+ use the `venue` field on notes
            submissions = self.client.get_all_notes(
                invitation=f"{venue_id}/-/Submission",
                details="directReplies"
            )
        except Exception as e:
            logger.warning(f"[OpenReview] API v2 failed ({e}), trying fallback ...")
            submissions = self._fallback_fetch(venue_id)

        papers = []
        for note in submissions:
            content = note.content if hasattr(note, "content") else {}
            title_raw = content.get("title", {})
            title = title_raw.get("value", "") if isinstance(title_raw, dict) else str(title_raw)

            # Determine decision
            decision = self._extract_decision(note)
            tier = _normalize_tier(decision)

            if tier == "rejected":
                continue  # Skip rejected papers (fetch separately for funnel)

            abstract_raw = content.get("abstract", {})
            abstract = abstract_raw.get("value", "") if isinstance(abstract_raw, dict) else str(abstract_raw)

            authors_raw = content.get("authors", {})
            if isinstance(authors_raw, dict):
                authors = authors_raw.get("value", [])
            elif isinstance(authors_raw, list):
                authors = authors_raw
            else:
                authors = []

            keywords_raw = content.get("keywords", {})
            if isinstance(keywords_raw, dict):
                keywords = keywords_raw.get("value", [])
            elif isinstance(keywords_raw, list):
                keywords = keywords_raw
            else:
                keywords = []

            # Reviews
            reviews = self._extract_reviews(note)

            papers.append({
                "id": note.id,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "keywords": keywords,
                "decision": decision,
                "tier": tier,
                "openreview_url": f"https://openreview.net/forum?id={note.id}",
                "year": year,
                "reviews": reviews,
            })

        logger.info(f"[OpenReview] Found {len(papers)} accepted papers for {self.conference} {year}")
        return papers

    def fetch_all_submissions_count(self, year: int) -> dict:
        """Return counts: {oral, spotlight, poster, rejected, total}"""
        vcfg = self._get_venue_cfg(year)
        venue_id = vcfg["venue_id"]
        try:
            submissions = self.client.get_all_notes(
                invitation=f"{venue_id}/-/Submission",
                details="directReplies"
            )
        except Exception:
            return {}

        counts = {"oral": 0, "spotlight": 0, "poster": 0, "rejected": 0, "total": 0}
        for note in submissions:
            decision = self._extract_decision(note)
            tier = _normalize_tier(decision)
            counts[tier] = counts.get(tier, 0) + 1
            counts["total"] += 1
        return counts

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_decision(self, note) -> str:
        """Look inside directReplies for a decision note."""
        try:
            replies = (note.details or {}).get("directReplies", []) if hasattr(note, "details") else []
            for reply in replies:
                inv = reply.get("invitations", [""])
                if any("decision" in i.lower() for i in inv):
                    content = reply.get("content", {})
                    decision_raw = content.get("decision", {})
                    if isinstance(decision_raw, dict):
                        return decision_raw.get("value", "Reject")
                    return str(decision_raw)
            # Fallback: check venue tag
            content = note.content if hasattr(note, "content") else {}
            venue_raw = content.get("venue", {})
            venue = venue_raw.get("value", "") if isinstance(venue_raw, dict) else str(venue_raw)
            if "oral" in venue.lower():
                return "Accept (oral)"
            if "spotlight" in venue.lower():
                return "Accept (spotlight)"
            if "poster" in venue.lower():
                return "Accept (poster)"
        except Exception:
            pass
        return "Reject"

    def _extract_reviews(self, note) -> List[dict]:
        reviews = []
        try:
            replies = (note.details or {}).get("directReplies", []) if hasattr(note, "details") else []
            for reply in replies:
                inv = reply.get("invitations", [""])
                if any("official_review" in i.lower() or "review" in i.lower() for i in inv):
                    content = reply.get("content", {})
                    rating_raw = content.get("rating", content.get("soundness", {}))
                    confidence_raw = content.get("confidence", {})

                    def _num(raw):
                        val = raw.get("value", 0) if isinstance(raw, dict) else raw
                        try:
                            return float(str(val).split(":")[0].strip())
                        except Exception:
                            return 0.0

                    reviews.append({
                        "rating": _num(rating_raw),
                        "confidence": _num(confidence_raw),
                    })
        except Exception:
            pass
        return reviews

    def _fallback_fetch(self, venue_id: str):
        """Try older API v1 style."""
        try:
            client_v1 = openreview.Client(baseurl="https://api.openreview.net")
            return client_v1.get_all_notes(invitation=f"{venue_id}/-/Blind_Submission")
        except Exception:
            return []
