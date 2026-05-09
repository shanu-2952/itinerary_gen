#!/usr/bin/env python3
"""
Stage 4: RAG Retrieval — Retrieve activities, dining, and logistics from ChromaDB.

Filters by tier, occasion, travel_type, and applies personalization
(excludes past trips/hotels). Day-type-aware queries for arrival/full/transit/departure.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import chromadb
except ImportError:
    chromadb = None

try:
    from scripts.rules_engine import DayConstraints
except ImportError:
    from rules_engine import DayConstraints

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_db")


@dataclass
class RetrievedItem:
    """A single item retrieved from the knowledge base."""
    item_id: str
    item_type: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class TravelKnowledgeBase:
    """Wrapper around ChromaDB for travel knowledge retrieval."""

    def __init__(self, chroma_dir: str = CHROMA_DIR):
        """Initialize ChromaDB client and load collections."""
        if chromadb is None:
            raise ImportError("chromadb not installed. Run: pip install chromadb")

        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.activities = self.client.get_collection("activities")
        self.dining = self.client.get_collection("dining")
        self.logistics = self.client.get_collection("logistics")

    def _query_collection(
        self,
        collection,
        query: str,
        n_results: int = 5,
        where_filter: dict = None
    ) -> List[RetrievedItem]:
        """Query a ChromaDB collection and return ranked results."""

        kwargs = {
            "query_texts": [query],
            "n_results": n_results
        }

        if where_filter:
            kwargs["where"] = where_filter

        try:
            results = collection.query(**kwargs)
        except Exception:
            # Retry without filter if metadata mismatch
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )

        items = []

        if not results or not results["ids"] or not results["ids"][0]:
            return items

        for i, doc_id in enumerate(results["ids"][0]):

            score = 1.0

            if results.get("distances") and results["distances"][0]:
                dist = results["distances"][0][i]
                score = max(0, 1.0 - dist)

            content = (
                results["documents"][0][i]
                if results.get("documents")
                else ""
            )

            meta = (
                results["metadatas"][0][i]
                if results.get("metadatas")
                else {}
            )

            items.append(
                RetrievedItem(
                    item_id=doc_id,
                    item_type=collection.name,
                    content=content,
                    score=score,
                    metadata=meta,
                )
            )

        return items

    def _build_query(
        self,
        day_type: str,
        city: str,
        tier: str,
        travel_type: str,
        occasion: str,
        area: str = ""
    ) -> str:
        """Build targeted retrieval query."""

        queries = {
            "arrival": (
                f"Easy welcome activity for {travel_type} in {city} {area}. "
                f"{tier} tier. First evening arrival. "
                f"Light, accessible, relaxing. Occasion: {occasion}."
            ),

            "full": (
                f"Best {tier} activity for {travel_type} in {city} {area}. "
                f"Full day exploration. Occasion: {occasion}. "
                f"Cultural, scenic, immersive luxury experiences."
            ),

            "transit": (
                f"Quick half-day activity near {city} center or hotel. "
                f"{tier} tier {travel_type}. "
                f"Short duration, easy access. Transit day."
            ),

            "departure": (
                f"Light morning activity in {city} before departure. "
                f"{tier} {travel_type}. Relaxed, near hotel or airport."
            ),
        }

        return queries.get(day_type, queries["full"])

    def retrieve_for_day(
        self,
        dc: DayConstraints,
        tier: str,
        travel_type: str,
        occasion: str,
        past_trips: List[str] = None,
        past_hotels: List[str] = None,
        already_planned: List[str] = None,
    ) -> Dict[str, List[RetrievedItem]]:
        """
        Retrieve activities, dining, and logistics for one day.
        """

        area_hint = ", ".join(dc.must_include) if dc.must_include else ""

        query = self._build_query(
            dc.day_type,
            dc.city,
            tier,
            travel_type,
            occasion,
            area_hint
        )

        # -------------------------
        # ACTIVITIES
        # -------------------------
        activities = self._query_collection(
            self.activities,
            query,
            n_results=8,
            where_filter={"city": dc.city},
        )

        # -------------------------
        # DINING
        # -------------------------
        dining_query = (
            f"Luxury dining for {travel_type} in {dc.city}. "
            f"{tier} tier restaurant. Occasion: {occasion}. "
            f"Romantic, premium, memorable experience."
        )

        dining = self._query_collection(
            self.dining,
            dining_query,
            n_results=5,
            where_filter={"city": dc.city},
        )

        # -------------------------
        # LOGISTICS
        # -------------------------
        logistics = self._query_collection(
            self.logistics,
            f"Transport and logistics for {dc.city}",
            n_results=2,
            where_filter={"city": dc.city},
        )

        # -------------------------
        # PERSONALIZATION FILTERS
        # -------------------------
        if past_trips:
            activities = self._filter_past_trips(
                activities,
                past_trips
            )

            dining = self._filter_past_trips(
                dining,
                past_trips
            )

        # -------------------------
        # STRONGER SIMILARITY FILTER
        # -------------------------
        activities = [
            a for a in activities
            if a.score >= 0.25
        ]

        dining = [
            d for d in dining
            if d.score >= 0.25
        ]

        # -------------------------
        # REMOVE REPETITION (FIXED FOR BOTH)
        # -------------------------
        if already_planned:
            used_names = set(x.lower().strip() for x in already_planned)
            
            # Filter Activities — match by item_id or normalized name
            filtered_activities = []
            for a in activities:
                a_id = a.item_id.lower().strip()
                # Extract name from content (first line or before " in ")
                a_name = a.content.split(".")[0].split(" in ")[0].lower().strip()
                if a_id not in used_names and a_name not in used_names:
                    filtered_activities.append(a)
            activities = filtered_activities

            # Filter Dining
            filtered_dining = []
            for d in dining:
                d_id = d.item_id.lower().strip()
                d_name = d.content.split(".")[0].split(" in ")[0].lower().strip()
                if d_id not in used_names and d_name not in used_names:
                    filtered_dining.append(d)
            dining = filtered_dining

        # -------------------------
        # SORT BY BEST SCORE
        # -------------------------
        activities.sort(
            key=lambda x: x.score,
            reverse=True
        )

        dining.sort(
            key=lambda x: x.score,
            reverse=True
        )

        # -------------------------
        # KEEP ONLY TOP RESULTS
        # -------------------------
        activities = activities[:4]
        dining = dining[:3]

        return {
            "activities": activities,
            "dining": dining,
            "logistics": logistics,
        }

    def _filter_past_trips(
        self,
        results: List[RetrievedItem],
        past_trips: List[str]
    ) -> List[RetrievedItem]:
        """Remove previously visited places."""

        filtered = []

        past_lower = [
            p.lower()
            for p in past_trips
        ]

        for item in results:

            city = item.metadata.get(
                "city",
                ""
            ).lower()

            country = item.metadata.get(
                "country",
                ""
            ).lower()

            if city not in past_lower and country not in past_lower:
                filtered.append(item)

        return filtered

    def format_for_llm(
        self,
        retrieved: Dict[str, List[RetrievedItem]],
        max_chars: int = 1800
    ) -> str:
        """
        Format compact retrieval context for the LLM.
        """

        parts = []
        char_count = 0

        # -------------------------
        # ACTIVITIES
        # -------------------------
        acts = retrieved.get("activities", [])

        if acts:

            parts.append(
                "== AVAILABLE ACTIVITIES =="
            )

            for a in acts:

                line = (
                    f"- [{a.item_id}] "
                    f"{a.content[:120]}"
                )

                if char_count + len(line) > max_chars:
                    break

                parts.append(line)
                char_count += len(line)

        # -------------------------
        # DINING
        # -------------------------
        dins = retrieved.get("dining", [])

        if dins:

            parts.append(
                "\n== DINING OPTIONS =="
            )

            for d in dins:

                line = (
                    f"- [{d.item_id}] "
                    f"{d.content[:120]}"
                )

                if char_count + len(line) > max_chars:
                    break

                parts.append(line)
                char_count += len(line)

        # -------------------------
        # LOGISTICS
        # -------------------------
        logs = retrieved.get("logistics", [])

        if logs:

            parts.append(
                "\n== LOGISTICS =="
            )

            for l in logs:

                line = f"- {l.content[:150]}"

                if char_count + len(line) > max_chars:
                    break

                parts.append(line)
                char_count += len(line)

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# TEST
# ---------------------------------------------------------------------------

def main():

    kb = TravelKnowledgeBase()

    dc = DayConstraints(
        day_number=1,
        date="2026-03-15",
        city="Tokyo",
        hotel_name="Hotel Gracery Shinjuku",
        day_type="arrival",
        energy_level="low",
        max_activities=2,
        usable_hours_start="11:00",
        usable_hours_end="21:00",
    )

    results = kb.retrieve_for_day(
        dc,
        tier="LuxPlus",
        travel_type="couple",
        occasion="anniversary",
    )

    print("[Stage 4] Knowledge retrieval test")
    print(f"  → Activities: {len(results['activities'])}")
    print(f"  → Dining: {len(results['dining'])}")
    print(f"  → Logistics: {len(results['logistics'])}")

    print()
    print(kb.format_for_llm(results))


if __name__ == "__main__":
    main()