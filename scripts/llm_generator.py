#!/usr/bin/env python3
"""
Stage 5: Smart Scheduler — Deterministic itinerary builder using KB data.

Replaces the LLM-per-day approach with a deterministic scheduler that SELECTS
activities/dining directly from the knowledge base. Zero hallucinations, zero
latency. LLM is an optional enrichment layer behind --use-llm flag.

Architecture:
    1. SmartScheduler.build_day() — deterministic day builder
       - Picks activities from KB matching city/tier/travel_type/occasion/energy
       - Picks dining from KB matching city/tier/occasion/meal_type
       - Tracks already_used set across all days — zero repeats guaranteed
       - Applies day_type templates (arrival/full/transit/departure)
       - Inserts occasion peak content on peak days

    2. LLMEnricher (optional) — enriches descriptions via Ollama
       - Single batch call for entire itinerary (not per-day)
       - Only modifies descriptions, summaries, tips
       - Falls back gracefully to deterministic descriptions

Usage:
    from scripts.llm_generator import SmartScheduler
    scheduler = SmartScheduler()
    days = scheduler.build_full_itinerary(constraints, ctx, activities, dining, logistics, spatial)
"""

import json
import os
import re
import requests
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

try:
    from scripts.deal_parser import ParsedItineraryContext
    from scripts.rules_engine import DayConstraints
    from scripts.spatial_sequencer import SpatialContext
except ImportError:
    from deal_parser import ParsedItineraryContext
    from rules_engine import DayConstraints
    from spatial_sequencer import SpatialContext

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "llama3.1:latest")


# ---------------------------------------------------------------------------
# Helper: description personalization (deterministic, no LLM needed)
# ---------------------------------------------------------------------------

def _personalize_description(base_desc: str, ctx: ParsedItineraryContext,
                             dc: DayConstraints) -> str:
    """Add occasion and travel-type flavor to a KB description."""
    suffix_parts = []

    if ctx.occasion == "anniversary":
        suffix_parts.append("A romantic experience to share with your partner.")
    elif ctx.occasion == "honeymoon":
        suffix_parts.append("A dreamy moment for your honeymoon.")
    elif ctx.occasion == "birthday":
        suffix_parts.append("A memorable way to celebrate your special day.")
    elif ctx.occasion == "school_holiday":
        suffix_parts.append("Fun for the whole family.")

    if dc.is_occasion_peak:
        suffix_parts.append("The highlight of your trip.")

    if suffix_parts:
        return f"{base_desc} {' '.join(suffix_parts)}"
    return base_desc


def _time_label(time_str: str) -> str:
    """Convert HH:MM to morning/afternoon/evening."""
    hour = int(time_str.split(":")[0])
    if hour < 12:
        return "morning"
    elif hour < 17:
        return "afternoon"
    return "evening"


# ---------------------------------------------------------------------------
# Smart Scheduler
# ---------------------------------------------------------------------------

class SmartScheduler:
    """
    Deterministic itinerary builder. Picks activities and dining directly
    from KB data using rule-based matching. Zero LLM calls, zero hallucinations.
    """

    def __init__(self):
        self.already_used: Set[str] = set()

    def _match_activity(self, poi: dict, city: str, tier: str,
                        travel_type: str, occasion: str,
                        energy: str) -> float:
        """
        Score a POI for relevance. Higher = better match.
        Returns 0 if hard-filtered out.
        """
        # Hard filter: wrong city
        if poi.get("city", "").lower() != city.lower():
            return 0.0

        # Hard filter: tier not supported
        poi_tiers = poi.get("tier", [])
        if tier not in poi_tiers:
            return 0.0

        # Hard filter: already used
        name_key = poi.get("name", "").lower().strip()
        if name_key in self.already_used:
            return 0.0

        score = 1.0

        # Travel type match
        poi_travel = poi.get("travel_type", [])
        if travel_type in poi_travel:
            score += 2.0

        # Occasion match
        poi_occasions = poi.get("occasion", [])
        if occasion in poi_occasions:
            score += 2.0

        # Energy match
        poi_energy = poi.get("energy", "moderate")
        if energy == "low" and poi_energy == "low":
            score += 1.0
        elif energy == "high" and poi_energy in ("moderate", "high"):
            score += 1.0
        elif energy == "moderate" and poi_energy == "moderate":
            score += 0.5

        # Rating bonus
        rating = poi.get("rating", 0)
        if rating >= 4.7:
            score += 1.0
        elif rating >= 4.5:
            score += 0.5

        return score

    def _match_dining(self, din: dict, city: str, tier: str,
                      travel_type: str, occasion: str,
                      meal: str = "dinner") -> float:
        """Score a dining option for relevance."""
        if din.get("city", "").lower() != city.lower():
            return 0.0

        din_tiers = din.get("tier", [])
        if tier not in din_tiers:
            return 0.0

        name_key = din.get("name", "").lower().strip()
        if name_key in self.already_used:
            return 0.0

        score = 1.0

        # Meal type match
        if din.get("meal", "") == meal:
            score += 2.0

        # Travel type match
        if travel_type in din.get("travel_type", []):
            score += 1.5

        # Occasion match
        if occasion in din.get("occasion", []):
            score += 2.0

        # Rating
        rating = din.get("rating", 0)
        if rating >= 4.7:
            score += 1.0

        return score

    def _pick_best(self, scored_items: List[tuple], n: int = 1) -> list:
        """Pick top-n items from (score, item) list, excluding 0-scored."""
        filtered = [(s, item) for s, item in scored_items if s > 0]
        filtered.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in filtered[:n]]

    def _make_schedule_item(self, source: dict, time: str,
                            item_type: str, ctx: ParsedItineraryContext,
                            dc: DayConstraints,
                            is_from_kb: bool = True) -> dict:
        """Create a schedule item dict from a KB source."""
        desc = source.get("description", "")
        if is_from_kb:
            desc = _personalize_description(desc, ctx, dc)

        return {
            "time": time,
            "time_label": _time_label(time),
            "type": item_type,
            "name": source.get("name", ""),
            "description": desc,
            "duration": source.get("duration", "1.5 hours"),
            "area": source.get("area", dc.city),
            "tier_note": f"Matches {ctx.tier} tier and {ctx.customer.travel_type} travel type",
            "is_from_knowledge_base": is_from_kb,
            "booking_required": source.get("booking_required", False),
            "booking_note": source.get("booking_note", ""),
        }

    def _build_arrival_day(self, dc: DayConstraints,
                           ctx: ParsedItineraryContext,
                           activities: list, dining: list,
                           logistics: list,
                           areas: List[str]) -> dict:
        """Build schedule for an arrival day."""
        schedule = []

        # Check-in
        checkin_time = dc.usable_hours_start
        schedule.append({
            "time": checkin_time,
            "time_label": _time_label(checkin_time),
            "type": "checkin",
            "name": f"Check-in at {dc.hotel_name}",
            "description": f"Arrive and settle into {dc.hotel_name}. Freshen up and relax after your journey.",
            "duration": "1.5 hours",
            "area": dc.hotel_name.split()[-1] if dc.hotel_name else dc.city,
            "tier_note": f"{ctx.tier} property",
            "is_from_knowledge_base": False,
            "booking_required": False,
            "booking_note": "",
        })

        # One light activity if time permits (energy=low, prefer evening/afternoon)
        scored = [(self._match_activity(a, dc.city, ctx.tier,
                                         ctx.customer.travel_type,
                                         ctx.occasion, "low"), a)
                  for a in activities]
        picks = self._pick_best(scored, 1)
        if picks:
            poi = picks[0]
            schedule.append(self._make_schedule_item(
                poi, "15:00", "activity", ctx, dc
            ))
            self.already_used.add(poi["name"].lower().strip())

        # Dinner
        scored_din = [(self._match_dining(d, dc.city, ctx.tier,
                                           ctx.customer.travel_type,
                                           ctx.occasion, "dinner"), d)
                      for d in dining]
        din_picks = self._pick_best(scored_din, 1)
        if din_picks:
            din = din_picks[0]
            schedule.append(self._make_schedule_item(
                din, "19:00", "dining", ctx, dc
            ))
            self.already_used.add(din["name"].lower().strip())
        else:
            schedule.append({
                "time": "19:00", "time_label": "evening", "type": "rest",
                "name": "Evening at leisure",
                "description": f"Explore the surroundings of {dc.hotel_name} at your own pace. Enjoy a relaxed first evening.",
                "duration": "2 hours", "area": dc.city,
                "tier_note": "Relaxed first evening",
                "is_from_knowledge_base": False,
                "booking_required": False, "booking_note": "",
            })

        return self._wrap_day(dc, ctx, schedule, areas,
                              f"Smooth arrival and settling into {dc.city}")

    def _build_full_day(self, dc: DayConstraints,
                        ctx: ParsedItineraryContext,
                        activities: list, dining: list,
                        logistics: list,
                        areas: List[str]) -> dict:
        """Build schedule for a full exploration day."""
        schedule = []

        # Morning activity
        energy = dc.energy_level
        scored = [(self._match_activity(a, dc.city, ctx.tier,
                                         ctx.customer.travel_type,
                                         ctx.occasion, energy), a)
                  for a in activities]
        picks = self._pick_best(scored, dc.max_activities)

        if picks:
            # First activity: morning
            poi1 = picks[0]
            schedule.append(self._make_schedule_item(
                poi1, "09:30", "activity", ctx, dc
            ))
            self.already_used.add(poi1["name"].lower().strip())

        # Lunch option (if we have lunch-type dining)
        scored_lunch = [(self._match_dining(d, dc.city, ctx.tier,
                                             ctx.customer.travel_type,
                                             ctx.occasion, "lunch"), d)
                        for d in dining]
        lunch_picks = self._pick_best(scored_lunch, 1)
        if lunch_picks:
            lunch = lunch_picks[0]
            schedule.append(self._make_schedule_item(
                lunch, "12:30", "dining", ctx, dc
            ))
            self.already_used.add(lunch["name"].lower().strip())

        # Downtime (Rule 7: mandatory on full days)
        schedule.append({
            "time": "14:00", "time_label": "afternoon", "type": "rest",
            "name": "Afternoon downtime",
            "description": f"Rest and recharge at {dc.hotel_name}. "
                           f"Take a nap, read, or explore the hotel amenities at your own pace.",
            "duration": "1.5 hours", "area": "Hotel",
            "tier_note": "Balanced pacing per Rule 7",
            "is_from_knowledge_base": False,
            "booking_required": False, "booking_note": "",
        })

        # Second activity (afternoon, if max_activities > 1 and available)
        if len(picks) > 1:
            poi2 = picks[1]
            schedule.append(self._make_schedule_item(
                poi2, "16:00", "activity", ctx, dc
            ))
            self.already_used.add(poi2["name"].lower().strip())

        # Dinner
        scored_din = [(self._match_dining(d, dc.city, ctx.tier,
                                           ctx.customer.travel_type,
                                           ctx.occasion, "dinner"), d)
                      for d in dining]
        din_picks = self._pick_best(scored_din, 1)
        if din_picks:
            din = din_picks[0]
            schedule.append(self._make_schedule_item(
                din, "19:00", "dining", ctx, dc
            ))
            self.already_used.add(din["name"].lower().strip())

        summary = f"Full day exploring {dc.city}"
        if dc.is_occasion_peak:
            summary = f"A special {ctx.occasion} celebration in {dc.city} — the highlight of your trip"

        return self._wrap_day(dc, ctx, schedule, areas, summary)

    def _build_transit_day(self, dc: DayConstraints,
                           ctx: ParsedItineraryContext,
                           activities: list, dining: list,
                           logistics: list,
                           areas: List[str]) -> dict:
        """Build schedule for a transit day (city transfer). Max 1 activity."""
        schedule = []

        # City transfer (morning)
        # Determine the previous city from context
        prev_city = dc.city  # Default
        schedule.append({
            "time": "09:00", "time_label": "morning", "type": "transport",
            "name": f"Transfer to {dc.city}",
            "description": f"Check out and transfer to {dc.city}. Settle into {dc.hotel_name}.",
            "duration": "2 hours", "area": dc.city,
            "tier_note": f"{ctx.tier_rules.transport_level} transport",
            "is_from_knowledge_base": False,
            "booking_required": False, "booking_note": "",
        })

        # Max 1 light activity (Rule 6)
        scored = [(self._match_activity(a, dc.city, ctx.tier,
                                         ctx.customer.travel_type,
                                         ctx.occasion, "low"), a)
                  for a in activities]
        picks = self._pick_best(scored, 1)
        if picks:
            poi = picks[0]
            schedule.append(self._make_schedule_item(
                poi, "15:00", "activity", ctx, dc
            ))
            self.already_used.add(poi["name"].lower().strip())
        else:
            schedule.append({
                "time": "15:00", "time_label": "afternoon", "type": "rest",
                "name": "Settling in",
                "description": f"Explore the surroundings of {dc.hotel_name} at your own pace. "
                               f"Relax after the transfer.",
                "duration": "2 hours", "area": dc.city,
                "tier_note": "Transit day — light schedule",
                "is_from_knowledge_base": False,
                "booking_required": False, "booking_note": "",
            })

        # Dinner
        scored_din = [(self._match_dining(d, dc.city, ctx.tier,
                                           ctx.customer.travel_type,
                                           ctx.occasion, "dinner"), d)
                      for d in dining]
        din_picks = self._pick_best(scored_din, 1)
        if din_picks:
            din = din_picks[0]
            schedule.append(self._make_schedule_item(
                din, "19:00", "dining", ctx, dc
            ))
            self.already_used.add(din["name"].lower().strip())

        return self._wrap_day(dc, ctx, schedule, areas,
                              f"Transit to {dc.city} — settling in and light exploration")

    def _build_departure_day(self, dc: DayConstraints,
                             ctx: ParsedItineraryContext,
                             activities: list, dining: list,
                             logistics: list,
                             areas: List[str]) -> dict:
        """Build schedule for departure day. Light morning + airport transfer."""
        schedule = []

        # Checkout
        schedule.append({
            "time": "08:00", "time_label": "morning", "type": "checkout",
            "name": f"Check-out from {dc.hotel_name}",
            "description": f"Pack and check out of {dc.hotel_name}. Enjoy a final breakfast at the hotel.",
            "duration": "1 hour", "area": "Hotel",
            "tier_note": "",
            "is_from_knowledge_base": False,
            "booking_required": False, "booking_note": "",
        })

        # Light morning activity if there's time (only if usable_hours_end > 12:00)
        end_hour = int(dc.usable_hours_end.split(":")[0])
        if end_hour >= 12:
            scored = [(self._match_activity(a, dc.city, ctx.tier,
                                             ctx.customer.travel_type,
                                             ctx.occasion, "low"), a)
                      for a in activities]
            picks = self._pick_best(scored, 1)
            if picks:
                poi = picks[0]
                schedule.append(self._make_schedule_item(
                    poi, "09:30", "activity", ctx, dc
                ))
                self.already_used.add(poi["name"].lower().strip())

        # Airport transfer
        transfer_time = dc.usable_hours_end
        schedule.append({
            "time": transfer_time, "time_label": _time_label(transfer_time),
            "type": "transport",
            "name": f"Airport transfer — departure to {ctx.flight.return_city}",
            "description": f"Transfer to the airport for your departure flight back to {ctx.flight.return_city}.",
            "duration": "1.5 hours", "area": "Airport",
            "tier_note": f"{ctx.tier_rules.transport_level} transfer",
            "is_from_knowledge_base": False,
            "booking_required": False, "booking_note": "",
        })

        return self._wrap_day(dc, ctx, schedule, areas,
                              f"Farewell to {dc.city} — departure to {ctx.flight.return_city}")

    def _wrap_day(self, dc: DayConstraints, ctx: ParsedItineraryContext,
                  schedule: list, areas: List[str], summary: str) -> dict:
        """Wrap a schedule into a full day dict."""
        # Get local tip from logistics if available
        local_tip = ""
        area_focus = areas[0] if areas else dc.city

        return {
            "day_number": dc.day_number,
            "date": dc.date,
            "city": dc.city,
            "hotel": dc.hotel_name,
            "day_type": dc.day_type,
            "area_focus": area_focus,
            "schedule": schedule,
            "day_summary": summary,
            "local_tip": local_tip,
        }

    def build_day(self, dc: DayConstraints, ctx: ParsedItineraryContext,
                  activities: list, dining: list, logistics: list,
                  areas: List[str]) -> dict:
        """Route to the appropriate day builder based on day_type."""
        builders = {
            "arrival": self._build_arrival_day,
            "full": self._build_full_day,
            "transit": self._build_transit_day,
            "departure": self._build_departure_day,
        }
        builder = builders.get(dc.day_type, self._build_full_day)
        return builder(dc, ctx, activities, dining, logistics, areas)

    def build_full_itinerary(self, constraints: List[DayConstraints],
                             ctx: ParsedItineraryContext,
                             all_activities: list,
                             all_dining: list,
                             all_logistics: list,
                             spatial: Optional[SpatialContext]) -> List[dict]:
        """
        Build the complete itinerary deterministically.

        Args:
            constraints: DayConstraints from Stage 2.
            ctx: ParsedItineraryContext from Stage 1.
            all_activities: Raw POI data from dummy_pois.json.
            all_dining: Raw dining data from dummy_dining.json.
            all_logistics: Raw logistics data from dummy_logistics.json.
            spatial: SpatialContext from Stage 3 (or None).

        Returns:
            List of day dicts matching the expected schema.
        """
        self.already_used = set()
        days = []

        for dc in constraints:
            areas = []
            if spatial and dc.day_number in spatial.day_clusters:
                areas = spatial.day_clusters[dc.day_number]

            day_data = self.build_day(
                dc, ctx, all_activities, all_dining, all_logistics, areas
            )

            # Add local tips from logistics
            for log in all_logistics:
                if log.get("city", "").lower() == dc.city.lower():
                    tips = log.get("local_tips", [])
                    if tips:
                        # Cycle through tips based on day number
                        tip_idx = (dc.day_number - 1) % len(tips)
                        day_data["local_tip"] = tips[tip_idx]
                    break

            days.append(day_data)

        return days


# ---------------------------------------------------------------------------
# Optional LLM Enricher (behind --use-llm flag)
# ---------------------------------------------------------------------------

class LLMEnricher:
    """
    Optional: Enrich deterministic itinerary with LLM-generated descriptions.
    Sends a SINGLE batch call for the entire itinerary (not per-day).
    """

    def __init__(self, model: str = MODEL_NAME, base_url: str = OLLAMA_URL):
        self.model = model
        self.base_url = base_url

    def enrich(self, days: List[dict], ctx: ParsedItineraryContext) -> List[dict]:
        """
        Enrich day summaries and local tips using a single LLM call.
        Falls back gracefully if LLM unavailable.
        """
        # Build a compact summary of all days for one batch call
        compact = []
        for d in days:
            activities = [s["name"] for s in d.get("schedule", [])
                         if s.get("type") in ("activity", "dining")]
            compact.append({
                "day": d["day_number"],
                "city": d["city"],
                "type": d["day_type"],
                "activities": activities,
            })

        prompt = f"""You are a luxury travel writer for Travel Jaunts.

Given this {ctx.tier} itinerary for a {ctx.customer.travel_type} trip to {ctx.destination}
(occasion: {ctx.occasion}), write a brief day_summary (1 sentence) and local_tip
(1 practical tip) for each day.

Itinerary:
{json.dumps(compact, indent=2)}

Respond as a JSON array of objects with keys: day, day_summary, local_tip.
JSON only, no markdown."""

        try:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "system": "You are a luxury travel content writer. Respond only with valid JSON.",
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.3, "num_predict": 1500},
            }
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            raw = resp.json().get("response", "")

            # Parse enrichment
            enrichments = self._extract_json(raw)
            if enrichments and isinstance(enrichments, list):
                for e in enrichments:
                    day_num = e.get("day", 0)
                    for d in days:
                        if d["day_number"] == day_num:
                            if e.get("day_summary"):
                                d["day_summary"] = e["day_summary"]
                            if e.get("local_tip"):
                                d["local_tip"] = e["local_tip"]
                            break

        except Exception as e:
            print(f"    LLM enrichment skipped: {e}")

        return days

    def _extract_json(self, raw: str) -> Any:
        """Extract JSON from LLM response."""
        text = raw.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Find [ to ]
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        return None


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

def main():
    """Test SmartScheduler with dummy data."""
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(PROJECT_ROOT, "data")

    with open(os.path.join(data_dir, "dummy_deals.json"), "r", encoding="utf-8") as f:
        deals = json.load(f)
    with open(os.path.join(data_dir, "dummy_customer_profiles.json"), "r", encoding="utf-8") as f:
        customers = json.load(f)
    with open(os.path.join(data_dir, "dummy_pois.json"), "r", encoding="utf-8") as f:
        activities = json.load(f)
    with open(os.path.join(data_dir, "dummy_dining.json"), "r", encoding="utf-8") as f:
        dining = json.load(f)
    with open(os.path.join(data_dir, "dummy_logistics.json"), "r", encoding="utf-8") as f:
        logistics = json.load(f)

    try:
        from scripts.deal_parser import parse_deal
        from scripts.rules_engine import apply_rules
    except ImportError:
        from deal_parser import parse_deal
        from rules_engine import apply_rules

    ctx = parse_deal(deals[0], customers[0])
    constraints = apply_rules(ctx)

    scheduler = SmartScheduler()
    print(f"[Stage 5] SmartScheduler test — {ctx.destination} ({ctx.tier})")
    days = scheduler.build_full_itinerary(
        constraints, ctx, activities, dining, logistics, None
    )

    for d in days:
        act_names = [s["name"] for s in d["schedule"]
                     if s["type"] in ("activity", "dining")]
        kb_count = sum(1 for s in d["schedule"] if s["is_from_knowledge_base"])
        print(f"  Day {d['day_number']} ({d['day_type']}): "
              f"{len(d['schedule'])} items, {kb_count} from KB — "
              f"{', '.join(act_names)}")


if __name__ == "__main__":
    main()
