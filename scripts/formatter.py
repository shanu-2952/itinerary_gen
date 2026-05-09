#!/usr/bin/env python3
"""
Stage 6: Formatter — Validate all rules and format final output.

Implements ItineraryValidator (checks all business rules) and
ItineraryFormatter (Pydantic models + human-readable + API output).

Usage:
    from scripts.formatter import ItineraryValidator, ItineraryFormatter
    validator = ItineraryValidator()
    result = validator.validate(itinerary, context)
    formatter = ItineraryFormatter()
    output = formatter.to_api_response(itinerary, result)
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

try:
    from scripts.deal_parser import ParsedItineraryContext
    from scripts.rules_engine import DayConstraints
except ImportError:
    from deal_parser import ParsedItineraryContext
    from rules_engine import DayConstraints


# ---------------------------------------------------------------------------
# Pydantic Models — API Contract with Frontend
# ---------------------------------------------------------------------------

class ScheduleItem(BaseModel):
    """A single scheduled item within a day."""
    time: str = Field(..., description="Start time in HH:MM format")
    time_label: str = Field(..., description="morning|afternoon|evening")
    type: str = Field(..., description="activity|dining|transport|rest|checkin|checkout")
    name: str = Field(..., description="Activity or venue name")
    description: str = Field(..., description="2-3 sentence personalized description")
    duration: str = Field(..., description="Duration string e.g. '2 hours'")
    area: str = Field("", description="Neighborhood/area name")
    tier_note: str = Field("", description="Why this fits the tier")
    is_from_knowledge_base: bool = Field(False, description="True if from KB")
    booking_required: bool = Field(False, description="Whether booking needed")
    booking_note: str = Field("", description="Booking instructions")


class DayItinerary(BaseModel):
    """Complete itinerary for a single day."""
    day_number: int
    date: str
    city: str
    hotel: str
    day_type: str
    area_focus: str = ""
    schedule: List[ScheduleItem] = []
    day_summary: str = ""
    local_tip: str = ""


class FullItinerary(BaseModel):
    """Complete multi-day itinerary — the final API output model."""
    deal_id: str
    deal_type: str
    tier: str
    destination: str
    customer_name: str
    travel_type: str
    occasion: str
    total_days: int
    total_nights: int
    checkin: str
    checkout: str
    days: List[DayItinerary] = []
    validation: Optional[Dict[str, Any]] = None
    generated_at: str = ""


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class ItineraryValidator:
    """Validate that the generated itinerary respects all business rules."""

    def validate(self, days: List[dict],
                 ctx: ParsedItineraryContext,
                 constraints: List[DayConstraints] = None
                 ) -> Dict[str, Any]:
        """
        Run all validation checks on the generated itinerary.

        Returns dict with: valid (bool), issues (list), warnings (list),
        activity_count (int).
        """
        issues: List[str] = []
        warnings: List[str] = []
        all_activities: List[str] = []

        # Check all days present
        expected_days = ctx.total_days
        if len(days) != expected_days:
            issues.append(
                f"Expected {expected_days} days, got {len(days)}"
            )

        for day in days:
            day_num = day.get("day_number", "?")
            day_type = day.get("day_type", "full")
            schedule = day.get("schedule", [])

            # Collect activity names for duplicate detection
            for item in schedule:
                name = item.get("name", "")
                itype = item.get("type", "")
                if itype in ("activity", "dining") and name:
                    all_activities.append(name)

            # --- Rule 7: Max activities per day ---
            activity_count = sum(
                1 for s in schedule if s.get("type") in ("activity", "dining")
            )
            max_act = 3  # default
            if constraints:
                for dc in constraints:
                    if dc.day_number == day_num:
                        max_act = dc.max_activities
                        break
            if activity_count > max_act + 1:  # +1 for separate dining
                warnings.append(
                    f"Day {day_num}: {activity_count} activities+dining "
                    f"(max recommended: {max_act})"
                )

            # --- Rule 6: Transit day max 1 activity ---
            if day_type == "transit":
                pure_activities = sum(
                    1 for s in schedule if s.get("type") == "activity"
                )
                if pure_activities > 1:
                    issues.append(
                        f"Day {day_num} (transit): {pure_activities} activities "
                        f"(max 1 on transit days per Rule 6)"
                    )

            # --- is_from_knowledge_base accuracy check ---
            for s in schedule:
                if s.get("type") in ("rest", "transport", "checkin", "checkout"):
                    if s.get("is_from_knowledge_base", False):
                        warnings.append(
                            f"Day {day_num}: '{s.get('name', '')}' "
                            f"(type={s.get('type')}) incorrectly marked "
                            f"is_from_knowledge_base=true"
                        )

            # --- Rule 2: Departure day must have airport transfer ---
            if day_type == "departure":
                has_transfer = any(
                    "airport" in s.get("name", "").lower() or
                    "airport" in s.get("type", "").lower() or
                    s.get("type") == "transport"
                    for s in schedule
                )
                if not has_transfer:
                    issues.append(
                        f"Day {day_num} (departure): missing airport transfer"
                    )

            # --- Rule 7: Downtime on full days ---
            if day_type == "full":
                has_downtime = any(
                    s.get("type") == "rest" or
                    "downtime" in s.get("name", "").lower() or
                    "leisure" in s.get("name", "").lower() or
                    "free" in s.get("name", "").lower()
                    for s in schedule
                )
                if not has_downtime:
                    warnings.append(
                        f"Day {day_num}: no downtime/rest period on full day"
                    )

            # --- Rule 1: Arrival day no heavy sightseeing ---
            if day_type == "arrival":
                heavy_items = [
                    s for s in schedule
                    if s.get("energy", "") == "high" or
                    "heavy" in s.get("description", "").lower() or
                    ("activity" == s.get("type") and
                     "explore" not in s.get("name", "").lower() and
                     "check" not in s.get("name", "").lower() and
                     "settle" not in s.get("name", "").lower() and
                     activity_count > 2)
                ]
                if len(heavy_items) > 0 and activity_count > 2:
                    warnings.append(
                        f"Day {day_num} (arrival): may have heavy sightseeing"
                    )

            # --- LuxUltra: no economy references ---
            if ctx.tier == "LuxUltra":
                for s in schedule:
                    text = json.dumps(s).lower()
                    if any(w in text for w in [
                        "economy", "budget", "group tour",
                        "crowded", "standard restaurant"
                    ]):
                        issues.append(
                            f"Day {day_num}: LuxUltra contains "
                            f"non-premium reference in '{s.get('name', '')}'"
                        )

        # --- Rule 7: No repeated activities (fuzzy matching) ---
        def _normalize_name(name: str) -> str:
            """Normalize an activity name for comparison."""
            n = name.lower().strip()
            # Remove common LLM suffixes like "in Tokyo, Japan"
            if " in " in n:
                n = n.split(" in ")[0].strip()
            # Remove KB ID prefixes like "[POI_003]"
            if n.startswith("["):
                bracket_end = n.find("]")
                if bracket_end != -1:
                    n = n[bracket_end + 1:].strip()
            return n

        seen = set()
        for name in all_activities:
            normalized = _normalize_name(name)
            if normalized in seen:
                issues.append(f"Repeated activity: '{name}'")
            seen.add(normalized)

        # --- Occasion peak moment check ---
        if ctx.occasion in ("anniversary", "honeymoon", "birthday"):
            has_special = any(
                "special" in item.get("name", "").lower() or
                "romantic" in item.get("description", "").lower() or
                "celebrat" in item.get("description", "").lower() or
                "anniversary" in item.get("description", "").lower() or
                "birthday" in item.get("description", "").lower()
                for day in days
                for item in day.get("schedule", [])
            )
            if not has_special:
                warnings.append(
                    f"No occasion peak moment found for '{ctx.occasion}'"
                )

        unique_count = len(set(n.lower().strip() for n in all_activities))

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "activity_count": unique_count,
        }


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

class ItineraryFormatter:
    """Format itinerary for human reading and API output."""

    def to_human_readable(self, itinerary: FullItinerary) -> str:
        """Convert a FullItinerary to a formatted text string."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"  {itinerary.destination} — {itinerary.tier} Itinerary")
        lines.append(f"  {itinerary.customer_name} | {itinerary.occasion.title()}")
        lines.append(f"  {itinerary.checkin} → {itinerary.checkout}")
        lines.append(f"  {itinerary.total_days} days | {itinerary.total_nights} nights")
        lines.append("=" * 60)

        for day in itinerary.days:
            lines.append("")
            emoji = {"arrival": "✈️", "departure": "✈️",
                     "transit": "🚂", "full": "🌟"}.get(day.day_type, "📅")
            lines.append(f"─── Day {day.day_number} | {day.date} | "
                         f"{day.city} ({day.day_type}) {emoji} ───")
            lines.append(f"  Hotel: {day.hotel}")
            if day.area_focus:
                lines.append(f"  Area: {day.area_focus}")
            lines.append("")

            for item in day.schedule:
                kb_marker = "📚" if item.is_from_knowledge_base else "💡"
                booking = " 🔖" if item.booking_required else ""
                lines.append(
                    f"  {item.time}  {kb_marker} [{item.type.upper()}] "
                    f"{item.name}{booking}"
                )
                lines.append(f"          {item.description[:120]}")
                lines.append(f"          Duration: {item.duration} | Area: {item.area}")
                if item.booking_note:
                    lines.append(f"          Note: {item.booking_note}")

            if day.day_summary:
                lines.append(f"\n  📝 {day.day_summary}")
            if day.local_tip:
                lines.append(f"  💡 Tip: {day.local_tip}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def to_api_response(self, days: List[dict],
                        ctx: ParsedItineraryContext,
                        validation: Dict[str, Any]) -> dict:
        """
        Build the final API response dict from raw days + context + validation.

        Args:
            days: List of day dicts from Stage 5.
            ctx: ParsedItineraryContext from Stage 1.
            validation: Validation result from ItineraryValidator.

        Returns:
            Complete API response dict matching FullItinerary schema.
        """
        day_models = []
        for d in days:
            schedule_items = []
            for s in d.get("schedule", []):
                schedule_items.append(ScheduleItem(
                    time=s.get("time", "00:00"),
                    time_label=s.get("time_label", "morning"),
                    type=s.get("type", "activity"),
                    name=s.get("name", ""),
                    description=s.get("description", ""),
                    duration=s.get("duration", ""),
                    area=s.get("area", ""),
                    tier_note=s.get("tier_note", ""),
                    is_from_knowledge_base=s.get("is_from_knowledge_base", False),
                    booking_required=s.get("booking_required", False),
                    booking_note=s.get("booking_note", ""),
                ))
            day_models.append(DayItinerary(
                day_number=d.get("day_number", 0),
                date=d.get("date", ""),
                city=d.get("city", ""),
                hotel=d.get("hotel", ""),
                day_type=d.get("day_type", "full"),
                area_focus=d.get("area_focus", ""),
                schedule=schedule_items,
                day_summary=d.get("day_summary", ""),
                local_tip=d.get("local_tip", ""),
            ))

        full = FullItinerary(
            deal_id=ctx.deal_id,
            deal_type=ctx.deal_type,
            tier=ctx.tier,
            destination=ctx.destination,
            customer_name=ctx.customer.name,
            travel_type=ctx.customer.travel_type,
            occasion=ctx.occasion,
            total_days=ctx.total_days,
            total_nights=ctx.total_nights,
            checkin=ctx.checkin,
            checkout=ctx.checkout,
            days=day_models,
            validation=validation,
            generated_at=datetime.now().isoformat(),
        )

        return full.model_dump()


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

def main():
    """Demonstrate validation and formatting with sample data."""
    print("[Stage 6] Formatter standalone test")

    sample_day = {
        "day_number": 1, "date": "2026-03-15", "city": "Tokyo",
        "hotel": "Hotel Gracery Shinjuku", "day_type": "arrival",
        "area_focus": "Shinjuku",
        "schedule": [
            {
                "time": "11:00", "time_label": "morning",
                "type": "checkin", "name": "Check-in at Hotel Gracery Shinjuku",
                "description": "Arrive and settle into your hotel. Freshen up after your journey.",
                "duration": "1.5 hours", "area": "Shinjuku",
                "tier_note": "LuxPlus premium hotel", "is_from_knowledge_base": False,
                "booking_required": False, "booking_note": ""
            },
            {
                "time": "18:00", "time_label": "evening",
                "type": "rest", "name": "Evening at leisure",
                "description": "Explore Shinjuku's neon-lit streets at your own pace. A romantic walk for your anniversary.",
                "duration": "2 hours", "area": "Shinjuku",
                "tier_note": "", "is_from_knowledge_base": False,
                "booking_required": False, "booking_note": ""
            }
        ],
        "day_summary": "Smooth arrival and relaxed first evening in Tokyo",
        "local_tip": "Convenience stores (konbini) are open 24/7 and have excellent food."
    }

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(PROJECT_ROOT, "data")

    with open(os.path.join(data_dir, "dummy_deals.json"), "r", encoding="utf-8") as f:
        deals = json.load(f)
    with open(os.path.join(data_dir, "dummy_customer_profiles.json"), "r", encoding="utf-8") as f:
        customers = json.load(f)

    try:
        from scripts.deal_parser import parse_deal
    except ImportError:
        from deal_parser import parse_deal

    ctx = parse_deal(deals[0], customers[0])

    validator = ItineraryValidator()
    result = validator.validate([sample_day], ctx)
    print(f"  Validation: {'✓ PASS' if result['valid'] else '✗ FAIL'}")
    print(f"  Issues: {result['issues']}")
    print(f"  Warnings: {result['warnings']}")
    print(f"  Activity count: {result['activity_count']}")


if __name__ == "__main__":
    main()
