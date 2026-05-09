#!/usr/bin/env python3
"""
Stage 2: Rules Engine — Apply all business rules to produce DayConstraints per day.

Implements itinerary rules mentionef in the documentation like: arrival handling, departure buffering,
no backtracking, buffer management, experience positioning, city sequencing, pacing.
Also handles occasion peak placement and energy arc management.

"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional

# Allow standalone or module import
try:
    from scripts.deal_parser import ParsedItineraryContext, classify_time
except ImportError:
    from deal_parser import ParsedItineraryContext, classify_time


@dataclass
class DayConstraints:
    """Constraints and metadata for a single day of the itinerary."""
    day_number: int
    date: str                              # YYYY-MM-DD
    city: str
    hotel_name: str
    day_type: str                          # arrival / full / transit / departure
    energy_level: str                      # low / moderate / high
    max_activities: int
    usable_hours_start: str                # HH:MM
    usable_hours_end: str                  # HH:MM
    must_include: List[str] = field(default_factory=list)
    must_avoid: List[str] = field(default_factory=list)
    is_premium_exp_day: bool = False
    has_city_transfer: bool = False
    needs_airport_buffer: bool = False
    is_occasion_peak: bool = False
    occasion_peak_time: Optional[str] = None  # morning / afternoon / evening
    area_cluster: Optional[int] = None
    notes: List[str] = field(default_factory=list)


def _time_str_to_minutes(t: str) -> int:
    """Convert HH:MM string to minutes since midnight."""
    h, m = map(int, t.split(":"))
    return h * 60 + m


def _minutes_to_time_str(minutes: int) -> str:
    """Convert minutes since midnight to HH:MM string."""
    minutes = max(0, min(1439, minutes))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _compute_occasion_peak_day(occasion: str, total_days: int,
                               transit_days: set = None) -> tuple:
    if occasion not in ("anniversary", "honeymoon", "birthday"):
        return (None, None)

    if occasion in ("anniversary", "honeymoon"):
        ideal = max(2, total_days - 2)
    else:  # birthday
        ideal = 2

    peak_time = "evening"

    # Set of blocked day offsets (0-based): arrival=0, departure=last, transits
    blocked_offsets = {0, total_days - 1}  # arrival, departure
    if transit_days:
        blocked_offsets.update(transit_days)

    # If ideal day (1-indexed) is blocked, find nearest unblocked full day
    if (ideal - 1) in blocked_offsets:
        # Search outward from ideal
        for delta in range(1, total_days):
            for candidate in [ideal + delta, ideal - delta]:
                if 1 <= candidate <= total_days and (candidate - 1) not in blocked_offsets:
                    return (candidate, peak_time)
        # All days blocked (shouldn't happen) — just skip peak
        return (None, None)

    return (ideal, peak_time)


def apply_rules(ctx: ParsedItineraryContext) -> List[DayConstraints]:
    """
    Apply all 7 business rules to produce one DayConstraints per day.

    Rules implemented:
        1. Day 1 arrival handling (early comfort, no heavy sightseeing)
        2. Last day departure (buffer, light schedule, airport transfer)
        3. No backtracking (flag for spatial sequencer)
        4. Buffer / risk management (airport buffers, available_until)
        5. Experience positioning (premium days only on rest days)
        6. City sequencing (transit day flagging)
        7. Pacing (max activities, downtime requirements)

    Also applies: occasion peak placement, tier enforcement, energy arc.

    Args:
        ctx: ParsedItineraryContext from Stage 1.

    Returns:
        List of DayConstraints, one per day (1-indexed day_number).
    """
    constraints: List[DayConstraints] = []
    max_act = ctx.tier_rules.activities_per_day

    # Precompute: which day_number is the first day in each city
    # and which day_number transitions to a new city
    day_to_city = []       # index = day offset (0-based), value = city name
    day_to_hotel = []
    transit_days = set()   # 0-based day offsets that are city-transfer days

    offset = 0
    for i, stay in enumerate(ctx.cities):
        for d in range(stay.days):
            day_to_city.append(stay.city)
            hotel_name = stay.hotel.name if stay.hotel else "N/A"
            day_to_hotel.append(hotel_name)
            # First day of a new city (except the very first city's first day)
            if d == 0 and i > 0:
                transit_days.add(offset)
            offset += 1

    total_travel_days = len(day_to_city)

    # Occasion peak
    peak_day_num, peak_time = _compute_occasion_peak_day(
        ctx.occasion, total_travel_days, transit_days
    )

    # Energy arc: balanced pacing throughout the trip
    def _energy_for_day(day_idx: int, day_type: str) -> str:
        if day_type in ("arrival", "departure", "transit"):
            return "low"

        # First full day should not be intense
        if day_idx <= 1:
            return "moderate"

        # Occasion peak day gets high energy
        if peak_day_num is not None and (day_idx + 1) == peak_day_num:
            return "high"

        # Prevent too many high-energy days
        if day_idx % 3 == 0:
            return "moderate"

        return "high"

    for day_idx in range(total_travel_days):
        day_num = day_idx + 1
        date = (datetime.strptime(ctx.checkin, "%Y-%m-%d") +
                timedelta(days=day_idx)).strftime("%Y-%m-%d")
        city = day_to_city[day_idx]
        hotel = day_to_hotel[day_idx]

        # Classify day type
        is_first = (day_idx == 0)
        is_last = (day_idx == total_travel_days - 1)
        is_transit = (day_idx in transit_days)

        if is_first:
            day_type = "arrival"
        elif is_last:
            day_type = "departure"
        elif is_transit:
            day_type = "transit"
        else:
            day_type = "full"

        energy = _energy_for_day(day_idx, day_type)
        must_include = list(ctx.occasion_rules.must_include)  # copy
        must_avoid = list(ctx.occasion_rules.must_avoid)

        # Defaults
        usable_start = "09:00"
        usable_end = "21:00"
        activities = max_act
        is_premium = False
        needs_buffer = False
        city_transfer = is_transit

        # ── RULE 1: Day 1 Arrival ─────────────────────────────────────
        if is_first:
            arr_class = ctx.flight.arrival_classification
            arr_minutes = _time_str_to_minutes(ctx.flight.arrival_time)

            if arr_class in ("very_early", "early"):
                # Before 10 AM: add early comfort
                must_include.append("early_comfort")
                must_include.append("light_relaxing_activity")
            usable_start = _minutes_to_time_str(arr_minutes + 90)  # 1.5h settle
            must_avoid.extend(["heavy_sightseeing", "premium_experience"])
            energy = "low"
            activities = min(activities, 2)

        # ── RULE 2: Last Day Departure ────────────────────────────────
        if is_last:
            dep_minutes = _time_str_to_minutes(ctx.flight.departure_time)
            buffer_minutes = 4 * 60  # 4 hours before flight
            available_until = dep_minutes - buffer_minutes
            usable_end = _minutes_to_time_str(max(available_until, 480))

            # Calculate usable hours
            start_min = _time_str_to_minutes(usable_start)
            end_min = _time_str_to_minutes(usable_end)
            usable_hours = max(0, (end_min - start_min) / 60)
            activities = min(activities, max(1, int(usable_hours / 2.5)))

            must_include.append("airport_transfer")
            needs_buffer = True
            energy = "low"

            # Validate last hotel city vs departure airport city
            last_city = ctx.cities[-1].city
            flight_arrival_city = ctx.flight.arrival_city
            if last_city.lower() != flight_arrival_city.lower():
                notes_list = [
                    f"Multi-city: last city is {last_city} but flight arrived in "
                    f"{flight_arrival_city}. Departure airport is in {last_city}."
                ]

        # ── RULE 3: No Backtracking ───────────────────────────────────
        # Handled by spatial sequencer (Stage 3) — no constraint modification needed

        # ── RULE 4: Buffer / Risk Management ──────────────────────────
        if is_last:
            needs_buffer = True
        if is_transit:
            # Keep buffer for inter-city transfer
            needs_buffer = True
            usable_end = "18:00"  # leave evening for settling in

        # ── RULE 5: Experience Positioning ────────────────────────────
        if (
            day_type == "full"
            and energy != "low"
            and not is_transit
        ):
            is_premium = True
            
        if day_type in ("arrival", "transit"):
            must_avoid.append("premium_experience")

        # ── RULE 6: City Sequencing / Transit ─────────────────────────
        if city_transfer:
            activities = min(activities, 1)

            must_include.append("city_transfer")
            must_include.append("easy_logistics")

            must_avoid.append("complex_transit")
            must_avoid.append("heavy_sightseeing")

            energy = "low"

        # ── RULE 7: Pacing ────────────────────────────────────────────
        activities = min(activities, max_act)

        if day_type == "full":
            must_include.append("downtime_period")

        # Recovery logic for high-energy days
        if energy == "high":
            must_include.append("late_start")
            must_include.append("downtime_period")

        # ── Occasion Peak ─────────────────────────────────────────────
        is_peak = (peak_day_num is not None and day_num == peak_day_num)

        if is_peak:
            must_include.append("signature_experience")
            must_include.append("premium_dining")

            # Allow one extra major activity on peak day
            activities = min(activities + 1, 4)

        # Build constraint
        dc = DayConstraints(
            day_number=day_num,
            date=date,
            city=city,
            hotel_name=hotel,
            day_type=day_type,
            energy_level=energy,
            max_activities=activities,
            usable_hours_start=usable_start,
            usable_hours_end=usable_end,
            must_include=list(set(must_include)),
            must_avoid=list(set(must_avoid)),
            is_premium_exp_day=is_premium,
            has_city_transfer=city_transfer,
            needs_airport_buffer=needs_buffer,
            is_occasion_peak=is_peak,
            occasion_peak_time=peak_time if is_peak else None,
        )
        constraints.append(dc)

    return constraints


# Standalone execution

def main():
    """Load first deal, parse, and print day constraints."""
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
    day_constraints = apply_rules(ctx)

    peak_day = next((d for d in day_constraints if d.is_occasion_peak), None)
    arrival = day_constraints[0] if day_constraints else None

    print(f"[Stage 2] Rules engine applied")
    print(f"  → {len(day_constraints)} day constraints built")
    if peak_day:
        print(f"  → Occasion peak: Day {peak_day.day_number} {peak_day.occasion_peak_time}")
    if arrival:
        print(f"  → Arrival: {ctx.flight.arrival_classification} "
              f"({ctx.flight.arrival_time}) → comfort rule applied")
    print()
    for dc in day_constraints:
        print(f"  Day {dc.day_number} | {dc.date} | {dc.city} | {dc.day_type} | "
              f"energy={dc.energy_level} | max_act={dc.max_activities} | "
              f"premium={dc.is_premium_exp_day} | peak={dc.is_occasion_peak}")


if __name__ == "__main__":
    main()