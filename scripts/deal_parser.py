#!/usr/bin/env python3
"""
Stage 1: Deal Parser — Parse deal JSON + customer profile into a unified context object.

Extracts and computes: deal fields, city stays with exact dates, flight classification,
customer profile, tier rules, and occasion rules.

"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


# Dataclasses

@dataclass
class FlightInfo:
    """Parsed flight details with time classification."""
    flight_class: str
    departure_city: str
    arrival_city: str
    return_city: str
    arrival_time: str          # HH:MM
    departure_time: str        # HH:MM
    is_direct: bool
    airline: str
    duration_hours: float
    num_stops: int
    price_inr: int
    arrival_classification: str   # very_early / early / morning / afternoon / evening / night
    departure_classification: str


@dataclass
class HotelInfo:
    """Parsed hotel details for a single city."""
    name: str
    lat: float
    lon: float
    stars: int
    area: str
    rating: float
    review_count: int
    property_type: str
    review_trend: str
    recent_complaints: List[str]


@dataclass
class CityStay:
    """A single city stay with computed check-in/check-out dates."""
    city: str
    days: int
    checkin: str   # YYYY-MM-DD
    checkout: str  # YYYY-MM-DD
    hotel: Optional[HotelInfo]


@dataclass
class TierRules:
    """Tier-specific constraints applied throughout the itinerary."""
    activities_per_day: int
    dining_level: str         # fine_dining / premium / good_value
    transport_level: str      # luxury / comfortable / standard
    avoid_list: List[str]


@dataclass
class OccasionRules:
    """Occasion-specific mood and content rules."""
    mood: str
    must_include: List[str]
    must_avoid: List[str]


@dataclass
class CustomerProfile:
    """Parsed customer profile."""
    customer_id: str
    name: str
    travel_type: str
    occasion: str
    interests: List[str]
    past_trips: List[str]
    past_hotels: List[str]
    pace: str
    dietary: str
    home_city: str
    preferred_tier: str
    children: Optional[List[Dict[str, Any]]]


@dataclass
class ParsedItineraryContext:
    """Complete context object passed to all downstream pipeline stages."""
    deal_id: str
    deal_type: str
    tier: str
    destination: str
    cities: List[CityStay]
    flight: FlightInfo
    hotels: Dict[str, HotelInfo]
    checkin: str
    checkout: str
    total_nights: int
    total_days: int
    occasion: str
    customer: CustomerProfile
    tier_rules: TierRules
    occasion_rules: OccasionRules


# Time classification

def classify_time(time_str: str) -> str:
    """
    Classify a time string (HH:MM) into a travel-relevant category.

    Categories:
        very_early: before 6AM
        early:      6AM - 10AM  (triggers Rule 1 special handling)
        morning:    10AM - 2PM
        afternoon:  2PM - 7PM
        evening:    7PM - 10PM
        night:      after 10PM
    """
    hour, minute = map(int, time_str.split(":"))
    total_minutes = hour * 60 + minute

    if total_minutes < 360:       # before 6:00
        return "very_early"
    elif total_minutes < 600:     # 6:00 - 9:59
        return "early"
    elif total_minutes < 840:     # 10:00 - 13:59
        return "morning"
    elif total_minutes < 1140:    # 14:00 - 18:59
        return "afternoon"
    elif total_minutes < 1320:    # 19:00 - 21:59
        return "evening"
    else:                         # 22:00+
        return "night"


# Tier / occasion rules builders

def build_tier_rules(tier: str) -> TierRules:
    """Build tier-specific rules object based on deal tier."""
    rules_map = {
        "LuxLite": TierRules(
            activities_per_day=3,
            dining_level="good_value",
            transport_level="standard",
            avoid_list=["exclusive_private", "first_class_only"],
        ),
        "LuxPlus": TierRules(
            activities_per_day=2,
            dining_level="premium",
            transport_level="comfortable",
            avoid_list=["budget_options", "crowded_group_tours"],
        ),
        "LuxUltra": TierRules(
            activities_per_day=2,
            dining_level="fine_dining",
            transport_level="luxury",
            avoid_list=[
                "economy_flights", "group_tours", "crowded_venues",
                "standard_restaurants", "budget_options",
            ],
        ),
    }
    return rules_map.get(tier, rules_map["LuxPlus"])


def build_occasion_rules(occasion: str) -> OccasionRules:
    """Build occasion-specific mood and content rules."""
    rules_map = {
        "anniversary": OccasionRules(
            mood="romantic",
            must_include=["romantic_dinner", "sunset_experience", "couple_spa"],
            must_avoid=["group_activities", "noisy_venues", "crowded_attractions"],
        ),
        "honeymoon": OccasionRules(
            mood="romantic",
            must_include=["romantic_dinner", "private_experience", "couple_spa"],
            must_avoid=["group_activities", "noisy_venues"],
        ),
        "birthday": OccasionRules(
            mood="celebratory",
            must_include=["special_dinner", "surprise_experience"],
            must_avoid=["somber_venues"],
        ),
        "school_holiday": OccasionRules(
            mood="fun_family",
            must_include=["family_activity", "kid_friendly_dining"],
            must_avoid=["adults_only", "late_night_events"],
        ),
        "general": OccasionRules(
            mood="exploratory",
            must_include=[],
            must_avoid=[],
        ),
    }
    return rules_map.get(occasion, rules_map["general"])


# Core parser

def compute_city_stays(cities_raw: List[Dict], checkin_str: str,
                       hotels_raw: Dict) -> List[CityStay]:
    """Compute exact check-in/check-out dates for each city stay."""
    checkin_date = datetime.strptime(checkin_str, "%Y-%m-%d")
    stays = []
    current_date = checkin_date

    for city_info in cities_raw:
        city_name = city_info["city"]
        days = city_info["days"]
        city_checkin = current_date.strftime("%Y-%m-%d")
        city_checkout = (current_date + timedelta(days=days)).strftime("%Y-%m-%d")

        hotel = None
        if city_name in hotels_raw:
            h = hotels_raw[city_name]
            hotel = HotelInfo(
                name=h["name"], lat=h["lat"], lon=h["lon"],
                stars=h["stars"], area=h["area"], rating=h["rating"],
                review_count=h["review_count"], property_type=h["property_type"],
                review_trend=h["review_trend"],
                recent_complaints=h.get("recent_complaints", []),
            )

        stays.append(CityStay(
            city=city_name, days=days,
            checkin=city_checkin, checkout=city_checkout,
            hotel=hotel,
        ))
        current_date += timedelta(days=days)

    return stays


def parse_customer(cust: Dict) -> CustomerProfile:
    """Parse a raw customer profile dictionary into a CustomerProfile dataclass."""
    return CustomerProfile(
        customer_id=cust["customer_id"],
        name=cust["name"],
        travel_type=cust["travel_type"],
        occasion=cust["occasion"],
        interests=cust.get("interests", []),
        past_trips=cust.get("past_trips", []),
        past_hotels=cust.get("past_hotels", []),
        pace=cust.get("pace", "moderate"),
        dietary=cust.get("dietary", "none"),
        home_city=cust.get("home_city", ""),
        preferred_tier=cust.get("preferred_tier", "LuxPlus"),
        children=cust.get("children"),
    )


def parse_deal(deal: Dict, customer: Dict) -> ParsedItineraryContext:
    """
    Parse a confirmed deal + customer profile into a single ParsedItineraryContext.

    This is the entry point for Stage 1. The returned context object is consumed
    by all downstream pipeline stages.

    Args:
        deal: Raw deal dictionary (from dummy_deals.json or API).
        customer: Raw customer profile dictionary.

    Returns:
        ParsedItineraryContext with all fields computed.
    """
    # Parse flight
    fl = deal["flight"]
    flight = FlightInfo(
        flight_class=fl["class"],
        departure_city=fl["departure_city"],
        arrival_city=fl["arrival_city"],
        return_city=fl["return_city"],
        arrival_time=fl["arrival_time"],
        departure_time=fl["departure_time"],
        is_direct=fl["is_direct"],
        airline=fl["airline"],
        duration_hours=fl["duration_hours"],
        num_stops=fl["num_stops"],
        price_inr=fl["price_inr"],
        arrival_classification=classify_time(fl["arrival_time"]),
        departure_classification=classify_time(fl["departure_time"]),
    )

    # Parse hotels dict
    hotels = {}
    for city_name, h in deal["hotels"].items():
        hotels[city_name] = HotelInfo(
            name=h["name"], lat=h["lat"], lon=h["lon"],
            stars=h["stars"], area=h["area"], rating=h["rating"],
            review_count=h["review_count"], property_type=h["property_type"],
            review_trend=h["review_trend"],
            recent_complaints=h.get("recent_complaints", []),
        )

    # Compute city stays with exact dates
    city_stays = compute_city_stays(
        deal["cities"], deal["checkin"], deal["hotels"]
    )

    # Build rules
    tier_rules = build_tier_rules(deal["tier"])
    occasion_rules = build_occasion_rules(deal.get("occasion", "general"))

    # Parse customer
    cust_profile = parse_customer(customer)

    total_nights = deal["total_nights"]
    total_days = total_nights  # travel days = nights (depart on last day)

    return ParsedItineraryContext(
        deal_id=deal["deal_id"],
        deal_type=deal["deal_type"],
        tier=deal["tier"],
        destination=deal["destination"],
        cities=city_stays,
        flight=flight,
        hotels=hotels,
        checkin=deal["checkin"],
        checkout=deal["checkout"],
        total_nights=total_nights,
        total_days=total_days,
        occasion=deal.get("occasion", "general"),
        customer=cust_profile,
        tier_rules=tier_rules,
        occasion_rules=occasion_rules,
    )


# Standalone execution

def main():
    """Load first deal + first customer and print parsed context."""
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(PROJECT_ROOT, "data")

    with open(os.path.join(data_dir, "dummy_deals.json"), "r", encoding="utf-8") as f:
        deals = json.load(f)
    with open(os.path.join(data_dir, "dummy_customer_profiles.json"), "r", encoding="utf-8") as f:
        customers = json.load(f)

    ctx = parse_deal(deals[0], customers[0])
    print(f"Deal: {ctx.deal_id} | {ctx.destination} | {ctx.tier}")
    print(f"Customer: {ctx.customer.name} | {ctx.customer.travel_type}")
    print(f"Total days: {ctx.total_days} | Cities: {len(ctx.cities)}")
    for stay in ctx.cities:
        hotel_name = stay.hotel.name if stay.hotel else "N/A"
        print(f"  {stay.city}: {stay.checkin} → {stay.checkout} ({stay.days}d) @ {hotel_name}")
    print(f"Flight: {ctx.flight.airline} | {ctx.flight.flight_class} | "
          f"Arrival {ctx.flight.arrival_time} ({ctx.flight.arrival_classification})")
    print(f"Tier rules: max {ctx.tier_rules.activities_per_day} activities/day")
    print(f"Occasion: {ctx.occasion} → mood={ctx.occasion_rules.mood}")


if __name__ == "__main__":
    main()
