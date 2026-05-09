#!/usr/bin/env python3
"""

manages all 6 stages in sequence:
  1. Deal Parser     
  2. Rules Engine     
  3. Spatial Sequencer
  4. RAG Retrieval     
  5. Smart Scheduler   
  6. Formatter         

Usage:
    python run_pipeline.py
    python run_pipeline.py --deal DEAL_001 --customer CUST_001
    python run_pipeline.py --deal DEAL_002 --customer CUST_004
    python run_pipeline.py --deal DEAL_001 --customer CUST_001 --use-llm
"""

import argparse
import json
import os
import sys
import time

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from scripts.deal_parser import parse_deal
from scripts.rules_engine import apply_rules
from scripts.spatial_sequencer import SpatialSequencer
from scripts.llm_generator import SmartScheduler, LLMEnricher
from scripts.formatter import (ItineraryValidator, ItineraryFormatter, FullItinerary)


DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_db")


def load_deal(deal_id: str) -> dict:
    """Load a specific deal from dummy_deals.json by deal_id."""
    path = os.path.join(DATA_DIR, "dummy_deals.json")
    with open(path, "r", encoding="utf-8") as f:
        deals = json.load(f)
    for d in deals:
        if d["deal_id"] == deal_id:
            return d
    raise ValueError(f"Deal '{deal_id}' not found in {path}")


def load_customer(customer_id: str) -> dict:
    """Load a specific customer profile by customer_id."""
    path = os.path.join(DATA_DIR, "dummy_customer_profiles.json")
    with open(path, "r", encoding="utf-8") as f:
        customers = json.load(f)
    for c in customers:
        if c["customer_id"] == customer_id:
            return c
    raise ValueError(f"Customer '{customer_id}' not found in {path}")


def load_pois() -> list:
    """Load POI data for spatial sequencing."""
    path = os.path.join(DATA_DIR, "dummy_pois.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dining() -> list:
    """Load dining data."""
    path = os.path.join(DATA_DIR, "dummy_dining.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_logistics() -> list:
    """Load logistics data."""
    path = os.path.join(DATA_DIR, "dummy_logistics.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(deal_id: str, customer_id: str, use_llm: bool = False):
    """Execute the full 6-stage pipeline."""
    start_time = time.time()

    # Load raw data
    deal_raw = load_deal(deal_id)
    customer_raw = load_customer(customer_id)

    # Header
    print()
    print("=" * 60)
    print("  TRAVEL JAUNTS — Itinerary Generator")
    print("=" * 60)
    print(f"  Deal:     {deal_id} | {deal_raw['destination']} | {deal_raw['tier']}")
    print(f"  Customer: {customer_id} | {customer_raw['name']}")
    print(f"  Tier:     {deal_raw['tier']} | Occasion: {deal_raw.get('occasion', 'general')}")
    mode = "SmartScheduler + LLM enrichment" if use_llm else "SmartScheduler (deterministic)"
    print(f"  Mode:     {mode}")
    print("=" * 60)

    # ── STAGE 1: Deal Parser ──────────────────────────────────
    print("\n[Stage 1] Parsing deal + customer profile...")
    ctx = parse_deal(deal_raw, customer_raw)
    cities_str = " -> ".join(f"{c.city}({c.days}d)" for c in ctx.cities)
    print(f"  -> {ctx.total_days} days | {cities_str}")
    print(f"  -> {ctx.customer.travel_type} | {ctx.occasion} | pace={ctx.customer.pace}")

    # ── STAGE 2: Rules Engine ─────────────────────────────────
    print("\n[Stage 2] Applying rules engine...")
    constraints = apply_rules(ctx)
    peak = next((d for d in constraints if d.is_occasion_peak), None)
    print(f"  -> {len(constraints)} day constraints built")
    if peak:
        print(f"  -> Occasion peak: Day {peak.day_number} ({peak.city}, {peak.occasion_peak_time})")
    print(f"  -> Arrival: {ctx.flight.arrival_classification} "
          f"({ctx.flight.arrival_time}) -> comfort rule applied")

    # ── STAGE 3: Spatial Sequencer ────────────────────────────
    print("\n[Stage 3] Spatial sequencing...")
    pois_data = load_pois()
    sequencer = SpatialSequencer()
    spatial = sequencer.build_spatial_context(ctx, pois_data)
    label = "resequenced" if spatial.city_was_resequenced else "validated"
    print(f"  -> City order {label}: {' -> '.join(spatial.ordered_cities)}")
    for city, areas in spatial.neighborhood_plan.items():
        area_strs = [", ".join(a) if a else "general" for a in areas]
        print(f"     {city}: {' | '.join(area_strs)}")

    # ── STAGE 4: Data Loading ─────────────────────────────────
    print("\n[Stage 4] Loading knowledge base...")
    dining_data = load_dining()
    logistics_data = load_logistics()

    # Count available data per destination city
    dest_cities = set(c.city for c in ctx.cities)
    act_count = sum(1 for p in pois_data if p.get("city") in dest_cities)
    din_count = sum(1 for d in dining_data if d.get("city") in dest_cities)
    log_count = sum(1 for l in logistics_data if l.get("city") in dest_cities)
    print(f"  -> Activities: {act_count} matching destination")
    print(f"  -> Dining:     {din_count} matching destination")
    print(f"  -> Logistics:  {log_count} matching destination")

    # ── STAGE 5: Smart Scheduler ──────────────────────────────
    print(f"\n[Stage 5] Building itinerary (SmartScheduler)...")
    gen_start = time.time()

    scheduler = SmartScheduler()
    days = scheduler.build_full_itinerary(
        constraints, ctx, pois_data, dining_data, logistics_data, spatial
    )

    gen_elapsed = time.time() - gen_start
    print(f"  -> {len(days)} days built in {gen_elapsed:.2f}s")

    # Show what was scheduled
    for d in days:
        items = [s["name"] for s in d["schedule"]
                 if s["type"] in ("activity", "dining")]
        kb_count = sum(1 for s in d["schedule"] if s["is_from_knowledge_base"])
        print(f"     Day {d['day_number']} ({d['day_type']:9s}) | "
              f"{kb_count} KB items | {', '.join(items) if items else 'no activities'}")

    # Optional LLM enrichment
    if use_llm:
        print(f"\n  [LLM] Enriching descriptions via Ollama...")
        enricher = LLMEnricher()
        enrich_start = time.time()
        days = enricher.enrich(days, ctx)
        enrich_elapsed = time.time() - enrich_start
        print(f"  [LLM] Enrichment completed in {enrich_elapsed:.1f}s")

    # ── STAGE 6: Validation + Formatting ─────────────────────
    print(f"\n[Stage 6] Validating + formatting...")
    validator = ItineraryValidator()
    validation = validator.validate(days, ctx, constraints)

    status = "PASS" if validation["valid"] else "FAIL"
    print(f"  -> Validation: {status}")
    print(f"  -> {validation['activity_count']} unique activities")
    print(f"  -> {len(validation['issues'])} issues, "
          f"{len(validation['warnings'])} warnings")

    if validation["issues"]:
        for issue in validation["issues"]:
            print(f"     ISSUE: {issue}")
    if validation["warnings"]:
        for warn in validation["warnings"]:
            print(f"     WARN:  {warn}")

    # Build final output
    formatter = ItineraryFormatter()
    api_response = formatter.to_api_response(days, ctx, validation)

    # Save to file
    output_path = os.path.join(PROJECT_ROOT, "output_itinerary.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(api_response, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Output saved to: {output_path}")
    print(f"  Total pipeline time: {elapsed:.2f}s")
    print(f"{'=' * 60}")

    # Print human-readable itinerary
    full = FullItinerary(**api_response)
    human = formatter.to_human_readable(full)
    print(human)

    return api_response


def main():
    parser = argparse.ArgumentParser(
        description="Travel Jaunts — Itinerary Pipeline Runner"
    )
    parser.add_argument(
        "--deal", default="DEAL_001",
        help="Deal ID to process (default: DEAL_001)"
    )
    parser.add_argument(
        "--customer", default="CUST_001",
        help="Customer ID to use (default: CUST_001)"
    )
    parser.add_argument(
        "--use-llm", action="store_true", default=False,
        help="Enable optional LLM enrichment for descriptions (requires Ollama)"
    )
    args = parser.parse_args()

    try:
        run_pipeline(args.deal, args.customer, args.use_llm)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Make sure to run 'python scripts/generate_dummy_data.py' first.")
        sys.exit(1)
    except ValueError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()