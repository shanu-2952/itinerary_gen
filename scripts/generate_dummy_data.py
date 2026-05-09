#!/usr/bin/env python3
"""
Generate dummy data files for the Travel Jaunts itinerary pipeline.

This script validates and copies the pre-authored JSON data files
into the data/ directory. In production, this would be replaced
by real data ingestion from APIs, databases, or partner feeds.

"""

import json
import os
import sys

# Resolve project root (one level up from scripts/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

EXPECTED_FILES = {
    "dummy_deals.json": {"min_count": 4, "label": "deals"},
    "dummy_customer_profiles.json": {"min_count": 4, "label": "customer profiles"},
    "dummy_pois.json": {"min_count": 15, "label": "POIs"},
    "dummy_dining.json": {"min_count": 8, "label": "dining options"},
    "dummy_logistics.json": {"min_count": 1, "label": "logistics entries"},
}


def validate_json_file(filepath: str, min_count: int, label: str) -> bool:
    """Validate that a JSON file exists, is parseable, and meets minimum record count."""
    if not os.path.exists(filepath):
        print(f"  ✗ MISSING: {filepath}")
        return False
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"  ✗ {label}: expected JSON array, got {type(data).__name__}")
            return False
        if len(data) < min_count:
            print(f"  ✗ {label}: expected >= {min_count} records, got {len(data)}")
            return False
        print(f"  ✓ {label}: {len(data)} records loaded")
        return True
    except json.JSONDecodeError as e:
        print(f"  ✗ {label}: JSON parse error — {e}")
        return False


def main():
    """Validate all dummy data files exist and meet requirements."""
    print("=" * 50)
    print("TRAVEL JAUNTS — Dummy Data Validator")
    print("=" * 50)
    print(f"\nData directory: {DATA_DIR}\n")

    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: Data directory not found: {DATA_DIR}")
        print("Please ensure the data/ folder contains all JSON files.")
        sys.exit(1)

    all_valid = True
    for filename, spec in EXPECTED_FILES.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not validate_json_file(filepath, spec["min_count"], spec["label"]):
            all_valid = False

    print()
    if all_valid:
        print("All dummy data files validated successfully!")
        print("Ready for pipeline execution.")
    else:
        print("Some data files are missing or invalid.")
        print("Please check the data/ directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
