#!/usr/bin/env python3
"""
Load Knowledge Base — Populate ChromaDB with POIs, dining, and logistics data.

Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings and ChromaDB
for vector storage. Creates collections: activities, dining, logistics.

"""

import json
import os
import sys

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("ERROR: chromadb not installed. Run: pip install chromadb")
    sys.exit(1)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_db")


def build_activity_text(poi: dict) -> str:
    """Build a searchable text string from a POI record."""
    tiers = ", ".join(poi.get("tier", []))
    types = ", ".join(poi.get("travel_type", []))
    occasions = ", ".join(poi.get("occasion", []))
    times = ", ".join(poi.get("time_of_day", []))
    return (
        f"{poi['name']} in {poi['city']}, {poi['country']}. "
        f"Category: {poi['category']}. Area: {poi['area']}. "
        f"Tiers: {tiers}. Travel types: {types}. "
        f"Occasions: {occasions}. Best time: {times}. "
        f"Duration: {poi['duration']}. Energy: {poi['energy']}. "
        f"{poi['description']}"
    )


def build_dining_text(dining: dict) -> str:
    """Build a searchable text string from a dining record."""
    tiers = ", ".join(dining.get("tier", []))
    types = ", ".join(dining.get("travel_type", []))
    occasions = ", ".join(dining.get("occasion", []))
    return (
        f"{dining['name']} in {dining['city']}, {dining['country']}. "
        f"Cuisine: {dining['cuisine']}. Meal: {dining['meal']}. "
        f"Tiers: {tiers}. Travel types: {types}. Occasions: {occasions}. "
        f"Price: {dining['price_range_inr']} INR. Area: {dining['area']}. "
        f"{dining['description']}"
    )


def build_logistics_text(logistics: dict) -> str:
    """Build a searchable text string from a logistics record."""
    transport = logistics.get("transport", {})
    timing = logistics.get("timing_advice", {})
    customs = ", ".join(logistics.get("local_customs", []))
    best = ", ".join(timing.get("best_months", []))
    return (
        f"Logistics for {logistics['city']}, {logistics['country']}. "
        f"Airport to city: {transport.get('airport_to_city', 'N/A')}. "
        f"Within city: {transport.get('within_city', 'N/A')}. "
        f"Best months: {best}. Timezone: {timing.get('timezone', 'N/A')}. "
        f"Local customs: {customs}"
    )


def load_collection(client, name: str, data: list,
                    text_builder, id_field: str):
    """Load records into a ChromaDB collection."""
    # Delete existing collection if it exists
    try:
        client.delete_collection(name)
    except Exception:
        pass

    collection = client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = []
    documents = []
    metadatas = []

    for record in data:
        record_id = record.get(id_field, record.get("city", "unknown"))
        ids.append(str(record_id))
        documents.append(text_builder(record))
        # Store key metadata for filtering
        meta = {"city": record.get("city", ""), "country": record.get("country", "")}
        if "tier" in record and isinstance(record["tier"], list):
            meta["tiers"] = ",".join(record["tier"])
        if "travel_type" in record and isinstance(record["travel_type"], list):
            meta["travel_types"] = ",".join(record["travel_type"])
        if "occasion" in record and isinstance(record["occasion"], list):
            meta["occasions"] = ",".join(record["occasion"])
        if "category" in record:
            meta["category"] = record["category"]
        if "cuisine" in record:
            meta["cuisine"] = record["cuisine"]
        if "meal" in record:
            meta["meal"] = record["meal"]
        metadatas.append(meta)

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


def main():
    """Load all knowledge base files into ChromaDB."""
    print("=" * 50)
    print("TRAVEL JAUNTS — Knowledge Base Loader")
    print("=" * 50)
    print(f"\nChromaDB path: {CHROMA_DIR}\n")

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Load POIs / activities
    pois_path = os.path.join(DATA_DIR, "dummy_pois.json")
    with open(pois_path, "r", encoding="utf-8") as f:
        pois = json.load(f)
    count = load_collection(client, "activities", pois,
                            build_activity_text, "poi_id")
    print(f"  ✓ Activities: {count} records loaded")

    # Load Dining
    dining_path = os.path.join(DATA_DIR, "dummy_dining.json")
    with open(dining_path, "r", encoding="utf-8") as f:
        dining = json.load(f)
    count = load_collection(client, "dining", dining,
                            build_dining_text, "dining_id")
    print(f"  ✓ Dining: {count} records loaded")

    # Load Logistics
    logistics_path = os.path.join(DATA_DIR, "dummy_logistics.json")
    with open(logistics_path, "r", encoding="utf-8") as f:
        logistics = json.load(f)
    count = load_collection(client, "logistics", logistics,
                            build_logistics_text, "city")
    print(f"  ✓ Logistics: {count} records loaded")

    print(f"\nKnowledge base loaded successfully!")
    print(f"Collections: {[c.name for c in client.list_collections()]}")


if __name__ == "__main__":
    main()
