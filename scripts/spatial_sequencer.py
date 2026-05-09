#!/usr/bin/env python3
"""
Stage 3: Spatial Sequencer — Order cities geographically, cluster POIs by proximity.

Uses DBSCAN clustering to group POIs/neighborhoods per day to prevent backtracking.
Uses GeoPy for accurate geographic distance calculation.
"""

import json
import os
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    DBSCAN = None

try:
    from geopy.distance import geodesic
except ImportError:
    geodesic = None

try:
    from scripts.deal_parser import ParsedItineraryContext, CityStay
except ImportError:
    from deal_parser import ParsedItineraryContext, CityStay


@dataclass
class POILocation:
    """A lightweight POI with location data for clustering."""
    poi_id: str
    name: str
    lat: float
    lon: float
    area: str
    city: str


@dataclass
class SpatialContext:
    """Output of the spatial sequencer consumed by downstream stages."""
    ordered_cities: List[str]
    city_was_resequenced: bool
    day_clusters: Dict[int, List[str]]       # day_number → list of area names
    poi_clusters: Dict[int, List[POILocation]]  # cluster_id → list of POIs
    neighborhood_plan: Dict[str, List[List[str]]]  # city → [day1_areas, day2_areas, ...]


def calculate_geo_distance(lat1: float, lon1: float,
                           lat2: float, lon2: float) -> float:
    """
    Calculate distance between two points in kilometers.

    Uses GeoPy geodesic if available, falls back to Haversine formula.
    """
    if geodesic is not None:
        return geodesic((lat1, lon1), (lat2, lon2)).km

    # Haversine fallback
    R = 6371.0
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class SpatialSequencer:
    """Handles city ordering and POI clustering for itinerary optimization."""

    def validate_and_resequence(self, cities: List[CityStay],
                                country: str) -> Tuple[List[str], bool]:
        """
        Check if city order is geographically optimal, resequence if not.

        Uses nearest-neighbor heuristic starting from the first city (arrival city).
        Returns (ordered_city_names, was_resequenced).
        """
        if len(cities) <= 2:
            return [c.city for c in cities], False

        # Build distance matrix
        city_coords = [(c.city, c.hotel.lat, c.hotel.lon)
                       for c in cities if c.hotel]
        if len(city_coords) < 2:
            return [c.city for c in cities], False

        n = len(city_coords)
        original_order = [c[0] for c in city_coords]

        # Nearest-neighbor TSP heuristic starting from first city
        visited = [False] * n
        order = [0]
        visited[0] = True

        for _ in range(n - 1):
            current = order[-1]
            best_dist = float('inf')
            best_idx = -1
            for j in range(n):
                if not visited[j]:
                    d = calculate_geo_distance(
                        city_coords[current][1], city_coords[current][2],
                        city_coords[j][1], city_coords[j][2],
                    )
                    if d < best_dist:
                        best_dist = d
                        best_idx = j
            if best_idx >= 0:
                order.append(best_idx)
                visited[best_idx] = True

        optimal_order = [city_coords[i][0] for i in order]
        was_resequenced = (optimal_order != original_order)
        return optimal_order, was_resequenced

    def cluster_pois_by_day(self, pois: List[POILocation],
                            num_days: int) -> Dict[int, List[POILocation]]:
        """
        Use DBSCAN to cluster POIs by geographic proximity.

        Returns dict mapping cluster_id → list of POIs.
        Cluster -1 = noise (unclustered POIs assigned to nearest cluster).
        """
        if not pois or num_days < 1:
            return {}

        if len(pois) < 2:
            return {0: pois}

        coords = np.array([[p.lat, p.lon] for p in pois])

        # DBSCAN with eps in degrees (~0.03° ≈ 3.3 km)
        eps = 0.03
        min_samples = max(1, len(pois) // (num_days + 1))

        if DBSCAN is not None:
            clustering = DBSCAN(
                eps=eps, min_samples=min_samples,
                metric='euclidean',
            ).fit(coords)
            labels = clustering.labels_
        else:
            # Fallback: simple round-robin assignment
            labels = np.array([i % num_days for i in range(len(pois))])

        # Group POIs by cluster
        clusters: Dict[int, List[POILocation]] = {}
        for i, label in enumerate(labels):
            cluster_id = int(label)
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(pois[i])

        # Assign noise POIs (-1) to nearest cluster
        if -1 in clusters and len(clusters) > 1:
            noise_pois = clusters.pop(-1)
            other_ids = [k for k in clusters.keys() if k != -1]
            for p in noise_pois:
                best_id = other_ids[0]
                best_dist = float('inf')
                for cid in other_ids:
                    center_lat = np.mean([pp.lat for pp in clusters[cid]])
                    center_lon = np.mean([pp.lon for pp in clusters[cid]])
                    d = calculate_geo_distance(p.lat, p.lon,
                                               center_lat, center_lon)
                    if d < best_dist:
                        best_dist = d
                        best_id = cid
                clusters[best_id].append(p)

        return clusters

    def assign_neighborhood_clusters(self, city: str,
                                     num_days: int,
                                     pois: List[POILocation] = None
                                     ) -> List[List[str]]:
        """
        Group neighborhoods per day for a single city.

        Returns list of area-name lists, one per day.
        """
        if not pois:
            return [[] for _ in range(num_days)]

        city_pois = [p for p in pois if p.city == city]
        if not city_pois:
            return [[] for _ in range(num_days)]

        clusters = self.cluster_pois_by_day(city_pois, num_days)

        # Map clusters to days (one cluster per day)
        day_areas: List[List[str]] = []
        
        # Sort cluster IDs by cluster size descending
        cluster_ids = sorted(
            clusters.keys(),
            key=lambda cid: len(clusters[cid]),
            reverse=True
        )

        for day_idx in range(num_days):
            if day_idx < len(cluster_ids):
                cid = cluster_ids[day_idx]
                areas = sorted(list(set(p.area for p in clusters[cid])))[:2]
            else:
                # More days than clusters: cycle
                cid = cluster_ids[day_idx % len(cluster_ids)]
                areas = sorted(list(set(p.area for p in clusters[cid])))[:2]
            day_areas.append(areas)

        return day_areas

    def build_spatial_context(self, ctx: ParsedItineraryContext,
                              pois_data: List[Dict]) -> SpatialContext:
        """
        Build complete spatial context from parsed deal and POI data.

        This is the main entry point for Stage 3.
        """
        # Convert raw POI dicts to POILocation objects
        all_pois = []
        for p in pois_data:
            all_pois.append(POILocation(
                poi_id=p["poi_id"], name=p["name"],
                lat=p["lat"], lon=p["lon"],
                area=p["area"], city=p["city"],
            ))

        # Step 1: Validate and resequence cities
        ordered_cities, was_resequenced = self.validate_and_resequence(
            ctx.cities, ctx.destination
        )

        # Step 2: Cluster POIs per city
        poi_clusters = {}
        neighborhood_plan = {}
        day_clusters = {}

        global_day = 1
        for city_stay in ctx.cities:
            city_name = city_stay.city
            city_pois = [p for p in all_pois if p.city == city_name]

            # Cluster POIs for this city
            clusters = self.cluster_pois_by_day(city_pois, city_stay.days)
            for cid, cluster_pois in clusters.items():
                poi_clusters[len(poi_clusters)] = cluster_pois

            # Assign neighborhoods per day
            day_areas = self.assign_neighborhood_clusters(
                city_name, city_stay.days, all_pois
            )
            neighborhood_plan[city_name] = day_areas

            # Map day numbers to area clusters
            for d_idx in range(city_stay.days):
                areas = day_areas[d_idx] if d_idx < len(day_areas) else []
                day_clusters[global_day] = areas
                global_day += 1

        return SpatialContext(
            ordered_cities=ordered_cities,
            city_was_resequenced=was_resequenced,
            day_clusters=day_clusters,
            poi_clusters=poi_clusters,
            neighborhood_plan=neighborhood_plan,
        )


# Standalone execution

def main():
    """Run spatial sequencer on first deal with all POIs."""
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(PROJECT_ROOT, "data")

    with open(os.path.join(data_dir, "dummy_deals.json"), "r", encoding="utf-8") as f:
        deals = json.load(f)
    with open(os.path.join(data_dir, "dummy_customer_profiles.json"), "r", encoding="utf-8") as f:
        customers = json.load(f)
    with open(os.path.join(data_dir, "dummy_pois.json"), "r", encoding="utf-8") as f:
        pois = json.load(f)

    try:
        from scripts.deal_parser import parse_deal
    except ImportError:
        from deal_parser import parse_deal

    ctx = parse_deal(deals[0], customers[0])
    sequencer = SpatialSequencer()
    spatial = sequencer.build_spatial_context(ctx, pois)

    reseq_label = "resequenced" if spatial.city_was_resequenced else "validated"
    print(f"[Stage 3] Spatial sequencing...")
    print(f"  → City order {reseq_label}: {' → '.join(spatial.ordered_cities)}")
    print(f"  → Neighborhood clusters assigned per city:")
    for city, areas in spatial.neighborhood_plan.items():
        print(f"    {city}: {len(areas)} day(s) of areas")
        for i, day_areas in enumerate(areas):
            print(f"      Day {i+1}: {', '.join(day_areas) if day_areas else 'general'}")


if __name__ == "__main__":
    main()