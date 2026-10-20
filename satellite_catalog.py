"""
satellite_catalog.py
====================
STAC Catalog Discovery Service for Space Netra.
Queries Microsoft Planetary Computer or reads local provenance manifests.
"""

import os
import json
import requests

STAC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
DEFAULT_BBOX = [75.4800, 19.8600, 75.5000, 19.8800] # Shendra MIDC, Chh. Sambhajinagar

def search_scenes(bbox=None, year_t1=2021, year_t2=2024, max_cloud=5.0):
    """
    Discovers candidate Sentinel-2 L2A anniversary scenes for the selected AOI.
    """
    if bbox is None:
        bbox = DEFAULT_BBOX

    results = {}
    windows = [
        ("T1", f"{year_t1}-03-01T00:00:00Z/{year_t1}-03-31T23:59:59Z"),
        ("T2", f"{year_t2}-03-01T00:00:00Z/{year_t2}-03-31T23:59:59Z")
    ]

    for label, dtime in windows:
        payload = {
            "collections": ["sentinel-2-l2a"],
            "bbox": bbox,
            "datetime": dtime,
            "query": {"eo:cloud_cover": {"lt": max_cloud}},
            "limit": 5
        }
        try:
            r = requests.post(STAC_ENDPOINT, json=payload, timeout=8)
            if r.status_code == 200:
                features = r.json().get("features", [])
                if features:
                    best = min(features, key=lambda f: f["properties"].get("eo:cloud_cover", 100))
                    results[label] = {
                        "id": best["id"],
                        "datetime": best["properties"]["datetime"],
                        "cloud_cover": round(best["properties"].get("eo:cloud_cover", 0.0), 3),
                        "platform": best["properties"].get("platform", "Sentinel-2A"),
                        "tile": best["properties"].get("s2:mgrs_tile", "T43QEC")
                    }
        except Exception:
            pass

    # If live search returned empty or network unavailable, load local manifest
    if not results or "T1" not in results or "T2" not in results:
        manifest_path = os.path.join("data", "change_detection", "manifests", "provenance_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                data = json.load(f)
                scenes = data.get("scenes", {})
                results["T1"] = scenes.get("T1_historical")
                results["T2"] = scenes.get("T2_recent")

    return results
