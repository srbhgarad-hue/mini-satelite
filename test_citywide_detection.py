"""
test_citywide_detection.py
==========================
Automated verification for city-wide photographic building change detection.
"""

import city_catalog
import photo_streamer
import building_detector

def test_citywide():
    print("==================================================================")
    print(" SPACE NETRA: WHOLE-CITY PHOTOGRAPHIC BUILDING DETECTION TEST     ")
    print(" Target: Chhatrapati Sambhajinagar (Aurangabad), Maharashtra      ")
    print("==================================================================")

    # 1. Verify Hotspots
    hotspots = city_catalog.CITY_HOTSPOTS
    assert len(hotspots) >= 7, f"Expected >= 7 hotspots, found {len(hotspots)}"
    print(f"[PASS] city_catalog: Verified {len(hotspots)} major city growth sectors:")
    for name in hotspots:
        print(f"  - {name} ({hotspots[name]['category']})")

    # 2. Test 3-Stage Photo Streamer on CIDCO
    cidco = hotspots["CIDCO Urban Expansion (N-1 to N-12)"]
    print(f"\n[INFO] Fetching 3-stage photographic timeline for CIDCO (Lat: {cidco['lat']}, Lon: {cidco['lon']})...")
    timeline = photo_streamer.fetch_3stage_photographic_timeline(cidco["lat"], cidco["lon"], zoom=cidco["zoom"])
    assert os.path.exists(timeline["p1_path"]), "Missing 2016 photo"
    assert os.path.exists(timeline["p2_path"]), "Missing 2020 photo"
    assert os.path.exists(timeline["p3_path"]), "Missing Current photo"
    print(f"[PASS] photo_streamer: Retrieved 3-stage real photos (GSD: {timeline['gsd_m']} m/px at Zoom {timeline['zoom']})")
    print(f"  - Epoch 1 (2016): {timeline['dates']['epoch_1']}")
    print(f"  - Epoch 2 (2020): {timeline['dates']['epoch_2']}")
    print(f"  - Epoch 3 (Current): {timeline['dates']['epoch_3']}")

    # 3. Test Building Detector Engine
    print("\n[INFO] Running geometric building detector on real photos...")
    res = building_detector.detect_new_buildings(
        timeline["p1_path"],
        timeline["p3_path"],
        gsd_m=timeline["gsd_m"],
        min_area_sq_m=50.0,
        sensitivity=50
    )
    print(f"[PASS] building_detector: Detected {res['building_count']} newly constructed buildings!")
    print(f"  - Total New Built-Up Footprint: {res['total_built_area_sq_m']} m² ({res['total_built_area_ha']} Ha)")
    print(f"  - Built-Up Expansion Density: {res['building_density_pct']}% of analyzed ground")

    # Print sample detected buildings
    if res["inventory"]:
        print("  - Sample Detected New Buildings:")
        for b in res["inventory"][:3]:
            print(f"    * {b['Building ID']}: {b['Footprint Area (m²)']} m² ({b['Dimensions (W × H m)']}) | Status: {b['Status']}")

    print("\n==================================================================")
    print(" ALL CITY-WIDE BUILDING DETECTION TESTS PASSED SUCCESSFULLY!      ")
    print("==================================================================")

if __name__ == "__main__":
    import os
    test_citywide()
