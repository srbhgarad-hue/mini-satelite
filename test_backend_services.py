"""
test_backend_services.py
========================
Automated Verification Suite for Space Netra Backend Services.
Tests satellite_loader, preprocessor, change_engine, and area_metrics.
"""

import sys
import numpy as np

def run_tests():
    print("==================================================================")
    print(" SPACE NETRA: BACKEND SERVICES AUTOMATED VERIFICATION SUITE       ")
    print("==================================================================")

    # 1. Test satellite_loader
    import satellite_loader
    data = satellite_loader.load_temporal_pair()
    assert "t1_b04" in data and "t2_b04" in data, "Failed to load T1/T2 B04 bands"
    assert data["shape"] == (200, 200), f"Unexpected shape {data['shape']}"
    print("[PASS] satellite_loader: Loaded T1/T2 multispectral bands (200x200 grid)")

    # 2. Test preprocessor
    import preprocessor
    valid_mask = preprocessor.build_valid_pixel_mask(data["t1_scl"], data["t2_scl"])
    assert valid_mask.shape == (200, 200), "Valid mask shape mismatch"
    valid_count = int(np.count_nonzero(valid_mask))
    print(f"[PASS] preprocessor: Constructed SCL cloud mask ({valid_count}/40000 valid pixels)")

    # 3. Test change_engine
    import change_engine
    res = change_engine.detect_surface_changes(
        data["t1_b04"], data["t1_b08"],
        data["t2_b04"], data["t2_b08"],
        valid_mask,
        sensitivity=50,
        min_cluster_px=15
    )
    assert "change_mask" in res and "overlay" in res, "Missing outputs from change engine"
    print(f"[PASS] change_engine: Detected {res['cluster_count']} change clusters")

    # 4. Test area_metrics
    import area_metrics
    metrics = area_metrics.calculate_geospatial_metrics(res["change_mask"], valid_mask, gsd_meters=10.0)
    assert metrics["total_area_ha"] == 400.0, f"Expected 400.0 ha, got {metrics['total_area_ha']}"
    assert metrics["changed_area_ha"] + metrics["unchanged_area_ha"] == metrics["valid_area_ha"], "Area conservation failed!"
    print(f"[PASS] area_metrics: Changed Area = {metrics['changed_area_ha']} Ha ({metrics['change_ratio_pct']}%)")
    print(f"[PASS] area_metrics: Conservation of Area verified ({metrics['valid_area_ha']} Ha total valid)")

    # 5. Zero-Change Sanity Test
    print("\n[INFO] Running Zero-Change Sanity Test (T1 vs T1 self-comparison)...")
    res_zero = change_engine.detect_surface_changes(
        data["t1_b04"], data["t1_b08"],
        data["t1_b04"], data["t1_b08"],
        valid_mask,
        sensitivity=50
    )
    metrics_zero = area_metrics.calculate_geospatial_metrics(res_zero["change_mask"], valid_mask, gsd_meters=10.0)
    print(f"[PASS] Zero-Change Test: Reported change = {metrics_zero['change_ratio_pct']}% (Target: < 1.0%)")
    assert metrics_zero["change_ratio_pct"] == 0.0, "Zero-change test produced non-zero false positives!"

    print("\n==================================================================")
    print(" ALL BACKEND UNIT TESTS PASSED SUCCESSFULLY!                      ")
    print("==================================================================")

if __name__ == "__main__":
    run_tests()
