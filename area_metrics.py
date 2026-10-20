"""
area_metrics.py
===============
Geospatially Rigorous Ground Area & Statistical Metrics Service for Space Netra.
Calculates ground surface area in Hectares and Sq Km derived from projected UTM Zone 43N grid specs.
"""

import numpy as np

def calculate_geospatial_metrics(change_mask, valid_mask, gsd_meters=10.0):
    """
    Computes exact physical ground area from confirmed pixel counts
    in a metric projected coordinate reference system (UTM Zone 43N).
    
    1 pixel at 10m GSD = 10m x 10m = 100 m^2 = 0.01 hectares.
    """
    total_pixels = int(change_mask.size)
    valid_pixels = int(np.count_nonzero(valid_mask))
    masked_pixels = total_pixels - valid_pixels
    
    # Changed pixels strictly within valid data
    changed_pixels = int(np.count_nonzero(change_mask > 0))
    unchanged_pixels = max(0, valid_pixels - changed_pixels)

    # Physical Area Math
    pixel_area_sq_m = float(gsd_meters * gsd_meters) # 100.0 m^2
    
    total_area_ha = round((total_pixels * pixel_area_sq_m) / 10000.0, 2)
    valid_area_ha = round((valid_pixels * pixel_area_sq_m) / 10000.0, 2)
    changed_area_ha = round((changed_pixels * pixel_area_sq_m) / 10000.0, 2)
    unchanged_area_ha = round((unchanged_pixels * pixel_area_sq_m) / 10000.0, 2)

    changed_sq_km = round(changed_area_ha / 100.0, 3)
    total_sq_km = round(total_area_ha / 100.0, 3)

    # Ratios
    if valid_pixels > 0:
        change_ratio_pct = round((changed_pixels / valid_pixels) * 100.0, 2)
        valid_coverage_pct = round((valid_pixels / total_pixels) * 100.0, 2)
    else:
        change_ratio_pct = 0.0
        valid_coverage_pct = 0.0

    unchanged_ratio_pct = round(max(0.0, 100.0 - change_ratio_pct), 2)

    return {
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "masked_pixels": masked_pixels,
        "changed_pixels": changed_pixels,
        "unchanged_pixels": unchanged_pixels,
        "total_area_ha": total_area_ha,
        "valid_area_ha": valid_area_ha,
        "changed_area_ha": changed_area_ha,
        "unchanged_area_ha": unchanged_area_ha,
        "changed_sq_km": changed_sq_km,
        "total_sq_km": total_sq_km,
        "change_ratio_pct": change_ratio_pct,
        "unchanged_ratio_pct": unchanged_ratio_pct,
        "valid_coverage_pct": valid_coverage_pct,
        "gsd_meters": gsd_meters,
        "crs": "EPSG:32643 (UTM Zone 43N)"
    }
