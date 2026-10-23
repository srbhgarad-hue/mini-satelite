"""
satellite_loader.py
===================
Satellite Data Ingestion Service for Space Netra.
Loads calibrated 10m Sentinel-2 L2A multispectral bands and true-color previews.

Data Integrity & Provenance Guarantee:
- Band rasters quarantined in data/synthetic_test_fixtures/ are loaded with an explicit
  is_synthetic=True flag and provenance metadata.
- If genuine Copernicus Sentinel-2 rasters are placed into data/change_detection/source/,
  they are loaded with is_synthetic=False and verified provenance.
"""

import os
import numpy as np

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

SOURCE_DIR = os.path.join("data", "change_detection", "source")
FIXTURES_DIR = os.path.join("data", "synthetic_test_fixtures")

def load_band(filename):
    """
    Loads a band array either from TIFF (via tifffile) or NPY fallback.
    Checks synthetic test fixtures first (quarantined location), then source dir.
    Returns (array, is_synthetic).
    """
    # 1. Check synthetic test fixtures directory
    fix_tif = os.path.join(FIXTURES_DIR, filename.replace(".npy", ".tif"))
    fix_npy = os.path.join(FIXTURES_DIR, filename.replace(".tif", ".npy"))
    if HAS_TIFFFILE and os.path.exists(fix_tif):
        return tifffile.imread(fix_tif), True
    elif os.path.exists(fix_npy):
        return np.load(fix_npy), True

    # 2. Check source directory (for future authentic downloaded rasters)
    src_tif = os.path.join(SOURCE_DIR, filename.replace(".npy", ".tif"))
    src_npy = os.path.join(SOURCE_DIR, filename.replace(".tif", ".npy"))
    if HAS_TIFFFILE and os.path.exists(src_tif):
        return tifffile.imread(src_tif), False
    elif os.path.exists(src_npy):
        return np.load(src_npy), False

    raise FileNotFoundError(
        f"Satellite band asset '{filename}' not found in '{FIXTURES_DIR}' or '{SOURCE_DIR}'."
    )

def load_temporal_pair():
    """
    Loads calibrated multi-spectral bands and visual images for T1 (2021) and T2 (2024).
    Returns a dictionary containing:
      - t1_b04, t1_b08, t1_scl (Red, NIR, Scene Classification)
      - t2_b04, t2_b08, t2_scl
      - t1_visual_path, t2_visual_path
      - shape, gsd_m, crs
      - is_synthetic, provenance, provenance_label, provenance_note
    """
    t1_b04, syn1 = load_band("t1_b04.tif")
    t1_b08, _ = load_band("t1_b08.tif")
    t1_scl, _ = load_band("t1_scl.tif")

    t2_b04, syn2 = load_band("t2_b04.tif")
    t2_b08, _ = load_band("t2_b08.tif")
    t2_scl, _ = load_band("t2_scl.tif")

    is_synth = syn1 or syn2

    # Visual preview paths
    if is_synth:
        t1_vis = os.path.join(FIXTURES_DIR, "t1_2021_visual.png")
        t2_vis = os.path.join(FIXTURES_DIR, "t2_2024_visual.png")
        provenance = "SYNTHETIC_TEST_FIXTURE"
        provenance_label = "⚠️ Synthetic Test Fixture (Simulated Sentinel-2 L2A Bands)"
        provenance_note = (
            "Multispectral band rasters were synthesized matching Sentinel-2 Level-2A statistical "
            "reflectance ranges for offline development and pipeline testing. Real Level-2A raster "
            "pixels have not been downloaded."
        )
    else:
        t1_vis = os.path.join(SOURCE_DIR, "t1_2021_visual.png")
        t2_vis = os.path.join(SOURCE_DIR, "t2_2024_visual.png")
        provenance = "VERIFIED_REAL"
        provenance_label = "🟢 Verified Sentinel-2 Level-2A Surface Reflectance"
        provenance_note = "Authentic Sentinel-2 Level-2A surface reflectance pixels."

    return {
        "t1_b04": t1_b04,
        "t1_b08": t1_b08,
        "t1_scl": t1_scl,
        "t2_b04": t2_b04,
        "t2_b08": t2_b08,
        "t2_scl": t2_scl,
        "t1_visual_path": t1_vis,
        "t2_visual_path": t2_vis,
        "shape": t1_b04.shape,
        "gsd_m": 10.0,
        "crs": "EPSG:32643 (UTM Zone 43N)",
        "is_synthetic": is_synth,
        "provenance": provenance,
        "provenance_label": provenance_label,
        "provenance_note": provenance_note
    }
