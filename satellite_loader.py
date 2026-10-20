"""
satellite_loader.py
===================
Satellite Data Ingestion Service for Space Netra.
Loads calibrated 10m Sentinel-2 L2A multispectral bands and true-color previews.
"""

import os
import numpy as np
from PIL import Image

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

SOURCE_DIR = os.path.join("data", "change_detection", "source")

def load_band(filename):
    """Loads a band array either from TIFF (via tifffile) or NPY fallback."""
    full_path_tif = os.path.join(SOURCE_DIR, filename.replace(".npy", ".tif"))
    full_path_npy = os.path.join(SOURCE_DIR, filename.replace(".tif", ".npy"))

    if HAS_TIFFFILE and os.path.exists(full_path_tif):
        return tifffile.imread(full_path_tif)
    elif os.path.exists(full_path_npy):
        return np.load(full_path_npy)
    else:
        raise FileNotFoundError(f"Satellite band asset '{filename}' not found in '{SOURCE_DIR}'.")

def load_temporal_pair():
    """
    Loads calibrated multi-spectral bands and visual images for T1 (2021) and T2 (2024).
    Returns a dictionary containing:
      - t1_b04, t1_b08, t1_scl (Red, NIR, Scene Classification)
      - t2_b04, t2_b08, t2_scl
      - t1_visual_path, t2_visual_path
    """
    t1_b04 = load_band("t1_b04.tif")
    t1_b08 = load_band("t1_b08.tif")
    t1_scl = load_band("t1_scl.tif")

    t2_b04 = load_band("t2_b04.tif")
    t2_b08 = load_band("t2_b08.tif")
    t2_scl = load_band("t2_scl.tif")

    t1_vis = os.path.join(SOURCE_DIR, "t1_2021_visual.png")
    t2_vis = os.path.join(SOURCE_DIR, "t2_2024_visual.png")

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
        "crs": "EPSG:32643 (UTM Zone 43N)"
    }
