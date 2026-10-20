"""
preprocessor.py
===============
Geospatial Preprocessing & Cloud/Shadow Screening Service for Space Netra.
Handles Sentinel-2 Scene Classification Layer (SCL) masking and surface reflectance normalization.
"""

import numpy as np

# Sentinel-2 Scene Classification Layer (SCL) classes:
# 0: NO_DATA
# 1: SATURATED_OR_DEFECTIVE
# 2: DARK_AREA_PIXELS
# 3: CLOUD_SHADOWS
# 4: VEGETATION
# 5: NOT_VEGETATED (Bare Soil)
# 6: WATER
# 7: UNCLASSIFIED
# 8: CLOUD_MEDIUM_PROBABILITY
# 9: CLOUD_HIGH_PROBABILITY
# 10: THIN_CIRRUS
# 11: SNOW
INVALID_SCL_CLASSES = {0, 1, 3, 8, 9, 10, 11}

def build_valid_pixel_mask(scl_t1, scl_t2):
    """
    Constructs a Boolean mask of valid pixels present in BOTH dates.
    Zeroes out clouds, cirrus, cloud shadows, and defective pixels.
    Returns:
      valid_mask (bool array, True = valid, False = cloud/shadow/nodata)
    """
    invalid_t1 = np.isin(scl_t1, list(INVALID_SCL_CLASSES))
    invalid_t2 = np.isin(scl_t2, list(INVALID_SCL_CLASSES))
    valid_mask = ~(invalid_t1 | invalid_t2)
    return valid_mask

def normalize_reflectance(band_16bit):
    """
    Converts 16-bit integer Sentinel-2 surface reflectance (0 - 10000)
    to physical float reflectance values (0.0 - 1.0).
    """
    arr = band_16bit.astype(np.float32) / 10000.0
    return np.clip(arr, 0.0, 1.0)
