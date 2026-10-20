"""
change_engine.py
================
Scientific Multi-Spectral & Structural Change Detection Engine for Space Netra.
Combines Delta-NDVI and Structural Similarity (SSIM) differencing with SCL cloud exclusion.
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_ndvi(b04_red, b08_nir):
    """
    Computes Normalized Difference Vegetation Index:
    NDVI = (NIR - Red) / (NIR + Red)
    """
    red = b04_red.astype(np.float32)
    nir = b08_nir.astype(np.float32)
    denom = nir + red + 1e-6
    ndvi = (nir - red) / denom
    return np.clip(ndvi, -1.0, 1.0)

def detect_surface_changes(t1_b04, t1_b08, t2_b04, t2_b08, valid_mask,
                           sensitivity=60, min_cluster_px=15, overlay_alpha=0.5):
    """
    Detects macro-scale land-surface modifications (urban expansion / grading)
    between two calibrated Sentinel-2 observation dates.
    """
    # 1. Multi-Spectral NDVI Transition Calculation
    ndvi_t1 = compute_ndvi(t1_b04, t1_b08)
    ndvi_t2 = compute_ndvi(t2_b04, t2_b08)
    delta_ndvi = ndvi_t2 - ndvi_t1

    # Conversion of vegetation to bare packed soil / industrial ground exhibits negative delta_ndvi
    spectral_diff = np.abs(delta_ndvi)
    # Scale to 0-255 uint8
    spectral_diff_uint8 = (spectral_diff * 255.0).clip(0, 255).astype(np.uint8)

    # 2. Structural Similarity (SSIM) Difference
    # Normalize Red bands to 8-bit for structural comparison
    gray_t1 = (t1_b04.astype(np.float32) / 4000.0 * 255.0).clip(0, 255).astype(np.uint8)
    gray_t2 = (t2_b04.astype(np.float32) / 4000.0 * 255.0).clip(0, 255).astype(np.uint8)
    score, diff = ssim(gray_t1, gray_t2, full=True)
    ssim_diff_uint8 = ((1.0 - diff) * 255.0).clip(0, 255).astype(np.uint8)

    # 3. Combined Change Magnitude Index (Dual Signal)
    combined_mag = (0.6 * spectral_diff_uint8 + 0.4 * ssim_diff_uint8).astype(np.uint8)

    # Apply valid-pixel mask (zero out clouds, shadows, nodata)
    combined_mag[~valid_mask] = 0

    # 4. Thresholding & Morphological Noise Suppression
    # sensitivity is from 30 to 150
    thresh_val = int(sensitivity)
    _, binary_raw = cv2.threshold(combined_mag, thresh_val, 255, cv2.THRESH_BINARY)
    binary_raw[~valid_mask] = 0

    # Morphological closing and dilation to remove isolated 1-pixel noise
    kernel = np.ones((3, 3), np.uint8)
    binary_clean = cv2.morphologyEx(binary_raw, cv2.MORPH_OPEN, kernel)
    binary_clean = cv2.dilate(binary_clean, kernel, iterations=1)
    binary_clean[~valid_mask] = 0

    # 5. Extract Contours & Bounding Zones
    contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by minimum cluster area
    change_mask_filtered = np.zeros_like(binary_clean)
    bbox_zones = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_cluster_px:
            cv2.drawContours(change_mask_filtered, [cnt], -1, 255, thickness=cv2.FILLED)
            x, y, w, h = cv2.boundingRect(cnt)
            bbox_zones.append((x, y, w, h, area))

    # 6. Generate Overlays for True-Color Image T2
    # Convert gray_t2 to RGB base
    img2_rgb = cv2.cvtColor(gray_t2, cv2.COLOR_GRAY2RGB)
    
    # Heatmap Overlay
    heatmap = cv2.applyColorMap(combined_mag, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img2_rgb, 1.0 - overlay_alpha, heatmap, overlay_alpha, 0)
    overlay[~valid_mask] = [80, 80, 80] # Grey out cloud/nodata regions

    # Bounding Box Highlight Image
    bbox_img = img2_rgb.copy()
    for idx, (x, y, w, h, _) in enumerate(bbox_zones, start=1):
        # Neon Yellow Bounding Box
        cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        # Translucent Red Highlight Fill
        sub = bbox_img[y:y+h, x:x+w]
        red_rect = np.full_like(sub, (255, 0, 0)) # Red in RGB
        bbox_img[y:y+h, x:x+w] = cv2.addWeighted(sub, 0.65, red_rect, 0.35, 0)
        # Tag Label
        cv2.putText(bbox_img, f"#{idx}", (x + 2, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    bbox_img[~valid_mask] = [80, 80, 80]

    # Cloud exclusion visual mask (Green = Valid, Magenta = Masked Out Cloud/Shadow)
    mask_visual = np.zeros((gray_t2.shape[0], gray_t2.shape[1], 3), dtype=np.uint8)
    mask_visual[valid_mask] = [34, 139, 34]      # Forest green
    mask_visual[~valid_mask] = [218, 112, 214]   # Orchid / Magenta for cloud shadow

    return {
        "change_mask": change_mask_filtered,
        "combined_magnitude": combined_mag,
        "overlay": overlay,
        "bbox_img": bbox_img,
        "cloud_mask_visual": mask_visual,
        "cluster_count": len(bbox_zones),
        "mean_divergence": round(float(np.mean(combined_mag[valid_mask])), 2) if np.any(valid_mask) else 0.0,
        "delta_ndvi": delta_ndvi
    }
