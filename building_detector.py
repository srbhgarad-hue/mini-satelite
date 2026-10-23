"""
building_detector.py
====================
Real-Photo Building-Level Change Detection Engine for Space Netra.
Extracts rectangular building contours, applies geometric filters,
and renders vector bounding polygons and labels directly on real satellite photos.
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def detect_new_buildings(img_t1_path, img_t3_path, gsd_m=2.24, min_area_sq_m=50.0, sensitivity=65):
    """
    Identifies newly constructed buildings between historical photo (T1)
    and current photo (T3) at full photographic resolution.
    
    Parameters:
      - img_t1_path: File path to historical aerial photo
      - img_t3_path: File path to current aerial photo
      - gsd_m: Ground sampling distance (meters/pixel)
      - min_area_sq_m: Minimum building footprint floor (filters cars, trees, lane lines)
      - sensitivity: Detection threshold
    """
    img1 = cv2.imread(img_t1_path)
    img3 = cv2.imread(img_t3_path)

    if img1 is None or img3 is None:
        raise FileNotFoundError(f"Cannot load aerial photos: '{img_t1_path}' or '{img_t3_path}'")

    if img1.shape != img3.shape:
        img3 = cv2.resize(img3, (img1.shape[1], img1.shape[0]))

    H, W = img1.shape[:2]
    pixel_area_sq_m = gsd_m * gsd_m
    min_pixels = max(8, int(min_area_sq_m / pixel_area_sq_m))

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

    # 1. Structural Edge Extraction (Canny / Sobel)
    edges1 = cv2.Canny(gray1, 50, 150)
    edges3 = cv2.Canny(gray3, 50, 150)

    # 2. Structural Dissimilarity (SSIM difference)
    score, diff = ssim(gray1, gray3, full=True)
    ssim_diff_uint8 = ((1.0 - diff) * 255.0).clip(0, 255).astype(np.uint8)

    # 3. Structural Gradient Difference (Highlighting new building walls and roof edges)
    edge_diff = cv2.subtract(edges3, edges1)
    combined = cv2.addWeighted(ssim_diff_uint8, 0.65, edge_diff, 0.35, 0)

    # 4. Threshold & Morphological Reconstruction
    _, thresh = cv2.threshold(combined, int(sensitivity), 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    # Morphological closing bridges building roof boundaries
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)

    # 5. Extract Contours & Apply Geometric Building Filter
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    building_inventory = []
    annotated_img = img3.copy()
    b_id = 0

    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px >= min_pixels:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / float(max(1, h))

            # Geometric Building Filter:
            # Human structures have aspect ratios between 0.2 and 5.0 (excluding endless thin lines)
            if 0.2 <= aspect_ratio <= 5.0 and w >= 5 and h >= 5:
                b_id += 1
                footprint_area_sq_m = round(area_px * pixel_area_sq_m, 1)
                width_m = round(w * gsd_m, 1)
                height_m = round(h * gsd_m, 1)

                building_inventory.append({
                    "Building ID": f"BLDG-{b_id:02d}",
                    "Status": "New Construction",
                    "Footprint Area (m²)": footprint_area_sq_m,
                    "Dimensions (W × H m)": f"{width_m}m × {height_m}m",
                    "Pixel Box (x,y,w,h)": f"({x}, {y}, {w}, {h})",
                    "Confidence": f"{min(98.5, round(75.0 + (area_px / 10.0), 1))}%"
                })

                # A. Draw High-Visibility Emerald Green Bounding Box (BGR: 45, 215, 80)
                cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (45, 215, 80), 2)

                # B. Translucent Fill on New Building
                sub = annotated_img[y:y+h, x:x+w]
                highlight_fill = np.full_like(sub, (35, 180, 70)) # Emerald green in BGR
                annotated_img[y:y+h, x:x+w] = cv2.addWeighted(sub, 0.70, highlight_fill, 0.30, 0)

                # C. Prominent Badge Tag Background
                label_text = f"NEW BLDG #{b_id}"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                cv2.rectangle(annotated_img, (x, max(0, y - 18)), (x + tw + 6, max(18, y)), (15, 20, 18), -1)
                # Badge Text (Emerald Green)
                cv2.putText(annotated_img, label_text, (x + 3, max(13, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (45, 215, 80), 1)

    # Thermal Heatmap Overlay for alternative inspection
    heatmap = cv2.applyColorMap(combined, cv2.COLORMAP_JET)
    heatmap_overlay = cv2.addWeighted(img3, 0.60, heatmap, 0.40, 0)

    # Convert BGR to RGB for Streamlit display
    annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2RGB)

    total_changed_footprint_sq_m = sum(b["Footprint Area (m²)"] for b in building_inventory)
    total_area_analyzed_sq_m = H * W * pixel_area_sq_m
    density_pct = round((total_changed_footprint_sq_m / max(1.0, total_area_analyzed_sq_m)) * 100.0, 2)

    return {
        "annotated_img": annotated_rgb,
        "heatmap_img": heatmap_rgb,
        "binary_mask": dilated,
        "building_count": b_id,
        "inventory": building_inventory,
        "total_built_area_sq_m": round(total_changed_footprint_sq_m, 1),
        "total_built_area_ha": round(total_changed_footprint_sq_m / 10000.0, 2),
        "building_density_pct": density_pct,
        "gsd_m": gsd_m
    }
