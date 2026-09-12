import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_change_confidence(img_t1_path, img_t2_path, resolution_m=10.0, sensitivity=60, overlay_alpha=0.6):
    img1 = cv2.imread(img_t1_path)
    img2 = cv2.imread(img_t2_path)
    
    if img1 is None or img2 is None:
        raise FileNotFoundError("Satellite patch images correctly read nahi ho paayi.")
        
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # SSIM Structural Change Calculation
    score, diff = ssim(gray1, gray2, full=True)
    diff_uint8 = (diff * 255).astype("uint8")
    diff_inv = 255 - diff_uint8
    
    # Noise Reduction & Morphological Dilation for Sharp Bounding Area
    _, thresh = cv2.threshold(diff_inv, sensitivity, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    thresh_dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    # 1. Vibrant Red Thermal Overlay
    heatmap = cv2.applyColorMap(diff_inv, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img2, 1 - overlay_alpha, heatmap, overlay_alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    # 2. High-Visibility Neon Bounding Boxes & Tags
    bbox_img = img2.copy()
    contours, _ = cv2.findContours(thresh_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bbox_count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 30:  # Minimum area threshold for valid structural change
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Thick Neon Yellow Bounding Box (BGR: 0, 255, 255)
            cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 255, 255), 3)
            
            # Red Highlight Fill over changed area inside box
            sub_rect = bbox_img[y:y+h, x:x+w]
            red_fill = np.full_like(sub_rect, (0, 0, 255))
            bbox_img[y:y+h, x:x+w] = cv2.addWeighted(sub_rect, 0.6, red_fill, 0.4, 0)
            
            # Label Tag Badge Background
            bbox_count += 1
            label = f"CHANGE #{bbox_count}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(bbox_img, (x, max(0, y - 20)), (x + text_w + 6, max(20, y)), (0, 0, 0), -1)
            cv2.putText(bbox_img, label, (x + 3, max(14, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            
    bbox_rgb = cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB)
    
    # Surface Area Metrics
    total_pixels = diff_uint8.size
    changed_pixels = np.count_nonzero(thresh)
    change_ratio = round((changed_pixels / total_pixels) * 100, 2)
    confidence_score = round((1 - score) * 100, 2)
    
    pixel_area_sq_m = resolution_m * resolution_m
    total_changed_area_sq_m = changed_pixels * pixel_area_sq_m
    area_hectares = round(total_changed_area_sq_m / 10000.0, 2)
    area_sq_km = round(total_changed_area_sq_m / 1000000.0, 3)

    return {
        "confidence": confidence_score,
        "is_real_change": confidence_score > 12.0,
        "overlay": overlay_rgb,
        "bbox_img": bbox_rgb,
        "thresh_mask": thresh_dilated,
        "change_ratio": change_ratio,
        "unchanged_ratio": round(100.0 - change_ratio, 2),
        "area_hectares": area_hectares,
        "area_sq_km": area_sq_km,
        "bbox_count": bbox_count
    }