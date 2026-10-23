"""
photo_streamer.py
=================
High-Resolution Multi-Temporal Photographic Satellite Streamer for Space Netra.
Fetches authentic optical aerial/satellite photography across Chhatrapati Sambhajinagar
for a 3-stage temporal evolution timeline (2016 -> 2020 -> Current 2025/2026).

Data Integrity & Provenance Guarantee:
- Real satellite observations are cached exclusively in data/change_detection/source/.
- Synthetic fallback generation is strictly quarantined to data/synthetic_test_fixtures/
  and will NEVER silently overwrite or masquerade as genuine satellite observations.
- If live acquisition fails and no genuine cached imagery is present, an explicit
  acquisition error status is returned unless allow_synthetic_demo=True is requested.
"""

import os
import math
import hashlib
import requests
import numpy as np
from PIL import Image, ImageDraw

CACHE_DIR = os.path.join("data", "change_detection", "source")
SYNTHETIC_FIXTURES_DIR = os.path.join("data", "synthetic_test_fixtures")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SYNTHETIC_FIXTURES_DIR, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# Reliable Esri Wayback snapshot M-IDs for Chhatrapati Sambhajinagar
WAYBACK_SNAPSHOTS = {
    "2016": {"m_id": "3515", "release_id": "WB_2016_R02", "date": "January 14, 2016"},
    "2020": {"m_id": "23001", "release_id": "WB_2020_R01", "date": "January 02, 2020"},
    "Current": {"date": "Current (2025/2026 High-Res Optical)"}
}

def deg2num(lat_deg, lon_deg, zoom=16):
    """Converts WGS84 lat/lon to Web Mercator tile x, y."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y

def calculate_ground_resolution(lat_deg, zoom=16):
    """
    Computes exact ground sampling distance (meters/pixel) for Web Mercator:
    Resolution = (156543.03392 * cos(latitude)) / (2 ^ zoom)
    """
    lat_rad = math.radians(lat_deg)
    meters_per_pixel = (156543.03392 * math.cos(lat_rad)) / (2.0 ** zoom)
    return round(meters_per_pixel, 2)

def is_procedural_synthetic(image_input):
    """
    Inspects an image path or PIL image to determine whether it is a procedural
    synthetic fallback drawing rather than authentic satellite photography.
    
    Detection signature:
    - Located in synthetic_test_fixtures or filename starting with synth_demo_
    - Flat asphalt road corridor drawn at y=118..124 with road_std < 10.0 and road_mean ~ 60
    - Exact horizontal road stripe at y=127 with stripe_mean > 180
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            return False
        basename = os.path.basename(image_input)
        if basename.startswith("synth_demo_") or "synthetic_test_fixtures" in os.path.abspath(image_input):
            return True
        try:
            img = Image.open(image_input)
        except Exception:
            return False
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        return False

    try:
        arr = np.array(img)
        if arr.shape != (256, 256, 3):
            return False
        
        band = arr[118:125, :]
        road_mean = band.mean(axis=(0, 1))
        road_std = band.std()
        stripe = arr[127, :]
        stripe_mean = stripe.mean(axis=0)

        # Procedural fallback signature check
        if road_std < 10.0 and abs(road_mean[0] - 60) < 15 and stripe_mean[0] > 180:
            return True
    except Exception:
        pass
    return False

def is_real_tile(image_path):
    """Returns True if the image exists and is verified non-synthetic satellite photography."""
    if not os.path.exists(image_path):
        return False
    return not is_procedural_synthetic(image_path)

def fetch_single_tile(url, file_path):
    """Fetches a single tile image over HTTP with error and integrity verification."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        if r.status_code == 200 and len(r.content) > 1000:
            with open(file_path, "wb") as f:
                f.write(r.content)
            # Verify file integrity after write
            if is_real_tile(file_path):
                return True
            else:
                # If downloaded content somehow matched procedural pattern, remove it
                if os.path.exists(file_path):
                    os.remove(file_path)
    except Exception:
        pass
    return False

def generate_photorealistic_fallback(lat, lon, zoom, epoch, file_path=None):
    """
    Generates a procedural demonstration raster for automated tests and offline simulation.
    Quarantined exclusively to data/synthetic_test_fixtures/.
    MUST NOT be saved to the live data/change_detection/source/ directory.
    """
    loc_hash = hashlib.md5(f"{lat:.4f}_{lon:.4f}_{zoom}".encode()).hexdigest()[:8]
    if file_path is None:
        file_path = os.path.join(SYNTHETIC_FIXTURES_DIR, f"synth_demo_{loc_hash}_{epoch}.jpg")

    # Safety check: Never allow saving into CACHE_DIR with csn_ prefix
    norm_path = os.path.abspath(file_path)
    if os.path.abspath(CACHE_DIR) in norm_path and "synth_demo_" not in os.path.basename(norm_path):
        raise ValueError(
            f"Provenance Violation: Attempted to save synthetic fallback into live source cache: {file_path}. "
            f"Synthetic data must be saved to {SYNTHETIC_FIXTURES_DIR} with 'synth_demo_' prefix."
        )

    seed = int(abs(lat * 1000 + lon * 1000) * 100) % 100000
    import random
    rng = random.Random(seed)

    W, H = 256, 256
    img = Image.new("RGB", (W, H), color=(145, 135, 120)) # Sandy soil base
    draw = ImageDraw.Draw(img)

    # Base ground texture (roads and natural vegetation)
    for _ in range(300):
        gx = rng.randint(0, W)
        gy = rng.randint(0, H)
        gw = rng.randint(10, 40)
        gh = rng.randint(10, 40)
        col = (rng.randint(90, 130), rng.randint(120, 160), rng.randint(70, 100)) # Greenery
        draw.rectangle([gx, gy, gx+gw, gy+gh], fill=col)

    # Main asphalt road corridor
    draw.rectangle([0, 115, W, 140], fill=(60, 60, 65))
    draw.line([0, 127, W, 127], fill=(220, 220, 220), width=1) # Road stripe

    # Cross street
    draw.rectangle([110, 0, 130, H], fill=(65, 65, 70))

    # Building Evolution across Epochs
    buildings_2016 = [
        (25, 30, 70, 75, (170, 165, 155)),
        (150, 40, 210, 85, (185, 180, 170)),
    ]
    buildings_2020 = buildings_2016 + [
        (30, 160, 85, 220, (195, 190, 180)),
        (160, 165, 225, 215, (210, 150, 130)),
    ]
    buildings_current = buildings_2020 + [
        (85, 25, 135, 70, (230, 225, 215)),
        (215, 35, 250, 80, (225, 220, 210)),
        (140, 150, 185, 230, (240, 235, 220)),
        (35, 80, 75, 110, (215, 210, 200)),
    ]

    selected_buildings = buildings_2016 if epoch == "2016" else (buildings_2020 if epoch == "2020" else buildings_current)

    for (bx1, by1, bx2, by2, bcolor) in selected_buildings:
        draw.rectangle([bx1+3, by1+3, bx2+3, by2+3], fill=(40, 40, 45)) # Shadow
        draw.rectangle([bx1, by1, bx2, by2], fill=bcolor, outline=(90, 85, 80)) # Roof
        draw.rectangle([bx1+8, by1+8, bx1+18, by1+18], fill=(120, 115, 110))

    img.save(file_path, "JPEG", quality=92)
    return file_path

def fetch_real_tile_with_pyramid_fallback(lat, lon, zoom, epoch, target_file, m_id=None):
    """
    Fetches authentic satellite photography for lat, lon, zoom.
    If the requested zoom level is not natively available on the tile server
    (e.g. Esri Wayback 2016 at zoom 18), falls back to the native zoom level (e.g. zoom 16),
    crops the exact geographical sub-quadrant, and resamples to 256x256 using Lanczos.
    Guarantees that the returned imagery is 100% authentic satellite photography.
    """
    x, y = deg2num(lat, lon, zoom)

    # 1. Direct fetch URL
    if epoch == "Current":
        url = f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={zoom}"
    else:
        url = f"https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/GoogleMapsCompatible/MapServer/tile/{m_id}/{zoom}/{y}/{x}"

    success = fetch_single_tile(url, target_file)
    if success and is_real_tile(target_file):
        return True

    # 2. Pyramid Fallback for High Zoom Levels (e.g., zoom 18 or 17 -> parent zoom 16 or 15)
    if zoom > 15:
        for p_zoom in [16, 15]:
            if p_zoom >= zoom:
                continue
            px, py = deg2num(lat, lon, p_zoom)
            p_hash = hashlib.md5(f"{lat:.4f}_{lon:.4f}_{p_zoom}".encode()).hexdigest()[:8]
            parent_cache_file = os.path.join(CACHE_DIR, f"csn_{p_hash}_{epoch}.jpg")

            # Fetch parent if not cached
            if not (os.path.exists(parent_cache_file) and is_real_tile(parent_cache_file)):
                if epoch == "Current":
                    p_url = f"https://mt1.google.com/vt/lyrs=s&x={px}&y={py}&z={p_zoom}"
                else:
                    p_url = f"https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/GoogleMapsCompatible/MapServer/tile/{m_id}/{p_zoom}/{py}/{px}"
                p_ok = fetch_single_tile(p_url, parent_cache_file)
                if not p_ok or not is_real_tile(parent_cache_file):
                    continue

            # Crop sub-quadrant from genuine parent satellite tile
            try:
                im_parent = Image.open(parent_cache_file)
                s = 2 ** (zoom - p_zoom)
                sub_w = 256 // s
                dx = (x % s) * sub_w
                dy = (y % s) * sub_w
                cropped = im_parent.crop((dx, dy, dx + sub_w, dy + sub_w)).resize((256, 256), Image.Resampling.LANCZOS)
                cropped.save(target_file, "JPEG", quality=92)
                if is_real_tile(target_file):
                    return True
            except Exception:
                continue

    return False

def fetch_3stage_photographic_timeline(lat, lon, zoom=16, allow_synthetic_demo=False):
    """
    Retrieves the 3-stage temporal photography for Chhatrapati Sambhajinagar:
      - Epoch 1: 2016 Historical (Esri Wayback Archive)
      - Epoch 2: 2020 Mid-Stage (Esri Wayback Archive)
      - Epoch 3: Current 2025/2026 (High-Res Optical Satellite)

    Provenance Guarantees:
    - Always prioritizes authentic satellite photography across all epochs.
    - If a tile server lacks native high-zoom tiles, smoothly downsamples/crops from
      verified parent real satellite tiles so human eyes see real terrain.
    - If live acquisition fails and allow_synthetic_demo is False (default):
      returns status='ACQUISITION_FAILED', is_real=False, with explicit error details.
      Does NOT generate or return artificial scenes masquerading as satellite imagery.
    - If allow_synthetic_demo is True and real satellite acquisition is impossible,
      generates procedural test fixtures strictly in data/synthetic_test_fixtures/.
    """
    x, y = deg2num(lat, lon, zoom)
    gsd_m = calculate_ground_resolution(lat, zoom)
    loc_hash = hashlib.md5(f"{lat:.4f}_{lon:.4f}_{zoom}".encode()).hexdigest()[:8]

    p1_file = os.path.join(CACHE_DIR, f"csn_{loc_hash}_2016.jpg")
    p2_file = os.path.join(CACHE_DIR, f"csn_{loc_hash}_2020.jpg")
    p3_file = os.path.join(CACHE_DIR, f"csn_{loc_hash}_current.jpg")

    epoch_configs = [
        ("p1", "2016", p1_file, WAYBACK_SNAPSHOTS["2016"]["m_id"]),
        ("p2", "2020", p2_file, WAYBACK_SNAPSHOTS["2020"]["m_id"]),
        ("p3", "Current", p3_file, None)
    ]

    resolved_paths = {}
    epoch_real_flags = {}
    missing_epochs = []

    for key, epoch, target_file, m_id in epoch_configs:
        # Check if already cached and verified genuine
        if is_real_tile(target_file):
            resolved_paths[key] = target_file
            epoch_real_flags[key] = True
            continue

        # Attempt remote acquisition with pyramid fallback for real satellite imagery
        success = fetch_real_tile_with_pyramid_fallback(lat, lon, zoom, epoch, target_file, m_id)
        if success and is_real_tile(target_file):
            resolved_paths[key] = target_file
            epoch_real_flags[key] = True
        else:
            # Real satellite acquisition failed
            if allow_synthetic_demo:
                synth_path = os.path.join(SYNTHETIC_FIXTURES_DIR, f"synth_demo_{loc_hash}_{epoch}.jpg")
                generate_photorealistic_fallback(lat, lon, zoom, epoch, synth_path)
                resolved_paths[key] = synth_path
                epoch_real_flags[key] = False
            else:
                resolved_paths[key] = None
                epoch_real_flags[key] = False
                missing_epochs.append(epoch)

    # Determine overall status and provenance
    all_real = len(epoch_real_flags) == 3 and all(epoch_real_flags.values())
    any_synthetic = any(not flag for flag in epoch_real_flags.values() if flag is not None)

    if all_real:
        status = "VERIFIED_REAL"
        provenance = "VERIFIED_REAL"
        provenance_label = "🟢 Verified Optical Satellite Imagery"
        acq_error = None
    elif allow_synthetic_demo:
        status = "SYNTHETIC_FALLBACK"
        provenance = "SYNTHETIC"
        provenance_label = "⚠️ Synthetic Demonstration Fixture (Simulated Fallback)"
        acq_error = "Real satellite acquisition was unavailable; procedural demonstration fixtures were generated by explicit user request."
    else:
        status = "ACQUISITION_FAILED"
        provenance = "UNAVAILABLE"
        provenance_label = "❌ Satellite Imagery Unavailable"
        acq_error = f"Real optical satellite imagery could not be retrieved from provider for epochs: {', '.join(missing_epochs)} at Lat: {lat:.4f}, Lon: {lon:.4f}. Synthetic fallbacks are disabled in live production mode."

    return {
        "status": status,
        "is_real": all_real,
        "provenance": provenance,
        "provenance_label": provenance_label,
        "acquisition_error": acq_error,
        "p1_path": resolved_paths.get("p1"),
        "p2_path": resolved_paths.get("p2"),
        "p3_path": resolved_paths.get("p3"),
        "gsd_m": gsd_m,
        "zoom": zoom,
        "x": x,
        "y": y,
        "loc_hash": loc_hash,
        "dates": {
            "epoch_1": WAYBACK_SNAPSHOTS["2016"]["date"],
            "epoch_2": WAYBACK_SNAPSHOTS["2020"]["date"],
            "epoch_3": WAYBACK_SNAPSHOTS["Current"]["date"]
        }
    }
