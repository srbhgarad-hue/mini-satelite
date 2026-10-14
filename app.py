import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import plotly.express as px
import numpy as np
import clip
import torch
import faiss
import os
import math
import re
import requests
from PIL import Image
from change_detector import calculate_change_confidence

# Ensure temp directory exists
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

st.set_page_config(layout="wide", page_title="Space Netra - Live Satellite Intelligence Engine")

st.title("🛰️ Space Netra: Live Satellite Imagery Change Detection")
st.caption("Focused Area: Chhatrapati Sambhajinagar, Maharashtra | Powered by Esri Wayback Archive + Google Satellite")

# ─────────────────────────────────────────────
# MODELS & RELEASE CATALOG
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    index_path = os.path.join("vector_db", "spatial.index")
    meta_path = os.path.join("vector_db", "metadata.npy")
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        return None, None, None, None, device
    index = faiss.read_index(index_path)
    metadata = np.load(meta_path, allow_pickle=True)
    return model, preprocess, index, metadata, device

@st.cache_data(ttl=86400)
def get_wayback_releases():
    """
    Fetch all Esri World Imagery Wayback snapshot releases (free, no key).
    Returns a dict: {year -> [(release_name, tile_M_id, date_str), ...]}
    Sorted newest-first within each year.
    """
    url = "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer?f=json"
    try:
        r = requests.get(url, timeout=10, headers=HEADERS)
        data = r.json()
        releases = data.get("Selection", [])
        year_map = {}
        for rel in releases:
            name = rel.get("Name", "")
            mid = rel.get("M", "")
            rid = rel.get("ID", "")
            m = re.search(r"Wayback (\d{4})-(\d{2}-\d{2})", name)
            if m:
                yr = int(m.group(1))
                date_str = f"{m.group(1)}-{m.group(2)}"
                if yr not in year_map:
                    year_map[yr] = []
                year_map[yr].append((rid, mid, date_str))
        return year_map
    except Exception as e:
        # Hardcoded fallback: one reliable release ID per year (earliest in year)
        return {
            2014: [("WB_2014_R01", "10", "2014-02-20")],
            2015: [("WB_2015_R01", "20222", "2015-01-01")],
            2016: [("WB_2016_R02", "3515", "2016-01-14")],
            2017: [("WB_2017_R01", "577", "2017-01-05")],
            2018: [("WB_2018_R01", "13161", "2018-01-04")],
            2019: [("WB_2019_R00", "6036", "2019-01-03")],
            2020: [("WB_2020_R01", "23001", "2020-01-02")],
            2021: [("WB_2021_R01", "1049", "2021-01-07")],
            2022: [("WB_2022_R00", "42663", "2022-01-06")],
            2023: [("WB_2023_R00", "11475", "2023-01-05")],
            2024: [("WB_2024_R00", "41468", "2024-01-04")],
            2025: [("WB_2025_R01", "36557", "2025-01-02")],
        }

def get_tile_id_for_year(year_map, target_year):
    """Return the earliest (first available) snapshot tile M-ID and date for the target year."""
    entries = year_map.get(target_year)
    if entries:
        # entries are newest-first from the API, so pick the last = earliest in that year
        earliest = entries[-1]
        return earliest[1], earliest[2]
    # Fallback: find nearest year
    available = sorted(year_map.keys())
    nearest = min(available, key=lambda y: abs(y - target_year))
    earliest = year_map[nearest][-1]
    return earliest[1], f"~{nearest} (nearest available)"

# ─────────────────────────────────────────────
# TILE UTILITIES
# ─────────────────────────────────────────────
def deg2num(lat_deg, lon_deg, zoom=16):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y

def fetch_temporal_tiles(lat, lon, tile_m_id, zoom=16):
    """
    T1 = Esri Wayback Archive tile (historical year chosen by user)
    T2 = Google Satellite (current ~2025/2026)
    """
    x, y = deg2num(lat, lon, zoom)

    # T1 — Esri Wayback historical tile
    t1_url = (
        f"https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/"
        f"WMTS/1.0.0/GoogleMapsCompatible/MapServer/tile/{tile_m_id}/{zoom}/{y}/{x}"
    )
    # T2 — Google Satellite (current)
    t2_url = f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={zoom}"

    t1_file = os.path.join(TEMP_DIR, "temp_t1_historical.jpg")
    t2_file = os.path.join(TEMP_DIR, "temp_t2_current.jpg")

    try:
        r1 = requests.get(t1_url, headers=HEADERS, timeout=12)
        r2 = requests.get(t2_url, headers=HEADERS, timeout=12)

        if r1.status_code == 200 and len(r1.content) > 500:
            with open(t1_file, "wb") as f:
                f.write(r1.content)
        else:
            return None, None, f"T1 fetch failed (status {r1.status_code})"

        if r2.status_code == 200 and len(r2.content) > 500:
            with open(t2_file, "wb") as f:
                f.write(r2.content)
        else:
            return None, None, f"T2 fetch failed (status {r2.status_code})"

        return t1_file, t2_file, None

    except Exception as e:
        return None, None, str(e)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.header("🕹️ Control Panel")
mode = st.sidebar.selectbox("Select Action Mode", [
    "🗺️ Interactive Map AOI Selection",
    "🔍 Vector-Based Semantic Search"
])

# ─────────────────────────────────────────────
# MODE 1: MAP + CHANGE DETECTION
# ─────────────────────────────────────────────
if mode == "🗺️ Interactive Map AOI Selection":

    st.sidebar.subheader("📅 Temporal Configuration")
    year_releases = get_wayback_releases()
    available_years = sorted(year_releases.keys())
    min_yr, max_yr = min(available_years), max(available_years) - 1  # T2 is current so T1 max is prev year
    year_t1 = st.sidebar.slider("Historical Year (T1)", min_yr, max_yr, 2016)
    tile_m_id, snapshot_date = get_tile_id_for_year(year_releases, year_t1)
    st.sidebar.info(f"📸 Using snapshot: **{snapshot_date}**")

    st.sidebar.subheader("🎛️ Analysis Adjustments")
    spatial_res = st.sidebar.slider("Sensor Resolution (m/px)", 1.0, 30.0, 10.0)
    sensitivity = st.sidebar.slider("Detection Sensitivity", 30, 150, 60)
    overlay_alpha = st.sidebar.slider("Heatmap Transparency", 0.1, 1.0, 0.5)
    zoom_level = st.sidebar.slider("Tile Zoom Level", 14, 18, 16)

    # Default to Chh. Sambhajinagar city centre
    DEFAULT_LAT, DEFAULT_LON = 19.8762, 75.3433
    search_location = st.text_input(
        "📍 Search Location",
        "Chhatrapati Sambhajinagar, Maharashtra"
    )

    current_lat, current_lon = DEFAULT_LAT, DEFAULT_LON

    if search_location:
        try:
            geolocator = Nominatim(user_agent="space_netra_csn_app")
            location = geolocator.geocode(search_location, timeout=10)
            if location:
                current_lat, current_lon = location.latitude, location.longitude
                st.success(f"📍 Located: Lat {current_lat:.4f}, Lon {current_lon:.4f}")
        except Exception:
            st.info("Using default coordinates for Chh. Sambhajinagar.")

    st.subheader("🗺️ Area of Interest — Satellite View")
    m = folium.Map(
        location=[current_lat, current_lon],
        zoom_start=zoom_level,
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite"
    )
    folium.Marker(
        [current_lat, current_lon],
        popup=f"AOI: {search_location}",
        tooltip="Selected Site"
    ).add_to(m)
    st_folium(m, width=1200, height=320)

    st.write("---")
    years_apart = 2026 - year_t1
    st.subheader(f"🛰️ Multi-Temporal Change Detection | {year_t1} → 2026 ({years_apart} years)")

    with st.spinner(f"⏳ Fetching {year_t1} archive tile (Esri Wayback) & 2026 tile (Google)..."):
        t1_path, t2_path, error = fetch_temporal_tiles(current_lat, current_lon, tile_m_id, zoom=zoom_level)

    if error:
        st.error(f"⚠️ Satellite Tile Fetch Error: {error}")
        st.info("Try a different zoom level or location. Some historical tiles may not have coverage for this exact coordinate.")
    elif t1_path and t2_path:
        res = calculate_change_confidence(
            t1_path, t2_path,
            resolution_m=spatial_res,
            sensitivity=sensitivity,
            overlay_alpha=overlay_alpha
        )

        # ── Metrics Row ──
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("🔬 Change Score", f"{res['confidence']}%")
        m2.metric("🌍 Changed Area", f"{res['change_ratio']}%")
        m3.metric("📐 Area Impacted", f"{res['area_hectares']} Ha")
        m4.metric("🔲 Clusters Found", f"{res['bbox_count']} Zones")
        m5.metric("📅 Years Apart", f"{years_apart} yrs")
        m6.metric("🚦 Status", "⚠️ Change" if res['is_real_change'] else "✅ Stable")

        tab1, tab2, tab3 = st.tabs([
            "🖼️ Visual Comparison & Highlights",
            "📊 Area Analytics",
            "📄 Executive Report"
        ])

        with tab1:
            view_mode = st.radio(
                "Select Output View:",
                ["Bounding Box Clusters", "Colored Heatmap Overlay", "Binary Change Mask"],
                horizontal=True
            )
            c1, c2, c3 = st.columns(3)
            c1.image(t1_path, caption=f"📅 T1: {snapshot_date} (Esri Wayback Archive)", width='stretch')
            c2.image(t2_path, caption="📅 T2: 2026 (Google Satellite — Current)", width='stretch')

            if view_mode == "Bounding Box Clusters":
                c3.image(res['bbox_img'], caption="🟨 Detected Change Clusters (Neon Bboxes)", width='stretch')
            elif view_mode == "Colored Heatmap Overlay":
                c3.image(res['overlay'], caption="🔴 Thermal Heatmap (Red/Yellow = High Change)", width='stretch')
            else:
                c3.image(res['thresh_mask'], caption="⚪ Binary Difference Mask", width='stretch')

        with tab2:
            cg1, cg2 = st.columns(2)
            with cg1:
                st.subheader("Land Surface Distribution")
                fig_pie = px.pie(
                    values=[res['change_ratio'], res['unchanged_ratio']],
                    names=[f'Changed ({year_t1}→2026)', 'Unchanged'],
                    color_discrete_sequence=['#E63946', '#2A9D8F'],
                    hole=0.4
                )
                st.plotly_chart(fig_pie, width='stretch')

            with cg2:
                st.subheader("Area Metric Breakdown")
                total_est = res['area_sq_km'] / (res['change_ratio'] / 100.0) if res['change_ratio'] > 0 else res['area_sq_km']
                fig_bar = px.bar(
                    x=["Total Zone (Sq Km)", "Changed (Sq Km)", "Impacted (Hectares)"],
                    y=[total_est, res['area_sq_km'], res['area_hectares']],
                    color=["Total", "Changed", "Hectares"],
                    color_discrete_map={"Total": "#457B9D", "Changed": "#E63946", "Hectares": "#F4A261"}
                )
                st.plotly_chart(fig_bar, width='stretch')

        with tab3:
            st.subheader("📋 Geospatial Intelligence Report")
            report_text = f"""==================================================
SPACE NETRA — GEOSPATIAL INTELLIGENCE REPORT
Chhatrapati Sambhajinagar, Maharashtra, India
==================================================
- Target Location : {search_location}
- Coordinates     : Lat {current_lat:.4f}, Lon {current_lon:.4f}
- T1 (Historical) : {snapshot_date} — Source: Esri World Imagery Wayback
- T2 (Current)    : 2026 — Source: Google Satellite
- Temporal Gap    : {years_apart} years
--------------------------------------------------
- Detection Status: {'⚠️  ACTION REQUIRED: Structural Change Detected' if res['is_real_change'] else '✅  NORMAL: No Major Change Detected'}
- Change Confidence     : {res['confidence']}%
- Changed Surface Ratio : {res['change_ratio']}%
- Impacted Area         : {res['area_hectares']} Hectares ({res['area_sq_km']} sq km)
- Hotspot Clusters      : {res['bbox_count']} distinct change zones
- Sensor Resolution     : {spatial_res} m/px
- Detection Sensitivity : {sensitivity}
=================================================="""
            st.code(report_text, language="text")
            st.download_button("📥 Download Report", report_text, file_name="space_netra_csn_report.txt")
    else:
        st.warning("⚠️ Could not retrieve satellite tiles. Please check internet connection or try a different zoom level.")

# ─────────────────────────────────────────────
# MODE 2: SEMANTIC SEARCH
# ─────────────────────────────────────────────
elif mode == "🔍 Vector-Based Semantic Search":
    st.subheader("🔍 Spatial Archive Semantic Search")
    try:
        model, preprocess, index, metadata, device = load_models()
        if index is None or metadata is None:
            st.warning("⚠️ Vector database not built yet. Run `python ingest.py` in terminal to index the image archive.")
        else:
            query_text = st.text_input(
                "Enter a natural language query to search satellite archive:",
                "urban area"
            )
            if st.button("🔍 Search Spatial Archive"):
                with st.spinner("Searching..."):
                    text_tokens = clip.tokenize([query_text]).to(device)
                    with torch.no_grad():
                        text_features = model.encode_text(text_tokens)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                    k_results = min(len(metadata), 3)
                    distances, indices = index.search(text_features.cpu().numpy().astype("float32"), k=k_results)

                cols = st.columns(k_results)
                for i, idx in enumerate(indices[0]):
                    if idx < len(metadata):
                        item = metadata[idx]
                        if os.path.exists(item["path"]):
                            img = Image.open(item["path"])
                            score = round(float(distances[0][i]) * 100, 2)
                            with cols[i]:
                                st.image(img, caption=f"📌 {item['filename']} | Score: {score}%", width='stretch')
                        else:
                            with cols[i]:
                                st.warning(f"File missing: {item['filename']}")
    except Exception as e:
        st.error(f"Search Error: {e}. Run 'python ingest.py' to rebuild the spatial index.")
