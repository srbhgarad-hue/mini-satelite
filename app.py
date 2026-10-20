import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from PIL import Image
import plotly.express as px
import torch
import clip
import faiss

# Import Space Netra City-Wide Services
import city_catalog
import photo_streamer
import building_detector

# ─────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Space Netra — City-Wide Photographic Building Intelligence",
    page_icon="🏙️"
)

st.title("🏙️ Space Netra: City-Wide Building Change Intelligence")
st.caption("Coverage: Full Municipal Region of Chhatrapati Sambhajinagar (Aurangabad), Maharashtra | Real Optical Satellite & Aerial Photography")

# ─────────────────────────────────────────────────────────────
# SIDEBAR CONTROLS
# ─────────────────────────────────────────────────────────────
st.sidebar.header("🕹️ Control Panel")
app_mode = st.sidebar.selectbox("Select Action Mode", [
    "🏙️ Whole-City Building Change Detection",
    "🔍 Vector-Based Semantic Search"
])

# ─────────────────────────────────────────────────────────────
# MODE 1: WHOLE-CITY PHOTOGRAPHIC BUILDING CHANGE DETECTION
# ─────────────────────────────────────────────────────────────
if app_mode == "🏙️ Whole-City Building Change Detection":
    st.sidebar.subheader("📍 City Navigator")
    
    # City Sector Hotspot Selector
    hotspot_names = list(city_catalog.CITY_HOTSPOTS.keys())
    selected_sector_name = st.sidebar.selectbox(
        "Choose City Growth Sector:",
        hotspot_names,
        index=0,
        help="Select any major urban expansion hub across Chhatrapati Sambhajinagar."
    )
    sector_meta = city_catalog.get_hotspot_info(selected_sector_name)

    # Custom address / landmark search
    custom_search = st.sidebar.text_input(
        "Or Search Any Custom Address in City:",
        value="",
        placeholder="e.g. Prozone Mall, Chikalthana, Waluj..."
    )

    current_lat = sector_meta["lat"]
    current_lon = sector_meta["lon"]
    zoom_level = sector_meta.get("zoom", 16)
    sector_display_name = sector_meta["name"]
    sector_category = sector_meta["category"]
    sector_desc = sector_meta["description"]

    if custom_search.strip():
        try:
            geolocator = Nominatim(user_agent="space_netra_city_navigator")
            full_query = f"{custom_search}, Chhatrapati Sambhajinagar, Maharashtra"
            loc = geolocator.geocode(full_query, timeout=8)
            if loc:
                current_lat, current_lon = loc.latitude, loc.longitude
                sector_display_name = custom_search
                sector_category = "Custom Searched Point"
                sector_desc = f"Coordinates: {current_lat:.4f}° N, {current_lon:.4f}° E"
                st.sidebar.success(f"📍 Found: Lat {current_lat:.4f}, Lon {current_lon:.4f}")
        except Exception:
            pass

    st.sidebar.subheader("🎛️ Building Detection Calibration")
    zoom_val = st.sidebar.slider("Aerial Zoom Level", 15, 18, zoom_level, help="Zoom 16: ~2.2m GSD | Zoom 17: ~1.1m GSD | Zoom 18: ~0.55m Sub-Meter GSD (Individual roofs and windows visible).")
    min_bldg_area = st.sidebar.slider("Min Building Footprint (m²)", 20, 250, 50, help="Filters out vehicles, trees, and road stripes. Standard house floor is ~50-100 m².")
    sensitivity = st.sidebar.slider("Structural Sensitivity", 30, 100, 55, help="Controls edge and structural difference threshold.")

    # ── City Location Context Banner ──
    col_info, col_map = st.columns([2, 3])
    
    with col_info:
        st.subheader(f"📍 {sector_display_name}")
        st.markdown(f"""
        - **Sector Category:** `{sector_category}`
        - **Description:** {sector_desc}
        - **Coordinates:** `{current_lat:.4f}° N, {current_lon:.4f}° E`
        - **Historical Baseline Epoch 1:** January 2016 (Esri Wayback Archive)
        - **Mid-Stage Epoch 2:** January 2020 (Esri Wayback Archive)
        - **Current Epoch 3:** 2025/2026 (High-Res Optical Satellite)
        """)

    with col_map:
        m = folium.Map(
            location=[current_lat, current_lon],
            zoom_start=zoom_val,
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google Satellite"
        )
        folium.Marker(
            [current_lat, current_lon],
            popup=f"Selected: {sector_display_name}",
            tooltip=f"{sector_display_name} (Click to inspect)"
        ).add_to(m)
        st_folium(m, width=650, height=220)

    # ── Fetch 3-Stage Photography & Run Building Detection ──
    with st.spinner(f"⏳ Ingesting high-resolution 3-stage photography for {sector_display_name}..."):
        timeline = photo_streamer.fetch_3stage_photographic_timeline(current_lat, current_lon, zoom=zoom_val)
        
        # Detect new buildings between Epoch 1 (2016) and Epoch 3 (Current)
        det_res = building_detector.detect_new_buildings(
            timeline["p1_path"],
            timeline["p3_path"],
            gsd_m=timeline["gsd_m"],
            min_area_sq_m=float(min_bldg_area),
            sensitivity=sensitivity
        )

    # ── Top Metrics Row ──
    st.write("---")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🏢 New Buildings Found", f"{det_res['building_count']} Structures")
    m2.metric("📐 New Built Footprint", f"{det_res['total_built_area_sq_m']:,.1f} m²")
    m3.metric("🌍 Built-Up Density", f"{det_res['building_density_pct']}% of ground")
    m4.metric("📸 Image Resolution", f"{det_res['gsd_m']} m/pixel", help="High-resolution optical ground sampling distance")
    m5.metric("🚦 Expansion Level", "🚨 High Growth" if det_res['building_count'] >= 3 else ("⚠️ Moderate Growth" if det_res['building_count'] > 0 else "✅ Stable Ground"))

    # ── Main View Selection ──
    tab_highlight, tab_timeline, tab_table, tab_report = st.tabs([
        "🏢 Photorealistic Building Detection & Outlines",
        "⏱️ 3-Stage Temporal Evolution Timeline (2016 → 2020 → 2026)",
        "📋 Detected Building Inventory & Dimensions",
        "📄 Official City Intelligence Report"
    ])

    with tab_highlight:
        st.subheader("High-Resolution Real Optical Satellite View with Labeled New Buildings")
        st.caption("Vector yellow polygons and labeled badges denote newly constructed building roofs and architectural structures.")

        view_choice = st.radio(
            "Select Display Mode:",
            [
                "🏢 Labeled Vector Outlines on Real Current Photo",
                "🔴 Thermal Construction Heatmap Overlay",
                "⚪ Clean Binary Building Footprint Mask"
            ],
            horizontal=True
        )

        c_left, c_right = st.columns(2)
        c_left.image(
            timeline["p1_path"],
            caption=f"📅 Epoch 1: {timeline['dates']['epoch_1']} (Historical Baseline Photo)",
            use_container_width=True
        )

        if view_choice == "🏢 Labeled Vector Outlines on Real Current Photo":
            c_right.image(
                det_res['annotated_img'],
                caption=f"📅 Epoch 3: {timeline['dates']['epoch_3']} (Real Photo with Detected New Buildings)",
                use_container_width=True
            )
        elif view_choice == "🔴 Thermal Construction Heatmap Overlay":
            c_right.image(
                det_res['heatmap_img'],
                caption="🔴 Thermal Construction Intensity (Red/Yellow = Major Structural Additions)",
                use_container_width=True
            )
        else:
            c_right.image(
                det_res['binary_mask'],
                caption="⚪ Binary Extracted Building Footprints",
                use_container_width=True
            )

    with tab_timeline:
        st.subheader(f"⏱️ 3-Stage Multi-Year Building Evolution: {sector_display_name}")
        st.caption("Observe the progressive emergence of new buildings, road networks, and compound walls across a 10-year timeline.")

        tc1, tc2, tc3 = st.columns(3)
        tc1.image(
            timeline["p1_path"],
            caption=f"1️⃣ Epoch 1: {timeline['dates']['epoch_1']}\n(Original Open Ground & Sparse Structures)",
            use_container_width=True
        )
        tc2.image(
            timeline["p2_path"],
            caption=f"2️⃣ Epoch 2: {timeline['dates']['epoch_2']}\n(Mid-Stage Construction & Initial Infill)",
            use_container_width=True
        )
        tc3.image(
            det_res['annotated_img'],
            caption=f"3️⃣ Epoch 3: {timeline['dates']['epoch_3']}\n(Current State with New Building Badges)",
            use_container_width=True
        )

    with tab_table:
        st.subheader("📋 Itemized Inventory of Detected New Buildings")
        if det_res["inventory"]:
            df = pd.DataFrame(det_res["inventory"])
            st.dataframe(df, use_container_width=True)

            # Chart of Building Footprint Sizes
            fig_bar = px.bar(
                df,
                x="Building ID",
                y="Footprint Area (m²)",
                title="Individual Building Footprint Areas (Square Meters)",
                color="Footprint Area (m²)",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("✅ No new building structures detected in this sector meeting the minimum area threshold.")

    with tab_report:
        st.subheader("📄 Chhatrapati Sambhajinagar City Building Intelligence Report")
        report_text = f"""================================================================================
SPACE NETRA — MUNICIPAL BUILDING CHANGE INTELLIGENCE REPORT
City: Chhatrapati Sambhajinagar (Aurangabad), Maharashtra, India
================================================================================
SECTOR & SPATIAL PARAMETERS:
- Sector Name                : {sector_display_name}
- Category                   : {sector_category}
- Coordinates                : Lat {current_lat:.4f}° N, Lon {current_lon:.4f}° E
- Ground Sampling Distance   : {det_res['gsd_m']} meters / pixel (High-Res Optical Aerial Photography)
- Web Mercator Tile Index    : X={timeline['x']}, Y={timeline['y']} (Zoom Level: {timeline['zoom']})
--------------------------------------------------------------------------------
TEMPORAL EVOLUTION TIMELINE:
- Epoch 1 (Historical)       : {timeline['dates']['epoch_1']} (Esri World Imagery Wayback)
- Epoch 2 (Mid-Stage)        : {timeline['dates']['epoch_2']} (Esri World Imagery Wayback)
- Epoch 3 (Current)          : {timeline['dates']['epoch_3']} (High-Resolution Optical Satellite)
--------------------------------------------------------------------------------
BUILDING DETECTION AUDIT RESULTS:
- Detected New Buildings     : {det_res['building_count']} distinct structures
- Total New Built-Up Area    : {det_res['total_built_area_sq_m']:,} m² ({det_res['total_built_area_ha']} Hectares)
- Built-Up Expansion Density : {det_res['building_density_pct']}% of analyzed visual frame
- Minimum Footprint Filter   : {min_bldg_area} m² (Geometric Aspect-Ratio Filter Active)
- Operational Verdict        : {'ACTION REQUIRED: Significant New Building Construction' if det_res['building_count'] > 0 else 'STABLE: No New Construction Detected'}
================================================================================"""
        st.code(report_text, language="text")
        st.download_button(
            "📥 Download Municipal Building Report (.txt)",
            report_text,
            file_name=f"space_netra_{sector_display_name.lower().replace(' ', '_')}_building_report.txt"
        )

# ─────────────────────────────────────────────────────────────
# MODE 2: VECTOR-BASED SEMANTIC SEARCH (PRESERVED CLIP + FAISS)
# ─────────────────────────────────────────────────────────────
elif app_mode == "🔍 Vector-Based Semantic Search":
    st.subheader("🔍 Spatial Archive Semantic Search (CLIP + FAISS)")
    st.caption("Query the local high-resolution spatial image archive using free-form natural language.")

    @st.cache_resource
    def load_clip_faiss():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        index_path = os.path.join("vector_db", "spatial.index")
        meta_path = os.path.join("vector_db", "metadata.npy")
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            return None, None, None, None, device
        index = faiss.read_index(index_path)
        metadata = np.load(meta_path, allow_pickle=True)
        return model, preprocess, index, metadata, device

    try:
        model, preprocess, index, metadata, device = load_clip_faiss()
        if index is None or metadata is None:
            st.warning("⚠️ Vector database not found. Run 'python ingest.py' to generate embeddings.")
        else:
            query_str = st.text_input(
                "Enter a natural language description to search satellite archive:",
                "urban area"
            )
            col_btn, _ = st.columns([1, 4])
            with col_btn:
                search_clicked = st.button("🔍 Search Spatial Archive")

            if search_clicked and query_str:
                with st.spinner("Embedding query & searching FAISS vector space..."):
                    tokens = clip.tokenize([query_str]).to(device)
                    with torch.no_grad():
                        text_feat = model.encode_text(tokens)
                        text_feat /= text_feat.norm(dim=-1, keepdim=True)
                    k = min(len(metadata), 3)
                    distances, indices = index.search(text_feat.cpu().numpy().astype("float32"), k=k)

                st.write("---")
                st.subheader(f"Top {k} Nearest Spatial Image Matches for: '{query_str}'")
                cols = st.columns(k)
                for i, idx in enumerate(indices[0]):
                    if idx < len(metadata):
                        item = metadata[idx]
                        if os.path.exists(item["path"]):
                            img = Image.open(item["path"])
                            score = round(float(distances[0][i]) * 100.0, 2)
                            with cols[i]:
                                st.image(img, caption=f"📌 {item['filename']} | Match: {score}%", use_container_width=True)
                        else:
                            with cols[i]:
                                st.warning(f"File missing: {item['filename']}")
    except Exception as e:
        st.error(f"Search Execution Error: {e}")
