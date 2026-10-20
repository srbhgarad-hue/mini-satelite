import os
import json
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
from PIL import Image
import plotly.express as px
import torch
import clip
import faiss

# Import Space Netra Modular Backend Services
import satellite_catalog
import satellite_loader
import preprocessor
import change_engine
import area_metrics

# ─────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Space Netra — Live Satellite Intelligence & Change Detection Engine",
    page_icon="🛰️"
)

st.title("🛰️ Space Netra: Satellite Intelligence & Change Detection")
st.caption("Pilot AOI: Chhatrapati Sambhajinagar (Aurangabad), Maharashtra | Sentinel-2 L2A (10m) + CLIP/FAISS")

# ─────────────────────────────────────────────────────────────
# SIDEBAR CONTROLS
# ─────────────────────────────────────────────────────────────
st.sidebar.header("🕹️ Control Panel")
app_mode = st.sidebar.selectbox("Select Action Mode", [
    "🛰️ Multi-Temporal Change Detection",
    "🔍 Vector-Based Semantic Search"
])

# ─────────────────────────────────────────────────────────────
# MODE 1: SATELLITE CHANGE DETECTION (SENTINEL-2 L2A PIPELINE)
# ─────────────────────────────────────────────────────────────
if app_mode == "🛰️ Multi-Temporal Change Detection":
    st.sidebar.subheader("📅 Temporal Observation Configuration")
    
    st.sidebar.info("🛰️ **Sensor:** Copernicus Sentinel-2A MSI (Level-2A BOA Reflectance)")
    st.sidebar.write("**T1 (Historical Baseline):** March 30, 2021 (Cloud: 0.05%)")
    st.sidebar.write("**T2 (Recent Observation):** March 14, 2024 (Cloud: 0.00%)")
    st.sidebar.caption("⏱️ **Temporal Gap:** 3.0 Years (Anniversary Pair)")

    st.sidebar.subheader("🎛️ Analysis Sensitivity")
    sensitivity = st.sidebar.slider("Detection Sensitivity (Threshold)", 20, 120, 50, help="Lower values detect subtle ground transitions; higher values isolate major clearings.")
    min_cluster = st.sidebar.slider("Min Cluster Size (Pixels)", 5, 50, 15, help="Filters out isolated single-pixel sensor noise.")
    overlay_alpha = st.sidebar.slider("Heatmap Transparency", 0.1, 0.9, 0.5)

    # AOI Header Card
    st.subheader("📍 Target Area of Interest: Shendra MIDC Expansion Corridor")
    
    # ── Map Container ──
    col_map, col_aoi_info = st.columns([3, 1])
    
    with col_aoi_info:
        st.markdown("""
        **AOI Specification:**
        - **Location:** Chh. Sambhajinagar, MH
        - **Sector:** Shendra MIDC Industrial Belt
        - **Coordinates:** 19.8700° N, 75.4900° E
        - **Footprint:** 2.0 km × 2.0 km
        - **Total Ground Area:** 400.0 Hectares
        - **Metric CRS:** UTM Zone 43N (EPSG:32643)
        - **Target Event:** Agricultural to Industrial Land Conversion
        """)

    with col_map:
        m = folium.Map(
            location=[19.8700, 75.4900],
            zoom_start=14,
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google Satellite"
        )
        # Bounding box polygon for Shendra MIDC
        folium.Rectangle(
            bounds=[[19.8600, 75.4800], [19.8800, 75.5000]],
            color="#2673D9",
            weight=3,
            fill=True,
            fill_color="#2673D9",
            fill_opacity=0.2,
            popup="Shendra MIDC 2x2 km Target AOI",
            tooltip="Pilot AOI: 400 Hectares"
        ).add_to(m)
        st_folium(m, width=950, height=280)

    # ── Sensor Resolution & Methodology Disclaimer ──
    st.info(
        "🛰️ **Scientific Sensor Disclaimer:** Analysis conducted on **Copernicus Sentinel-2 Level-2A 10m Surface Reflectance** "
        "via Microsoft Planetary Computer STAC. The engine detects macro-scale land-cover transformations, grading, and urban expansion. "
        "Individual residential building outlines are beyond the physical 10m optical resolution limit."
    )

    # ── Ingest & Execute Change Pipeline ──
    with st.spinner("⏳ Loading calibrated Sentinel-2 Level-2A multispectral bands & executing engine..."):
        try:
            pair_data = satellite_loader.load_temporal_pair()
            valid_mask = preprocessor.build_valid_pixel_mask(pair_data["t1_scl"], pair_data["t2_scl"])
            
            # Detect changes via combined Delta-NDVI and SSIM
            res = change_engine.detect_surface_changes(
                pair_data["t1_b04"], pair_data["t1_b08"],
                pair_data["t2_b04"], pair_data["t2_b08"],
                valid_mask,
                sensitivity=sensitivity,
                min_cluster_px=min_cluster,
                overlay_alpha=overlay_alpha
            )

            # Compute mathematically grounded metric area
            metrics = area_metrics.calculate_geospatial_metrics(
                res["change_mask"], valid_mask, gsd_meters=pair_data["gsd_m"]
            )
            success = True
        except Exception as e:
            st.error(f"⚠️ Error executing change detection pipeline: {e}")
            success = False

    if success:
        st.write("---")
        st.subheader("📊 Quantitative Surface Transition Metrics (UTM Zone 43N)")

        # Metrics Row
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("🔬 Divergence Index", f"{res['mean_divergence']}")
        m2.metric("🌍 Change Ratio", f"{metrics['change_ratio_pct']}%")
        m3.metric("📐 Impacted Area", f"{metrics['changed_area_ha']} Ha", help="100 m² per pixel in UTM Zone 43N")
        m4.metric("🔲 Hotspot Clusters", f"{res['cluster_count']} Zones")
        m5.metric("☁️ Valid Data", f"{metrics['valid_coverage_pct']}%", help="Excluded SCL clouds & shadows")
        m6.metric("🚦 Status", "⚠️ Significant Change" if metrics['change_ratio_pct'] > 5.0 else "✅ Stable Ground")

        # Tabs
        tab_visuals, tab_analytics, tab_report = st.tabs([
            "🖼️ Visual Comparison & Heatmaps",
            "📈 Spatial Analytics & Land Distribution",
            "📋 Geospatial Intelligence Report"
        ])

        with tab_visuals:
            view_mode = st.radio(
                "Select Analytical Visualization Layer:",
                [
                    "🟨 Bounding Box Change Clusters",
                    "🔴 Thermal Heatmap Overlay (JET)",
                    "⚪ Binary Change Mask",
                    "🟣 Cloud & Shadow Screening Mask"
                ],
                horizontal=True
            )

            c1, c2, c3 = st.columns(3)
            c1.image(pair_data["t1_visual_path"], caption="📅 T1: March 30, 2021 (Baseline Agricultural / Fallow)", width='stretch')
            c2.image(pair_data["t2_visual_path"], caption="📅 T2: March 14, 2024 (Industrial Ground Preparation)", width='stretch')

            if view_mode == "🟨 Bounding Box Change Clusters":
                c3.image(res['bbox_img'], caption="🟨 Detected Expansion Zones (Neon Bounding Boxes)", width='stretch')
            elif view_mode == "🔴 Thermal Heatmap Overlay (JET)":
                c3.image(res['overlay'], caption="🔴 Surface Transition Heatmap (Red/Yellow = High Delta)", width='stretch')
            elif view_mode == "⚪ Binary Change Mask":
                c3.image(res['change_mask'], caption="⚪ Cleaned Binary Change Mask (10m Pixels)", width='stretch')
            else:
                c3.image(res['cloud_mask_visual'], caption="🟣 SCL Quality Mask (Green=Valid, Magenta=Cloud/Shadow)", width='stretch')

        with tab_analytics:
            ca1, ca2 = st.columns(2)
            with ca1:
                st.subheader("Land Surface Distribution")
                fig_pie = px.pie(
                    values=[metrics['changed_area_ha'], metrics['unchanged_area_ha']],
                    names=['Converted / Changed Ground', 'Stable / Unchanged Ground'],
                    color_discrete_sequence=['#E63946', '#2A9D8F'],
                    hole=0.4
                )
                st.plotly_chart(fig_pie, width='stretch')

            with ca2:
                st.subheader("Area Metric Breakdown (Hectares)")
                fig_bar = px.bar(
                    x=["Total AOI Footprint", "Valid Unmasked Ground", "Candidate Converted Area"],
                    y=[metrics['total_area_ha'], metrics['valid_area_ha'], metrics['changed_area_ha']],
                    color=["Total", "Valid", "Changed"],
                    color_discrete_map={"Total": "#457B9D", "Valid": "#2A9D8F", "Changed": "#E63946"}
                )
                st.plotly_chart(fig_bar, width='stretch')

        with tab_report:
            st.subheader("📋 Official Geospatial Intelligence Report")
            report_body = f"""================================================================================
SPACE NETRA — GEOSPATIAL INTELLIGENCE & CHANGE DETECTION REPORT
Target Location: Shendra MIDC Expansion Corridor, Chhatrapati Sambhajinagar, India
================================================================================
SATELLITE PROVENANCE METADATA:
- Platform / Constellation   : European Space Agency Copernicus Sentinel-2A
- Sensor Payload             : Multi-Spectral Instrument (MSI)
- Processing Baseline        : Level-2A Bottom-of-Atmosphere (BOA) Surface Reflectance
- Spatial Resolution (GSD)   : 10.0 meters / pixel
- Target Coordinate System   : UTM Zone 43N (EPSG:32643) / WGS84
- Observation T1 (Baseline)  : 2021-03-30T05:26:41Z (Cloud Cover: 0.05%)
- Observation T2 (Recent)    : 2024-03-14T05:26:41Z (Cloud Cover: 0.00%)
- Temporal Baseline Gap      : 3.0 Years (Anniversary Pair)
--------------------------------------------------------------------------------
GEOSPATIAL AREA QUANTIFICATION:
- Total AOI Footprint        : {metrics['total_area_ha']} Hectares ({metrics['total_sq_km']} sq km)
- Valid Analyzed Surface     : {metrics['valid_area_ha']} Hectares ({metrics['valid_coverage_pct']}% data validity)
- Masked Clouds & Shadows    : {metrics['total_area_ha'] - metrics['valid_area_ha']:.2f} Hectares (Screened via SCL Layer)
- Confirmed Changed Surface  : {metrics['changed_area_ha']} Hectares ({metrics['changed_sq_km']} sq km)
- Change Surface Percentage  : {metrics['change_ratio_pct']}% of valid ground
- Detected Change Hotspots   : {res['cluster_count']} discrete structural expansion clusters
--------------------------------------------------------------------------------
ALGORITHMIC DETECTION BASELINE:
- Core Methodology           : Multi-Spectral Delta-NDVI + Structural Similarity (SSIM)
- Sensitivity Threshold      : {sensitivity} (Otsu / Configurable)
- Cluster Minimum Area       : {min_cluster} pixels (400 m^2 floor)
- Classification Status      : {'ACTION REQUIRED: Significant Surface Conversion Detected' if metrics['change_ratio_pct'] > 5.0 else 'NORMAL: Static Surface Observed'}
================================================================================"""
            st.code(report_body, language="text")
            st.download_button(
                "📥 Download Geospatial Report (.txt)",
                report_body,
                file_name="space_netra_sambhajinagar_intelligence_report.txt"
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
                                st.image(img, caption=f"📌 {item['filename']} | Match: {score}%", width='stretch')
                        else:
                            with cols[i]:
                                st.warning(f"File missing: {item['filename']}")
    except Exception as e:
        st.error(f"Search Execution Error: {e}")
