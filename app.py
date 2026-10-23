"""
app.py
======
Space Netra — Professional Satellite Intelligence & Change Detection Platform
Coverage: Full Municipal Region of Chhatrapati Sambhajinagar (Aurangabad), Maharashtra
Includes: High-Resolution Photographic Building Detection, Sentinel-2 Multispectral Differencing,
          Experimental S2Looking Bitemporal U-Net Model, and CLIP+FAISS Semantic Search.
"""

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
import plotly.graph_objects as go
import torch
import clip
import faiss

# Import Space Netra Backend Services
import city_catalog
import photo_streamer
import building_detector
import satellite_loader
import preprocessor
import change_engine
import area_metrics
import spacenetra_model

# ─────────────────────────────────────────────────────────────
# 1. PAGE CONFIGURATION & GEOSPATIAL DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Space Netra — Geospatial Change Intelligence",
    page_icon="🛰️",
    initial_sidebar_state="expanded"
)

# Custom Dark Charcoal / Deep Navy CSS with Cyan and Emerald Accents
st.markdown("""
<style>
    /* Global Base */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Top App Header Banner */
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }
    .app-title {
        font-size: 1.65rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .app-subtitle {
        font-size: 0.90rem;
        color: #94a3b8;
        margin-top: 0;
    }
    
    /* Card Containers */
    .geo-card {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 16px;
    }
    .geo-card-header {
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #38bdf8;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Coordinate & Metadata Badges */
    .coord-badge {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        background: #0f172a;
        border: 1px solid #334155;
        padding: 5px 12px;
        border-radius: 6px;
        color: #38bdf8;
        font-size: 0.88rem;
        font-weight: 500;
        display: inline-block;
    }
    .provenance-tag {
        font-size: 0.78rem;
        color: #94a3b8;
        background: #1e293b;
        padding: 3px 8px;
        border-radius: 4px;
        display: inline-block;
        margin-right: 6px;
    }
    
    /* Status Badges */
    .status-badge-active {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-standby {
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    
    /* Metrics Customization */
    div[data-testid="stMetricValue"] {
        color: #f8fafc;
        font-weight: 700;
        font-size: 1.55rem;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-size: 0.80rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    
    /* Tab Styling */
    button[data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-weight: 500;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom-color: #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 2. SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────
default_hotspot_key = "CIDCO Urban Expansion (N-1 to N-12)"
default_meta = city_catalog.get_hotspot_info(default_hotspot_key)

if "active_area_name" not in st.session_state:
    st.session_state["active_area_name"] = default_hotspot_key
    st.session_state["target_lat"] = default_meta["lat"]
    st.session_state["target_lon"] = default_meta["lon"]
    st.session_state["target_zoom"] = default_meta.get("zoom", 16)
    st.session_state["target_category"] = default_meta["category"]
    st.session_state["target_desc"] = default_meta["description"]
    st.session_state["detection_executed"] = False
    st.session_state["detection_results"] = None
    st.session_state["timeline_results"] = None

# ─────────────────────────────────────────────────────────────
# 3. SIDEBAR NAVIGATION & PARAMETER CONTROLS
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="padding: 10px 0 16px 0;">
    <div style="font-size: 1.25rem; font-weight: 700; color: #ffffff; letter-spacing: -0.01em;">
        🛰️ SPACE NETRA
    </div>
    <div style="font-size: 0.78rem; color: #38bdf8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
        Geospatial Change Platform
    </div>
</div>
""", unsafe_allow_html=True)

app_mode = st.sidebar.radio(
    "Navigation Mode",
    [
        "🛰️ Satellite Change Detection",
        "🔍 Semantic Archive Search"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# ─────────────────────────────────────────────────────────────
# MODE 1: SATELLITE CHANGE DETECTION
# ─────────────────────────────────────────────────────────────
if app_mode == "🛰️ Satellite Change Detection":
    
    # ── Sidebar Area & Detection Parameters ──
    st.sidebar.markdown("<div style='font-size: 0.82rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.04em;'>Detection Settings</div>", unsafe_allow_html=True)
    
    zoom_val = st.sidebar.slider(
        "Optical Zoom Level",
        min_value=15,
        max_value=18,
        value=int(st.session_state.get("target_zoom", 16)),
        help="Zoom 16: ~2.25 m/px | Zoom 17: ~1.12 m/px | Zoom 18: ~0.56 m/px Ground Sampling Distance."
    )
    st.session_state["target_zoom"] = zoom_val
    
    min_bldg_area = st.sidebar.slider(
        "Min Footprint Floor (m²)",
        min_value=20,
        max_value=250,
        value=50,
        step=10,
        help="Filters out non-building noise, individual cars, and tree clusters. Standard residential footprint starts at 50 m²."
    )
    
    sensitivity = st.sidebar.slider(
        "Structural Sensitivity",
        min_value=30,
        max_value=100,
        value=55,
        step=5,
        help="Threshold for Canny edge gradient dissimilarity."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("<div style='font-size: 0.82rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.04em;'>Data Provenance Control</div>", unsafe_allow_html=True)
    allow_demo_mode = st.sidebar.checkbox(
        "🧪 Allow Synthetic Demo Fallback",
        value=False,
        help="If genuine satellite imagery cannot be acquired, permits generating quarantined procedural fallback fixtures in data/synthetic_test_fixtures/. Synthetic data is clearly labeled."
    )

    # Sidebar Status Card
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div style="background: #111827; border: 1px solid #1e293b; border-radius: 6px; padding: 12px;">
        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Active Pipeline</div>
        <div style="font-size: 0.88rem; color: #f8fafc; font-weight: 600; margin-top: 2px;">
            {st.session_state['active_area_name'][:28]}...
        </div>
        <div style="margin-top: 6px;">
            <span class="{'status-badge-active' if st.session_state['detection_executed'] else 'status-badge-standby'}">
                {'● Ready / Results Loaded' if st.session_state['detection_executed'] else '○ Standby / Pending Run'}
            </span>
        </div>
        <div style="font-size: 0.74rem; color: #64748b; margin-top: 8px;">
            CRS: EPSG:4326 (WGS 84)<br>
            Grid: Web Mercator Zoom {zoom_val}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Main Header ──
    st.markdown("""
    <div class="app-header">
        <div class="app-title">
            <span>🛰️ Space Netra: Satellite Change-Detection Platform</span>
        </div>
        <p class="app-subtitle">
            Municipal Scope: Chhatrapati Sambhajinagar (Aurangabad), Maharashtra | Multi-Temporal Optical Aerial & Satellite Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Search & Location Workflow Section (Phase 4) ──
    st.markdown("<div class='geo-card-header'>📍 Area Selection & Geographic Search</div>", unsafe_allow_html=True)
    
    col_search1, col_search2 = st.columns([3, 2])
    
    with col_search1:
        hotspot_list = list(city_catalog.CITY_HOTSPOTS.keys())
        current_idx = hotspot_list.index(st.session_state["active_area_name"]) if st.session_state["active_area_name"] in hotspot_list else 0
        
        selected_hotspot = st.selectbox(
            "Select Pre-Verified Municipal Sector:",
            hotspot_list,
            index=current_idx,
            help="Choose from 7 verified expansion hubs with pre-calibrated baseline imagery."
        )
        
        if selected_hotspot != st.session_state["active_area_name"]:
            st.session_state["active_area_name"] = selected_hotspot
            h_info = city_catalog.get_hotspot_info(selected_hotspot)
            st.session_state["target_lat"] = h_info["lat"]
            st.session_state["target_lon"] = h_info["lon"]
            st.session_state["target_category"] = h_info["category"]
            st.session_state["target_desc"] = h_info["description"]
            st.session_state["detection_executed"] = False
            st.session_state["detection_results"] = None

    with col_search2:
        custom_query = st.text_input(
            "Or Search Any Address in City:",
            value="",
            placeholder="e.g. Kranti Chowk, Waluj MIDC, Airport...",
            help="Queries OpenStreetMap / Nominatim within Chhatrapati Sambhajinagar bounds."
        )
        col_btn_geo, col_btn_coord = st.columns(2)
        with col_btn_geo:
            geocode_clicked = st.button("🔎 Geocode Address", width="stretch")

    if geocode_clicked and custom_query.strip():
        try:
            with st.spinner("Resolving coordinates via OpenStreetMap Nominatim..."):
                geolocator = Nominatim(user_agent="space_netra_audit_app")
                full_q = f"{custom_query}, Chhatrapati Sambhajinagar, Maharashtra"
                loc = geolocator.geocode(full_q, timeout=8)
                if loc:
                    # Municipal bounding box validation: [75.2000, 19.8000, 75.5200, 19.9200]
                    min_lon, min_lat, max_lon, max_lat = city_catalog.CITY_BOUNDS_BBOX
                    if (min_lat <= loc.latitude <= max_lat) and (min_lon <= loc.longitude <= max_lon):
                        st.session_state["active_area_name"] = custom_query
                        st.session_state["target_lat"] = loc.latitude
                        st.session_state["target_lon"] = loc.longitude
                        st.session_state["target_category"] = "Geocoded Urban Point"
                        st.session_state["target_desc"] = f"Geocoded landmark address: {loc.address}"
                        st.session_state["detection_executed"] = False
                        st.session_state["detection_results"] = None
                        st.success(f"📍 Resolved: {loc.latitude:.6f}° N, {loc.longitude:.6f}° E")
                    else:
                        st.warning(f"⚠️ Location resolved to ({loc.latitude:.4f}, {loc.longitude:.4f}), which is outside Chhatrapati Sambhajinagar municipal bounds.")
                else:
                    st.error(f"Could not locate '{custom_query}' in Chhatrapati Sambhajinagar. Select from verified sectors.")
        except Exception as e:
            st.error(f"Geocoding service unavailable: {e}")

    # ── Prominent Area Information & Verified Coordinates Panel (Phase 3) ──
    cur_lat = float(st.session_state["target_lat"])
    cur_lon = float(st.session_state["target_lon"])
    
    st.markdown(f"""
    <div class="geo-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
            <div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff;">
                    {st.session_state['active_area_name']}
                </div>
                <div style="font-size: 0.84rem; color: #94a3b8; margin-top: 2px;">
                    Category: <strong style="color: #cbd5e1;">{st.session_state['target_category']}</strong> • {st.session_state['target_desc']}
                </div>
                <div style="margin-top: 10px; display: flex; flex-wrap: wrap; gap: 8px; align-items: center;">
                    <span class="coord-badge">Latitude: {cur_lat:.6f}° N</span>
                    <span class="coord-badge">Longitude: {cur_lon:.6f}° E</span>
                    <span class="provenance-tag">CRS: EPSG:4326 (WGS 84)</span>
                    <span class="provenance-tag">Grid: Web Mercator Zoom {zoom_val}</span>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.78rem; color: #64748b; text-transform: uppercase; font-weight: 600;">Temporal Epoch Coverage</div>
                <div style="font-size: 0.84rem; color: #38bdf8; margin-top: 3px;">
                    Baseline: <strong>Jan 2016</strong> → Mid-Stage: <strong>Jan 2020</strong> → Recent: <strong>2025/2026</strong>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Interactive Map & Verification Trigger (Two-Step Workflow) ──
    col_map_view, col_action = st.columns([3, 2])
    
    with col_map_view:
        m = folium.Map(
            location=[cur_lat, cur_lon],
            zoom_start=zoom_val,
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google Satellite"
        )
        folium.CircleMarker(
            location=[cur_lat, cur_lon],
            radius=8,
            color="#06b6d4",
            fill=True,
            fill_color="#06b6d4",
            fill_opacity=0.8,
            popup=f"AOI: {st.session_state['active_area_name']}",
            tooltip=f"{st.session_state['active_area_name']} ({cur_lat:.6f}°, {cur_lon:.6f}°)"
        ).add_to(m)
        st_folium(m, width=640, height=210)

    with col_action:
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1e293b; border-radius: 8px; padding: 16px; height: 210px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-size: 0.88rem; font-weight: 600; color: #f8fafc;">
                    Ready for Change Detection
                </div>
                <p style="font-size: 0.80rem; color: #94a3b8; margin: 6px 0 12px 0;">
                    Verify the geographic coordinates and zoom level above. Click the button below to ingest the 3-epoch imagery and isolate newly built structures.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        run_detection_btn = st.button("🚀 Find Changes / Run Detection", type="primary", width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Run Change Detection Execution ──
    if run_detection_btn or (st.session_state.get("detection_executed") and st.session_state.get("timeline_results") is not None):
        if run_detection_btn:
            with st.spinner(f"Ingesting multi-epoch photography and extracting building contours for {st.session_state['active_area_name']}..."):
                timeline = photo_streamer.fetch_3stage_photographic_timeline(
                    cur_lat, cur_lon, zoom=zoom_val, allow_synthetic_demo=allow_demo_mode
                )
                st.session_state["timeline_results"] = timeline
                st.session_state["detection_executed"] = True

                if timeline["status"] != "ACQUISITION_FAILED" and timeline.get("p1_path") and timeline.get("p3_path"):
                    det_res = building_detector.detect_new_buildings(
                        timeline["p1_path"],
                        timeline["p3_path"],
                        gsd_m=timeline["gsd_m"],
                        min_area_sq_m=float(min_bldg_area),
                        sensitivity=sensitivity
                    )
                    st.session_state["detection_results"] = det_res
                    st.session_state["acquisition_failed"] = False
                    if timeline["is_real"]:
                        st.success("Analysis complete. Verified real optical satellite photography rendered below.")
                    else:
                        st.warning("Analysis complete using explicit synthetic demonstration fallback fixtures.")
                else:
                    st.session_state["detection_results"] = None
                    st.session_state["acquisition_failed"] = True

        timeline = st.session_state["timeline_results"]
        det_res = st.session_state.get("detection_results")
        acquisition_failed = st.session_state.get("acquisition_failed", False)

        if acquisition_failed or det_res is None:
            # ── Honest Acquisition Failure Alert (Phase 3 & 6) ──
            st.markdown("---")
            st.error("❌ Optical Satellite Imagery Unavailable for this AOI")
            st.markdown(f"""
            <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 8px; padding: 20px; margin: 16px 0;">
                <h4 style="color: #ef4444; margin-top: 0; font-size: 1.05rem;">🛰️ Remote Satellite Acquisition Incomplete</h4>
                <p style="font-size: 0.88rem; color: #e2e8f0; margin-bottom: 8px;">
                    {timeline.get('acquisition_error', 'Authentic optical satellite tiles could not be fetched from the remote provider.')}
                </p>
                <p style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 14px;">
                    <strong>Data Provenance Policy:</strong> Space Netra adheres to strict data integrity standards and refuses to silently substitute synthetic drawings as genuine satellite imagery.
                </p>
                <div style="background: rgba(15, 23, 42, 0.7); border-radius: 6px; padding: 14px; font-size: 0.84rem; color: #cbd5e1;">
                    <strong style="color: #38bdf8;">Recommended Next Steps:</strong>
                    <ul style="margin: 6px 0 0 18px; padding: 0; line-height: 1.6;">
                        <li>Select one of the verified pre-cached city growth sectors: <strong>CIDCO Urban Expansion</strong>, <strong>Shendra MIDC</strong>, <strong>Kranti Chowk</strong>, <strong>Railway Station</strong>, <strong>Waluj MIDC</strong>, or <strong>Beed Bypass</strong>.</li>
                        <li>Or enable <strong>'Allow Synthetic Demo Fallback'</strong> in the sidebar under <em>Data Provenance Control</em> to test the processing pipeline using clearly marked procedural fixtures.</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # ── Provenance Status Banner ──
            st.markdown("---")
            if timeline["is_real"]:
                st.markdown("""
                <div style="display: flex; align-items: center; justify-content: space-between; background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 6px; padding: 10px 16px; margin-bottom: 16px;">
                    <span style="color: #10b981; font-weight: 700; font-size: 0.88rem;">
                        🟢 VERIFIED REAL SATELLITE IMAGERY — Genuine Historical & Current Optical Observations
                    </span>
                    <span style="background: #10b981; color: #0b0f19; font-weight: 700; font-size: 0.70rem; padding: 2px 8px; border-radius: 4px;">
                        VERIFIED PROVENANCE
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="display: flex; align-items: center; justify-content: space-between; background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.35); border-radius: 6px; padding: 10px 16px; margin-bottom: 16px;">
                    <span style="color: #f59e0b; font-weight: 700; font-size: 0.88rem;">
                        ⚠️ SYNTHETIC DEMONSTRATION FIXTURE — Procedurally Generated Fallback (Not Real Observation)
                    </span>
                    <span style="background: #f59e0b; color: #0b0f19; font-weight: 700; font-size: 0.70rem; padding: 2px 8px; border-radius: 4px;">
                        SIMULATED TEST FIXTURE
                    </span>
                </div>
                """, unsafe_allow_html=True)

            # ── Metrics Row (Phase 6 & 7: Units, Distinguishable Colors, No Clutter) ──
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("New Buildings Found", f"{det_res['building_count']} Units")
            m2.metric("New Built Footprint", f"{det_res['total_built_area_sq_m']:,.1f} m²", help="Sum of extracted building contours meeting minimum area floor")
            m3.metric("Built-Up Density", f"{det_res['building_density_pct']}% of AOI", help="Percentage of analyzed ground occupied by new building roofs")
            m4.metric("Ground Resolution", f"{det_res['gsd_m']} m/px", help="Metric Ground Sampling Distance for selected zoom level")
            m5.metric("Expansion Level", "High Growth" if det_res['building_count'] >= 3 else ("Moderate Growth" if det_res['building_count'] > 0 else "Stable Surface"))

            # ── Primary Analysis Tabs ──
            tab_outlines, tab_timeline, tab_inventory, tab_sentinel2, tab_ml_unet, tab_report = st.tabs([
                "🏢 Building Outlines & Verification",
                "⏱️ 3-Stage Multi-Year Timeline",
                "📋 Detected Building Inventory & Dimensions",
                "🛰️ Sentinel-2 Multispectral Surface Differencing",
                "🧪 Experimental ML Change Detection",
                "📄 Municipal Intelligence Report"
            ])

            with tab_outlines:
                provenance_badge = "🟢 Verified Optical Imagery" if timeline["is_real"] else "⚠️ Synthetic Fallback Fixture"
                st.subheader(f"Optical Imagery with Labeled New Construction ({provenance_badge})")
                st.caption("Emerald green vector bounding boxes and badges denote newly constructed building roofs and architectural structures.")

                view_mode = st.radio(
                    "Display Overlay Option:",
                    [
                        "Labeled Emerald Green Outlines on Current Real Photo",
                        "Thermal Construction Intensity Heatmap",
                        "Binary Building Footprint Mask"
                    ],
                    horizontal=True
                )

            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(
                    timeline["p1_path"],
                    caption=f"Baseline Photo — {timeline['dates']['epoch_1']} (Esri World Imagery Wayback)",
                    width="stretch"
                )
            with col_img2:
                if view_mode == "Labeled Emerald Green Outlines on Current Real Photo":
                    st.image(
                        det_res["annotated_img"],
                        caption=f"Current Photo with Detected Buildings — {timeline['dates']['epoch_3']}",
                        width="stretch"
                    )
                elif view_mode == "Thermal Construction Intensity Heatmap":
                    st.image(
                        det_res["heatmap_img"],
                        caption="Thermal Structural Difference Intensity (Red/Yellow = Major Structural Modifications)",
                        width="stretch"
                    )
                else:
                    st.image(
                        det_res["binary_mask"],
                        caption="Extracted Binary Building Roof Footprint Mask",
                        width="stretch"
                    )

        with tab_timeline:
            st.subheader(f"3-Stage Multi-Year Building Evolution: {st.session_state['active_area_name']}")
            st.caption("Side-by-side temporal progression across a 10-year span showing authentic construction and infill emergence.")

            t_col1, t_col2, t_col3 = st.columns(3)
            with t_col1:
                st.image(
                    timeline["p1_path"],
                    caption=f"Epoch 1 (Baseline): {timeline['dates']['epoch_1']}\nOriginal Ground & Sparse Structures",
                    width="stretch"
                )
            with t_col2:
                st.image(
                    timeline["p2_path"],
                    caption=f"Epoch 2 (Mid-Stage): {timeline['dates']['epoch_2']}\nMid-Stage Groundwork & Infill",
                    width="stretch"
                )
            with t_col3:
                st.image(
                    det_res["annotated_img"],
                    caption=f"Epoch 3 (Current): {timeline['dates']['epoch_3']}\nCurrent Structures with Building Tags",
                    width="stretch"
                )

        with tab_inventory:
            st.subheader("Itemized Inventory of Detected New Buildings")
            if det_res["inventory"]:
                df_bldgs = pd.DataFrame(det_res["inventory"])
                st.dataframe(df_bldgs, width="stretch")

                # Emerald Green Plotly Chart following Semantic Colors (Phase 6)
                fig_bar = px.bar(
                    df_bldgs,
                    x="Building ID",
                    y="Footprint Area (m²)",
                    title=f"New Construction Footprint Breakdown — {st.session_state['active_area_name']}",
                    labels={"Building ID": "Detected Structure ID", "Footprint Area (m²)": "Footprint Area (Square Meters)"},
                    color_discrete_sequence=["#10b981"]
                )
                fig_bar.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="#0f172a",
                    paper_bgcolor="#111827",
                    font=dict(color="#f8fafc"),
                    margin=dict(l=40, r=40, t=50, b=40),
                    xaxis=dict(gridcolor="#1e293b"),
                    yaxis=dict(gridcolor="#1e293b")
                )
                st.plotly_chart(fig_bar, width="stretch")
            else:
                st.info("No newly constructed building structures detected in this sector meeting the minimum area threshold.")

        with tab_sentinel2:
            st.subheader("Calibrated Sentinel-2 Multispectral Surface Differencing")
            st.caption("Physics-based 10m Level-2A BOA surface reflectance analysis over Shendra MIDC Industrial Corridor.")

            try:
                pairs = satellite_loader.load_temporal_pair()
                
                # Provenance Integrity Banner
                if pairs.get("is_synthetic", False):
                    st.markdown("""
                    <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.4); border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;">
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <span style="font-weight: 700; color: #f59e0b; font-size: 0.92rem;">
                                ⚠️ SYNTHETIC TEST FIXTURE — Calibrated Simulated 10m Sentinel-2 L2A Multispectral Bands
                            </span>
                            <span style="background: #f59e0b; color: #0b0f19; font-weight: 700; font-size: 0.72rem; padding: 2px 8px; border-radius: 4px;">
                                QUARANTINED BENCHMARK
                            </span>
                        </div>
                        <p style="font-size: 0.82rem; color: #cbd5e1; margin: 8px 0 4px 0;">
                            <strong>Data Provenance Notice:</strong> Authentic Sentinel-2 Level-2A scene metadata (IDs, timestamps, coordinates) exists in the catalog manifest, but the 10m multispectral rasters displayed below are <strong>synthetic benchmark test arrays</strong> quarantined in <code>data/synthetic_test_fixtures/</code>. They are used to verify cloud masking (SCL), band-differencing math (NDVI/NDBI), and area conservation algorithms. Real Level-2A raster pixels have not yet been downloaded.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;">
                        <span style="font-weight: 700; color: #10b981; font-size: 0.92rem;">
                            🟢 VERIFIED REAL SENTINEL-2 OBSERVATION — Authentic Level-2A BOA Reflectance
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                t1_clean, t1_mask = preprocessor.clean_scene(pairs["t1_b04"], pairs["t1_b08"], pairs["t1_scl"])
                t2_clean, t2_mask = preprocessor.clean_scene(pairs["t2_b04"], pairs["t2_b08"], pairs["t2_scl"])
                scl_combined = preprocessor.combine_masks(t1_mask, t2_mask)
                s2_res = change_engine.detect_multispectral_changes(t1_clean, t2_clean, scl_combined)
                s2_metrics = area_metrics.compute_area_metrics(s2_res["change_mask"], scl_combined)

                col_s2_1, col_s2_2 = st.columns(2)
                with col_s2_1:
                    t1_caption = "Synthetic Preview T1 (Simulated March 30, 2021 — Benchmark Fixture)" if pairs.get("is_synthetic") else "Sentinel-2 L2A BOA True Color (March 30, 2021)"
                    st.image(pairs["t1_visual_path"], caption=t1_caption, width="stretch")
                with col_s2_2:
                    t2_caption = "Synthetic Preview T2 (Simulated March 14, 2024 — Benchmark Fixture)" if pairs.get("is_synthetic") else "Sentinel-2 L2A BOA True Color (March 14, 2024)"
                    st.image(pairs["t2_visual_path"], caption=t2_caption, width="stretch")

                st.write("---")
                sm1, sm2, sm3, sm4 = st.columns(4)
                sm1.metric("Changed Surface", f"{s2_metrics['changed_area_hectares']} Ha")
                sm2.metric("Change Ratio", f"{s2_metrics['change_ratio_pct']}% of AOI")
                sm3.metric("Clear Sky Confidence", f"{s2_metrics['valid_ratio_pct']}%")
                sm4.metric("Change Clusters", f"{s2_res['cluster_count']} Regions")

                # Land Cover Donut Chart
                donut_df = pd.DataFrame([
                    {"Category": "Changed Surface", "Area (Ha)": s2_metrics["changed_area_hectares"]},
                    {"Category": "Unchanged Ground", "Area (Ha)": s2_metrics["unchanged_area_hectares"]},
                    {"Category": "Masked (Cloud/Shadow)", "Area (Ha)": s2_metrics["masked_area_hectares"]}
                ])
                fig_donut = px.pie(
                    donut_df,
                    names="Category",
                    values="Area (Ha)",
                    hole=0.55,
                    title="Surface Classification Distribution (Shendra MIDC AOI)",
                    color="Category",
                    color_discrete_map={
                        "Changed Surface": "#10b981",
                        "Unchanged Ground": "#64748b",
                        "Masked (Cloud/Shadow)": "#475569"
                    }
                )
                fig_donut.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#111827",
                    font=dict(color="#f8fafc"),
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig_donut, width="stretch")

            except Exception as e:
                st.warning(f"Calibrated Sentinel-2 data unavailable or requires pre-staging: {e}")

        # ── NEW TAB: EXPERIMENTAL ML CHANGE DETECTION (S2LOOKING U-NET) ──
        with tab_ml_unet:
            st.subheader("🧪 Experimental ML Change Detection (S2Looking Bitemporal U-Net)")
            st.caption("Deep-learning binary change-detection neural network trained on a subset of the S2Looking benchmark dataset.")

            # Honest Model-Status Notice (Sections 4 & 5)
            st.markdown("""
            <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.35); border-radius: 8px; padding: 14px 18px; margin-bottom: 18px;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="font-weight: 700; color: #f59e0b; font-size: 0.95rem;">
                        ⚠️ EXPERIMENTAL RESEARCH PREVIEW — S2Looking Bitemporal U-Net
                    </span>
                    <span style="background: #f59e0b; color: #0b0f19; font-weight: 700; font-size: 0.72rem; padding: 2px 8px; border-radius: 4px;">
                        EPOCH 2 CHECKPOINT
                    </span>
                </div>
                <p style="font-size: 0.82rem; color: #cbd5e1; margin: 8px 0 10px 0;">
                    This deep-learning model was trained in Google Colab on a sampled subset (500 train pairs, 100 val pairs) of the <strong>S2Looking</strong> satellite building change benchmark. Validation metrics indicate preliminary, exploratory detection capability. Output is a <strong>binary change probability mask only</strong> (it does not distinguish construction from demolition).
                </p>
                <div style="display: flex; gap: 16px; flex-wrap: wrap; font-size: 0.80rem; color: #94a3b8;">
                    <span>Benchmark Precision: <strong style="color: #f8fafc;">0.2993</strong></span>
                    <span>Recall: <strong style="color: #f8fafc;">0.1534</strong></span>
                    <span>Validation F1: <strong style="color: #f8fafc;">0.2028</strong></span>
                    <span>Validation IoU: <strong style="color: #f8fafc;">0.1129</strong></span>
                    <span>Loss: <strong style="color: #f8fafc;">1.1263</strong></span>
                </div>
                <div style="font-size: 0.74rem; color: #64748b; margin-top: 6px;">
                    * Note: These are benchmark validation results across 100 test pairs, not performance measurements for the currently selected AOI.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Model Loading via Streamlit Cache
            @st.cache_resource
            def load_cached_unet():
                return spacenetra_model.load_model()

            ml_model, ml_meta, ml_err = load_cached_unet()

            if ml_err:
                st.error(f"Could not load ML checkpoint: {ml_err}")
            else:
                # Source Selection
                ml_source_mode = st.radio(
                    "Image Input Source for Model Inference:",
                    [
                        "Use Current Active Area Images (Epoch 1 2016 vs. Epoch 3 Current)",
                        "Upload Custom Image Pair (Before / After)"
                    ],
                    horizontal=True
                )

                input_img1 = None
                input_img2 = None
                provenance_note = ""

                if ml_source_mode == "Use Current Active Area Images (Epoch 1 2016 vs. Epoch 3 Current)":
                    input_img1 = timeline["p1_path"]
                    input_img2 = timeline["p3_path"]
                    input_prov = "🟢 Verified Optical Tile" if timeline.get("is_real") else "⚠️ Synthetic Fallback Fixture"
                    provenance_note = f"Active Sector: {st.session_state['active_area_name']} | Earlier: {timeline['dates']['epoch_1']} | Later: {timeline['dates']['epoch_3']} | Input Provenance: {input_prov}"
                    st.info(f"📍 {provenance_note}")
                else:
                    st.write("Upload custom Earlier ($T_1$) and Later ($T_2$) images (PNG or JPG):")
                    u_col1, u_col2 = st.columns(2)
                    with u_col1:
                        f1 = st.file_uploader("Upload Earlier Image (T1)", type=["png", "jpg", "jpeg"], key="ml_upl_1")
                    with u_col2:
                        f2 = st.file_uploader("Upload Later Image (T2)", type=["png", "jpg", "jpeg"], key="ml_upl_2")

                    if f1 and f2:
                        input_img1 = Image.open(f1)
                        input_img2 = Image.open(f2)
                        provenance_note = f"Analyzing User-Uploaded Images: '{f1.name}' vs. '{f2.name}'"
                    else:
                        st.caption("Upload both an earlier and later image to enable custom inference.")

                # Controls
                c_thresh, c_btn = st.columns([3, 2])
                with c_thresh:
                    thresh_val = st.slider(
                        "Prediction Threshold",
                        min_value=0.20,
                        max_value=0.80,
                        value=0.50,
                        step=0.05,
                        help="Default 0.50. Pixels with change probability >= threshold are classified as changed."
                    )
                with c_btn:
                    st.write("") # spacing
                    run_ml_btn = st.button("🔬 Run U-Net ML Inference", type="primary", width="stretch", disabled=(input_img1 is None or input_img2 is None))

                if run_ml_btn and input_img1 is not None and input_img2 is not None:
                    with st.spinner("Preprocessing image pair (bilinear 256×256, 6-channel tensor) and computing U-Net forward pass..."):
                        tensor_6ch, im1_prep, im2_prep = spacenetra_model.preprocess_pair(input_img1, input_img2, target_size=256)
                        pred = spacenetra_model.predict_change(
                            ml_model,
                            tensor_6ch,
                            im2_prep,
                            threshold=thresh_val,
                            device=ml_meta["device"]
                        )
                        st.session_state["ml_last_pred"] = pred
                        st.session_state["ml_im1_prep"] = im1_prep
                        st.session_state["ml_im2_prep"] = im2_prep
                        st.session_state["ml_provenance"] = provenance_note

                if "ml_last_pred" in st.session_state and st.session_state["ml_last_pred"] is not None:
                    pred = st.session_state["ml_last_pred"]
                    im1_p = st.session_state["ml_im1_prep"]
                    im2_p = st.session_state["ml_im2_prep"]

                    st.markdown("---")
                    pm1, pm2, pm3, pm4 = st.columns(4)
                    pm1.metric("Predicted Changed Pixels", f"{pred['changed_pixels']:,} px")
                    pm2.metric("Predicted Change Fraction", f"{pred['change_ratio_pct']}% of frame")
                    pm3.metric("Decision Threshold", f"{pred['threshold']}")
                    pm4.metric("Model Status", "Inference OK", help="Strict checkpoint execution completed with no NaNs or non-finite values.")

                    # 4-column visual display
                    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                    with col_p1:
                        st.image(im1_p, caption="Earlier Image (T1, 256×256)", width="stretch")
                    with col_p2:
                        st.image(im2_p, caption="Later Image (T2, 256×256)", width="stretch")
                    with col_p3:
                        st.image(pred["binary_mask"] * 255, caption=f"Predicted Binary Mask (Threshold ≥ {pred['threshold']})", width="stretch")
                    with col_p4:
                        st.image(pred["overlay_img"], caption="Highlighted Change Overlay (Coral Red)", width="stretch")

                    # Distinct Legend
                    st.markdown("""
                    <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 6px; padding: 10px 14px; margin-top: 10px; font-size: 0.80rem; display: flex; gap: 20px; align-items: center;">
                        <span style="font-weight: 600; color: #94a3b8;">Legend:</span>
                        <span><strong style="color: #f5503c;">■ Coral Red:</strong> Predicted Change (Probability ≥ Threshold)</span>
                        <span><strong style="color: #64748b;">■ Natural Imagery:</strong> Unchanged / Background</span>
                        <span style="color: #94a3b8;">• Output is a binary change mask only (does not distinguish new construction vs. demolition).</span>
                    </div>
                    """, unsafe_allow_html=True)

        with tab_report:
            st.subheader("Official Municipal Intelligence Report")
            rep_text = f"""================================================================================
SPACE NETRA — MUNICIPAL BUILDING CHANGE INTELLIGENCE AUDIT
Municipality: Chhatrapati Sambhajinagar (Aurangabad), Maharashtra, India
Audit Generated: Synchronized Operational Session
================================================================================
GEOGRAPHIC & SPATIAL PARAMETERS:
- Sector Designation         : {st.session_state['active_area_name']}
- Sector Classification      : {st.session_state['target_category']}
- Geographic Center Lat/Lon  : {cur_lat:.6f}° N, {cur_lon:.6f}° E
- Coordinate Reference (CRS) : EPSG:4326 (WGS 84)
- Web Mercator Tile Grid     : Zoom {zoom_val} (X={timeline['x']}, Y={timeline['y']})
- Metric Ground Sampling     : {det_res['gsd_m']} meters / pixel
--------------------------------------------------------------------------------
TEMPORAL IMAGERY PROVENANCE:
- Provenance Classification : {'VERIFIED_REAL (Authentic Satellite Photography)' if timeline.get('is_real') else 'SYNTHETIC_FALLBACK (Procedural Simulation Test Fixture)'}
- Epoch 1 (Historical)       : {timeline['dates']['epoch_1']} (Esri World Imagery Wayback)
- Epoch 2 (Mid-Stage)        : {timeline['dates']['epoch_2']} (Esri World Imagery Wayback)
- Epoch 3 (Current)          : {timeline['dates']['epoch_3']} (High-Resolution Optical Satellite)
--------------------------------------------------------------------------------
BUILDING DETECTION METRICS:
- Detected New Structures    : {det_res['building_count']} distinct units
- Total New Built-Up Footprint: {det_res['total_built_area_sq_m']:,} m² ({det_res['total_built_area_ha']} Hectares)
- Built-Up Expansion Density : {det_res['building_density_pct']}% of analyzed ground
- Minimum Area Floor Filter  : {min_bldg_area} m² (Aspect-Ratio Constrained: 0.2 - 5.0)
- Operational Verdict        : {'ACTION: Significant New Building Construction' if det_res['building_count'] > 0 else 'STABLE: No Structural Alteration Detected'}
--------------------------------------------------------------------------------
EXPERIMENTAL ML MODEL PROVENANCE (S2LOOKING U-NET):
- Model Architecture        : 6-Channel Bitemporal U-Net (256x256 resolution)
- Training Dataset          : S2Looking Benchmark Subset (Google Colab, Epoch 2)
- Benchmark Val F1 / IoU    : 0.2028 / 0.1129 (Exploratory Binary Change Mask Only)
================================================================================"""
            st.code(rep_text, language="text")
            st.download_button(
                "📥 Download Official Municipal Report (.txt)",
                rep_text,
                file_name=f"space_netra_{st.session_state['active_area_name'].lower().replace(' ', '_')}_audit.txt",
                width="stretch"
            )

    else:
        st.info("👆 Click **'Find Changes / Run Detection'** above to fetch imagery and execute building change detection.")

# ─────────────────────────────────────────────────────────────
# MODE 2: VECTOR-BASED SEMANTIC SEARCH (PRESERVED CLIP + FAISS)
# ─────────────────────────────────────────────────────────────
elif app_mode == "🔍 Semantic Archive Search":
    
    st.markdown("""
    <div class="app-header">
        <div class="app-title">
            <span>🔍 Spatial Archive Semantic Search</span>
        </div>
        <p class="app-subtitle">
            Natural-Language Vector Search over Local Satellite Image Patches via OpenAI CLIP (ViT-B/32) and FAISS
        </p>
        <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 6px; padding: 8px 14px; margin-top: 10px; font-size: 0.78rem; color: #94a3b8;">
            📁 <strong>Dataset Provenance:</strong> Local archive consists of benchmark remote-sensing scene classification image patches (<code>data/raw/</code>: airport, forest, harbor, stadium, urban) indexed via 512-dimensional CLIP embeddings. Used for semantic visual concept indexing, not temporal physical change detection.
        </div>
    </div>
    """, unsafe_allow_html=True)

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
            st.warning("Vector database not found at 'vector_db/'. Run 'python ingest.py' to generate embeddings.")
        else:
            st.markdown("""
            <div class="geo-card">
                <div class="geo-card-header">🔎 Natural Language Query</div>
                <div style="font-size: 0.84rem; color: #94a3b8; margin-bottom: 12px;">
                    Enter a natural language description to search the local high-resolution spatial image archive:
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_q, col_btn = st.columns([4, 1])
            with col_q:
                query_str = st.text_input(
                    "Search Query:",
                    "urban area",
                    label_visibility="collapsed",
                    placeholder="e.g. urban area, harbor and ships, airport runway, forest..."
                )
            with col_btn:
                search_clicked = st.button("Search Archive", type="primary", width="stretch")

            # Sample query chips
            st.caption("Quick search suggestions: `urban area` • `harbor` • `airport` • `stadium` • `forest`")

            if (search_clicked or query_str) and query_str.strip():
                with st.spinner("Computing CLIP text embedding and querying FAISS vector index..."):
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
                                st.image(img, caption=f"Match Score: {score}% | {item['filename']}", width="stretch")
                        else:
                            with cols[i]:
                                st.warning(f"File missing: {item['filename']}")
    except Exception as e:
        st.error(f"Semantic search execution error: {e}")
