"""
views/optical_detection.py
==========================
High-Resolution Photographic Building-Level Change Detection.
Processes 3-epoch optical satellite imagery across Chhatrapati Sambhajinagar.
"""

import os
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import plotly.express as px

import city_catalog
import photo_streamer
import building_detector
from views.theme import render_page_header

def render_optical_detection():
    """Renders the optical building change detection module."""
    render_page_header(
        title="Optical Change Detection — Multi-Temporal Building Extraction",
        subtitle="High-Resolution Photographic Evolution (2016 → 2020 → Current) with Canny/SSIM Structural Contour Analysis",
        badge_text="Photographic Pipeline",
        badge_type="ready"
    )

    # ── Sidebar Controls (Isolated to this page) ──
    st.sidebar.markdown("<div class='workstation-card-title'>Optical Settings</div>", unsafe_allow_html=True)

    hotspot_list = list(city_catalog.CITY_HOTSPOTS.keys())
    if "active_area_name" not in st.session_state or st.session_state["active_area_name"] not in hotspot_list:
        st.session_state["active_area_name"] = "CIDCO Urban Expansion (N-1 to N-12)"

    current_idx = hotspot_list.index(st.session_state["active_area_name"])

    selected_hotspot = st.sidebar.selectbox(
        "Municipal Growth Sector:",
        hotspot_list,
        index=current_idx,
        help="Select from 7 pre-calibrated expansion corridors in Chhatrapati Sambhajinagar."
    )

    if selected_hotspot != st.session_state["active_area_name"]:
        st.session_state["active_area_name"] = selected_hotspot
        h_info = city_catalog.get_hotspot_info(selected_hotspot)
        st.session_state["target_lat"] = h_info["lat"]
        st.session_state["target_lon"] = h_info["lon"]
        st.session_state["target_zoom"] = h_info.get("zoom", 17)
        st.session_state["target_category"] = h_info["category"]
        st.session_state["target_desc"] = h_info["description"]
        st.session_state["optical_detection_executed"] = False
        st.session_state["optical_detection_results"] = None
        st.session_state["optical_timeline_results"] = None

    # Geocoding custom query in sidebar
    with st.sidebar.expander("Search Custom Landmark / Address", expanded=False):
        custom_query = st.text_input(
            "Address Search:",
            value="",
            placeholder="e.g. Kranti Chowk, Waluj...",
            help="Geocodes within municipal bounds via OpenStreetMap Nominatim."
        )
        if st.button("Geocode Address", width="stretch") and custom_query.strip():
            try:
                with st.spinner("Resolving coordinates..."):
                    geolocator = Nominatim(user_agent="space_netra_workstation")
                    full_q = f"{custom_query}, Chhatrapati Sambhajinagar, Maharashtra"
                    loc = geolocator.geocode(full_q, timeout=8)
                    if loc:
                        min_lon, min_lat, max_lon, max_lat = city_catalog.CITY_BOUNDS_BBOX
                        if (min_lat <= loc.latitude <= max_lat) and (min_lon <= loc.longitude <= max_lon):
                            st.session_state["active_area_name"] = custom_query
                            st.session_state["target_lat"] = loc.latitude
                            st.session_state["target_lon"] = loc.longitude
                            st.session_state["target_category"] = "Geocoded Urban Location"
                            st.session_state["target_desc"] = f"Geocoded address: {loc.address}"
                            st.session_state["optical_detection_executed"] = False
                            st.session_state["optical_detection_results"] = None
                            st.session_state["optical_timeline_results"] = None
                            st.success(f"Resolved: {loc.latitude:.5f}° N, {loc.longitude:.5f}° E")
                        else:
                            st.warning("Location lies outside Chhatrapati Sambhajinagar bounds.")
                    else:
                        st.error("Address not found within municipal boundary.")
            except Exception as e:
                st.error(f"Geocoding service unavailable: {e}")

    zoom_val = st.sidebar.slider(
        "Optical Zoom Level:",
        min_value=15,
        max_value=18,
        value=int(st.session_state.get("target_zoom", 17)),
        help="Zoom 16: ~2.25 m/px | Zoom 17: ~1.12 m/px | Zoom 18: ~0.56 m/px Ground Sampling Distance."
    )
    st.session_state["target_zoom"] = zoom_val

    min_bldg_area = st.sidebar.slider(
        "Min Footprint Floor (m²):",
        min_value=20,
        max_value=250,
        value=50,
        step=10,
        help="Filters out transient noise, vehicles, and small trees. Residential footprints typically start at 50 m²."
    )

    sensitivity = st.sidebar.slider(
        "Structural Sensitivity:",
        min_value=30,
        max_value=100,
        value=55,
        step=5,
        help="Higher values require stronger structural gradient contrast to isolate building roofs."
    )

    # ── Geographic Metadata Banner ──
    cur_lat = float(st.session_state.get("target_lat", 19.8856))
    cur_lon = float(st.session_state.get("target_lon", 75.3689))
    cat_desc = st.session_state.get("target_desc", "Urban Expansion Zone")
    cat_type = st.session_state.get("target_category", "Residential & Commercial")

    st.markdown(f"""
    <div class="workstation-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
            <div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff;">
                    {st.session_state['active_area_name']}
                </div>
                <div style="font-size: 0.84rem; color: #94a3b8; margin-top: 2px;">
                    Category: <strong style="color: #cbd5e1;">{cat_type}</strong> — {cat_desc}
                </div>
                <div style="margin-top: 10px; display: flex; flex-wrap: wrap; gap: 8px; align-items: center;">
                    <span class="meta-chip">Lat: {cur_lat:.6f}° N</span>
                    <span class="meta-chip">Lon: {cur_lon:.6f}° E</span>
                    <span class="tag-chip">CRS: EPSG:4326 (WGS 84)</span>
                    <span class="tag-chip">Web Mercator Zoom {zoom_val}</span>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.75rem; color: #64748b; text-transform: uppercase; font-weight: 600;">Temporal Epochs</div>
                <div style="font-size: 0.82rem; color: #38bdf8; margin-top: 2px;">
                    Baseline: <strong>2016</strong> → Mid: <strong>2020</strong> → Current: <strong>2025/2026</strong>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Interactive Folium Map & Trigger Action ──
    col_map, col_act = st.columns([3, 2])

    with col_map:
        m = folium.Map(
            location=[cur_lat, cur_lon],
            zoom_start=zoom_val,
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google Satellite"
        )
        folium.CircleMarker(
            location=[cur_lat, cur_lon],
            radius=8,
            color="#38bdf8",
            fill=True,
            fill_color="#38bdf8",
            fill_opacity=0.75,
            tooltip=f"{st.session_state['active_area_name']} ({cur_lat:.5f}°, {cur_lon:.5f}°)"
        ).add_to(m)
        st_folium(m, width=640, height=210)

    with col_act:
        st.markdown("""
        <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 18px; height: 210px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-size: 0.90rem; font-weight: 600; color: #f8fafc;">
                    Photographic Change Pipeline
                </div>
                <p style="font-size: 0.80rem; color: #94a3b8; margin: 8px 0 12px 0; line-height: 1.5;">
                    Fetches 3-epoch calibrated satellite tiles (Esri Wayback 2016 & 2020 + Current High-Res Optical)
                    and executes geometric building contour extraction.
                </p>
            </div>
        """, unsafe_allow_html=True)

        run_btn = st.button("Run Optical Detection", type="primary", width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Pipeline Execution ──
    if run_btn or (st.session_state.get("optical_detection_executed") and st.session_state.get("optical_timeline_results") is not None):
        if run_btn:
            with st.spinner(f"Acquiring multi-epoch imagery and extracting building contours for {st.session_state['active_area_name']}..."):
                timeline = photo_streamer.fetch_3stage_photographic_timeline(
                    cur_lat, cur_lon, zoom=zoom_val, allow_synthetic_demo=False
                )
                st.session_state["optical_timeline_results"] = timeline
                st.session_state["optical_detection_executed"] = True

                if timeline["status"] != "ACQUISITION_FAILED" and timeline.get("p1_path") and timeline.get("p3_path"):
                    det_res = building_detector.detect_new_buildings(
                        timeline["p1_path"],
                        timeline["p3_path"],
                        gsd_m=timeline["gsd_m"],
                        min_area_sq_m=float(min_bldg_area),
                        sensitivity=sensitivity
                    )
                    st.session_state["optical_detection_results"] = det_res
                    st.session_state["optical_acq_failed"] = False
                else:
                    st.session_state["optical_detection_results"] = None
                    st.session_state["optical_acq_failed"] = True

        timeline = st.session_state.get("optical_timeline_results")
        det_res = st.session_state.get("optical_detection_results")
        acq_failed = st.session_state.get("optical_acq_failed", False)

        if acq_failed or det_res is None or timeline is None:
            st.error("Optical Satellite Imagery Acquisition Incomplete for this location.")
            st.info("Select one of the verified pre-cached sectors (CIDCO, Shendra MIDC, Kranti Chowk, Waluj MIDC, Beed Bypass).")
            return

        # Provenance Banner
        if timeline["is_real"]:
            st.markdown("""
            <div style="display: flex; align-items: center; justify-content: space-between; background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 6px; padding: 10px 16px; margin: 16px 0;">
                <span style="color: #10b981; font-weight: 700; font-size: 0.85rem;">
                    AUTHENTIC SATELLITE IMAGERY — Verified Optical Observations (EPSG:4326)
                </span>
                <span class="status-pill-ready">Verified Real</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display: flex; align-items: center; justify-content: space-between; background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.35); border-radius: 6px; padding: 10px 16px; margin: 16px 0;">
                <span style="color: #f59e0b; font-weight: 700; font-size: 0.85rem;">
                    BENCHMARK TEST FIXTURE — Simulated Verification Imagery
                </span>
                <span class="status-pill-warning">Test Fixture</span>
            </div>
            """, unsafe_allow_html=True)

        # Metrics Row
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("New Buildings Found", f"{det_res['building_count']} Units")
        m2.metric("New Built Footprint", f"{det_res['total_built_area_sq_m']:,.1f} m²", help="Sum of discrete building contours meeting minimum area floor")
        m3.metric("Built-Up Density", f"{det_res['building_density_pct']}% of AOI", help="Percentage of analyzed ground occupied by new building roofs")
        m4.metric("Ground Resolution", f"{det_res['gsd_m']} m/px", help="Metric Ground Sampling Distance for selected zoom level")
        m5.metric("Expansion Level", "High Growth" if det_res['building_count'] >= 3 else ("Moderate Growth" if det_res['building_count'] > 0 else "Stable Surface"))

        st.markdown("---")

        # ── 3-Stage Temporal Evolution Timeline (Clean, Equal Width Row) ──
        st.markdown("##### 3-Stage Multi-Year Photographic Evolution")
        st.caption("Temporal progression across a 10-year baseline displaying ground transformation and infill emergence.")

        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            st.image(
                timeline["p1_path"],
                caption=f"Epoch 1 (Baseline): {timeline['dates']['epoch_1']}\nEsri Wayback Historical",
                width="stretch"
            )
        with col_t2:
            st.image(
                timeline["p2_path"],
                caption=f"Epoch 2 (Mid-Stage): {timeline['dates']['epoch_2']}\nEsri Wayback Mid-Stage",
                width="stretch"
            )
        with col_t3:
            st.image(
                det_res["annotated_img"],
                caption=f"Epoch 3 (Current): {timeline['dates']['epoch_3']}\nCurrent High-Res Optical with Building Tags",
                width="stretch"
            )

        st.markdown("---")

        # ── Visual Overlays Section ──
        st.markdown("##### Visual Inspection & Structural Overlays")
        view_opt = st.radio(
            "Select Inspection Overlay:",
            [
                "Labeled Outlines on Current Satellite Photo",
                "Thermal Structural Dissimilarity Heatmap",
                "Binary Building Footprint Mask"
            ],
            horizontal=True
        )

        ov_col1, ov_col2 = st.columns(2)
        with ov_col1:
            st.image(
                timeline["p1_path"],
                caption=f"Baseline Reference — {timeline['dates']['epoch_1']}",
                width="stretch"
            )
        with ov_col2:
            if view_opt == "Labeled Outlines on Current Satellite Photo":
                st.image(
                    det_res["annotated_img"],
                    caption=f"Current Photo with Detected Buildings — {timeline['dates']['epoch_3']}",
                    width="stretch"
                )
            elif view_opt == "Thermal Structural Dissimilarity Heatmap":
                st.image(
                    det_res["heatmap_img"],
                    caption="Thermal Structural Difference Intensity (Warmer colors = Major structural shifts)",
                    width="stretch"
                )
            else:
                st.image(
                    det_res["binary_mask"],
                    caption="Extracted Binary Building Footprint Mask",
                    width="stretch"
                )

        # ── Itemized Building Inventory Table ──
        st.markdown("---")
        st.markdown("##### Itemized Inventory of Detected Structures")

        if det_res["inventory"]:
            df_bldgs = pd.DataFrame(det_res["inventory"])
            st.dataframe(df_bldgs, width="stretch", hide_index=True)

            fig_bar = px.bar(
                df_bldgs,
                x="Building ID",
                y="Footprint Area (m²)",
                title=f"New Building Footprint Breakdown — {st.session_state['active_area_name']}",
                labels={"Building ID": "Structure ID", "Footprint Area (m²)": "Footprint Area (Square Meters)"},
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

        # ── Municipal Report Export ──
        with st.expander("Municipal Intelligence Audit Report", expanded=False):
            rep_text = f"""================================================================================
SPACE NETRA — MUNICIPAL BUILDING CHANGE INTELLIGENCE AUDIT
Municipality: Chhatrapati Sambhajinagar (Aurangabad), Maharashtra, India
Operational Session: Calibrated Workstation Run
================================================================================
GEOGRAPHIC & SPATIAL PARAMETERS:
- Sector Designation         : {st.session_state['active_area_name']}
- Sector Classification      : {cat_type}
- Geographic Center Lat/Lon  : {cur_lat:.6f}° N, {cur_lon:.6f}° E
- Coordinate Reference (CRS) : EPSG:4326 (WGS 84)
- Web Mercator Tile Grid     : Zoom {zoom_val} (X={timeline['x']}, Y={timeline['y']})
- Metric Ground Sampling     : {det_res['gsd_m']} meters / pixel
--------------------------------------------------------------------------------
TEMPORAL IMAGERY PROVENANCE:
- Provenance Classification : {'VERIFIED_REAL (Authentic Satellite Photography)' if timeline.get('is_real') else 'BENCHMARK_TEST_FIXTURE'}
- Epoch 1 (Historical)       : {timeline['dates']['epoch_1']} (Esri World Imagery Wayback)
- Epoch 2 (Mid-Stage)        : {timeline['dates']['epoch_2']} (Esri World Imagery Wayback)
- Epoch 3 (Current)          : {timeline['dates']['epoch_3']} (High-Resolution Optical Satellite)
--------------------------------------------------------------------------------
BUILDING DETECTION METRICS:
- Detected New Structures    : {det_res['building_count']} distinct units
- Total New Built-Up Footprint: {det_res['total_built_area_sq_m']:,} m² ({det_res['total_built_area_ha']} Hectares)
- Built-Up Expansion Density : {det_res['building_density_pct']}% of analyzed ground
- Minimum Area Floor Filter  : {min_bldg_area} m²
- Operational Verdict        : {'ACTION: Significant New Building Construction' if det_res['building_count'] > 0 else 'STABLE: No Structural Alteration Detected'}
================================================================================"""
            st.code(rep_text, language="text")
            st.download_button(
                "Download Municipal Report (.txt)",
                rep_text,
                file_name=f"space_netra_{st.session_state['active_area_name'].lower().replace(' ', '_')}_audit.txt",
                width="stretch"
            )

    else:
        st.info("Click 'Run Optical Detection' above to fetch imagery and execute building change detection.")
