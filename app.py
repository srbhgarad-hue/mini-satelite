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
import requests
from PIL import Image
from change_detector import calculate_change_confidence

st.set_page_config(layout="wide", page_title="Space Netra - Live Satellite Intelligence Engine")

st.title("🛰️ Space Netra: Live Satellite Imagery Change Detection")

# Sidebar Setup
st.sidebar.header("🕹️ Control Panel")
mode = st.sidebar.selectbox("Select Action Mode", [
    "🗺️ Interactive Map AOI Selection", 
    "🔍 Vector-Based Semantic Search"
])

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    index = faiss.read_index("vector_db/spatial.index")
    metadata = np.load("vector_db/metadata.npy", allow_pickle=True)
    return model, preprocess, index, metadata, device

def deg2num(lat_deg, lon_deg, zoom=16):
    lat_rad = math.radians(lat_deg)
    
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def fetch_esri_wayback_tiles(lat, lon, zoom=16):
    x, y = deg2num(lat, lon, zoom)
    
    t1_url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
    t2_url = f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={zoom}"

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    try:
        r1 = requests.get(t1_url, headers=headers, timeout=10)
        r2 = requests.get(t2_url, headers=headers, timeout=10)
        
        if r1.status_code == 200 and r2.status_code == 200:
            with open("temp_real_t1.jpg", "wb") as f:
                f.write(r1.content)
            with open("temp_real_t2.jpg", "wb") as f:
                f.write(r2.content)
            return "temp_real_t1.jpg", "temp_real_t2.jpg"
    except Exception as e:
        st.error(f"Live Satellite Fetching Exception: {e}")
        
    return None, None

if mode == "🗺️ Interactive Map AOI Selection":
    st.sidebar.subheader("🎛️ Analysis Adjustments")
    spatial_res = st.sidebar.slider("Sensor Resolution (m/px)", 1.0, 30.0, 10.0)
    sensitivity = st.sidebar.slider("Detection Sensitivity", 30, 150, 60)
    overlay_alpha = st.sidebar.slider("Heatmap Transparency", 0.1, 1.0, 0.5)

    search_location = st.text_input("📍 Search Location (e.g. Prozone Mall Chhatrapati Sambhajinagar)", "Prozone Mall Chhatrapati Sambhajinagar")

    default_lat, default_lon = 19.8732, 75.3614
    
    if search_location:
        try:
            geolocator = Nominatim(user_agent="space_netra_app")
            location = geolocator.geocode(search_location)
            if location:
                default_lat, default_lon = location.latitude, location.longitude
                st.success(f"📍 Real Coordinates Located: Lat {default_lat:.4f}, Lon {default_lon:.4f}")
        except Exception as e:
            st.info("Using Fallback Coordinates for Selected Area.")

    st.subheader("🗺️ Satellite Targeted Area of Interest")
    m = folium.Map(location=[default_lat, default_lon], zoom_start=16, tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", attr="Google Satellite")
    folium.Marker([default_lat, default_lon], popup=f"AOI: {search_location}", tooltip="Selected Site").add_to(m)
    st_folium(m, width=1200, height=320)

    st.write("---")
    st.subheader("🛰️ Multi-Temporal Satellite Change Detection Engine")

    with st.spinner("Fetching Satellite Archive Imagery..."):
        t1_path, t2_path = fetch_esri_wayback_tiles(default_lat, default_lon)

    if t1_path and t2_path:
        res = calculate_change_confidence(t1_path, t2_path, resolution_m=spatial_res, sensitivity=sensitivity, overlay_alpha=overlay_alpha)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Change Score", f"{res['confidence']}%")
        m2.metric("Changed Surface Ratio", f"{res['change_ratio']}%")
        m3.metric("Est. Area Impacted", f"{res['area_hectares']} Ha")
        m4.metric("Identified Clusters", f"{res['bbox_count']} Zones")
        m5.metric("Status", "Real Change" if res['is_real_change'] else "Stable Zone")

        tab1, tab2, tab3 = st.tabs(["🖼️ Spatial Multi-View & Highlighting", "📊 Area Analytics", "📄 Executive Report"])

        with tab1:
            view_mode = st.radio("Select Output View:", ["Bounding Box Clusters", "Colored Heatmap Overlay", "Binary Change Mask"], horizontal=True)
            
            c1, c2, c3 = st.columns(3)
            
            c1.image(t1_path, caption="📅 T1 Historical Base: Jan 2024 (ArcGIS Archive)", use_container_width=True)
            c2.image(t2_path, caption="📅 T2 Recent Observation: Sep 2026 (Google Satellite)", use_container_width=True)
            
            if view_mode == "Bounding Box Clusters":
                c3.image(res['bbox_img'], caption="🟨 High-Contrast Bounding Boxes (Neon Bboxes + Red Fills)", use_container_width=True)
            elif view_mode == "Colored Heatmap Overlay":
                c3.image(res['overlay'], caption="🔴 Heatmap View (Red/Yellow = Structural Shifts)", use_container_width=True)
            else:
                c3.image(res['thresh_mask'], caption="⚪ High-Contrast Binary Difference Mask", use_container_width=True)

        with tab2:
            cg1, cg2 = st.columns(2)
            with cg1:
                st.subheader("Land Surface Distribution")
                fig_pie = px.pie(
                    values=[res['change_ratio'], res['unchanged_ratio']],
                    names=['Changed Surface', 'Unchanged Area'],
                    color_discrete_sequence=['#E63946', '#2A9D8F'],
                    hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with cg2:
                st.subheader("Quantitative Metric Breakdown")
                fig_bar = px.bar(
                    x=["Total Zone (Sq Km)", "Changed Area (Sq Km)", "Impacted (Hectares)"],
                    y=[res['area_sq_km'] / (res['change_ratio']/100 if res['change_ratio']>0 else 1), res['area_sq_km'], res['area_hectares']],
                    color=["Total", "Changed", "Hectares"],
                    color_discrete_map={"Total": "#457B9D", "Changed": "#E63946", "Hectares": "#F4A261"}
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        with tab3:
            st.subheader("📋 SIH AI Analytical Summary")
            report_text = f"""
            ==================================================
            SPACE NETRA LIVE GEOSPATIAL INTELLIGENCE REPORT
            ==================================================
            - Target Location Search: {search_location}
            - Temporal Span: T1 (Jan 2024 Archive) vs T2 (Sep 2026 Recent)
            - Coordinates Center: Lat {default_lat:.4f}, Lon {default_lon:.4f}
            - Detection Status: {'ACTION REQUIRED: Structural Change Detected' if res['is_real_change'] else 'NORMAL: No Major Change'}
            - Change Confidence Score: {res['confidence']}%
            - Impacted Area: {res['area_hectares']} Hectares ({res['area_sq_km']} sq km)
            - Hotspot Clusters: {res['bbox_count']} zones
            ==================================================
            """
            st.code(report_text, language="text")
            st.download_button("📥 Download Analysis Report", report_text, file_name="space_netra_report.txt")

elif mode == "🔍 Vector-Based Semantic Search":
    st.subheader("🔍 Localized Spatial Archive Semantic Search")
    try:
        model, preprocess, index, metadata, device = load_models()
        query_text = st.text_input("Enter Prompt Query (e.g. 'Prozone Mall Sambhajinagar', 'airport', 'lake'):")
        
        if st.button("Search Spatial Archive"):
            text_tokens = clip.tokenize([query_text]).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
            distances, indices = index.search(text_features.cpu().numpy().astype("float32"), k=3)
            
            cols = st.columns(3)
            for i, idx in enumerate(indices[0]):
                item = metadata[idx]
                img = Image.open(item["path"])
                with cols[i]:
                    st.image(img, caption=f"Matched: {item['filename']} | Score: {round(float(distances[0][i])*100, 2)}%", use_container_width=True)
    except Exception as e:
        st.error(f"Index Load Error: Pehle terminal me 'python ingest.py' run karke database index update karein. Details: {e}")