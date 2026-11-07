"""
app.py
======
Space Netra — Geospatial Change Intelligence Workstation
Coverage: Chhatrapati Sambhajinagar (Aurangabad), Maharashtra

Central application controller and router. Provides strict page-level isolation
and modular view orchestration across 5 analytical subsystems:
1. Overview
2. Optical Change Detection
3. ML Change Detection
4. Sentinel-2 Analysis
5. Semantic Archive Search
"""

import streamlit as st

# Import theme & modular view controllers
from views.theme import inject_theme_css
from views.overview import render_overview
from views.optical_detection import render_optical_detection
from views.ml_detection import render_ml_detection
from views.sentinel2_analysis import render_sentinel2_analysis
from views.semantic_search import render_semantic_search

# ─────────────────────────────────────────────────────────────
# 1. PAGE CONFIGURATION & THEME
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Space Netra — Geospatial Intelligence Workstation",
    page_icon="🛰️",
    initial_sidebar_state="expanded"
)

# Apply unified dark charcoal/navy workstation theme
inject_theme_css()

# ─────────────────────────────────────────────────────────────
# 2. SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="padding: 10px 0 16px 0;">
    <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; letter-spacing: -0.01em;">
        SPACE NETRA
    </div>
    <div style="font-size: 0.72rem; color: #38bdf8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
        Geospatial Change Platform
    </div>
</div>
""", unsafe_allow_html=True)

PAGE_OPTIONS = [
    "Overview",
    "Optical Change Detection",
    "ML Change Detection",
    "Sentinel-2 Analysis",
    "Semantic Archive Search"
]

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Overview"

selected_page = st.sidebar.radio(
    "Navigation",
    PAGE_OPTIONS,
    index=PAGE_OPTIONS.index(st.session_state["active_page"]),
    label_visibility="collapsed"
)

st.session_state["active_page"] = selected_page
st.sidebar.markdown("---")

# ─────────────────────────────────────────────────────────────
# 3. STRICT PAGE DISPATCH & VIEW ORCHESTRATION
# ─────────────────────────────────────────────────────────────
if selected_page == "Overview":
    render_overview()

elif selected_page == "Optical Change Detection":
    render_optical_detection()

elif selected_page == "ML Change Detection":
    render_ml_detection()

elif selected_page == "Sentinel-2 Analysis":
    render_sentinel2_analysis()

elif selected_page == "Semantic Archive Search":
    render_semantic_search()
