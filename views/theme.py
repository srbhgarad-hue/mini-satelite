"""
views/theme.py
==============
Design System & CSS Styling for Space Netra Geospatial Workstation.
Provides dark charcoal/navy palette, restrained teal/blue accents,
symmetric card layouts, and responsive spacing.
"""

import streamlit as st

def inject_theme_css():
    """Injects workstation stylesheet for professional GIS appearance."""
    st.markdown("""
    <style>
        /* Base Canvas */
        .stApp {
            background-color: #0b0f19;
            color: #f1f5f9;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0f172a !important;
            border-right: 1px solid #1e293b !important;
        }

        /* Workstation Page Header */
        .workstation-header {
            background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 18px 22px;
            margin-bottom: 20px;
        }
        .workstation-title {
            font-size: 1.45rem;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: -0.015em;
            margin: 0 0 4px 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .workstation-subtitle {
            font-size: 0.85rem;
            color: #94a3b8;
            margin: 0;
            line-height: 1.4;
        }

        /* Workstation Cards */
        .workstation-card {
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 16px 18px;
            margin-bottom: 16px;
        }
        .workstation-card-title {
            font-size: 0.88rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #38bdf8;
            margin-bottom: 10px;
        }

        /* Monospace Coordinate & Metric Chips */
        .meta-chip {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            background: #111827;
            border: 1px solid #334155;
            padding: 4px 10px;
            border-radius: 5px;
            color: #38bdf8;
            font-size: 0.82rem;
            font-weight: 500;
            display: inline-block;
        }
        .tag-chip {
            font-size: 0.76rem;
            color: #94a3b8;
            background: #1e293b;
            border: 1px solid #334155;
            padding: 3px 8px;
            border-radius: 4px;
            display: inline-block;
        }

        /* Status Pills */
        .status-pill-ready {
            background: rgba(16, 185, 129, 0.12);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.35);
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-block;
        }
        .status-pill-standby {
            background: rgba(56, 189, 248, 0.10);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.25);
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-block;
        }
        .status-pill-warning {
            background: rgba(245, 158, 11, 0.12);
            color: #f59e0b;
            border: 1px solid rgba(245, 158, 11, 0.35);
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-block;
        }

        /* Streamlit Metrics */
        div[data-testid="stMetricValue"] {
            color: #f8fafc !important;
            font-weight: 700 !important;
            font-size: 1.45rem !important;
        }
        div[data-testid="stMetricLabel"] {
            color: #94a3b8 !important;
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.04em !important;
        }

        /* Clean Image Cards with Equal Heights */
        div[data-testid="stImage"] {
            border-radius: 6px;
            overflow: hidden;
            border: 1px solid #1e293b;
            background: #0f172a;
        }
        div[data-testid="stImage"] figcaption {
            color: #94a3b8 !important;
            font-size: 0.80rem !important;
            padding: 6px 4px !important;
        }

        /* Button Styling */
        button[kind="primary"] {
            background-color: #0284c7 !important;
            border-color: #0284c7 !important;
            color: #ffffff !important;
            font-weight: 600 !important;
        }
        button[kind="primary"]:hover {
            background-color: #0369a1 !important;
            border-color: #0369a1 !important;
        }

        /* Navigation Radio */
        div[data-testid="stRadio"] > div {
            gap: 4px;
        }
    </style>
    """, unsafe_allow_html=True)

def render_page_header(title: str, subtitle: str, badge_text: str = None, badge_type: str = "ready"):
    """Renders a standardized page header across all modules."""
    badge_html = ""
    if badge_text:
        cls_map = {
            "ready": "status-pill-ready",
            "standby": "status-pill-standby",
            "warning": "status-pill-warning"
        }
        cls_name = cls_map.get(badge_type, "status-pill-ready")
        badge_html = f'<span class="{cls_name}">{badge_text}</span>'

    st.markdown(f"""
    <div class="workstation-header">
        <div class="workstation-title">
            <span>{title}</span>
            {badge_html}
        </div>
        <p class="workstation-subtitle">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)
