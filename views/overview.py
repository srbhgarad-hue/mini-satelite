"""
views/overview.py
=================
Space Netra Platform Overview & System Health Status.
Displays workstation identity, scope, operational capabilities,
and module status matrix. Contains no heavy computation or detection leakage.
"""

import streamlit as st
from views.theme import render_page_header

def render_overview():
    """Renders the central platform overview and system status matrix."""
    render_page_header(
        title="Space Netra — Geospatial Intelligence Workstation",
        subtitle="Autonomous Urban Growth Monitoring, High-Resolution Building Infill Detection & Remote-Sensing Analysis",
        badge_text="Operational Platform",
        badge_type="ready"
    )

    # Mission & Scope Card
    st.markdown("""
    <div class="workstation-card">
        <div class="workstation-card-title">Geospatial Scope & Objectives</div>
        <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6;">
            Space Netra is a specialized geospatial intelligence workstation designed for municipal remote sensing
            over <strong>Chhatrapati Sambhajinagar (Aurangabad), Maharashtra</strong> (EPSG:4326 / UTM Zone 43N).
            The platform provides independent, non-destructive methodologies to track urban expansion, detect newly constructed
            structures, monitor large-scale land modifications, and query spatial image archives using natural language.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Core Analytical Modules Grid
    st.markdown("##### Operational Modules")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="workstation-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 700; color: #f8fafc; font-size: 0.95rem;">1. Optical Change Detection</span>
                <span class="status-pill-ready">Photographic</span>
            </div>
            <p style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5; margin-bottom: 12px;">
                Three-stage photographic timeline analysis comparing historical baselines (2016, 2020) with current high-resolution
                optical satellite imagery. Employs Canny structural gradient analysis and structural dissimilarity (SSIM)
                to isolate discrete building roofs with metric ground sampling distance (GSD).
            </p>
            <div style="display: flex; gap: 6px; flex-wrap: wrap;">
                <span class="tag-chip">Resolution: ~1.12 m/px</span>
                <span class="tag-chip">Baseline: Esri Wayback</span>
                <span class="tag-chip">CRS: EPSG:4326</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="workstation-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 700; color: #f8fafc; font-size: 0.95rem;">2. ML Change Detection</span>
                <span class="status-pill-ready">Deep Learning</span>
            </div>
            <p style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5; margin-bottom: 12px;">
                Experimental 6-channel bitemporal U-Net deep-learning architecture trained on S2Looking and LEVIR-CD benchmark datasets.
                Generates continuous sigmoid probability heatmaps and dynamic binary change decision masks with interactive threshold control.
            </p>
            <div style="display: flex; gap: 6px; flex-wrap: wrap;">
                <span class="tag-chip">U-Net 6-Channel</span>
                <span class="tag-chip">S2Looking + LEVIR-CD</span>
                <span class="tag-chip">Probability Maps</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="workstation-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 700; color: #f8fafc; font-size: 0.95rem;">3. Sentinel-2 Analysis</span>
                <span class="status-pill-standby">Multispectral</span>
            </div>
            <p style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5; margin-bottom: 12px;">
                Physics-based 10m Level-2A Bottom-Of-Atmosphere (BOA) surface reflectance analysis over industrial growth corridors.
                Features Scene Classification Layer (SCL) cloud/shadow screening, Delta-NDVI vegetation transitions, and area conservation math.
            </p>
            <div style="display: flex; gap: 6px; flex-wrap: wrap;">
                <span class="tag-chip">GSD: 10m Ground Grid</span>
                <span class="tag-chip">SCL Cloud Masking</span>
                <span class="tag-chip">UTM Zone 43N</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="workstation-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 700; color: #f8fafc; font-size: 0.95rem;">4. Semantic Archive Search</span>
                <span class="status-pill-ready">Vector Search</span>
            </div>
            <p style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5; margin-bottom: 12px;">
                Natural-language concept retrieval across local spatial archives using OpenAI CLIP (ViT-B/32) multi-modal embeddings
                and a persistent FAISS vector index. Enables rapid indexing and retrieval of satellite imagery features.
            </p>
            <div style="display: flex; gap: 6px; flex-wrap: wrap;">
                <span class="tag-chip">CLIP ViT-B/32</span>
                <span class="tag-chip">FAISS Index</span>
                <span class="tag-chip">512-Dim Vector Space</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # System Status & Provenance Matrix
    st.markdown("##### System Status & Data Provenance Matrix")
    st.markdown("""
    <div class="workstation-card" style="padding: 0; overflow: hidden;">
        <table style="width: 100%; border-collapse: collapse; font-size: 0.84rem; color: #cbd5e1;">
            <thead>
                <tr style="background: #111827; border-bottom: 1px solid #1e293b; text-align: left;">
                    <th style="padding: 10px 14px; color: #94a3b8; font-weight: 600;">System Subsystem</th>
                    <th style="padding: 10px 14px; color: #94a3b8; font-weight: 600;">Status</th>
                    <th style="padding: 10px 14px; color: #94a3b8; font-weight: 600;">Operational Source & Provenance</th>
                    <th style="padding: 10px 14px; color: #94a3b8; font-weight: 600;">Coverage / Target</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid #1e293b;">
                    <td style="padding: 10px 14px; font-weight: 600; color: #f8fafc;">Optical Streamer</td>
                    <td style="padding: 10px 14px;"><span class="status-pill-ready">Active</span></td>
                    <td style="padding: 10px 14px;">Esri Wayback Historical + Current High-Res Satellite</td>
                    <td style="padding: 10px 14px;">7 Chhatrapati Sambhajinagar Growth Hubs</td>
                </tr>
                <tr style="border-bottom: 1px solid #1e293b;">
                    <td style="padding: 10px 14px; font-weight: 600; color: #f8fafc;">Bitemporal U-Net ML</td>
                    <td style="padding: 10px 14px;"><span class="status-pill-ready">Active</span></td>
                    <td style="padding: 10px 14px;">S2Looking + LEVIR-CD Mixed Model (Default Epoch 5)</td>
                    <td style="padding: 10px 14px;">256×256 6-Channel Inference</td>
                </tr>
                <tr style="border-bottom: 1px solid #1e293b;">
                    <td style="padding: 10px 14px; font-weight: 600; color: #f8fafc;">Sentinel-2 Analytics</td>
                    <td style="padding: 10px 14px;"><span class="status-pill-standby">Test Fixtures</span></td>
                    <td style="padding: 10px 14px;">Authentic L2A Metadata; Calibrated Benchmark Arrays</td>
                    <td style="padding: 10px 14px;">Shendra MIDC Industrial AOI</td>
                </tr>
                <tr>
                    <td style="padding: 10px 14px; font-weight: 600; color: #f8fafc;">Semantic Search</td>
                    <td style="padding: 10px 14px;"><span class="status-pill-ready">Active</span></td>
                    <td style="padding: 10px 14px;">CLIP ViT-B/32 + FAISS Index (spatial.index)</td>
                    <td style="padding: 10px 14px;">Local Remote-Sensing Archive</td>
                </tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # Navigation Guidance Callout
    st.info("Select a workflow from the left sidebar navigation to begin an analysis.")
