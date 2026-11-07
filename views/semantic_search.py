"""
views/semantic_search.py
========================
Semantic Archive Search via OpenAI CLIP (ViT-B/32) and FAISS.
Enables natural-language concept search across indexed satellite image patches.
Strictly preserves vector_db and vector_db_backup assets.
"""

import os
import numpy as np
from PIL import Image
import streamlit as st
import torch
import clip
import faiss

from views.theme import render_page_header

@st.cache_resource
def load_clip_faiss():
    """Loads CLIP model and reads FAISS vector index."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    index_path = os.path.join("vector_db", "spatial.index")
    meta_path = os.path.join("vector_db", "metadata.npy")
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        return None, None, None, None, device
    index = faiss.read_index(index_path)
    metadata = np.load(meta_path, allow_pickle=True)
    return model, preprocess, index, metadata, device

def render_semantic_search():
    """Renders the semantic archive vector search module."""
    render_page_header(
        title="Semantic Archive Search — Multi-Modal Vector Retrieval",
        subtitle="Natural-Language Image Retrieval over Spatial Image Archive via OpenAI CLIP (ViT-B/32) & FAISS",
        badge_text="Vector Pipeline",
        badge_type="ready"
    )

    # ── Sidebar Controls (Isolated to Semantic Search) ──
    st.sidebar.markdown("<div class='workstation-card-title'>Retrieval Parameters</div>", unsafe_allow_html=True)

    top_k = st.sidebar.slider(
        "Top-K Matches:",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
        help="Number of nearest semantic neighbor image patches to retrieve."
    )

    # ── Provenance & Archive Scope Card ──
    st.markdown("""
    <div class="workstation-card">
        <div style="font-weight: 700; color: #38bdf8; font-size: 0.88rem; margin-bottom: 6px;">
            Dataset Provenance & Index Information
        </div>
        <div style="font-size: 0.80rem; color: #cbd5e1; line-height: 1.5;">
            The spatial archive indexes benchmark remote-sensing scene classification image patches
            located in <code>data/raw/</code> (airport, forest, harbor, stadium, urban) transformed into
            512-dimensional vector embeddings. This subsystem performs semantic visual concept indexing,
            distinct from physical multi-temporal change detection.
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        model, preprocess, index, metadata, device = load_clip_faiss()
        if index is None or metadata is None:
            st.warning("Vector database not found at 'vector_db/'. Preserved database required.")
            return

        # Query Input
        st.markdown("##### Query Spatial Archive")
        col_q, col_b = st.columns([4, 1])
        with col_q:
            query_str = st.text_input(
                "Enter concept or description:",
                value="urban area",
                placeholder="e.g. urban area, harbor, airport runway, forest...",
                label_visibility="collapsed"
            )
        with col_b:
            search_clicked = st.button("Search Archive", type="primary", width="stretch")

        st.caption("Suggested queries: `urban area` • `harbor` • `airport` • `stadium` • `forest`")

        if (search_clicked or query_str) and query_str.strip():
            with st.spinner("Embedding query and executing nearest neighbor search in FAISS..."):
                tokens = clip.tokenize([query_str]).to(device)
                with torch.no_grad():
                    text_feat = model.encode_text(tokens)
                    text_feat /= text_feat.norm(dim=-1, keepdim=True)
                k = min(len(metadata), top_k)
                distances, indices = index.search(text_feat.cpu().numpy().astype("float32"), k=k)

            st.markdown("---")
            st.markdown(f"##### Top {k} Nearest Spatial Image Matches for: '{query_str}'")

            cols = st.columns(k)
            for i, idx in enumerate(indices[0]):
                if idx < len(metadata):
                    item = metadata[idx]
                    if os.path.exists(item["path"]):
                        img = Image.open(item["path"])
                        score = round(float(distances[0][i]) * 100.0, 1)
                        with cols[i]:
                            st.image(img, caption=f"Score: {score}% | {item['filename']}", width="stretch")
                    else:
                        with cols[i]:
                            st.warning(f"File missing: {item['filename']}")

    except Exception as e:
        st.error(f"Semantic search execution error: {e}")
