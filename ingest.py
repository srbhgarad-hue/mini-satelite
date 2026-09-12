import os
import glob
import numpy as np
import torch
from PIL import Image
import faiss
import clip

# Load CLIP Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def process_and_index_images(image_dir="data/raw/", index_save_path="vector_db/spatial.index"):
    image_paths = glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(os.path.join(image_dir, "*.jpg"))
    
    if not image_paths:
        print("Error: data/raw/ folder mein koi images nahi mili!")
        return

    embeddings = []
    metadata = []

    print(f"Processing {len(image_paths)} images...")

    for idx, path in enumerate(image_paths):
        img = Image.open(path).convert("RGB")
        tensor_img = preprocess(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            img_feature = model.encode_image(tensor_img)
            img_feature /= img_feature.norm(dim=-1, keepdim=True)
            
        embeddings.append(img_feature.cpu().numpy()[0])
        metadata.append({"id": idx, "path": path, "filename": os.path.basename(path)})

    embeddings_np = np.array(embeddings).astype("float32")
    dimension = embeddings_np.shape[1]
    
    # FAISS Index
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np)
    
    os.makedirs(os.path.dirname(index_save_path), exist_ok=True)
    faiss.write_index(index, index_save_path)
    np.save("vector_db/metadata.npy", metadata)
    print(f"Indexed {len(image_paths)} images successfully!")

if __name__ == "__main__":
    process_and_index_images()import streamlit as st
import plotly.express as px
import numpy as np
import clip
import torch
import faiss
from PIL import Image
from change_detector import calculate_change_confidence

st.set_page_config(layout="wide", page_title="Space Netra Intelligence Engine")

st.title("🛰️ Space Netra: Satellite Change Detection & Semantic Intelligence")

# Sidebar Setup
st.sidebar.header("🕹️ Control Panel")
mode = st.sidebar.selectbox("Select Action Mode", ["Multi-Temporal Change Analysis", "Semantic Search"])

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    index = faiss.read_index("vector_db/spatial.index")
    metadata = np.load("vector_db/metadata.npy", allow_pickle=True)
    return model, preprocess, index, metadata, device

if mode == "Multi-Temporal Change Analysis":
    st.sidebar.subheader("🎛️ Analysis Adjustments")
    spatial_res = st.sidebar.slider("Sensor Spatial Resolution (m/px)", 1.0, 30.0, 10.0)
    sensitivity = st.sidebar.slider("Detection Sensitivity Threshold", 30, 150, 80)
    overlay_alpha = st.sidebar.slider("Heatmap Overlay Opacity", 0.1, 1.0, 0.4)

    col1, col2 = st.columns(2)
    with col1:
        file_t1 = st.file_uploader("Upload T1 (Before Image)", type=["png", "jpg"])
    with col2:
        file_t2 = st.file_uploader("Upload T2 (After Image)", type=["png", "jpg"])

    if file_t1 and file_t2:
        with open("temp_t1.png", "wb") as f: f.write(file_t1.getbuffer())
        with open("temp_t2.png", "wb") as f: f.write(file_t2.getbuffer())

        res = calculate_change_confidence("temp_t1.png", "temp_t2.png", resolution_m=spatial_res, sensitivity=sensitivity, overlay_alpha=overlay_alpha)

        # Executive Metrics Header
        st.write("---")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Confidence Score", f"{res['confidence']}%")
        m2.metric("Changed Ratio", f"{res['change_ratio']}%")
        m3.metric("Est. Changed Area", f"{res['area_hectares']} Ha")
        m4.metric("Detected Clusters", f"{res['bbox_count']} Zones")
        m5.metric("Status", "Real Change" if res['is_real_change'] else "Noise / Micro Change")

        # Tabs Interface
        tab1, tab2, tab3 = st.tabs(["🖼️ Multi-View Visuals", "📊 Quantitative Graphs", "📄 Executive Summary Report"])

        with tab1:
            view_mode = st.radio("Select Output View:", ["Colored Heatmap Overlay", "Bounding Box Detection", "Binary Change Mask"], horizontal=True)
            
            c1, c2, c3 = st.columns(3)
            c1.image("temp_t1.png", caption="T1 Image (Before)")
            c2.image("temp_t2.png", caption="T2 Image (After)")
            
            if view_mode == "Colored Heatmap Overlay":
                c3.image(res['overlay'], caption="Heatmap Overlay View")
            elif view_mode == "Bounding Box Detection":
                c3.image(res['bbox_img'], caption="Bounding Box Highlight View")
            else:
                c3.image(res['thresh_mask'], caption="Binary Change Mask")

        with tab2:
            cg1, cg2 = st.columns(2)
            with cg1:
                st.subheader("Land Area Share")
                fig_pie = px.pie(
                    values=[res['change_ratio'], res['unchanged_ratio']],
                    names=['Changed Surface', 'Unchanged Surface'],
                    color_discrete_sequence=['#E63946', '#2A9D8F'],
                    hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with cg2:
                st.subheader("Area Metrics Breakdown")
                fig_bar = px.bar(
                    x=["Total Area (Sq Km)", "Changed Area (Sq Km)", "Impacted Area (Hectares)"],
                    y=[res['area_sq_km'] / (res['change_ratio']/100 if res['change_ratio']>0 else 1), res['area_sq_km'], res['area_hectares']],
                    color=["Total", "Changed", "Hectares"],
                    color_discrete_map={"Total": "#457B9D", "Changed": "#E63946", "Hectares": "#F4A261"}
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        with tab3:
            st.subheader("📋 SIH AI Analytical Summary")
            report_text = f"""
            ==================================================
            SPACE NETRA AUTOMATED CHANGE INTELLIGENCE REPORT
            ==================================================
            - Detection Status: {'ACTION REQUIRED: Real Change Detected' if res['is_real_change'] else 'NORMAL: Micro/No Structural Change'}
            - Confidence Level: {res['confidence']}%
            - Estimated Surface Area Changed: {res['area_hectares']} Hectares ({res['area_sq_km']} sq km)
            - Total Distinct Change Hotspots Identified: {res['bbox_count']} clusters
            - Resolution Parameter Applied: {spatial_res} meters per pixel
            - Sensitivity Threshold Applied: {sensitivity}
            ==================================================
            """
            st.code(report_text, language="text")
            st.download_button("📥 Download Analysis Report", report_text, file_name="space_netra_report.txt")

elif mode == "Semantic Search":
    st.subheader("🔍 Vector-Based Semantic Search Engine")
    try:
        model, preprocess, index, metadata, device = load_models()
        query_text = st.text_input("Enter Prompt Query (e.g. 'coastal area', 'airport', 'lake'):")
        
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
                    st.image(img, caption=f"Match Confidence: {round(float(distances[0][i])*100, 2)}%", use_container_width=True)
    except Exception as e:
        st.error(f"Index Load Error: Pehle terminal me 'python ingest.py' run karke database index create karein. Details: {e}")