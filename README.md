# 🛰️ Space Netra: Live Satellite Intelligence & Change Detection Engine

**Space Netra** is an AI-powered geospatial intelligence and multi-temporal satellite change detection platform built for the **Smart India Hackathon (SIH)**. It combines computer vision (SSIM structural anomaly detection) and deep learning (OpenAI CLIP + FAISS vector indexing) to provide real-time spatial monitoring and natural-language satellite archive search.

---

## ✨ Key Features

1. **🗺️ Interactive Map AOI Selection & Live Analysis**
   - Geocode any location name or landmark across the globe with OpenStreetMap/Nominatim.
   - Interactive Folium satellite view to pinpoint the Area of Interest (AOI).
   - Live multi-temporal imagery ingestion (ArcGIS Historical Archives vs. Google Satellite feeds).

2. **🔍 Advanced Change Detection Engine**
   - **Structural Similarity (SSIM)** & morphological dilation for noise reduction.
   - **High-Visibility Visualizations**:
     - 🟨 Neon Yellow Bounding Boxes & tagged change cluster counters.
     - 🔴 JET Heatmap overlays showing intensity of ground modifications.
     - ⚪ Binary Difference Masks.
   - **Quantitative Metrics**: Change score (%), surface change ratio, impacted area in Hectares / Sq Km, and cluster count.

3. **📊 Quantitative Analytics & Executive Reporting**
   - Plotly dynamic pie charts and distribution metrics.
   - One-click exportable AI Geospatial Intelligence Reports (`.txt`).

4. **🔎 Vector-Based Semantic Search**
   - Natural language search over satellite image archives using **CLIP (ViT-B/32)** embeddings.
   - Ultra-fast nearest neighbor spatial retrieval powered by **FAISS**.

---

## 🏗️ Architecture Pipeline

```
[Satellite Tile Services / Local Archive]
               │
               ▼
      [data/raw/ Images]
               │
               ├───► [ingest.py] ───► CLIP ViT-B/32 ───► FAISS Index (vector_db/)
               │
               ▼
           [app.py] (Streamlit Dashboard)
               │
               ├── Mode 1: Live AOI Fetch ──► [change_detector.py (SSIM + Contours)]
               └── Mode 2: Text Search   ──► [vector_db/ Spatial Search]
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.9+
- Git

### 2. Installation
Clone the repository and install all dependencies:

```bash
git clone https://github.com/vinaya-git-code/SIH-satelite.git
cd SIH-satelite
pip install -r requirements.txt
```

### 3. (Optional) Download Sample Imagery & Ingest Vector DB
To populate the semantic search spatial archive:

```bash
# Download sample high-res satellite tiles
python download_images.py

# Generate CLIP embeddings and FAISS index
python ingest.py
```

### 4. Launch the Application
Run the Streamlit dashboard:

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
SIH-satelite/
│
├── app.py                  # Main Streamlit Geospatial Web Application
├── change_detector.py      # Core Computer Vision Change Detection Engine (SSIM)
├── ingest.py               # CLIP + FAISS Vector Ingestion Script
├── download_images.py      # Automated Satellite Sample Tile Downloader
├── requirements.txt        # Python Dependencies Manifest
├── .gitignore              # Ignored Build/Temp Artifacts
│
├── data/
│   └── raw/                # Satellite image archive (PNG / JPG)
│
├── vector_db/
│   ├── spatial.index       # FAISS Vector Index
│   └── metadata.npy        # Image Metadata Store
│
└── temp/                   # Temporary cache directory for live tile processing
```

---

## 🧰 Tech Stack

- **Frontend / UI**: [Streamlit](https://streamlit.io/), [Folium](https://python-visualization.github.io/folium/), [Streamlit-Folium](https://github.com/randyzwitch/streamlit-folium)
- **Computer Vision & Metrics**: [OpenCV](https://opencv.org/), [scikit-image](https://scikit-image.org/)
- **Deep Learning & Vector Search**: [OpenAI CLIP](https://github.com/openai/CLIP), [PyTorch](https://pytorch.org/), [FAISS](https://github.com/facebookresearch/faiss)
- **Geocoding & Visualization**: [GeoPy](https://geopy.readthedocs.io/), [Plotly](https://plotly.com/)