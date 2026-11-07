# Dispatch Log

## 2026-11-07T16:12:06Z
From: Sentinel (f857d917-43f0-4567-92b3-b3af60351b5e)

Transform Space Netra into a professional, reliable production geospatial change-detection platform across five independent modules.

Working directory: D:\PythonProjects\mini-satelite
Integrity mode: development

## Requirements

### R1. Five-Page Application Separation and State Management
Separate the existing Streamlit application into five independent pages: Overview, Optical Change Detection, ML Change Detection, Sentinel-2 Analysis, and Semantic Archive Search. Each page must render only its own components and controls without duplicate sections, cross-page content leaks, repeated model loading, or shared session-state collisions. The Overview page must display only the Space Netra title, a short description, five navigation cards, and live module status.

### R2. GIS UI Overhaul and Layout Alignment
Implement a clean, professional dark-navy/charcoal GIS interface with subtle teal/blue accents, consistent typography, and clean spacing. Eliminate comparison image overlapping across two-stage and three-stage comparisons by displaying imagery in equal-width responsive cards in a single row (each containing the image, verified date, source, and concise caption) with the subsequent section starting strictly below the containers, avoiding negative margins or absolute positioning hacks. Contextualize sidebars so each page displays only its relevant controls, moving advanced settings to expanders.

### R3. Technical Pipeline Decoupling
Keep the four technical systems strictly independent:
- Optical Change Detection: Traditional computer vision comparison (image registration, SSIM, Canny, contours, area calculation); do not automatically classify or label all detected changes as buildings.
- ML Change Detection: Dedicated Space Netra U-Net binary change detection pipeline using the trained checkpoint at `models/spacenetra_s2looking_levir_20261030_031904_best.pth`. Support two-image upload/selection, validation, 6-channel concatenation input, tiled inference for large images, binary mask and overlay output with adjustable opacity, prediction download, and a compact expandable model benchmark information section with initial threshold set to 0.35.
- Sentinel-2 Analysis: Multispectral raster processing (NDVI, NDBI, NDWI) with real CRS, transform, pixel dimensions, and SCL/cloud masking, kept isolated from optical and ML workflows.
- Semantic Archive Search: Dedicated CLIP + FAISS text-to-image and image-to-image retrieval preserving existing indices and metadata; do not present similarity scores as change-detection confidence.

### R4. Data Provenance and Synthetic Data Isolation
Audit all visible metrics, charts, and provenance claims. Remove hardcoded, synthetic, or unverified numbers (including unbacked "New Buildings Found", "Built-Up Density", "Ground Resolution", "Expansion Level", and single-building oversized charts) and replace them with "Not available" or "Awaiting verified data". Isolate synthetic test fixtures under a visible developer-only TEST/SYNTHETIC label, ensuring synthetic data is never silently substituted for real imagery.

### R5. Sentinel-2 Interface Resolution and Operational Isolation
Resolve module interface mismatches, including `preprocessor.py` attribute errors (`clean_scene`). If verified Level-2A raster pixels are absent, mark and isolate the Sentinel-2 module as "Experimental — Awaiting verified raster data" with clean fixture separation and zero synthetic data masquerading as real analysis.

### R6. Data and Checkpoint Protection
Strictly preserve existing trained model checkpoints, original datasets, FAISS index files, FAISS metadata, and existing backups. Never overwrite or delete these assets, and do not execute `ingest.py`, `download_images.py`, or `download_satellite_data.py` without prior auditing of their write operations.

## Verification Resources
- Existing repository test scripts: `test_backend_services.py`, `test_citywide_detection.py`, `test_ml_model.py`, `test_provenance_safety.py`, `test_semantic_search.py`.
- Interactive inspection via browser on the running Streamlit application.

## Acceptance Criteria

### Navigation & Routing
- [ ] Navigating between Overview, Optical Change Detection, ML Change Detection, Sentinel-2 Analysis, and Semantic Archive Search renders only the selected page's components with zero content bleeding.
- [ ] The Overview page renders exclusively the title, description, five module navigation cards, and live module status.

### UI & Image Comparison Layout
- [ ] Three-stage and two-stage comparison cards render in a single horizontal row of equal-width cards without overlapping subsequent sections on desktop and laptop screen widths.
- [ ] Sidebar controls dynamically switch to display only controls relevant to the active page (Overview: navigation only; Optical: optical settings; ML: model status, threshold, and inference options; Sentinel-2: AOI/date/cloud options; Semantic Search: query controls).

### ML Model Inference
- [ ] Space Netra U-Net loads strictly from `models/spacenetra_s2looking_levir_20261030_031904_best.pth` using strict checkpoint loading without retraining or overwriting files.
- [ ] Tiled inference processes paired registered images via 6-channel concatenation, outputting a binary change mask and an overlay with adjustable opacity and download functionality at a default 0.35 threshold.

### Sentinel-2 Pipeline
- [ ] Preprocessor and satellite loader interfaces execute without `AttributeError` or interface mismatch exceptions.
- [ ] In the absence of genuine downloaded Level-2A rasters, the Sentinel-2 page displays "Experimental — Awaiting verified raster data", and synthetic test data is never presented as verified satellite imagery.

### Semantic Search & Safety Safeguards
- [ ] Semantic Archive Search queries the existing FAISS index and metadata without executing ingestion or index rebuilding.
- [ ] All existing checkpoints in `models/`, index files in `vector_db/`, datasets in `data/`, and backup folders remain completely unaltered.
- [ ] Existing test suites in the repository pass.
