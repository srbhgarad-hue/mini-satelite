# Orchestrator Context

## Project Overview
Space Netra is a geospatial change-detection platform. The goal is to transform it into a reliable production platform across five independent modules:
1. Overview
2. Optical Change Detection
3. ML Change Detection
4. Sentinel-2 Analysis
5. Semantic Archive Search

## Key Requirements & Boundaries
- R1: 5-page separation, clean session state, no leaks, Overview has title/desc/cards/live module status.
- R2: GIS dark navy/charcoal UI with subtle teal/blue accents, equal-width cards in single horizontal row, no negative margins or absolute positioning hacks. Contextual sidebars.
- R3: Four technical pipelines decoupled. Optical CV (no auto building label), ML U-Net (spacenetra_s2looking_levir_20261030_031904_best.pth, 6-channel concat, tiled inference, binary mask/overlay, download, benchmark expander, 0.35 default threshold), Sentinel-2 (NDVI, NDBI, NDWI, real CRS/transform, cloud masking), Semantic search (CLIP + FAISS, existing index/metadata, similarity not change confidence).
- R4: Data provenance & synthetic data isolation. Replace unbacked claims with "Not available" / "Awaiting verified data". Isolate synthetic fixtures under developer TEST/SYNTHETIC label.
- R5: Sentinel-2 interface resolution (preprocessor.py clean_scene attribute errors). If real Level-2A raster absent, mark as "Experimental — Awaiting verified raster data".
- R6: Strict preservation of checkpoints, datasets, FAISS index/metadata, backups. No execution of ingest.py, download_images.py, download_satellite_data.py without prior auditing.

## Existing Test Scripts
- `test_backend_services.py`
- `test_citywide_detection.py`
- `test_ml_model.py`
- `test_provenance_safety.py`
- `test_semantic_search.py`
