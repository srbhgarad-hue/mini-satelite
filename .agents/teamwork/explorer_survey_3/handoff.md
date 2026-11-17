# Space Netra Handoff Report: Sentinel-2 Pipeline, Semantic Archive Search, Data Provenance & Safety

- **Author**: `explorer_survey_3` (teamwork_explorer)
- **Target Role / Agent**: `parent` (orchestrator: `c8b0d635-6058-4219-9931-c46dafaa9330`)
- **Working Directory**: `D:\PythonProjects\mini-satelite\.agents\teamwork\explorer_survey_3\`
- **Date**: 2026-11-07
- **Mission**: Thorough investigation of Sentinel-2 analysis pipeline, Semantic Archive Search (FAISS/CLIP), data provenance, synthetic data isolation, and safety safeguards (R3, R4, R5, R6).

---

## 1. Observation

### 1.1 The `AttributeError: clean_scene` Root Cause & Module Interface Mismatches
1. In `preprocessor.py` (lines 1-44), the module only defines two functions:
   - `build_valid_pixel_mask(scl_t1, scl_t2)` (line 25)
   - `normalize_reflectance(band_16bit)` (line 37)
   There is **no** `clean_scene` function or method in `preprocessor.py`.
2. In git history, commit `d141026` ("feat: update app UI styling, layout, and add sidebar controls for satellite change detection") in `app.py` introduced calls to `clean_scene`:
   ```python
   # app.py (commit d141026, lines 549-552 / lines 739-743 in tab_sentinel2):
   pairs = satellite_loader.load_temporal_pair()
   t1_clean, t1_mask = preprocessor.clean_scene(pairs["t1_b04"], pairs["t1_b08"], pairs["t1_scl"])
   t2_clean, t2_mask = preprocessor.clean_scene(pairs["t2_b04"], pairs["t2_b08"], pairs["t2_scl"])
   scl_combined = preprocessor.combine_masks(t1_mask, t2_mask)
   s2_res = change_engine.detect_multispectral_changes(t1_clean, t2_clean, scl_combined)
   s2_metrics = area_metrics.compute_area_metrics(s2_res["change_mask"], scl_combined)
   ```
   Because `preprocessor.py` lacked `clean_scene` and `combine_masks`, any execution following this pattern threw `AttributeError: module 'preprocessor' has no attribute 'clean_scene'`.
3. In `views/sentinel2_analysis.py` (lines 94-110), the code was refactored to:
   ```python
   # views/sentinel2_analysis.py:96-110
   valid_mask = preprocessor.build_valid_pixel_mask(pairs["t1_scl"], pairs["t2_scl"])
   s2_res = change_engine.detect_surface_changes(
       pairs["t1_b04"], pairs["t1_b08"],
       pairs["t2_b04"], pairs["t2_b08"],
       valid_mask,
       sensitivity=sensitivity,
       min_cluster_px=min_cluster_px
   )
   s2_metrics = area_metrics.calculate_geospatial_metrics(
       s2_res["change_mask"], valid_mask, gsd_meters=10.0
   )
   ```
   However, `preprocessor.py` still lacks `clean_scene(b04, b08, scl)` and `combine_masks(mask1, mask2)`, leaving legacy scripts, external callers, or tests vulnerable to `AttributeError`.

### 1.2 Sentinel-2 Level-2A Data Status, Geometry & Isolation
1. **Asset Location**:
   - `data/synthetic_test_fixtures/` contains 29 files, including:
     - 16-bit GeoTIFF and NumPy arrays: `t1_b04.tif`, `t1_b04.npy`, `t1_b08.tif`, `t1_b08.npy`, `t1_scl.tif`, `t1_scl.npy`, `t2_b04.tif`, `t2_b04.npy`, `t2_b08.tif`, `t2_b08.npy`, `t2_scl.tif`, `t2_scl.npy`.
     - Visual previews: `t1_2021_visual.png`, `t2_2024_visual.png`.
   - `data/change_detection/source/` contains 42 optical imagery files (`csn_*.jpg`), but **0** Sentinel-2 band raster files (`b04`, `b08`, `scl`).
2. **Satellite Loader Behavior (`satellite_loader.py:load_band`)**:
   - Lines 32-50 check `FIXTURES_DIR` (`data/synthetic_test_fixtures`) first, returning `(array, True)`.
   - `load_temporal_pair()` returns:
     - `is_synthetic`: `True`
     - `provenance`: `"SYNTHETIC_TEST_FIXTURE"`
     - `provenance_label`: `"⚠️ Synthetic Test Fixture (Simulated Sentinel-2 L2A Bands)"`
     - `provenance_note`: `"Multispectral band rasters were synthesized matching Sentinel-2 Level-2A statistical reflectance ranges for offline development and pipeline testing. Real Level-2A raster pixels have not been downloaded."`
     - `shape`: `(200, 200)`
     - `gsd_m`: `10.0`
     - `crs`: `"EPSG:32643 (UTM Zone 43N)"`
3. **Download Script Status (`download_satellite_data.py`)**:
   - `download_real_sentinel2_rasters(...)` (lines 97-107) explicitly raises `NotImplementedError`:
     `"Real Sentinel-2 raster pixel ingestion is not yet operational. Space Netra currently queries catalogue metadata via STAC, while raster math is tested using quarantined synthetic benchmark fixtures in data/synthetic_test_fixtures/."`
4. **Current UI Display (`views/sentinel2_analysis.py`)**:
   - Lines 21-26 render header badge: `badge_text="Multispectral Pipeline"`, `badge_type="standby"`.
   - Lines 57-73 display banner: `BENCHMARK TEST FIXTURE — Calibrated Simulated 10m Sentinel-2 L2A Rasters` with pill `Quarantined Benchmark`.
   - Requirement R5 specifies: In the absence of genuine Level-2A rasters, the module must be marked and isolated as `"Experimental — Awaiting verified raster data"` with developer-only `TEST/SYNTHETIC` labeling.
5. **Multispectral Index Coverage (`change_engine.py`)**:
   - `change_engine.py` only implements `compute_ndvi(b04_red, b08_nir)` (lines 12-21).
   - R3 requires multispectral raster processing including **NDVI**, **NDBI**, and **NDWI**. Neither NDBI (Normalized Difference Built-up Index) nor NDWI (Normalized Difference Water Index) is currently implemented in `change_engine.py`.

### 1.3 Semantic Archive Search (CLIP + FAISS) Status & Scoring Decoupling
1. **Index & Metadata Preservation**:
   - `vector_db/spatial.index` (10,285 bytes) and `vector_db/metadata.npy` (531 bytes) are intact.
   - `vector_db_backup/` contains identical safety copies.
   - `data/raw/` contains 5 image patches: `airport.png`, `forest.png`, `harbor.png`, `stadium.png`, `urban.png`.
2. **Query Pipeline (`views/semantic_search.py`)**:
   - Uses `@st.cache_resource def load_clip_faiss()` (lines 19-30) loading CLIP `ViT-B/32` and FAISS `spatial.index` in read-only mode without executing index reconstruction.
   - Computes query text embedding: `text_feat = model.encode_text(tokens)`, normalizes via `norm(dim=-1, keepdim=True)`.
   - Queries FAISS: `distances, indices = index.search(text_feat.cpu().numpy().astype("float32"), k=k)`.
3. **Similarity Score Presentation**:
   - Line 107-109: `score = round(float(distances[0][i]) * 100.0, 1)`, captioned as `caption=f"Score: {score}% | {item['filename']}"`.
   - In `test_semantic_search.py` line 24: `print(f"  {i+1}. {item['filename']} (Score: {dist[0][i]*100:.1f}%)")`.
   - Requirement R3 explicitly mandates: *"Dedicated CLIP + FAISS text-to-image and image-to-image retrieval preserving existing indices and metadata; do not present similarity scores as change-detection confidence."*
   - Currently, `views/semantic_search.py` does not offer image-to-image querying (only text prompt query) and uses ambiguous caption `"Score: XX%"`.

### 1.4 Data Provenance Audit & Unbacked Codebase Metrics
1. **Unbacked Building Metrics in Optical Detection (`views/optical_detection.py`)**:
   - Lines 244-248 render:
     - `m1.metric("New Buildings Found", f"{det_res['building_count']} Units")`
     - `m2.metric("New Built Footprint", f"{det_res['total_built_area_sq_m']:,.1f} m²", help="Sum of discrete building contours meeting minimum area floor")`
     - `m3.metric("Built-Up Density", f"{det_res['building_density_pct']}% of AOI", help="Percentage of analyzed ground occupied by new building roofs")`
     - `m4.metric("Ground Resolution", f"{det_res['gsd_m']} m/px", help="Metric Ground Sampling Distance for selected zoom level")`
     - `m5.metric("Expansion Level", "High Growth" if det_res['building_count'] >= 3 else ("Moderate Growth" if det_res['building_count'] > 0 else "Stable Surface"))`
   - In `building_detector.py` (lines 91-98):
     - Every contour meeting aspect ratio 0.2-5.0 is automatically labeled `"Status": "New Construction"`.
     - Fabricated confidence: `"Confidence": f"{min(98.5, round(75.0 + (area_px / 10.0), 1))}%"`.
   - In `views/optical_detection.py` (lines 325-342):
     - An oversized Plotly bar chart plots single-building footprint areas by `Building ID` (`BLDG-01`, etc.).
   - In `views/optical_detection.py` (lines 348-384):
     - "Municipal Intelligence Audit Report" exports unverified claims: `"Total Newly Built Structures Detected: {det_res['building_count']} Units"`, `"Built-Up Surface Density: {det_res['building_density_pct']}%"`.
   - These directly conflict with R3 ("do not automatically classify or label all detected changes as buildings") and R4 ("Remove hardcoded, synthetic, or unverified numbers (including unbacked 'New Buildings Found', 'Built-Up Density', 'Ground Resolution', 'Expansion Level', and single-building oversized charts) and replace them with 'Not available' or 'Awaiting verified data'").
2. **Overview Page Claims (`views/overview.py`)**:
   - Line 51: `<span class="tag-chip">Resolution: ~1.12 m/px</span>` (hardcoded estimate).
   - Module cards render only 4 modules, not the 5 navigation cards mandated by R1.
3. **Asset Protection Audits (`R6`)**:
   - `ingest.py` lines 50-52: `faiss.write_index(index, index_save_path)` and `np.save("metadata.npy", metadata)`. Executing this rewrites the vector index.
   - `download_images.py` line 25: overwrites `data/raw/*.png`.
   - `download_satellite_data.py`: `synthesize_calibrated_rasters()` writes synthetic arrays to output directory.

---

## 2. Logic Chain

1. **Interface Contract Resolution**:
   - From Obs 1.1, `preprocessor.py` was authored without `clean_scene` or `combine_masks`, whereas earlier commits (`d141026`) and external tools assumed their presence.
   - In modern `views/sentinel2_analysis.py`, `preprocessor.build_valid_pixel_mask(scl1, scl2)` is called directly.
   - Therefore, providing `clean_scene(b04, b08, scl)` and `combine_masks(mask1, mask2)` in `preprocessor.py` as forward-compatible wrappers around `build_valid_pixel_mask` eliminates any potential `AttributeError: clean_scene` across tests, legacy routes, and external modules while preserving existing functionality.

2. **Sentinel-2 Operational Isolation**:
   - From Obs 1.2, authentic Level-2A Copernicus rasters cannot currently be streamed because `download_real_sentinel2_rasters()` is not yet implemented (requires Azure SAS tokens and rasterio/GDAL).
   - Only synthetic benchmark arrays exist in `data/synthetic_test_fixtures/`.
   - Presenting these numbers as authentic ground changes violates scientific integrity and R4/R5.
   - Therefore, the Sentinel-2 page must be strictly tagged as `"Experimental — Awaiting verified raster data"` with clear developer-only `TEST/SYNTHETIC` callouts, and the metrics must be explicitly captioned as benchmark validation metrics rather than municipal ground observations.
   - In addition, implementing `compute_ndbi` and `compute_ndwi` in `change_engine.py` fulfills requirement R3.

3. **Semantic Search Decoupling**:
   - From Obs 1.3, CLIP ViT-B/32 produces normalized vector embeddings whose dot product with archive patch embeddings represents visual semantic similarity.
   - In `views/semantic_search.py`, labeling results as `"Score: {score}%"` allows users to confuse semantic retrieval with change-detection confidence.
   - Therefore, the caption must be explicitly changed to `"Cosine Similarity: {score}%"` or `"Semantic Match: {score}%"`, accompanied by an explicit disclaimer decoupling semantic indexing from bitemporal change detection.
   - Adding an optional image upload query input implements the image-to-image retrieval requirement of R3 without modifying or re-indexing `vector_db/`.

4. **Provenance & Unbacked Metric Replacement**:
   - From Obs 1.4, optical change detection (SSIM + Canny) detects structural pixel differences, not municipal building construction permits.
   - The labels `"New Buildings Found"`, `"Built-Up Density"`, `"Ground Resolution"`, `"Expansion Level"`, and the single-building bar chart represent unverified assertions.
   - Therefore, per R3 and R4, optical change detection must report detected changes as general `"Optical Change Regions"` and `"Changed Surface Footprint"`, replacing unbacked building counts and expansion levels with `"Not available"` or `"Awaiting verified data"`, and replacing the single-building bar chart with a change area distribution or summary.

5. **Asset Preservation**:
   - From Obs 1.3 and Obs 1.4, all trained models in `models/` (`spacenetra_s2looking_levir_20261030_031904_best.pth`, `spacenetra_s2looking_unet_fulltrain_best.pth`, `spacenetra_s2looking_unet.pth`), the FAISS index (`vector_db/spatial.index`), and backups (`vector_db_backup/`) are intact.
   - Maintaining complete protection requires strictly forbidding any automated invocation of `ingest.py`, `download_images.py`, or `download_satellite_data.py`.

---

## 3. Caveats

1. **Remote Cloud Providers**: We did not initiate network connections to Microsoft Planetary Computer or Copernicus Data Space Ecosystem because live network downloads of multi-gigabyte SAFE/COG rasters require authenticated credentials/SAS tokens and rasterio dependencies not installed in the environment.
2. **Third-Party Model Checkpoints**: We verified file presence and byte sizes (> 7.8 MB) for all model checkpoints, but did not retrain or alter any model weights.
3. **No Code Modification**: In accordance with the Explorer archetype instructions, zero production code files were edited during this survey. Proposed modifications are documented below for subsequent implementation.

---

## 4. Conclusion

The Sentinel-2, Semantic Search, and Data Provenance subsystems are architecturally decoupled and structurally intact, but require specific interface updates, isolation banners, and metric cleanups to satisfy requirements R1-R6:

1. **`preprocessor.py` Interface Contract**:
   Add `clean_scene(b04, b08, scl)` and `combine_masks(mask1, mask2)` to `preprocessor.py` to prevent any `AttributeError: clean_scene`.
   ```python
   def clean_scene(b04, b08, scl):
       """
       Screens cloud, shadow, and defective pixels for a single scene using SCL.
       Returns:
           cleaned_scene (dict): {"b04": b04, "b08": b08}
           valid_mask (np.ndarray): Boolean mask (True = valid, False = cloud/shadow)
       """
       invalid = np.isin(scl, list(INVALID_SCL_CLASSES))
       valid_mask = ~invalid
       cleaned_b04 = b04.copy()
       cleaned_b08 = b08.copy()
       cleaned_b04[invalid] = 0
       cleaned_b08[invalid] = 0
       return {"b04": cleaned_b04, "b08": cleaned_b08}, valid_mask

   def combine_masks(mask1, mask2):
       """Combines two single-scene valid masks into a joint valid mask."""
       return mask1 & mask2
   ```

2. **Multispectral Engine Enhancements (`change_engine.py`)**:
   Add `compute_ndbi(swir, nir)` and `compute_ndwi(green, nir)` to complete the multispectral index suite required by R3.

3. **Sentinel-2 UI Isolation (`views/sentinel2_analysis.py`)**:
   Update page badge to:
   - `badge_text="Experimental — Awaiting verified raster data"`, `badge_type="warning"`.
   - Update provenance banner with explicit `DEVELOPER-ONLY TEST/SYNTHETIC BENCHMARK FIXTURE` labeling, stating that genuine Level-2A COG pixels are awaiting ingestion and current metrics reflect synthetic fixture validation.

4. **Semantic Archive Search Decoupling (`views/semantic_search.py`)**:
   - Change result captions from `Score: {score}%` to `Cosine Similarity: {score}%`.
   - Add explicit visual disclaimer: *"Notice: Cosine similarity represents multi-modal embedding alignment in CLIP feature space, decoupled from physical change-detection confidence."*
   - Add image-to-image search capability via `model.encode_image()` alongside text search.
   - Strictly keep `vector_db/spatial.index` and `metadata.npy` read-only.

5. **Optical Detection & Provenance Audit (`views/optical_detection.py`)**:
   - Reframe `"New Buildings Found"` to `"Detected Change Zones"` / `"Unclassified Optical Changes"`.
   - Replace unverified claims (`"Built-Up Density"`, `"Expansion Level"`, heuristic building confidence percentages) with `"Not available"` or `"Awaiting verified cadastre data"`.
   - Replace the single-building oversized bar chart with a general change-size distribution or remove per R4.

6. **Safety Safeguards (`R6`)**:
   - Ensure `ingest.py`, `download_images.py`, and `download_satellite_data.py` remain unexecuted during normal application operation and tests.

---

## 5. Verification Method

### 5.1 Automated Test Execution
Run the existing repository test suites to verify backend services, data provenance, and semantic search:
```powershell
# 1. Backend services verification (satellite_loader, preprocessor, change_engine, area_metrics)
python test_backend_services.py

# 2. Data provenance and synthetic isolation safety verification
python test_provenance_safety.py

# 3. Semantic vector search verification (CLIP + FAISS)
python test_semantic_search.py

# 4. City-wide optical detection verification
python test_citywide_detection.py

# 5. ML model and checkpoint verification
python test_ml_model.py
```

### 5.2 Interactive Streamlit Inspection
Launch the Streamlit application:
```powershell
streamlit run app.py
```
Inspect:
1. **Overview Page**: Verify only 5 navigation cards, description, title, and live module status are displayed.
2. **Optical Change Detection**: Verify 3-stage timeline renders in equal-width cards in a single row without overlapping, and unbacked building metrics are replaced with "Not available" / "Awaiting verified data".
3. **ML Change Detection**: Verify Space Netra U-Net loads from `spacenetra_s2looking_levir_20261030_031904_best.pth`, threshold defaults to 0.35, and benchmark section is in an expander.
4. **Sentinel-2 Analysis**: Verify "Experimental — Awaiting verified raster data" badge and developer-only TEST/SYNTHETIC labeling. Verify no `AttributeError` occurs.
5. **Semantic Archive Search**: Verify text search and image search query `vector_db/spatial.index` without rebuilding, and scores are labeled as "Cosine Similarity" with change-confidence decoupling disclaimer.

### 5.3 Invalidation Conditions
This assessment is invalidated if:
- Any file in `vector_db/`, `vector_db_backup/`, or `models/` is deleted, modified, or overwritten.
- Real Sentinel-2 Level-2A GeoTIFFs are downloaded into `data/change_detection/source/` (in which case Sentinel-2 should transition from "Experimental" to "Verified Real").
- `preprocessor.py` fails when called via either `clean_scene()` or `build_valid_pixel_mask()`.
