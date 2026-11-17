# Exploration Report: Optical and ML Change Detection Pipelines

- **Agent**: `explorer_survey_2` (teamwork_explorer)
- **Working Directory**: `D:\PythonProjects\mini-satelite\.agents\teamwork\explorer_survey_2\`
- **Date**: 2026-11-07
- **Mission**: Thorough investigation of Optical and ML Change Detection pipelines, model checkpoints, inference pipelines, test suites, and requirements R3 and R6 compliance.

---

## 1. Observation

### 1.1 Python Execution Environment & Dependencies
- System `PATH` default Python is `Python 3.14` (`C:\Users\ranav\AppData\Local\Programs\Python\Python314\python.exe`), which lacks `torch`:
  ```
  ModuleNotFoundError: No module named 'torch'
  ```
- The project virtual environment is located at:
  ```
  D:\PythonProjects\mini-satelite\.venv\Scripts\python.exe
  ```
- Verified environment packages:
  - `PyTorch`: `2.14.1+cpu`
  - `Streamlit`: `1.66.0`
  - `OpenCV`: `opencv-python 4.13.0`
  - `scikit-image`: `0.26.0`
  - `PIL`: `12.2.0`
  - `CLIP`: `1.0` (ViT-B/32)
  - `FAISS`: `faiss-cpu 1.13.2`

### 1.2 Model Checkpoints in `models/` (Requirement R6)
Direct inspection of `D:\PythonProjects\mini-satelite\models` revealed:
| Checkpoint File | Size (Bytes) | Epoch | Configuration ID | Purpose / Datasets |
|---|---|---|---|---|
| `spacenetra_s2looking_levir_20261030_031904_best.pth` | 7,839,903 | 5 | `"mixed"` | S2Looking (3,500 pairs) + LEVIR-CD (7,120 patches). **Mandated default model (R3)** |
| `spacenetra_s2looking_unet_fulltrain_best.pth` | 7,838,589 | 9 | `"fulltrain"` | S2Looking full-train (3,500 train, 500 val, 1,000 test pairs) |
| `spacenetra_s2looking_unet.pth` | 7,836,395 | 2 | `"initial"` | S2Looking baseline (500 train, 100 val pairs) |
| `spacenetra_s2looking_unet_fulltrain_best1 .pth` | 7,838,589 | 9 | - | Existing backup copy of fulltrain checkpoint |

- Verified integrity: Loaded `spacenetra_s2looking_levir_20261030_031904_best.pth` via `spacenetra_model.load_model(..., device="cpu")` with `strict=True`. Checkpoint contains valid `model_state_dict`, with keys matching `SpaceNetraUNet` exactly.
- Benchmark metadata associated with `mixed` checkpoint:
  - **S2Looking Test**: Precision 33.04%, Recall 30.50%, F1 31.72%, IoU 18.85% (at threshold 0.50)
  - **LEVIR-CD Test**: Precision 68.16%, Recall 63.49%, F1 65.75%, IoU 48.97% (at threshold 0.50)
  - **Validation Metrics**: S2Looking (loss 0.8336, F1 0.3635, IoU 0.2221), LEVIR-CD (loss 0.5449, F1 0.5921, IoU 0.4206), Mean F1: 0.4778.

### 1.3 ML Change Detection Pipeline (`spacenetra_model.py` & `views/ml_detection.py`)
- **Model Architecture** (`spacenetra_model.py:135-183`):
  - U-Net with 6-channel input: `conv_block(6, 16)` -> MaxPool -> `conv_block(16, 32)` -> MaxPool -> `conv_block(32, 64)` -> MaxPool -> `conv_block(64, 128)` -> MaxPool -> Bottleneck `conv_block(128, 256)`.
  - Decoder: 4 stages of `ConvTranspose2d` + skip connections + `conv_block` -> `Conv2d(16, 1, kernel_size=1)` yielding single-channel raw change logits.
- **Preprocessing** (`spacenetra_model.py:289-332`):
  - Resizes earlier and later images to `target_size=256` via bilinear interpolation.
  - Normalizes pixel values to float32 in `[0.0, 1.0]`.
  - Concatenates earlier RGB (channels 0..2) and later RGB (channels 3..5) to shape `(1, 6, 256, 256)`.
- **Inference & Thresholding** (`spacenetra_model.py:345-410`):
  - Model forward pass + sigmoid yields continuous probability map `[0.0, 1.0]`.
  - `apply_threshold_to_prob_map` produces binary mask `(prob_map >= threshold).astype(np.uint8)`.
  - Overlay highlights changed pixels in Coral Red (`[245, 80, 60]`). Currently hardcoded alpha 0.55 (`spacenetra_model.py:364`):
    ```python
    overlay[change_pixels] = (
        0.45 * overlay[change_pixels] + 0.55 * np.array([245, 80, 60], dtype=np.uint8)
    ).astype(np.uint8)
    ```
- **Threshold Sensitivity & Alignment Diagnostic** (`spacenetra_model.py:508-620`):
  - `compute_threshold_sensitivity_diagnostic`: Computes pixel counts and frame % across thresholds `[0.10, 0.20, 0.30, 0.40, 0.50, 0.65]` from the single cached probability map without re-running inference.
  - `verify_input_consistency_and_alignment`: Uses Hann-windowed 2D FFT phase correlation (`cv2.phaseCorrelate`) to estimate sub-pixel displacement `(dx, dy)`, Pearson spatial correlation `r`, and MAE intensity difference.
- **Current State in `views/ml_detection.py`**:
  - Threshold slider initialized to `value=0.35` (`views/ml_detection.py:101`).
  - Supports "Single Checkpoint Evaluation" and "Side-by-Side Dual-Model Comparison".
  - **GAPS IDENTIFIED RELATIVE TO R3**:
    1. *Tiled Inference*: Not implemented. Large images are resized down to 256x256, losing spatial resolution.
    2. *Adjustable Overlay Opacity*: Not available in UI; fixed at 0.55 in backend.
    3. *Prediction Download*: Not implemented; no `st.download_button` for binary mask or overlay.
    4. *Benchmark Expander*: The benchmark cards are rendered open in the main body (lines 113-154) rather than in a compact collapsible `st.expander`.

### 1.4 Optical Change Detection Pipeline (`change_detector.py`, `building_detector.py`, `views/optical_detection.py`)
- **`change_detector.py`** (`lines 5-84`):
  - `calculate_change_confidence(img_t1_path, img_t2_path, resolution_m=10.0, sensitivity=60, overlay_alpha=0.6)`.
  - Computes SSIM diff: `score, diff = ssim(gray1, gray2, full=True)`.
  - Thresholds inverted diff (`diff_inv`) + dilation -> contours.
  - Generates JET colormap heatmap overlay.
  - Bounding boxes labeled `CHANGE #{bbox_count}` (generic change, not building!).
  - Surface area math: `changed_pixels * resolution_m^2`, hectares, sq km.
  - *Missing*: Canny edge integration, geometric sub-pixel registration.
- **`building_detector.py`** (`lines 13-138`):
  - `detect_new_buildings(img_t1_path, img_t3_path, gsd_m=2.24, min_area_sq_m=50.0, sensitivity=65)`.
  - Computes Canny edges on T1 and T3 (`cv2.Canny(gray, 50, 150)`).
  - Computes SSIM difference.
  - Blends SSIM diff (0.65) + Canny edge difference (0.35).
  - Filters contours by area and aspect ratio (0.2 to 5.0).
  - Automatically classifies all valid contours as `"New Construction"` / `"BLDG-01"`.
- **`views/optical_detection.py`**:
  - Currently imports `building_detector` (line 18) and calls `building_detector.detect_new_buildings` (line 200).
  - Automatically labels all changes as "New Buildings Found", "New Built Footprint", "Built-Up Density", "Expansion Level" (lines 243-249).
  - Renders itemized building table and an oversized Plotly bar chart (lines 321-344).
  - *Direct Conflict with R3 & R4*:
    - R3: "Traditional computer vision comparison (image registration, SSIM, Canny, contours, area calculation); do not automatically classify or label all detected changes as buildings."
    - R4: "Remove hardcoded, synthetic, or unverified numbers (including unbacked 'New Buildings Found', 'Built-Up Density', 'Ground Resolution', 'Expansion Level', and single-building oversized charts) and replace them with 'Not available' or 'Awaiting verified data'."

### 1.5 Automated Verification Suites
Executed all 5 project test scripts using `D:\PythonProjects\mini-satelite\.venv\Scripts\python.exe`:
| Test Script | Execution Command | Result | Details |
|---|---|---|---|
| `test_ml_model.py` | `& ".venv\Scripts\python.exe" test_ml_model.py` | **10/10 PASS** | Checkpoints, preprocessing, inference, slider, switching, heatmap, morphology, reference mask eval, traditional detection, alignment/sensitivity. |
| `test_citywide_detection.py` | `& ".venv\Scripts\python.exe" test_citywide_detection.py` | **PASS** | 7 hotspots in `city_catalog`, 3-stage photo streaming, building detector smoke test. |
| `test_backend_services.py` | `& ".venv\Scripts\python.exe" test_backend_services.py` | **PASS** | Sentinel-2 loader, preprocessor SCL mask, change engine, area metrics conservation. |
| `test_provenance_safety.py` | `& ".venv\Scripts\python.exe" test_provenance_safety.py` | **PASS** | Quarantine isolation, acquisition safety, checkpoint protection. |
| `test_semantic_search.py` | `& ".venv\Scripts\python.exe" test_semantic_search.py` | **PASS** | CLIP ViT-B/32 + FAISS spatial index search. |

All key modules compile cleanly with `py_compile`:
```
& "D:\PythonProjects\mini-satelite\.venv\Scripts\python.exe" -m py_compile app.py change_detector.py change_engine.py building_detector.py area_metrics.py city_catalog.py spacenetra_model.py views/optical_detection.py views/ml_detection.py views/overview.py views/sentinel2_analysis.py views/semantic_search.py views/theme.py
(Exit code 0, no syntax errors)
```

---

## 2. Logic Chain

1. **Environment Consistency** (from Obs 1.1):
   - The system Python 3.14 lacks packages. Any verification or runtime command must explicitly use `D:\PythonProjects\mini-satelite\.venv\Scripts\python.exe` (or `streamlit run app.py` from within `.venv`).
2. **Model Integrity & Protection** (from Obs 1.2 & R6):
   - The primary checkpoint `models/spacenetra_s2looking_levir_20261030_031904_best.pth` is verified intact, strictly matches `SpaceNetraUNet`, and all 10 unit tests in `test_ml_model.py` pass with it. It must remain strictly read-only and never overwritten.
3. **Decoupling Optical Change from Building Labeling** (from Obs 1.4, R3, R4):
   - In `views/optical_detection.py`, the existing workflow calls `building_detector.detect_new_buildings`, resulting in unverified "New Buildings Found", "Built-Up Density", and single-building bar charts.
   - Per R3, optical change detection must perform traditional computer vision comparison:
     1. Image registration (e.g. 2D FFT phase correlation displacement check or alignment transform)
     2. SSIM structural similarity differencing
     3. Canny edge extraction and edge gradient differencing
     4. Morphological contour isolation and bounding boxes labeled generically as `"CHANGE #{id}"`
     5. Surface area calculation (pixels, m², hectares)
   - It must NOT automatically classify or label changes as buildings.
   - However, `test_ml_model.py` and `test_citywide_detection.py` import `building_detector.detect_new_buildings`. Therefore, `building_detector.py` must be retained without breaking its signature, while `views/optical_detection.py` and/or `change_detector.py` should implement the decoupled optical change detection pipeline conforming to R3 and R4.
4. **ML Change Detection Feature Enhancements** (from Obs 1.3 & R3):
   - R3 requires:
     1. Dedicated Space Netra U-Net binary change detection using `models/spacenetra_s2looking_levir_20261030_031904_best.pth` (already default in `views/ml_detection.py`).
     2. Two-image upload/selection and input validation (already supported via municipal sector and custom upload).
     3. 6-channel concatenation input (already verified).
     4. Tiled inference for large images: When input images exceed 256x256, a tiling engine should split the image into 256x256 patches (with step/overlap), run inference on each patch, and stitch/blend predictions into a full-resolution probability map, avoiding downsampling distortion.
     5. Binary mask and overlay output with adjustable opacity: Add an opacity slider (e.g. 0.10 to 1.00, default ~0.55) so the user controls overlay translucency in real time without re-running the neural network.
     6. Prediction download: Provide download buttons for the binary mask (PNG) and the change overlay (PNG/JPEG).
     7. Compact expandable model benchmark section: Wrap benchmark cards and disclaimers in `st.expander("Model Benchmark & Validation Information", expanded=False)`.
     8. Initial threshold set to 0.35: Slider already defaults to 0.35 in `views/ml_detection.py`.

---

## 3. Caveats

- **No Caveats** on test execution: All 5 existing test suites were executed directly and passed 100%.
- **GPU Availability**: The workstation uses CPU for PyTorch inference (`torch.device("cpu")`), which executes a 256x256 forward pass in ~80-120 ms. Tiled inference on large images (e.g. 1024x1024 = 16 tiles) will take ~1.5-2 seconds on CPU, which is acceptable for interactive Streamlit use.
- **Backward Compatibility**: `test_ml_model.py:test_traditional_detection_and_semantic_search` calls `building_detector.detect_new_buildings`. Downstream implementers must not delete or rename this function in `building_detector.py`.

---

## 4. Conclusion

1. **Readiness**: The ML and Optical subsystems are structurally sound and have 100% passing test baselines.
2. **Specific Action Items for Implementation Agent**:
   - **Optical Change Detection (`change_detector.py` / `views/optical_detection.py`)**:
     - Upgrade `change_detector.py` to integrate Canny edge detection, SSIM dissimilarity, phase-correlation registration check, contours, and metric area calculation.
     - In `views/optical_detection.py`, replace `building_detector` calls with the generic optical change pipeline.
     - Eliminate unbacked "New Buildings Found", "Built-Up Density", "Expansion Level", and single-building bar charts; replace with generic verified change metrics (Changed Area in m²/Ha, Frame Change %, Confidence Score, Bounding Box Count).
     - Display equal-width responsive comparison cards in a single row without negative margin hacks.
   - **ML Change Detection (`spacenetra_model.py` / `views/ml_detection.py`)**:
     - Implement `predict_change_tiled(model, earlier_img, later_img, tile_size=256, stride=256, threshold=0.35, device="cpu")` in `spacenetra_model.py` for full-resolution inference on large images.
     - Add opacity slider to `views/ml_detection.py` and pass `alpha` to overlay generation.
     - Add download buttons (`st.download_button`) for binary change mask and overlay.
     - Move dataset-level benchmark cards into a collapsible `st.expander("Model Benchmark Information", expanded=False)`.
     - Maintain strict loading of `models/spacenetra_s2looking_levir_20261030_031904_best.pth` with initial threshold 0.35.

---

## 5. Verification Method

To independently verify these findings, run the following commands using the project virtual environment:

1. **Verify Python & Dependencies**:
   ```powershell
   & "D:\PythonProjects\mini-satelite\.venv\Scripts\python.exe" -c "import torch, streamlit, cv2, PIL, faiss, clip; print('Environment OK')"
   ```
2. **Verify Checkpoint Presence and Properties**:
   ```powershell
   Get-Item "D:\PythonProjects\mini-satelite\models\spacenetra_s2looking_levir_20261030_031904_best.pth" | Select-Object Name, Length, LastWriteTime
   ```
3. **Run ML Model Test Suite**:
   ```powershell
   & "D:\PythonProjects\mini-satelite\.venv\Scripts\python.exe" test_ml_model.py
   ```
4. **Run Citywide Detection Test Suite**:
   ```powershell
   & "D:\PythonProjects\mini-satelite\.venv\Scripts\python.exe" test_citywide_detection.py
   ```
5. **Run All Repository Tests**:
   ```powershell
   & "D:\PythonProjects\mini-satelite\.venv\Scripts\python.exe" test_backend_services.py
   & "D:\PythonProjects\mini-satelite\.venv\Scripts\python.exe" test_provenance_safety.py
   & "D:\PythonProjects\mini-satelite\.venv\Scripts\python.exe" test_semantic_search.py
   ```
6. **Files to Inspect**:
   - `spacenetra_model.py`: Lines 135-287, 345-410, 508-620
   - `views/ml_detection.py`: Lines 64-154, 214-265, 365-379
   - `change_detector.py`: Lines 5-84
   - `building_detector.py`: Lines 13-138
   - `views/optical_detection.py`: Lines 200-249, 318-345
