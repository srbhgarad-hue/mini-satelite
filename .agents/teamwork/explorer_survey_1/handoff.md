# Space Netra Streamlit Workstation Architecture, Navigation & GIS UI Layout Analysis

## 1. Observation

### 1.1 Architecture & Page Routing
- **Entry point router (`app.py`, lines 16–91)**:
  - Dispatches across five views imported from `views/`:
    ```python
    from views.theme import inject_theme_css
    from views.overview import render_overview
    from views.optical_detection import render_optical_detection
    from views.ml_detection import render_ml_detection
    from views.sentinel2_analysis import render_sentinel2_analysis
    from views.semantic_search import render_semantic_search
    ```
  - Navigation is currently controlled solely by `st.sidebar.radio` (lines 53–71):
    ```python
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
    ```
  - Page rendering is an exclusive `if-elif` chain (lines 77–90) rendering only the selected page's function.

### 1.2 Overview View Deficiencies (`views/overview.py`, lines 12–158)
- **Mandate in R1**: "The Overview page must display only the Space Netra title, a short description, five navigation cards, and live module status."
- **Direct Observations in `views/overview.py`**:
  1. Lines 21–32 render a redundant section: `<div class="workstation-card-title">Geospatial Scope & Objectives</div>` containing 63 words of introductory prose.
  2. Lines 35–112 render only **4 module cards** (Optical Change Detection, ML Change Detection, Sentinel-2 Analysis, Semantic Archive Search) under `col1, col2 = st.columns(2)`. The Overview/Hub module itself is omitted, failing the 5 navigation cards requirement.
  3. All module cards are static HTML `<div>` blocks. There are no interactive Streamlit navigation buttons or cards that update `st.session_state["active_page"]` or route to the target view.
  4. Lines 113–154 render a static HTML table ("System Status & Data Provenance Matrix") with hardcoded status strings (`<span class="status-pill-ready">Active</span>`), rather than dynamic live health/availability status queries.
  5. Line 157 has an unneeded static callout `st.info("Select a workflow from the left sidebar navigation to begin an analysis.")`.

### 1.3 Session State Collisions & Variable Leaks
- **State cross-talk between Optical and ML (`views/optical_detection.py` and `views/ml_detection.py`)**:
  - In `views/optical_detection.py` (lines 34–57, 75–82):
    ```python
    st.session_state["active_area_name"] = selected_hotspot
    st.session_state["target_lat"] = h_info["lat"]
    st.session_state["target_lon"] = h_info["lon"]
    st.session_state["target_zoom"] = h_info.get("zoom", 17)
    ```
  - In `views/ml_detection.py` (lines 161–166):
    ```python
    area_name = st.session_state.get("active_area_name", "CIDCO Urban Expansion (N-1 to N-12)")
    h_info = city_catalog.get_hotspot_info(area_name)
    tl = photo_streamer.fetch_3stage_photographic_timeline(h_info["lat"], h_info["lon"], zoom=h_info.get("zoom", 17))
    ```
  - **Collision consequence**:
    - When a user performs a custom address geocoding search in Optical Detection, `active_area_name` is set to the raw search string (e.g., `"Kranti Chowk"`).
    - When the user switches to ML Change Detection, `city_catalog.get_hotspot_info(area_name)` fails to match `CITY_HOTSPOTS` and falls back to `CITY_CENTER_LAT` (19.8762) / `CITY_CENTER_LON` (75.3433), ignoring `target_lat` and `target_lon` entirely!
    - Changing the municipal sector in Optical Detection silently modifies the input sector for ML Change Detection without the user explicitly choosing it.
  - **Unnamespaced keys polluting global session state**:
    `target_lat`, `target_lon`, `target_zoom`, `target_category`, `target_desc` are stored without a namespace prefix.

### 1.4 Comparison Image Layouts & Overlap Vulnerabilities (`views/theme.py`, `views/optical_detection.py`, `views/sentinel2_analysis.py`, `views/ml_detection.py`)
- **Mandate in R2**: "Eliminate comparison image overlapping across two-stage and three-stage comparisons by displaying imagery in equal-width responsive cards in a single row (each containing the image, verified date, source, and concise caption) with the subsequent section starting strictly below the containers, avoiding negative margins or absolute positioning hacks."
- **Direct Observations**:
  - In `views/optical_detection.py` (lines 256–274):
    ```python
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.image(
            timeline["p1_path"],
            caption=f"Epoch 1 (Baseline): {timeline['dates']['epoch_1']}\nEsri Wayback Historical",
            width="stretch"
        )
    ```
    The images are passed raw `width="stretch"` inside `st.columns(3)`. There is no enclosing card wrapper with standardized metadata slots (verified date badge, source chip, and concise caption).
  - In `views/optical_detection.py` (lines 290–316): Two-stage visual comparison uses raw `st.columns(2)` with `st.image(..., width="stretch")`.
  - In `views/sentinel2_analysis.py` (lines 116–123 and 135–142): Two-stage true color previews (`st.columns(2)`) and three-stage analytical overlays (`st.columns(3)`) are raw images without card containment.
  - In `views/theme.py` (lines 138–149):
    ```css
    div[data-testid="stImage"] {
        border-radius: 6px;
        overflow: hidden;
        border: 1px solid #1e293b;
        background: #0f172a;
    }
    ```
    While `overflow: hidden` is defined, there are no layout classes for responsive comparison cards (`.comparison-card`, `.comparison-meta`, `.comparison-caption`), risking height desynchronization and vertical overlap when captions wrap unevenly.

### 1.5 Sidebar Control Contextualization Audit
- **Overview (`views/overview.py`)**:
  - Contains zero sidebar elements. Clean compliance with "Overview: navigation only".
- **Optical Detection (`views/optical_detection.py`, lines 31–117)**:
  - Sidebar contains: growth sector selectbox, geocoding address expander, optical zoom level slider (15–18), min footprint floor slider (20–250 m²), and structural sensitivity slider (30–100).
  - Only the geocoding input is inside an expander; zoom level, footprint floor, and sensitivity are exposed at root sidebar level rather than collapsed into an "Advanced Optical Parameters" expander.
- **ML Detection (`views/ml_detection.py`, lines 64–110)**:
  - Sidebar contains: evaluation mode radio ("Single Checkpoint" vs "Side-by-Side Dual-Model"), checkpoint selectbox, decision threshold slider (0.05–0.90), and imagery source radio.
  - Fails R3 dedication mandate: R3 specifies a single dedicated Space Netra U-Net checkpoint at `models/spacenetra_s2looking_levir_20261030_031904_best.pth`. Exposing multiple checkpoints and dual-model comparison clutter the sidebar and violate the dedicated pipeline requirement.
  - Missing overlay opacity slider (needed per R3).
- **Sentinel-2 Analysis (`views/sentinel2_analysis.py`, lines 28–48)**:
  - Sidebar contains only: spectral sensitivity slider (30–120) and min cluster area slider (5–50).
  - Missing acceptance criteria controls: AOI corridor selection, temporal date options, and SCL cloud masking threshold.
- **Semantic Archive Search (`views/semantic_search.py`, lines 41–52)**:
  - Sidebar contains only top-K slider. Query controls are placed on the main page canvas instead of contextual sidebar placement.

### 1.6 Interface Gaps & Upstream Dependencies
- **R5 Interface Bug (`preprocessor.py`)**:
  - `preprocessor.py` contains only `build_valid_pixel_mask` and `normalize_reflectance`. It does NOT define `clean_scene(band, scl)`. Any caller expecting `preprocessor.clean_scene` will crash with `AttributeError`.
- **R3 Tiling Gap (`spacenetra_model.py`)**:
  - `spacenetra_model.py` only implements `preprocess_pair(img1, img2, target_size=256)` which bilinearly resizes the whole image down to 256×256. It has no tiled sliding-window inference function for full-resolution evaluation of larger imagery.

---

## 2. Logic Chain

1. **Premise 1**: R1 mandates 5 independent pages (Overview, Optical Change Detection, ML Change Detection, Sentinel-2 Analysis, Semantic Archive Search) with zero cross-page content leaks or shared session-state collisions, and the Overview page rendering strictly the title, description, five navigation cards, and live module status.
2. **Step 1 (Overview Page)**: In `views/overview.py`, the extra "Geospatial Scope" card, the omitted 5th navigation card, the lack of clickable navigation triggers, and the static HTML matrix directly conflict with R1. Therefore, `views/overview.py` must be refactored to render solely: (1) title & description header, (2) five interactive navigation cards with button triggers that switch `st.session_state["active_page"]` and call `st.rerun()`, and (3) a dynamic live status monitor querying subsystem readiness.
3. **Step 2 (State Isolation)**: `active_area_name` is written in `views/optical_detection.py` and read in `views/ml_detection.py`. Because custom geocoded queries cause `city_catalog.get_hotspot_info` to fail in ML detection, and because R1 forbids shared state collisions, each page must operate within its own namespaced session keys (`optical_*`, `ml_*`, `s2_*`, `semantic_*`).
4. **Step 3 (GIS UI & Image Comparison Layouts)**: In R2, overlapping between multi-stage images and subsequent content occurs when elements lack flex/grid containment, use unconstrained captions, or apply negative margins. Encapsulating each image in a `.comparison-card` component with equal flex basis, fixed meta headers (`date`, `source`), standard image display (`use_container_width=True`), and concise captions, followed by standard Streamlit vertical flow without negative margins, completely eliminates layout overlap across two-stage and three-stage comparisons.
5. **Step 4 (Sidebar Contextualization)**: Under R2, sidebars must be contextualized so each page shows only its required primary controls, with advanced settings collapsed in expanders. Currently, `views/optical_detection.py` leaves sliders uncollapsed, `views/ml_detection.py` exposes redundant multi-checkpoint options instead of dedicated status and opacity controls, and `views/sentinel2_analysis.py` lacks AOI/date/cloud controls. Contextualizing each view's sidebar resolves R2.

---

## 3. Caveats

- **No Caveats on Local Code**: All code files, templates, styles, and test scripts were directly inspected via `view_file` and verified against git history.
- **Python Execution Environment**: System default `python` (Python 3.14) lacks geospatial dependencies; all tests and application executions must strictly run with the local virtual environment at `.venv\Scripts\python.exe` (Python 3.12).
- **Asset Integrity**: Checkpoints in `models/`, vector index in `vector_db/`, and fixtures in `data/` were verified read-only and intact.

---

## 4. Conclusion

The application already possesses a clean 5-page router foundation in `app.py` and modular views in `views/`. However, it requires targeted refactoring across three areas to satisfy R1 and R2:

1. **Overview Page Alignment (R1)**:
   - Strip out redundant "Scope & Objectives" card and static markdown tables.
   - Implement **5 responsive navigation cards** (Overview, Optical Change Detection, ML Change Detection, Sentinel-2 Analysis, Semantic Archive Search) with interactive "Open Module" buttons that update `st.session_state["active_page"]` and call `st.rerun()`.
   - Implement a dynamic live status indicator querying `photo_streamer` readiness, ML checkpoint existence (`models/spacenetra_s2looking_levir_20261030_031904_best.pth`), Sentinel-2 L2A raster availability (displaying "Experimental — Awaiting verified raster data"), and FAISS index existence (`vector_db/spatial.index`).

2. **Session State Decoupling (R1)**:
   - Isolate optical detection parameters under `optical_active_sector`, `optical_target_lat`, `optical_target_lon`, `optical_target_zoom`, `optical_results`, `optical_timeline`.
   - Isolate ML parameters under `ml_sector`, `ml_input_mode`, `ml_t1_file`, `ml_t2_file`, `ml_threshold` (default 0.35), `ml_opacity` (default 0.50), `ml_prob_map`.
   - Prevent any cross-module reads of `active_area_name`.

3. **GIS UI Styling & Layout Overlap Elimination (R2)**:
   - Add `.comparison-card`, `.comparison-meta`, `.comparison-caption` CSS to `views/theme.py`.
   - Refactor two-stage and three-stage comparisons in `views/optical_detection.py` and `views/sentinel2_analysis.py` to use structured cards in single equal-width rows with `use_container_width=True` and clear vertical clearance beneath containers.
   - Contextualize all 5 sidebars:
     - Overview: Navigation radio only.
     - Optical: Sector selector + quick info; expander for geocoding, zoom, and sensitivity.
     - ML: Dedicated checkpoint status (`spacenetra_s2looking_levir_...`), decision threshold slider (0.35 default), opacity slider, image source; expander for tiling and alignment diagnostics.
     - Sentinel-2: AOI corridor, epoch/date selection, SCL cloud screening options; expander for sensitivity and min cluster area.
     - Semantic Search: Query input mode, query string, top-K slider; expander for vector index metadata.

4. **Technical Interfaces (R3 & R5)**:
   - Add `clean_scene` function to `preprocessor.py` to prevent `AttributeError`.
   - Add tiled inference helper to `spacenetra_model.py` for large image support.

---

## 5. Verification Method

To independently verify these findings and subsequent implementation:

1. **Verify Unit Tests via Local Virtualenv**:
   ```powershell
   & .\.venv\Scripts\python.exe test_backend_services.py
   & .\.venv\Scripts\python.exe test_citywide_detection.py
   & .\.venv\Scripts\python.exe test_ml_model.py
   & .\.venv\Scripts\python.exe test_provenance_safety.py
   & .\.venv\Scripts\python.exe test_semantic_search.py
   ```
   All 5 suites must output `ALL ... TESTS PASSED SUCCESSFULLY!`.

2. **Inspect Streamlit App Interactively**:
   ```powershell
   & .\.venv\Scripts\streamlit.exe run app.py
   ```
   - Navigate to Overview: verify exactly title, description, 5 navigation cards, and live module status are displayed without extra sections.
   - Click each module card's navigation button: verify immediate transition to that page with zero content leakage.
   - Check Optical and Sentinel-2 comparison rows: verify 2-stage and 3-stage cards render on a single horizontal row without vertical overlap into following sections.
   - Check sidebars on each page: verify only page-specific controls appear.

3. **Invalidation Conditions**:
   - Any test failure in the 5 test suites.
   - Overview page displaying fewer or more than 5 navigation cards.
   - Image cards wrapping or overlapping on desktop screens (>1200px width).
   - Changing optical sector altering ML input without explicit user selection.
