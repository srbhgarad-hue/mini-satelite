# Space Netra — Mini-Project Plan

## 1. Project purpose

Build a demonstrable geospatial application that compares satellite observations from two dates over a small Area of Interest (AOI) near Chhatrapati Sambhajinagar and highlights areas with potential changes. Retain the existing CLIP + FAISS semantic image search as a separate feature when it is already working.

**Working project statement:** Space Netra is a satellite imagery analysis application that visualizes potential land-surface changes between comparable dates and supports natural-language retrieval of indexed satellite images.

This is a college mini project and a one-week pilot, not an SIH submission. Favor sound data handling, explainable results, reproducibility, and an honest demo over a long feature list.

## 2. Recommended scope for the first pilot

### In scope

- One small, explicitly drawn AOI (start with roughly 1–5 km² if data availability and machine performance allow; reduce if needed).
- Search for usable Sentinel-2 Level-2A scenes for this AOI across multiple dates.
- Select two comparable, sufficiently clear observations from similar seasons when possible.
- Clip/crop the data consistently, apply cloud/shadow masking, and ensure both dates share the same CRS, grid, pixel size, and bounds.
- Run an explainable baseline change detector using spectral differences and/or SSIM, plus noise cleanup.
- Show before/after imagery, a change mask/overlay, a heatmap, change-pixel statistics, and metadata describing the date pair and resolution.
- Save the outputs and configuration for reproducibility.
- Preserve the current CLIP/FAISS database and semantic-search path.

### Explicitly out of scope for the one-week pilot

- Training a custom neural network from scratch.
- Reliable individual-building construction detection from 10 m Sentinel-2 imagery.
- Claims of real-time monitoring, automated enforcement, or operational accuracy.
- Unverified hectares/km² statistics derived from screenshots, web-map tiles, or images without georeferencing.
- A full backend/API rewrite, React front end, cloud deployment, user accounts, or production database unless the current application already requires them.
- Treating a detected pixel difference as proof of construction, demolition, illegal activity, or any other specific cause.

## 3. Why a small AOI is the right pilot — and the limitation

A small AOI reduces download volume, processing time, and debugging scope. It does **not** increase a satellite's ground sampling distance. Sentinel-2 bands B02/B03/B04/B08 are 10 m per pixel, so narrow roads and individual buildings may occupy too few pixels for dependable object-level conclusions. Scope the first result as **potential land-surface/urban-area change**, not precise building detection.

If building-level change is the required output, pause before promising it: obtain imagery around 0.5–4 m/pixel with clear rights for the intended use, or constrain that model experiment to an existing benchmark dataset and label the result as a separate research demonstration.

## 4. One-week implementation schedule

### Day 1 — Baseline and data feasibility

- Create a Git branch and a backup of the existing `vector_db/` and current working app.
- Record existing file names, how the FAISS index and metadata are loaded, and current known-good commands.
- Draw a small AOI polygon or rectangle and record its GeoJSON plus CRS.
- Query Sentinel-2 L2A metadata for the AOI over a practical date window.
- Inspect candidate scene dates, cloud metadata, scene footprints, and available bands.
- **Go/no-go gate:** identify at least one usable image pair (or a plan to build a seasonal composite). If no usable pair exists, change the AOI/date range before implementing the detector.

### Day 2 — Acquire and inspect imagery

- Download or stream only the required assets/windows; avoid pulling complete scenes unnecessarily.
- Save source, item ID, sensing time, band names, resolution, CRS, cloud metadata, license/source URL, and processing parameters.
- Create quick-look before/after images and a metadata record.
- Reject unsuitable pairs with clouds, major seasonal differences, different footprints, or incompatible quality.

### Day 3 — Geospatial preprocessing

- Apply scene-classification/cloud-shadow masks where available.
- Clip to the AOI.
- Reproject/resample to a shared metric CRS/grid and identical dimensions.
- Check alignment using roads/landmarks and raster transforms; do not assume same-sized images are aligned.
- Save aligned analysis rasters separately from the original downloaded assets.

### Day 4 — Baseline change detection

- Implement a transparent baseline: reflectance normalization, valid-pixel mask, spectral/image difference score, threshold, connected components/morphological cleanup, and optional SSIM view.
- Keep thresholds configurable and record them with each run.
- Add safeguards for nodata/cloud pixels and empty masks.
- Generate a binary mask, heatmap, overlay, and a statistics JSON/CSV.

### Day 5 — Streamlit integration

- Add a dedicated “Change Detection” section/page in the current app.
- Display date pair, AOI, source, nominal pixel size, cloud/valid-pixel fraction, and the limitations alongside result images.
- Present area estimates only when derived from georeferenced pixels in a suitable metric projection; explain that detected changes are candidate changes.
- Keep the current semantic-search mode untouched.

### Day 6 — Quality checks and mini validation

- Inspect a set of known unchanged and visibly changed patches with a team member/annotator.
- Record false positives due to cloud/shadow, seasonal vegetation, water, shadows, sensor/processing differences, and alignment.
- If no ground-truth mask is available, report qualitative review and do not invent precision/accuracy percentages.
- Test no-data, bad-date-pair, missing asset, and empty-change cases.

### Day 7 — Freeze a reproducible demo

- Freeze the selected AOI, source scenes, parameters, and output artifacts.
- Create a small technical report describing data source, dates, resolution, method, observed limits, and example results.
- Re-run from a clean environment/known setup.
- Verify the old FAISS index still loads and semantic search still works.
- Prepare a short demo: select AOI/pair → inspect imagery → run analysis → review candidate changes → export report.

## 5. Proposed success criteria

The one-week pilot is successful when:

1. The app can identify and load a documented image pair for one AOI.
2. The pair is cloud/shadow-screened, clipped, and aligned to a common grid.
3. The detector produces a visible candidate-change mask and overlay without crashing.
4. The dashboard clearly shows scene dates, source, resolution, and limitations.
5. Any physical area figure is derived from georeferenced pixels and an appropriate metric CRS; otherwise show pixel counts only.
6. Outputs and parameters can be reproduced from saved metadata.
7. Existing CLIP/FAISS search and its 4–5 indexed images remain intact.
8. The report does not claim building identification or model accuracy unsupported by validation.

## 6. Recommended order of priorities

**P0:** preserve existing app and DB; confirm source-data availability; define AOI; correct geospatial pair preparation.  
**P1:** baseline detector, visual output, metadata, and basic validation.  
**P2:** Streamlit integration, reports, and visual polish.  
**P3:** supervised model experiments, large external datasets, production-style APIs, deployment.

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| No useful clear image pair for the chosen AOI | Query availability before implementing; widen the date window or choose a nearby AOI. |
| User expects high detail because the AOI is small | State resolution explicitly; AOI size and pixel size are different things. |
| Clouds, shadows, dry/wet season, or vegetation trigger false positives | Apply cloud/shadow masks, compare similar seasons, and present results as candidate changes. |
| Misaligned dates create edge artifacts | Reproject to the same grid and inspect visual registration. |
| Area statistics are incorrect | Calculate from georeferenced pixels in a metric CRS; otherwise report only pixel/percentage values. |
| Existing vector database is corrupted | Back it up and keep its ingestion and storage path separate. |
| Model demo overstates capability | Do not train from 4–5 unlabelled images; validate with held-out labelled data before reporting metrics. |
| External imagery terms are unclear | Prefer Sentinel open-data terms; check each provider/dataset terms before storing or redistributing imagery. |

## 8. Deliverables

- `plan.md`, `architecture.md`, `tech.md`, `pipeline.md`, `data_sources.md`, `model_strategy.md`, `open_questions.md`.
- An AOI GeoJSON and a data manifest for the selected scene pair.
- Original scene assets (or persistent source IDs/URLs), masked/warped rasters, and analysis outputs in separate directories.
- Baseline detection module and a Streamlit results view.
- A short reproducible demo report with dates, resolution, method, limitations, and examples.
- Evidence the existing semantic-search mode still functions.
