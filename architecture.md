# Space Netra — Proposed Architecture

## 1. Architecture principles

1. **Extend, do not replace.** Keep the current Streamlit app and working CLIP/FAISS path. Add change detection as a separate feature/module.
2. **Separate two data problems.** Semantic image retrieval indexes individual images and text-image embeddings; temporal change detection needs a comparable, georeferenced image pair with acquisition dates and alignment. They must not share a storage directory by accident.
3. **Make every run reproducible.** Store the AOI, item IDs, dates, resolution, CRS, masks, thresholds, and output paths.
4. **Treat output as candidate change.** The baseline does not identify the cause of a change.
5. **Make geospatial metadata first-class.** A normal PNG/JPG screenshot is not enough for a physically meaningful area calculation.

## 2. Logical architecture

```text
                           SPACE NETRA / Streamlit
                                      |
                   +------------------+-------------------+
                   |                                      |
                   v                                      v
          SEMANTIC SEARCH MODE                  CHANGE DETECTION MODE
          (existing path)                       (new path)
                   |                                      |
          data/raw/ image archive                 AOI + date selection
                   |                                      |
          ingest.py / CLIP                           STAC search
                   |                                      |
          image embeddings                          Sentinel-2 L2A items
                   |                                      |
          vector_db/ FAISS index                    band loading / crop
                   |                                      |
          text-to-image search                      masks + grid alignment
                   |                                      |
          retrieved images                          baseline change detector
                   |                                      |
                   v                                      v
            search results                      map / mask / heatmap / metrics
                   |                                      |
                   +------------------+-------------------+
                                      |
                                      v
                         shared UI shell / report links
```

## 3. Components and responsibilities

| Component | Responsibility | Status |
|---|---|---|
| `app.py` | Streamlit navigation and UI; connects user controls to modules | Existing per README; inspect before editing |
| `ingest.py` | Generate CLIP embeddings and build FAISS index for search archive | Existing per README; preserve behavior |
| `vector_db/` | Existing vector index and metadata for semantic search | Existing state; back up and do not overwrite during pilot |
| `change_detector.py` | Current or evolving image-comparison core | Existing per README; inspect and avoid silently breaking interfaces |
| `download_images.py` | Existing sample-image downloader | Existing per README; do not assume it fetches current/historical Sentinel data until code is inspected |
| `satellite_catalog.py` (proposed) | Search STAC, filter candidate observations, return item/asset metadata | New module |
| `satellite_loader.py` (proposed) | Load needed Sentinel-2 bands, crop/window to AOI, persist source metadata | New module |
| `preprocess.py` (proposed) | Cloud/shadow mask, CRS/grid harmonization, nodata handling, normalization | New module |
| `change_pipeline.py` (proposed) | Orchestrate preprocessing, detector, metrics, output saving | New module |
| `area_metrics.py` (proposed) | Geospatially sound pixel counts/area calculations | New module |
| `reporting.py` (proposed) | Build result JSON/CSV and downloadable human-readable report | New module or existing report helper |
| `config.py` (proposed) | AOI, source endpoint, thresholds, output folders and safe defaults | New module |

The proposed filenames are a design suggestion, not a claim that these files already exist. Adapt them to the actual project after inspecting the IDE tree.

## 4. Proposed folder layout

Keep existing paths as they are until a backup and code inspection are complete. Add new folders rather than moving the live project around.

```text
SpaceNetra/
├── app.py
├── change_detector.py
├── ingest.py
├── download_images.py
├── satellite_catalog.py       # proposed
├── satellite_loader.py        # proposed
├── preprocess.py              # proposed
├── change_pipeline.py         # proposed
├── area_metrics.py            # proposed
├── reporting.py               # proposed
├── config.py                  # proposed
├── data/
│   ├── raw/                   # existing archive used by current workflow; keep intact
│   └── change_detection/      # new temporal imagery workflow only
│       ├── aoi/
│       │   └── sambhajinagar_aoi.geojson
│       ├── source/            # source rasters or documented asset refs
│       ├── aligned/            # masked/reprojected pair
│       ├── masks/              # valid/cloud/shadow/change masks
│       ├── outputs/            # overlays, heatmaps, metrics
│       └── manifests/          # JSON/CSV metadata per run
├── vector_db/                 # existing FAISS index + metadata; preserve
├── reports/                   # optional exported analysis reports
├── tests/                     # proposed unit/smoke tests
├── docs/                      # this planning pack, if copied into the repo
├── requirements.txt            # inspect and update deliberately
└── .gitignore
```

Do not commit large raw imagery or the full FAISS index unless that is deliberately intended and compatible with the repository's sharing policy. Use `.gitignore` for generated raster files and local cache, while committing small AOI/manifests and documented example outputs where appropriate.

## 5. Data contracts

### AOI input

- GeoJSON polygon or bounding box in longitude/latitude (EPSG:4326) at input boundaries.
- Store the exact AOI and an identifier so future runs compare the same place.
- Check that the AOI is valid and intersects the selected image footprint.

### Scene record

Store at minimum:

- provider/catalog and collection;
- STAC item ID and source asset URLs;
- observation datetime;
- AOI coverage percentage;
- scene-level cloud metadata and AOI-level valid-pixel/cloud estimate if calculated;
- chosen band names and native spatial resolution;
- license/source link;
- CRS, transform, bounds, width, height, nodata, and processing version.

### Processed pair

Each date must use the same target CRS, transform/grid, dimensions, AOI, pixel size, and valid-data logic. Store the pair manifest and reference it from all outputs.

### Result record

Store pixel counts, valid-pixel count, candidate-change count, percentage of valid pixels flagged, algorithm name/version, thresholds, preprocessing parameters, output paths, and validation notes. Store area in m²/ha/km² only when correctly derived from the georeferenced raster grid in a metric CRS.

## 6. Main sequence

1. UI obtains or loads the AOI and date window.
2. Catalog service searches Sentinel-2 L2A observations and returns candidates.
3. User or deterministic selection logic chooses a suitable pair/composite.
4. Loader retrieves only required bands/windows and records provenance.
5. Preprocessor masks clouds/shadows, clips, reprojects/resamples, and checks alignment.
6. Baseline detector generates change score and mask.
7. Metrics component excludes invalid pixels and computes counts/physical area where valid.
8. Reporting module saves outputs/manifest and returns result paths.
9. Streamlit shows imagery, overlay, metrics, metadata, and caveats.
10. Existing semantic-search mode continues to use its current image archive, CLIP embeddings, and FAISS metadata unchanged.

## 7. Reliability and failure behavior

- Fail clearly if no scenes are found or the selected image pair has inadequate AOI coverage.
- Do not interpret nodata/clouds as change.
- Do not silently interpolate/resample different dates to convenient dimensions without retaining CRS/resolution metadata.
- If registration/alignment checks fail, stop and ask for another pair.
- If no valid pixels remain, return “insufficient valid imagery,” not “no change.”
- If the algorithm flags zero changes, distinguish that result from a failed download or empty mask.
- Log the source items and configuration for every run.

## 8. Architecture decision record

**Decision:** Use a modular Python/Streamlit application for the one-week pilot, not a separate Flask service or a new frontend.  
**Reason:** The current product is already documented as Streamlit; an API rewrite adds scope without improving the core geospatial result.  
**Revisit when:** another application needs to call the analysis independently, the work becomes a multi-user service, or batch processing must run outside the UI.
