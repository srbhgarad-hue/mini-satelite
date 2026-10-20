# Space Netra — Recommended Tech Stack and Application

## Recommendation in one sentence

Keep **Python + Streamlit + Folium + Plotly + OpenCV/scikit-image + CLIP + FAISS** for the current product, and add a small geospatial acquisition/preprocessing layer built with **PySTAC Client, Rasterio, GeoPandas/Shapely, PyProj, and optionally planetary-computer**. Do not retrain a model for the first-week pilot.

## 1. Technology decisions

| Layer / technology | Recommended use | Why it belongs | Priority |
|---|---|---|---|
| Python 3.11 or the project's currently compatible supported Python version | Main implementation | Fits current code and the scientific Python ecosystem. Verify actual installed Python/PyTorch/FAISS compatibility before changing versions. | P0 |
| Streamlit | Dashboard and interaction | Existing documented frontend; fast to extend with a dedicated change-detection section. | P0 |
| Folium + `streamlit-folium` | Draw/inspect AOI and display location context | Fits current map UI; use a GeoJSON AOI as the analytical boundary. | P0 |
| Plotly | Change distribution charts and simple metrics | Existing dependency; helps show change statistics and comparison summaries. | P1 |
| `pystac-client` | Search a STAC catalog for date/location-filtered scenes | Standard Python client pattern for spatial-temporal catalog discovery. | P0 |
| `planetary-computer` | Sign/access Planetary Computer asset URLs when needed | Useful when consuming Microsoft's public STAC catalog and protected asset URLs; obey catalog rate limits. | P0 if using this provider |
| `rasterio` | Read GeoTIFF/COG windows, preserve transforms/CRS, reproject, write GeoTIFFs | Core geospatial raster IO for defensible pixel and area calculations. | P0 |
| `geopandas`, `shapely`, `pyproj` | Validate AOI, geometry operations, CRS transformations and metric calculations | Needed for robust geospatial clipping/geometry handling. | P0/P1 |
| `numpy`, `pandas` | Array computation, manifests, summary tables | Already likely central to project and broadly suitable. | P0 |
| `opencv-python` | Threshold cleanup, contours, morphology, image alignment diagnostics if useful | Already in requirements; retain for baseline processing. | P0 |
| `scikit-image` | SSIM and image-processing utilities | Already in requirements; use SSIM as one change signal rather than treating it as semantic classification. | P0 |
| `torch`, `torchvision`, `CLIP` | Existing semantic image search | Keep isolated in the existing search path. Do not load CLIP in the change-detection run unless there is a specific feature that needs it. | Existing |
| `faiss-cpu` | Existing vector nearest-neighbor search | Preserve the current index and metadata. It is not the temporal change-detection engine. | Existing |
| `rioxarray`, `xarray`, `stackstac` | Optional for multi-date cube/composite workflows | Useful when processing more dates or many bands; likely unnecessary for one image pair if Rasterio is enough. | Later/optional |
| `pytest` | Automated tests for preprocessing/metrics and no-data paths | Helps avoid breaking existing behavior. | P1 |
| SQLite | Optional simple run-history metadata | Add only if flat JSON/CSV manifests no longer suffice; not needed for the first pilot. | Later |
| Flask/FastAPI | Separate API only when required | Not needed for a one-week Streamlit-only demo; avoid extra service layer for now. | Not now |
| Docker | Reproducible packaging after core pipeline works | Do not spend first-week time moving the current setup into containers unless installation reproducibility becomes a blocking issue. | Later |

## 2. Satellite data access choice

### Primary recommended route: Microsoft Planetary Computer STAC

Use its public STAC catalog to discover Sentinel-2 L2A items by AOI/date and filter candidates by scene-level cloud metadata. Its documentation describes public STAC access and Sentinel-2/Landsat datasets, with anonymous access subject to throttling. Use only the window/bands needed for the AOI.

- Catalog: <https://planetarycomputer.microsoft.com/api/stac/v1/>
- Docs: <https://planetarycomputer.microsoft.com/docs/>
- Sentinel-2 L2A overview: <https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a>

The exact collection and item assets must be confirmed in the catalog at implementation time. The code should not hard-code assumptions about every band being stored in the same way.

### Fallback route: Copernicus Data Space Ecosystem

Use the official Copernicus data space search/download APIs or browser if the first catalog path is inaccessible or its limits block the workflow. Sentinel data is available on a free/full/open basis subject to the official legal notice and service rules.

- Portal: <https://dataspace.copernicus.eu/>
- API documentation: <https://documentation.dataspace.copernicus.eu/>
- Terms: <https://dataspace.copernicus.eu/terms-and-conditions>

### Caution for Bhuvan and web-map basemaps

ISRO/NRSC's Bhuvan terms restrict copying, modifying, creating derivative works, and redistribution without the required authorization. Do not automate tile-scraping or use Bhuvan/Google/other map screenshots as a training archive unless applicable terms and permissions clearly allow it. Use Bhuvan as a viewing/reference portal or follow its formal data-access process. The source of every image used in a demo or dataset should be recorded.

## 3. Resolution and algorithm decisions

| Question | Pilot decision |
|---|---|
| Image resolution | Start with Sentinel-2 10 m visible/NIR bands for a broad land-surface/urban-change demo. |
| Small AOI | Use it to lower data volume and simplify review; do not describe it as a resolution enhancement. |
| Change detector | Begin with valid-pixel mask + reflectance/spectral difference score + threshold/morphology. Add SSIM as a complementary structural signal/visualization. |
| Cloud/shadow | Use Sentinel-2 scene classification / available quality assets where available, plus item metadata. Scene-level cloud cover alone is not sufficient. |
| Area metrics | Use Rasterio affine transform and metric CRS; mask nodata/clouds first. Do not calculate hectares from an unreferenced PNG. |
| Model training | No custom training in week one. Benchmark/open-data training is a later workstream. |
| Existing vector DB | Leave as-is. Keep CLIP/FAISS indexing path separate from geospatial temporal imagery. |

## 4. Dependency additions to consider

Inspect the existing `requirements.txt` and Python environment before installing. Likely additions for the new workflow are:

```text
pystac-client
planetary-computer
rasterio
geopandas
shapely
pyproj
```

Optional only if the implementation uses them:

```text
rioxarray
xarray
stackstac
pytest
```

These are candidate dependencies, not a tested lockfile. Install into the existing project's intended virtual environment, then test imports and the current app before continuing. Avoid upgrading torch/torchvision/FAISS/NumPy blindly; resolve the current environment first.

## 5. What not to add yet

- No custom CNN/Transformer training loop.
- No new React/Vue frontend.
- No separate Flask API service.
- No cloud database, Kubernetes, or complex orchestration.
- No vector database replacement; FAISS is already present.
- No large full-scene downloads when only a small AOI/window is required.

## 6. Research references

- Sentinel-2 band resolution and harmonized time series in the Earth Engine dataset catalog: <https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED>
- Microsoft Planetary Computer documentation: <https://planetarycomputer.microsoft.com/docs/>
- STAC Python search walkthrough: <https://pystac-client.readthedocs.io/en/stable/quickstart.html>
- `stackstac` example filtering by time and stacking Sentinel-2 data: <https://stackstac.readthedocs.io/en/stable/basic.html>
- Copernicus data legal notice: <https://cds.climate.copernicus.eu/licences/ec-sentinel>
