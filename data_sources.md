# Space Netra — Satellite Imagery and Dataset Sources

## Decision summary

For the first local pilot, begin with **Sentinel-2 Level-2A** imagery and search by a small AOI near Chhatrapati Sambhajinagar. Sentinel-2 is open and practical to acquire, but its 10 m bands support broad land-surface or larger urban-area change analysis—not confident individual-building detection. First confirm that at least two suitable observations exist for the exact AOI.

A smaller AOI makes the data lighter to process; it does **not** make the source imagery higher resolution.

## A. Sources for local pilot imagery

### 1. Microsoft Planetary Computer — recommended first path

- Catalog: <https://planetarycomputer.microsoft.com/api/stac/v1/>
- Documentation: <https://planetarycomputer.microsoft.com/docs/>
- Sentinel-2 L2A dataset: <https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a>
- STAC tutorial: <https://stackstac.readthedocs.io/en/stable/basic.html>

**Use:** programmatically discover Sentinel-2 L2A scenes by AOI and date; inspect item metadata and load only the required data/bands. The Planetary Computer docs state that the STAC API is publicly accessible and most data can be downloaded anonymously, subject to throttling. Confirm current catalog collection/asset names and access limits when implementing.

**Strengths:** useful for quick Python/STAC discovery; no new app rewrite; supports a repeatable data query.  
**Limitations:** access may be throttled; remote assets can change; scene-level cloud percent may not reflect cloud coverage inside the AOI. Cache responsibly and save IDs/source metadata.

### 2. Copernicus Data Space Ecosystem — official fallback/primary alternative

- Portal: <https://dataspace.copernicus.eu/>
- Documentation: <https://documentation.dataspace.copernicus.eu/>
- Terms: <https://dataspace.copernicus.eu/terms-and-conditions>
- Sentinel legal notice: <https://cds.climate.copernicus.eu/licences/ec-sentinel>

**Use:** search, visualize, and download official Sentinel products or integrate its supported APIs. Copernicus Sentinel data is provided on a free/full/open basis under its legal notice, subject to service access limits and legal conditions.

**Strengths:** authoritative distribution source, broad archive, clear Sentinel-data policy.  
**Limitations:** API authentication/setup may take time; avoid spending the whole week on account/API configuration if the first catalog path works.

### 3. Google Earth Engine — optional route, not first choice for the one-week path

- Access guide: <https://developers.google.com/earth-engine/guides/access>
- Sentinel-2 catalog: <https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED>
- Terms: <https://earthengine.google.com/terms/>

Google Earth Engine can support research workflows and analysis, but access requires a configured Cloud project and registration for noncommercial/commercial use. Do not start with it unless you already have access and understand the terms. Do not scrape Google satellite-map tiles or treat a browser basemap screenshot as an open training dataset.

### 4. ISRO/NRSC Bhuvan and Bhoonidhi — useful India-specific context, check permissions

- Bhuvan: <https://bhuvan.nrsc.gov.in/home/>
- Bhuvan terms: <https://bhuvan.nrsc.gov.in/bhuvan/bhuvan_terms/>
- Bhoonidhi: <https://bhoonidhi.nrsc.gov.in/>
- ISRO Earth-observation overview: <https://www.isro.gov.in/SpaceBasedEarthObservationServices.html>

Bhuvan provides thematic/geospatial services and is valuable for India-specific reference layers. However, its terms restrict copying, modifying, creating derivative works, and redistribution without applicable authorization. Do not use its visual tiles as a scraped training corpus. Follow its formal access conditions for any data intended for processing or publication. Bhoonidhi may be relevant when requesting specific Indian remote-sensing products; availability, resolution, price, and permissions must be checked for the requested product.

## B. Model-training and benchmarking datasets

These datasets help develop or evaluate a model. They are not a substitute for finding a comparable temporal image pair of the local pilot AOI.

### 1. Onera Satellite Change Detection (OSCD) — best conceptual match to Sentinel-2 baseline

- Dataset site: <https://rcdaudt.github.io/oscd/>
- IEEE DataPort: <https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection>

The dataset describes 24 pairs of multispectral Sentinel-2 images from different world locations, with 13 bands at 10/20/60 m resolutions and pixel-level change ground truth (14 train and 10 test pairs on the project page). It is particularly relevant to testing a multi-spectral change-detection pipeline because it uses Sentinel-2-like data and labels. It is small and geographically diverse, so it should not be treated as proof of performance in Chhatrapati Sambhajinagar.

### 2. LEVIR-CD — high-resolution building-change benchmark

- Dataset site: <https://justchenhao.github.io/LEVIR/>
- Associated code/data description: <https://github.com/ChenHongruixuan/ChangeDetectionRepository/blob/master/Dataset/LEVIR-CD/README.md>
- Paper: <https://www.mdpi.com/2072-4292/12/10/1662>

LEVIR-CD contains 637 pairs of very-high-resolution 0.5 m/pixel images, 1024×1024 patches, with binary change masks focusing on building growth/decline. The dataset page states that the images and annotations are for academic use only and prohibits commercial use; its images originate from Google Earth, so the stated Google Earth terms also matter. It is a useful benchmark for an academic building-change model but is not local to Maharashtra, and its high resolution differs sharply from Sentinel-2 10 m data. Do not mix this training data with local Sentinel-2 inputs without a clear domain/resolution strategy.

### 3. SpaceNet 7 — strongest future option for temporal urban development/building tracking

- Challenge/dataset page: <https://www.spacenet.ai/sn7-challenge/>
- SpaceNet dataset catalogue: <https://spacenet.ai/datasets/>
- AWS Open Data registry: <https://registry.opendata.aws/spacenet/>
- Paper: <https://openaccess.thecvf.com/content/CVPR2021/html/Van_Etten_The_Multi-Temporal_Urban_Development_SpaceNet_Dataset_CVPR_2021_paper.html>

SpaceNet 7 provides around 24 monthly observations at about 4 m/pixel across roughly 100 diverse geographies, with time-series building-footprint labels. It is well suited to research on urban development/building tracking and much closer to a temporal time-series task than a single-image building detector. It is still not a local Chhatrapati Sambhajinagar dataset and does not magically guarantee performance on local imagery. Check the dataset's license, download size, access path, and label/task format before use.

### 4. SpaceNet general datasets — for building/road extraction tasks

- Official catalogue: <https://spacenet.ai/datasets/>

SpaceNet includes high-resolution imagery with building-footprint and road-network labels; the official page lists multiple geographic areas, including Mumbai in its geographical dataset catalogue. Check each specific dataset's license and task before training. Building/road extraction is not automatically temporal change detection; for a change model, you need paired time points and compatible change/temporal labels.

### 5. WHU Building Dataset — building extraction, not a direct temporal pair benchmark

- Official dataset page: <https://gpcv.whu.edu.cn/data/building_dataset.html>

This dataset includes building samples/footprints at high resolution. It can be useful for a future building segmentation experiment, but it is not the first choice for training a temporal change detector unless you deliberately construct and label matched before/after data. Check license and terms for each subset.

## C. What source should the pilot use?

Recommended order:

1. **Search Sentinel-2 L2A in Planetary Computer** for the exact local AOI and inspect candidate dates.
2. If access or results are inadequate, search/download via Copernicus Data Space.
3. Use Bhuvan/Bhoonidhi only under the correct data-access terms.
4. Use OSCD as a backup validation/benchmark dataset for pipeline testing, clearly separated from the local-pilot claim.
5. Defer LEVIR-CD/SpaceNet 7 training until the target task, resolution, labels, compute budget, and license have been decided.

## D. Required data provenance manifest

For every source image/pair, record:

- provider, product/collection, item/product ID;
- source URL and access date;
- sensing date/time and, for composites, date window and composition method;
- AOI identifier and coverage;
- native spatial resolution and bands;
- CRS/grid metadata;
- cloud metadata and applied quality mask;
- license/terms reference and attribution requirement;
- any modifications, crops, reprojection, scaling or derived masks;
- whether the data is local pilot, training, validation, or demonstration-only.

## E. Practical interpretation of “high resolution”

- Sentinel-2: 10 m for its standard visible/NIR bands, suitable for larger landscape patterns and larger changes; not an individual-building solution in dense urban settings.
- SpaceNet 7: about 4 m imagery with temporal building-footprint labels; useful for research, but not automatically local.
- LEVIR-CD: 0.5 m/pixel high-resolution paired benchmark, academic-use restriction on the dataset page and non-local domain.
- Commercial/provider-specific high-resolution imagery: potentially needed for local building-level pilot; availability, price, rights to process/store/share, and acquisition dates must be confirmed first.

The “small area” is an engineering scoping decision. The sensor and product determine pixel resolution.
