# Space Netra — Model Strategy and Training Decision

## Decision for the first week

**Do not train a new deep-learning model for the first-week Chhatrapati Sambhajinagar pilot.** Build and validate an explainable image-processing baseline first. This is the fastest way to establish that the source data, geo-alignment, masking, and UI work correctly.

The existing four or five images in the CLIP/FAISS vector database are not a change-detection training set unless they are paired by time, spatially matched, consistently processed, and supplied with trustworthy change masks. An embedding index does not contain change labels.

## 1. What the pilot baseline should do

1. Input two observations of the same AOI from different dates.
2. Use equivalent bands/resolution and a valid-pixel mask.
3. Remove clouds/shadows/nodata where relevant.
4. Align the images to the same geospatial grid.
5. Compute a transparent change score from reflectance/spectral differences and optionally SSIM.
6. Threshold the score and apply modest noise cleanup.
7. Display candidate areas and report parameters/limits.

This is a **candidate-change detector**, not a classifier that can reliably label “new building,” “road,” “demolition,” or “illegal activity.” At 10 m resolution, limit interpretation to broader land-surface/urban-area changes.

## 2. Why SSIM alone is not enough

SSIM measures structural similarity between image windows. It is useful for comparison/visualization, but its output can be affected by registration, illumination, clouds/shadows, seasonal vegetation, water, atmospheric/processing differences, and display stretching. It also does not assign semantic classes to changes.

Use a geospatially controlled pipeline first, and compare more than one signal where helpful. Keep the formulas/thresholds visible and configurable. Never interpret every thresholded pixel as verified ground change.

## 3. When model training becomes justified

Move to supervised training only if all are true:

- The desired target is specific, e.g. “new building footprint” rather than generic image difference.
- There are paired temporal images with compatible bands/resolution and accurate labels or masks.
- Train/validation/test splits can be made by location or acquisition group to reduce leakage.
- The training data's license permits the intended academic and future pilot use.
- A baseline score has been recorded so the trained model can be compared honestly.
- The team's one-week project deadline no longer depends on the model being trainable and debugged.

## 4. Dataset choices by task

| Target task | Recommended dataset | Why | Major caveat |
|---|---|---|---|
| Multispectral changes at Sentinel-2-like resolution | OSCD | Paired multispectral images and change masks; good pipeline benchmark. | Only 24 image pairs, diverse places, not specific to Maharashtra. |
| Building growth/decline at very high resolution | LEVIR-CD | 637 paired 0.5 m image patches and binary change masks. | Dataset page says academic use only/no commercial use; non-local source imagery and strong resolution/domain mismatch. |
| Temporal urban development/building tracking | SpaceNet 7 | About 24 monthly ~4 m observations across many regions with building footprints over time. | Significant download and processing work; labels/task differ from simple binary mask; not local imagery. |
| Single-image building/road segmentation | Selected SpaceNet/WHU subsets | Useful for extracting objects from a single date. | Object extraction is not the same as detecting change between dates. Check per-subset license. |

Links and further source details are in [`data_sources.md`](data_sources.md).

## 5. Potential later model architecture

If the team chooses building-level change detection, evaluate a Siamese U-Net or a Transformer-based change-detection model (e.g. ChangeFormer) on a suitable paired dataset. Do not lock an architecture before defining the data, target labels, and output resolution. Training from scratch should not be the default; transfer learning or a published baseline is a more reasonable experiment.

A locally trained model will need a clear path from downloaded data to chips, labels, splits, augmentations, training, evaluation, and inference. It should not be bolted into `app.py` without a standalone evaluation script.

## 6. Recommended evaluation

When ground-truth binary change masks are available, calculate precision, recall, F1/Dice, and IoU on a held-out test split. State the number of pairs/tiles, resolution, geographic source, date range, class imbalance, and split method. Avoid random tile splits when neighboring tiles from the same scene can leak similar content into both train and test sets.

For the local pilot without ground truth:

- use a manually reviewed set of changed/unchanged patches;
- record the review protocol and common false positives;
- report qualitative review and candidate-change coverage;
- do not claim “95% accurate” or similar numbers without defensible labels and metrics.

## 7. Local-data plan for future training

If a later local building-level model is approved, collect or obtain temporally matched image pairs and label a representative subset of the actual target AOI. Record permissions and source metadata. Labels should distinguish at least change / unchanged / ignore-invalid where appropriate, and should exclude cloudy/unusable regions. A pre-trained model evaluated on local pairs may be the first sensible step before training local weights.

## 8. Stop conditions

Do not train yet if any applies:

- The only data is the current 4–5 unlabelled image archive.
- There are no change masks and no annotation capacity.
- The local data is 10 m but the success criterion is precise individual-building detection.
- No held-out evaluation plan exists.
- The source imagery terms are unclear.
- Training would threaten completing the working pilot.
