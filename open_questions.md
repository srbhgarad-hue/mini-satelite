# Space Netra — Open Questions That Must Be Resolved

This document is a decision log. The recommended defaults below keep the project moving; the team should confirm or change them deliberately. Do not treat a question as resolved merely because the UI can display a result.

## Priority 0 — Decide before writing the new pipeline

### 1. What exact change are we trying to detect?

**Options:** broad land-surface change; urban expansion; new buildings; vegetation/water change; generic visual differences.  
**Recommended default:** broad candidate land-surface/urban-area change for the first pilot.  
**Why it matters:** the target controls image resolution, dataset choice, algorithm, label needs, and what you can responsibly claim.

### 2. What does “small scale with better resolution” mean?

**Recommended interpretation:** a small AOI to reduce data and make manual validation possible. Do not assume it increases satellite spatial resolution.  
**Decision needed:** is 10 m imagery sufficient, or does the evaluator expect individual buildings?  
**Recommended default:** accept Sentinel-2 10 m for a broad-change demo; if the requirement is building-level, first prove that suitable finer imagery and rights are available.

### 3. What exact AOI within Chhatrapati Sambhajinagar?

**Decision needed:** polygon/rectangle, approximate area, and whether it contains urban, peri-urban, agricultural, or water features.  
**Recommended default:** choose one clearly bounded 1–5 km² test area, then save its GeoJSON. Do not decide based only on map appearance; verify imagery coverage and visible change.

### 4. What time span should be compared?

**Decision needed:** dates and minimum time gap.  
**Recommended default:** search enough years to find a meaningful, reasonably cloud-free pair; choose similar seasons where possible. Record the selected dates and why they were chosen. The exact dates should be data-driven, not invented in advance.

### 5. What constitutes a “successful” result?

**Recommended default:** a reproducible detector produces a plausible overlay, valid-pixel statistics, and a limitations statement on one AOI. Do not define success as a particular accuracy percentage until suitable ground truth exists.

## Priority 1 — Decide during data feasibility work

### 6. Which satellite source should be primary?

**Recommended default:** Microsoft Planetary Computer STAC for discovery and Sentinel-2 L2A asset access, then Copernicus Data Space as fallback.  
**Decision needed:** whether catalog access works in the current network/environment, and whether data access limits fit the pilot.

### 7. Are there at least two usable scenes for the AOI?

**Decision needed:** item IDs, dates, scene footprints, cloud/shadow quality inside the AOI, bands, and download behavior.  
**Recommended action:** query first on Day 1, before building the detector. This remains open until inspected against the actual AOI.

### 8. Are we using single acquisitions or seasonal composites?

**Recommended default:** two individual acquisitions if each is sufficiently clear. If not, use comparable seasonal composites and clearly label them as composites.

### 9. What is the cloud/shadow strategy?

**Recommended default:** use available classification/quality assets plus item-level metadata; calculate or inspect AOI-level valid-pixel coverage; masked pixels must be excluded from difference scores and metrics.

### 10. What output resolution and CRS will be used?

**Recommended default:** one declared metric analysis grid with matching transform, width, height and resolution for both dates. Store native resolution and the resampling method. Select a suitable projected CRS after confirming the AOI.

## Priority 2 — Decisions for algorithm and reporting

### 11. Is SSIM the primary change detector or just one feature?

**Recommended default:** use a transparent spectral/image difference baseline and consider SSIM as a complementary structural signal. Compare outputs on reviewed patches before deciding. Avoid naming SSIM output as semantic or object recognition.

### 12. How will the threshold be chosen?

**Recommended default:** configurable threshold tuned on a small manually reviewed sample and saved with each run. Do not tune only for visually impressive output.

### 13. Can the system calculate actual affected area?

**Recommended default:** yes, but only from georeferenced pixels after invalid-pixel exclusion and in an appropriate metric CRS. Otherwise show counts/percentages and explicitly omit hectares/km².

### 14. What does a detected region mean?

**Recommended default:** “candidate visual/spectral change.” Only call it a building, road, vegetation loss, or construction if supported by a suitable classifier/validated method. Do not infer legality, intent, or a specific event from image difference alone.

### 15. What validation ground truth exists?

**Decision needed:** none, manual patch labels, locally annotated mask, or external benchmark.  
**Recommended default:** start with a manual review checklist for the first week; use OSCD for benchmark pipeline testing where useful; do not report model metrics without ground-truth masks.

## Priority 3 — Model training and future pilot

### 16. Do we actually need to train a model?

**Recommended default:** no for week one. Training needs labeled pairs and evaluation, not just indexed images.

### 17. What output category should a future model predict?

**Options:** binary change mask; new building footprint; building growth/demolition tracking; land-cover class transitions.  
**Recommended default:** choose one task before choosing a model architecture.

### 18. Which dataset is suitable and permitted?

**Recommended default:** OSCD for Sentinel-2-like change-detection benchmarking; SpaceNet 7 for temporal urban building tracking; LEVIR-CD for high-resolution building-change research with its academic-use restriction respected. Check licenses and domain/resolution differences.

### 19. How will the local pilot be validated?

**Recommended default:** reserve areas/dates not used for threshold tuning, or label a small set of local patches. If collecting labels, record who labeled them, definitions, ambiguous pixels, and agreement procedure.

### 20. What does “pilot within a week” include?

**Recommended default:** one AOI, one usable pair/composite, baseline mask, transparent metrics, a working dashboard section, and a reproducible report. A trained custom model, general city coverage, and production deployment are future milestones.

## Priority 4 — Product and engineering

### 21. Should the existing CLIP/FAISS search remain in the mini project?

**Recommended default:** yes, preserve the current feature if it works, but do not let it block change detection. Demonstrate it as a separate mode.

### 22. Can the current vector DB be safely modified?

**Recommended default:** do not modify it during change-pipeline development. Make a backup, inspect its metadata schema, and use a separate output directory. The reported 4–5 indexed images can support a smoke test, not training.

### 23. Do we need Flask/FastAPI, a SQL database, or Docker now?

**Recommended default:** no. Stay with Streamlit and JSON/CSV run manifests in the first week. Add services only for a concrete need.

### 24. What is allowed to be stored/shared in GitHub or the report?

**Recommended default:** save source IDs/metadata and small reproducible outputs; check each data provider's license before committing raw imagery. Never share access keys or personal credentials.

### 25. What is the next milestone after the one-week pilot?

**Recommended default:** decide based on evidence: (A) improve 10 m land-surface change quality, (B) obtain suitable finer imagery for building change, or (C) train/evaluate a benchmark model. Do not combine all three into the initial deliverable.

## Meeting decision record template

Copy this block for each team decision:

```text
Decision ID:
Question:
Options considered:
Decision:
Reason/evidence:
Who approved it:
Date:
Impact on scope/architecture:
What evidence would make us revisit it:
```

## Current unresolved facts

- Exact AOI coordinates/polygon: not provided yet.
- Actual local Sentinel-2 scene availability and AOI cloud conditions: not queried/verified yet.
- Current source-code implementation and environment compatibility: not inspected yet.
- Whether the academic evaluator expects building-level detection or broad land-cover change: not confirmed yet.
- Local labeled before/after imagery: not known to be available.
