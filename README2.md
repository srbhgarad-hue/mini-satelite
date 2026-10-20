# Space Netra — Mini-Project Planning Pack

**Project status:** Existing student project being refined into a small-area pilot for Chhatrapati Sambhajinagar, Maharashtra. This is **not an SIH submission plan**.

## Read these documents in order

1. [`plan.md`](plan.md) — scope, priorities, one-week schedule, acceptance criteria.
2. [`architecture.md`](architecture.md) — system components, data boundaries, proposed folder layout.
3. [`tech.md`](tech.md) — recommended technologies and the role of each one.
4. [`pipeline.md`](pipeline.md) — end-to-end implementation pipeline.
5. [`data_sources.md`](data_sources.md) — satellite sources, access notes, and training datasets.
6. [`model_strategy.md`](model_strategy.md) — whether to train a model, baseline choice, future training plan.
7. [`open_questions.md`](open_questions.md) — decisions that still need answers, with a recommended default for each.

## The key decision

For the first-week pilot, prioritize a reproducible change-detection workflow on **one small, precisely defined AOI** near Chhatrapati Sambhajinagar. Keep the current Streamlit app and CLIP/FAISS semantic-search feature intact. Add the new temporal imagery workflow as a separate module and store its imagery separately from the existing vector-search archive.

Start with free Sentinel-2 Level-2A imagery at 10 m for the selected visible bands. This is suitable for demonstrating broad land-cover or larger urban-area changes, but **reducing the AOI does not increase image resolution**. Do not claim reliable individual-building detection from 10 m imagery. A building-level pilot needs suitable finer-resolution imagery and appropriate usage rights.

## Existing project assets to preserve

The current project documentation describes `app.py`, `change_detector.py`, `ingest.py`, `download_images.py`, `data/raw/`, and `vector_db/`. The project owner reports that their IDE project already has a vector database indexed from approximately 4–5 images. Treat that database as valuable existing state:

- Back it up before changing ingestion code or folder paths.
- Do not delete, regenerate, or silently replace `vector_db/spatial.index` or its metadata.
- Do not mix temporal change-detection raster products into the CLIP/FAISS image archive by default.
- Keep image-search ingestion and temporal-pair processing as two separate pipelines.
- Four or five indexed images are enough to smoke-test retrieval plumbing, not enough to train or validate a change-detection model.

## Important limitation

These are plans based on the currently shared README and requirements list, plus external research. The actual Python implementation, the current FAISS index schema, and local satellite-data availability have not yet been inspected or run. Confirm them in the IDE before integrating changes.
