# Dispatch to Explorer 3 (Sentinel-2, Semantic Archive Search, Data Provenance & Safety)

- **Target Agent**: explorer_survey_3 (teamwork_explorer)
- **Working Directory**: D:\PythonProjects\mini-satelite\.agents\teamwork\explorer_survey_3\
- **Original Request Path**: D:\PythonProjects\mini-satelite\.agents\teamwork\ORIGINAL_REQUEST.md
- **Parent Conversation ID**: c8b0d635-6058-4219-9931-c46dafaa9330

## Objective
Thoroughly explore the Sentinel-2 analysis pipeline, Semantic Archive Search (FAISS/CLIP), data provenance, synthetic data isolation, and safety safeguards to support requirements R3 (Sentinel-2 & Semantic Search), R4, R5, and R6.

## Tasks
1. Read `D:\PythonProjects\mini-satelite\.agents\teamwork\ORIGINAL_REQUEST.md`.
2. Inspect `satellite_loader.py`, `preprocessor.py`, `satellite_catalog.py`, `photo_streamer.py`, and `data_sources.md`.
3. Investigate the `AttributeError: clean_scene` and any interface mismatches between `preprocessor.py` and `satellite_loader.py`.
4. Inspect handling of real vs synthetic Sentinel-2 Level-2A data, CRS/transform/pixel dimension requirements, SCL/cloud masking, and isolation as "Experimental — Awaiting verified raster data" with developer-only TEST/SYNTHETIC labeling.
5. Inspect `vector_db/`, `vector_db_backup/`, `ingest.py`, and semantic search implementation (CLIP + FAISS) to ensure index preservation, safe querying without re-indexing, and similarity scores decoupled from change-detection confidence.
6. Inspect `DATA_PROVENANCE.md` and codebase metrics to identify unverified numbers ("New Buildings Found", "Built-Up Density", "Ground Resolution", "Expansion Level") that must be replaced with "Not available" or "Awaiting verified data".
7. Inspect existing tests `test_backend_services.py`, `test_provenance_safety.py`, and `test_semantic_search.py`.
8. Enumerate all features, constraints, risks, and interface contracts for Sentinel-2, Semantic Search, and Data Provenance.
9. Write your structured findings to `D:\PythonProjects\mini-satelite\.agents\teamwork\explorer_survey_3\handoff.md`.


## 2026-11-07T16:18:17Z
[Message] sender=c8b0d635-6058-4219-9931-c46dafaa9330 priority=MESSAGE_PRIORITY_HIGH
You are explorer_survey_3.
Working directory: D:\PythonProjects\mini-satelite\.agents\teamwork\explorer_survey_3\
Original request path: D:\PythonProjects\mini-satelite\.agents\teamwork\ORIGINAL_REQUEST.md
Parent conversation ID: c8b0d635-6058-4219-9931-c46dafaa9330

Please read your task in D:\PythonProjects\mini-satelite\.agents\teamwork\explorer_survey_3\DISPATCH.md.
Explore the Sentinel-2 analysis pipeline (preprocessor, loader, AttributeError), Semantic Archive Search (FAISS/CLIP), data provenance, synthetic data isolation, and safety safeguards.
Follow the Explorer archetype instructions, create your progress.md, investigate code thoroughly without modifying any files, and write a structured handoff report to D:\PythonProjects\mini-satelite\.agents\teamwork\explorer_survey_3\handoff.md. When done, send a message to parent with the summary and path to your handoff.md.
