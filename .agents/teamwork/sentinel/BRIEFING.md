# BRIEFING — 2026-11-07T16:12:00Z

## Mission
Coordinate the Space Netra professional application finalization run.

## 🔒 My Identity
- Archetype: sentinel
- Working directory: d:\PythonProjects\mini-satelite\.agents\teamwork\sentinel
- Orchestrator: c8b0d635-6058-4219-9931-c46dafaa9330
- Victory Auditor: to be spawned on victory claim

## 🔒 Key Constraints
- No technical decisions — relay only
- Victory Audit is MANDATORY before reporting completion
- R1: Five-page application separation (Overview, Optical, ML, Sentinel-2, Semantic Search) with zero bleeding
- R2: GIS dark navy/charcoal UI overhaul, fix comparison image overlapping in responsive 1-row cards, contextual sidebars
- R3: Technical decoupling of 4 systems (Optical, ML U-Net, Sentinel-2, Semantic Search FAISS)
- R4: Data provenance safety, remove hardcoded/unbacked metrics, isolate synthetic fixtures with TEST/SYNTHETIC label
- R5: Fix Sentinel-2 preprocessor interface (`clean_scene`), isolate/mark experimental if no verified L2A raster pixels
- R6: Strictly preserve checkpoints, datasets, FAISS index/metadata; no overwriting or unauthorized ingestion scripts

## User Context
- **Last user request**: Approved task prompt received
- **Pending clarifications**: none
- **Delivered results**: None

## Project Status
- **Phase**: in progress
- **Route**: General (teamwork_orchestrator)
- **Routing Rationale**: Multi-component SWE project across frontend, backend, ML inference, and raster pipelines.

## Victory Audit Status
- **Triggered**: no
- **Verdict**: pending
- **Retry count**: 0

## Artifact Index
- .agents/teamwork/ORIGINAL_REQUEST.md — Authoritative approved task prompt
