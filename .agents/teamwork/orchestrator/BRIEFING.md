# BRIEFING — 2026-11-07T16:19:00Z

## Mission
Transform Space Netra into a professional, reliable production geospatial change-detection platform across five independent modules.

## 🔒 My Identity
- Archetype: teamwork_orchestrator
- Roles: orchestrator, user_liaison, human_reporter, successor
- Working directory: D:\PythonProjects\mini-satelite\.agents\teamwork\orchestrator
- Original parent: Sentinel
- Original parent conversation ID: f857d917-43f0-4567-92b3-b3af60351b5e

## 🔒 My Workflow
- **Pattern**: Project
- **Scope document**: D:\PythonProjects\mini-satelite\PROJECT.md
1. **Decompose**: Map features from Survey phase into milestones along module boundaries (3-7 milestones); establish interface contracts.
2. **Dispatch & Execute**:
   - **Direct (iteration loop)**: Explorer (3) -> Worker (1) -> Reviewer (2) -> Challenger (2) -> Forensic Auditor (1) -> Gate (all must pass).
   - **Delegate (sub-orchestrator)**: Spawn a sub-orchestrator per milestone for multi-milestone project execution. Parallel E2E Testing Orchestrator for independent test suite.
3. **On failure**: Retry -> Replace -> Skip (non-critical) -> Redistribute -> Redesign -> Escalate (sub-orchestrator only).
4. **Succession**: At 16 spawns, write handoff.md, cancel crons, spawn successor, exit.
- **Work items**:
  1. Survey phase: Map features and existing architecture [in-progress]
- **Current phase**: 0 (Survey)
- **Current focus**: Step 0 - Survey existing codebase and map full scope before decomposing

## 🔒 Key Constraints
- NEVER write, modify, or create source code files directly.
- NEVER run build/test commands yourself — require workers to do so.
- NEVER investigate or explore the problem at the code level — dispatch Explorers for technical investigation.
- File-editing tools ONLY for metadata/state files (.md) in .agents/teamwork/ folder.
- Binary veto on Forensic Auditor integrity violations.
- Strictly preserve existing model checkpoints, datasets, FAISS indices/metadata, backups.
- Never reuse a subagent after it has delivered its handoff — always spawn fresh.

## Current Parent
- Conversation ID: f857d917-43f0-4567-92b3-b3af60351b5e
- Updated: 2026-11-07T16:15:00Z

## Key Decisions Made
- Initiated Project pattern with parallel Survey phase.
- Dispatched 3 parallel Explorers for full scope survey (UI/Nav, CV/ML, Sentinel-2/VectorDB/Provenance).

## Team Roster
| Agent | Type | Work Item | Status | Conv ID |
|-------|------|-----------|--------|---------|
| explorer_survey_1 | teamwork_explorer | App Navigation & GIS UI Survey | in-progress | 13fe8e13-6d39-45e8-826f-b9e777263cff |
| explorer_survey_2 | teamwork_explorer | Optical & ML Pipeline Survey | in-progress | d9881254-3127-4dac-94d3-f21b9177eb29 |
| explorer_survey_3 | teamwork_explorer | Sentinel-2 & Semantic Search Survey | in-progress | 17a4159a-b643-439b-8526-97a051fe36dd |

## Succession Status
- Succession required: no
- Spawn count: 3 / 16
- Pending subagents: 13fe8e13-6d39-45e8-826f-b9e777263cff, d9881254-3127-4dac-94d3-f21b9177eb29, 17a4159a-b643-439b-8526-97a051fe36dd
- Predecessor: none
- Successor: not yet spawned

## Active Timers
- Heartbeat cron: c8b0d635-6058-4219-9931-c46dafaa9330/task-34
- Safety timer: c8b0d635-6058-4219-9931-c46dafaa9330/task-61
- On succession: kill all timers before spawning successor
- On context truncation: handle timers as step 4 of the State Recovery section of your prompt says

## Artifact Index
- .agents/teamwork/ORIGINAL_REQUEST.md — Authoritative user request
- .agents/teamwork/orchestrator/DISPATCH.md — Dispatch log from Sentinel
- .agents/teamwork/orchestrator/plan.md — Orchestrator plan
- .agents/teamwork/orchestrator/progress.md — Liveness & status tracking
- .agents/teamwork/orchestrator/context.md — Context log
- PROJECT.md — High-level architecture, milestones, feature inventory (to be synthesized)
