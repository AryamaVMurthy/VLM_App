# GraphPilot Characterization and Ablations Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add paper-grade characterization, ablation, and baseline-comparison artifacts for GraphPilot-Edge across the workload universe and current support-safe baseline suite.

**Architecture:** Add a dedicated analysis script that consumes the workload universe, surrogate hardware topology, baseline registry, candidate plans, profiler evidence, and calibration artifacts to generate machine-readable characterization and ablation summaries. Then extend the artifact-pack builder so these new summaries become first-class plots, tables, report sections, and audit evidence.

**Tech Stack:** Python 3, unittest, JSON registries, GraphPilot workload/model simulators, surrogate hardware simulator, artifact-pack Markdown/SVG generation.

---

### Task 1: Add failing tests for characterization summary generation

**Files:**
- Create: `scripts/tests/test_run_graphpilot_characterization.py`
- Modify: `scripts/run_graphpilot_characterization.py`

**Step 1: Write failing tests**
- Assert the characterization script emits machine-readable sections for backend affinity, baseline comparisons, ablations, transfer matrix, batching curves, thermal curves, memory/KV curves, and knob frontiers.
- Assert the compound workflow baseline table includes `cpu_only`, `stage_greedy`, `static_best_map`, and `no_pipeline`.

**Step 2: Run the targeted tests and verify RED**
- Run: `env -u PYTHONHOME -u PYTHONPATH python3 -m unittest scripts.tests.test_run_graphpilot_characterization -v`

**Step 3: Implement the minimal characterization script**
- Create the loader/helpers and summary writer.

**Step 4: Run the targeted tests and verify GREEN**
- Run: `env -u PYTHONHOME -u PYTHONPATH python3 -m unittest scripts.tests.test_run_graphpilot_characterization -v`

### Task 2: Add failing tests for artifact-pack integration

**Files:**
- Modify: `scripts/tests/test_build_graphpilot_artifact_pack.py`
- Modify: `scripts/build_graphpilot_artifact_pack.py`

**Step 1: Write failing tests**
- Assert the artifact pack includes characterization/ablation sections in `report.md`, `paper_tables.md`, `paper_draft.md`, `summary.json`, and plot registry entries.

**Step 2: Run targeted tests and verify RED**
- Run: `env -u PYTHONHOME -u PYTHONPATH python3 -m unittest scripts.tests.test_build_graphpilot_artifact_pack -v`

**Step 3: Implement the minimal integration**
- Load latest characterization summary, generate SVGs, and propagate tables/report sections.

**Step 4: Run targeted tests and verify GREEN**
- Run: `env -u PYTHONHOME -u PYTHONPATH python3 -m unittest scripts.tests.test_build_graphpilot_artifact_pack -v`

### Task 3: Generate fresh characterization artifacts and verify the wider test surface

**Files:**
- Modify: `scripts/generate_graphpilot_baseline_registry.py` only if needed for wider workload coverage
- Create or update: `artifacts/graphpilot_edge/analysis/graphpilot_characterization_*`

**Step 1: Run the new script on the current registries**
- Run: `env -u PYTHONHOME -u PYTHONPATH python3 scripts/run_graphpilot_characterization.py`

**Step 2: Rebuild the artifact pack**
- Run: `env -u PYTHONHOME -u PYTHONPATH python3 scripts/build_graphpilot_artifact_pack.py`

**Step 3: Run the full verification surface**
- Run: `env -u PYTHONHOME -u PYTHONPATH python3 -m unittest discover -s tests/graphpilot_edge -v`
- Run: `env -u PYTHONHOME -u PYTHONPATH python3 -m unittest discover -s scripts/tests -v`

### Task 4: Track, commit, and push

**Files:**
- Update Beads notes and close or advance `fvlm-i6n.10`
- Commit all tracked changes
- Push branch `codex/litert-deepdive`

**Step 1: Update Beads with artifact paths and verification evidence**
**Step 2: Commit**
**Step 3: Push**
