# GraphPilot-Edge Runtime Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build GraphPilot-Edge as a measured, fallback-aware multimodal assistant runtime on SM8750, backed by explicit feasibility evidence, profiler data, planning/scheduling components, end-to-end workflow harnesses, and a reproducible artifact bundle.

**Architecture:** Preserve the existing FastVLM LiteRT NPU/CPU paths as the VLM anchor, add explicit stage adapters for ASR/planner/responder/TTS, and build a Python-side planning/simulation layer that consumes a machine-readable feasibility matrix and profiler registry. Integrate the chosen plans with Android instrumentation workflows for voice-only and voice+vision execution, while recording explicit infeasibility for retrieval if a real on-device embedder artifact is unavailable.

**Tech Stack:** Python 3, Android/Kotlin instrumentation, LiteRT-LM, FastVLM bridge, adb, Beads, JSON registries, unittest.

---

### Task 1: Close feasibility gaps and registry integrity

**Files:**
- Modify: `artifacts/graphpilot_edge/registries/backend_feasibility_matrix.json`
- Modify: `artifacts/graphpilot_edge/state/state_ledger.json`
- Modify: `artifacts/graphpilot_edge/reports/phase2_stage_feasibility_status.md`
- Create: `scripts/probe_graphpilot_retrieval_artifact.py`
- Test: `scripts/tests/test_probe_graphpilot_retrieval_artifact.py`

**Step 1:** Write failing tests for retrieval-artifact probing and explicit infeasibility output.
**Step 2:** Run the tests and verify failure.
**Step 3:** Implement the probe script with explicit host/device evidence and no silent fallback.
**Step 4:** Run the tests and the probe on the connected device.
**Step 5:** Update the feasibility matrix and state ledger using probe evidence.

### Task 2: Expand profiler coverage and profile ingestion

**Files:**
- Modify: `scripts/run_graphpilot_stage_profiler.py`
- Create: `scripts/ingest_graphpilot_profiler_run.py`
- Test: `scripts/tests/test_run_graphpilot_stage_profiler.py`
- Test: `scripts/tests/test_ingest_graphpilot_profiler_run.py`
- Modify: `artifacts/graphpilot_edge/registries/profiler_registry.json`

**Step 1:** Write failing tests for profiler-run ingestion and summary generation.
**Step 2:** Implement ingestion and reporting.
**Step 3:** Run profiler on all feasible stages/workflows.
**Step 4:** Verify profiler registry entries and derived summaries.

### Task 3: Implement Phase 4 planning core

**Files:**
- Create: `graphpilot_edge/catalog.py`
- Create: `graphpilot_edge/profiles.py`
- Create: `graphpilot_edge/planner.py`
- Create: `graphpilot_edge/scheduler.py`
- Create: `graphpilot_edge/memory.py`
- Create: `graphpilot_edge/kv.py`
- Create: `tests/graphpilot_edge/test_catalog.py`
- Create: `tests/graphpilot_edge/test_profiles.py`
- Create: `tests/graphpilot_edge/test_planner.py`
- Create: `tests/graphpilot_edge/test_scheduler.py`
- Create: `tests/graphpilot_edge/test_memory.py`
- Create: `tests/graphpilot_edge/test_kv.py`

**Step 1:** Add failing tests for config/registry loading, fallback-aware costed planning, list scheduling, interval reuse, and KV admission/degradation.
**Step 2:** Implement minimal production code to satisfy each test in red-green-refactor order.
**Step 3:** Run the full GraphPilot Python test suite.

### Task 4: Integrate GraphPilot runtime harness for workflows A and B

**Files:**
- Create: `android-app/app/src/main/java/com/qidk/fastvlm/core/graphpilot/GraphPilotCoordinator.kt`
- Create: `android-app/app/src/androidTest/java/com/qidk/fastvlm/graphpilot/GraphPilotCoordinatorInstrumentedTest.kt`
- Modify: `android-app/app/src/main/java/com/qidk/fastvlm/core/text/LiteRtLmTextStageAdapter.kt`
- Modify: `android-app/app/src/main/java/com/qidk/fastvlm/core/bridge/FastVlmNativeBridge.kt`
- Modify: `scripts/run_graphpilot_stage_feasibility.py`
- Modify: `scripts/run_graphpilot_stage_profiler.py`

**Step 1:** Write failing instrumentation tests for voice-only and voice+vision GraphPilot execution.
**Step 2:** Implement coordinator with explicit backend map, stage timings, TTFT/TTFS capture, and no silent fallback.
**Step 3:** Run the instrumentation tests and capture artifacts.

### Task 5: Baselines, candidate search, and artifact pack

**Files:**
- Create: `scripts/run_graphpilot_experiments.py`
- Create: `scripts/build_graphpilot_artifact_pack.py`
- Create: `scripts/tests/test_run_graphpilot_experiments.py`
- Create: `scripts/tests/test_build_graphpilot_artifact_pack.py`
- Modify: `artifacts/graphpilot_edge/registries/experiment_registry.json`
- Modify: `artifacts/graphpilot_edge/registries/plot_registry.json`
- Create: `artifacts/graphpilot_edge/reports/final_audit_report.md`

**Step 1:** Write failing tests for experiment-plan materialization and artifact-pack manifests.
**Step 2:** Implement experiment runner over feasible plans and sustained-run support.
**Step 3:** Generate plots/tables/report bundle.
**Step 4:** Run audit against success criteria and narrow claims if retrieval remains infeasible.
