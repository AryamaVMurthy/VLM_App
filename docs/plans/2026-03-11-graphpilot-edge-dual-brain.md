# GraphPilot-Edge Dual-Brain Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade GraphPilot-Edge from the current prototype into the formula-driven dual-brain system specified in the approved design, with calibrated cost models, richer scheduling logic, and hyperparameter tuning scripts.

**Architecture:** Keep the Android runtime as the measured execution backend and preserve the existing FastVLM LiteRT NPU path. Upgrade the Python offline brain so candidate plans are scored with explicit latency, TTFS, energy, memory, copy, and quality terms; add calibration from actual experiment batches and sustained runs; then drive plan search and paper artifacts from the calibrated model.

**Tech Stack:** Python 3, unittest, JSON registries, Android instrumentation outputs, LiteRT/LiteRT-LM artifacts, adb, SVG/Markdown artifact generation.

---

### Task 1: Add formula-driven cost and objective modules

**Files:**
- Create: `graphpilot_edge/cost_model.py`
- Modify: `graphpilot_edge/profiles.py`
- Modify: `graphpilot_edge/models.py`
- Test: `tests/graphpilot_edge/test_cost_model.py`
- Test: `tests/graphpilot_edge/test_profiles.py`

**Step 1:** Write failing tests for execution cost, transfer cost, energy proxy, and hybrid prefill/decode threshold helpers.
**Step 2:** Run the focused tests and confirm they fail for missing behavior.
**Step 3:** Implement minimal cost-model production code and richer stage-profile fields.
**Step 4:** Re-run the focused tests and keep them green.

### Task 2: Upgrade simulation and ranking to use J(Pi)

**Files:**
- Modify: `graphpilot_edge/simulation.py`
- Modify: `graphpilot_edge/planner.py`
- Modify: `graphpilot_edge/scheduler.py`
- Test: `tests/graphpilot_edge/test_simulation.py`
- Test: `tests/graphpilot_edge/test_planner.py`
- Test: `tests/graphpilot_edge/test_scheduler.py`

**Step 1:** Write failing tests for transfer-aware simulation, objective scoring, and HEFT-style dynamic priority helpers.
**Step 2:** Implement the minimal simulation/ranking changes to satisfy those tests.
**Step 3:** Verify the updated planner still ranks feasible plans deterministically and exposes the objective breakdown.

### Task 3: Add calibration and hyperparameter tuning scripts

**Files:**
- Create: `scripts/calibrate_graphpilot_cost_model.py`
- Create: `scripts/tune_graphpilot_hyperparameters.py`
- Create: `scripts/tests/test_calibrate_graphpilot_cost_model.py`
- Create: `scripts/tests/test_tune_graphpilot_hyperparameters.py`
- Create: `configs/graphpilot_edge/optimization_defaults.json`

**Step 1:** Write failing tests for calibration summary generation and tuning output.
**Step 2:** Implement calibration from measured experiment batch + sustained-load summary.
**Step 3:** Implement hyperparameter search over objective/scheduler weights using only measured artifacts.
**Step 4:** Run the scripts on the real GraphPilot artifact set and save results.

### Task 4: Integrate calibration/tuning into candidate-plan generation and artifact pack

**Files:**
- Modify: `scripts/generate_graphpilot_candidate_plans.py`
- Modify: `scripts/build_graphpilot_artifact_pack.py`
- Modify: `scripts/run_graphpilot_experiments.py`
- Modify: `scripts/tests/test_generate_graphpilot_candidate_plans.py`
- Modify: `scripts/tests/test_build_graphpilot_artifact_pack.py`
- Modify: `scripts/tests/test_run_graphpilot_experiments.py`

**Step 1:** Write failing tests for calibrated plan generation and artifact-pack inclusion of calibration/tuning outputs.
**Step 2:** Implement the smallest changes needed to keep the pipeline deterministic and machine-readable.
**Step 3:** Regenerate candidate plans and the artifact pack from the calibrated model.

### Task 5: Re-verify end to end and update the machine-readable state

**Files:**
- Modify: `artifacts/graphpilot_edge/registries/candidate_plan_registry.json`
- Modify: `artifacts/graphpilot_edge/registries/experiment_registry.json`
- Modify: `artifacts/graphpilot_edge/state/state_ledger.json`

**Step 1:** Run the focused unit suites for the new offline-brain modules.
**Step 2:** Run calibration, tuning, plan generation, and artifact-pack regeneration.
**Step 3:** Update the state ledger with exact commands, outputs, and current limitations.
