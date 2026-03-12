# GraphPilot Stream Scheduler Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend GraphPilot-Edge from single-request static plan execution into a continuous-stream heterogeneous scheduling system with multi-request simulation, queue/deadline metrics, runtime-facing scheduling metadata, and updated artifact reporting.

**Architecture:** Preserve the current support-safe execution core and FastVLM NPU path. Add the missing continuous-stream layer in the offline brain first: multi-request discrete-event simulation, queue-aware metrics, memory/KV-aware request accounting, and knob-aware plan metadata. Then propagate those results into runtime-facing plan artifacts and the paper bundle.

**Tech Stack:** Python 3, unittest, JSON registries, Android/Kotlin GraphPilot runtime, LiteRT/LiteRT-LM, adb, Markdown/SVG artifact generation.

---

### Task 1: Continuous multi-request simulator core

**Files:**
- Modify: `graphpilot_edge/models.py`
- Modify: `graphpilot_edge/simulation.py`
- Test: `tests/graphpilot_edge/test_simulation.py`

**Step 1: Write the failing test**
- Add tests for overlapping requests sharing CPU/NPU backends.
- Add tests for queue delay, per-request makespan, and deadline-miss-rate reporting.
- Add tests for stream-level peak memory and copy accounting.

**Step 2: Run test to verify it fails**
- Run: `env -u PYTHONHOME -u PYTHONPATH python3 -m unittest tests.graphpilot_edge.test_simulation`
- Expected: FAIL for missing stream-simulation API and missing queue/deadline metrics.

**Step 3: Write minimal implementation**
- Add request-level simulation dataclasses.
- Implement `simulate_request_stream(...)` over shared backend queues and per-request DAG dependencies.
- Aggregate P95 queue delay, deadline miss rate, peak memory, copy bytes, and energy.

**Step 4: Run test to verify it passes**
- Run: `env -u PYTHONHOME -u PYTHONPATH python3 -m unittest tests.graphpilot_edge.test_simulation`
- Expected: PASS.

### Task 2: Candidate-plan generation and reporting for stream metrics

**Files:**
- Modify: `scripts/generate_graphpilot_candidate_plans.py`
- Modify: `scripts/build_graphpilot_artifact_pack.py`
- Modify: `scripts/tests/test_generate_graphpilot_candidate_plans.py`
- Modify: `scripts/tests/test_build_graphpilot_artifact_pack.py`

**Step 1: Write the failing test**
- Add tests requiring predicted `p95_queue_ms` and `deadline_miss_rate` in generated plan cost blocks.
- Add tests requiring artifact-pack summaries to surface these fields when present.

**Step 2: Run test to verify it fails**
- Run the focused script tests and confirm missing stream metrics cause failure.

**Step 3: Write minimal implementation**
- Feed stream-simulation metrics into the candidate registry.
- Expose them in report and paper-draft summaries.

**Step 4: Run test to verify it passes**
- Re-run focused tests.

### Task 3: Runtime-facing scheduling metadata and state tracking

**Files:**
- Modify: `artifacts/graphpilot_edge/state/state_ledger.json`
- Modify: `artifacts/graphpilot_edge/registries/candidate_plan_registry.json`
- Modify: `artifacts/graphpilot_edge/registries/experiment_registry.json`

**Step 1: Regenerate stream-aware plan metadata**
- Run plan generation with stream metrics enabled.

**Step 2: Update the ledger**
- Record the new simulator capability, limitations, and next runtime integration step.

**Step 3: Verify outputs exist**
- Confirm the updated registries remain machine-readable JSON.

### Task 4: Runtime execution follow-on slice

**Files:**
- Modify: `android-app/app/src/main/java/com/qidk/fastvlm/core/graphpilot/GraphPilotCoordinator.kt`
- Modify: `android-app/app/src/main/java/com/qidk/fastvlm/core/graphpilot/GraphPilotPlanStore.kt`
- Test: `android-app/app/src/androidTest/java/com/qidk/fastvlm/graphpilot/GraphPilotCoordinatorInstrumentedTest.kt`

**Step 1: Add runtime-visible queue/deadline metadata consumption**
- Load stream-aware plan fields without changing support-safe backend behavior.

**Step 2: Add instrumentation hooks**
- Record request admission time, queue wait, and deadline class in logs.

**Step 3: Verify instrumentation**
- Run the focused Android tests after the offline slice is stable.

### Task 5: Verify and checkpoint

**Files:**
- Modify: `docs/plans/2026-03-11-graphpilot-stream-scheduler.md`
- Modify: Beads issues under `fvlm-97e`

**Step 1: Run verification**
- Run focused unit tests for stream simulation and registry generation.

**Step 2: Update task tracker**
- Add notes with actual artifact paths and remaining blockers.

**Step 3: Commit checkpoint when green**
- Commit only after verification evidence is fresh.
