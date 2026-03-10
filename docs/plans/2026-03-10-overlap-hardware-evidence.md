# Overlap Hardware Evidence Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add first-token/runtime evidence, adaptive per-request budgeting, four benchmark variants, and paper-facing figures/counters for the FastVLM heterogeneous overlap runtime.

**Architecture:** Extend the overlap runtime to emit all authoritative stream events directly from the NPU-prefill/CPU-decode path. Keep adaptive token budgeting in the runner/runtime boundary so each request can choose a concrete visual-token bucket from a small allowed set using queue pressure and rolling stage timings. Build analysis scripts on top of the structured event stream to generate figures/tables without re-parsing informal logs.

**Tech Stack:** C++, Abseil, existing LiteRT-LM runtime, Python 3 analysis scripts, adb/device shell sampling.

---

### Task 1: Add overlap TTFT and controller event tests

**Files:**
- Modify: `scripts/tests/test_fastvlm_host_edge_diff.py`
- Modify: `scripts/tests/test_fastvlm_litert_gqa_overlap_eval.py`
- Modify: `scripts/tests/test_fastvlm_budget_controller.py`

**Step 1: Write failing tests**
- Add parser tests for `FIRST_TOKEN`, controller-decision payloads, and timeline summaries.
- Add controller tests for bucket reduction under queue pressure and decode-window matching.

**Step 2: Run tests to verify they fail**
Run: `env -u PYTHONHOME -u PYTHONPATH python3 -m unittest scripts.tests.test_fastvlm_host_edge_diff scripts.tests.test_fastvlm_litert_gqa_overlap_eval scripts.tests.test_fastvlm_budget_controller`
Expected: parser/controller tests fail because the new event fields are not handled yet.

**Step 3: Implement minimal parser/controller updates**
- Update the Python parsers/controllers only enough to satisfy the new tests.

**Step 4: Re-run the tests**
Run the same unittest command and confirm green.

### Task 2: Add overlap-runtime TTFT emission and adaptive budget hooks

**Files:**
- Modify: `third_party/litert-lm/runtime/core/session_basic.h`
- Modify: `third_party/litert-lm/runtime/engine/litert_lm_overlap_main.cc`
- Modify: `third_party/litert-lm/runtime/engine/shared_flags.cc`
- Modify: `third_party/litert-lm/runtime/engine/engine_settings_test.cc`
- Modify: `third_party/litert-lm/runtime/core/session_basic_test.cc`

**Step 1: Write failing runtime tests**
- Add tests for mutable per-request visual-token budget updates.
- Add overlap parser/fixture expectations for `FIRST_TOKEN` and controller decision events.

**Step 2: Run targeted tests and verify failure**
Run targeted Bazel/gtest plus Python tests for the touched files.

**Step 3: Implement runtime changes**
- Add per-request budget selection inputs/flags.
- Emit `FIRST_TOKEN` and controller decision events from the overlap runtime.
- Emit process id / stream metadata once at startup.

**Step 4: Re-run tests**
Run targeted Bazel tests and the Python parser tests.

### Task 3: Build stream benchmark + figure generation tooling

**Files:**
- Create: `scripts/fastvlm_overlap_stream_benchmark.py`
- Create: `scripts/generate_fastvlm_paper_figures.py`
- Modify: `scripts/fastvlm_host_edge_diff.py`
- Modify: `scripts/run_fastvlm_litert_overlap_adb.sh`

**Step 1: Write failing tool tests**
- Add unit coverage for timeline aggregation and baseline summary generation.

**Step 2: Run tests to verify failure**
Run the new/updated Python tests.

**Step 3: Implement minimal scripts**
- Benchmark runner for the four variants.
- Figure generator for timeline + partition map + baseline charts.

**Step 4: Re-run tests**
Confirm green.

### Task 4: Add hardware counter collection

**Files:**
- Create: `scripts/collect_fastvlm_hardware_counters.py`
- Create: `scripts/tests/test_collect_fastvlm_hardware_counters.py`

**Step 1: Write failing tests**
- Cover parser/collector logic for `/proc` CPU snapshots, battery rail samples, and explicit unavailable-counter handling.

**Step 2: Run tests to verify failure**
Run the new unittest target.

**Step 3: Implement minimal collector**
- Process CPU utilization sampling.
- Sync/wait/copy counters from overlap events.
- Optional battery current/voltage sampling with explicit unavailability reporting.

**Step 4: Re-run tests**
Confirm green.

### Task 5: Run end-to-end device validation and build final bundle

**Files:**
- Output under: `artifacts/analysis/`

**Step 1: Run targeted quick stream benchmarks**
- Run the four variants on the same repeated-image stream.
- Verify runtime outputs are sensible.

**Step 2: Run hardware counter collection**
- Capture CPU utilization and any available battery/power rails during the same stream runs.

**Step 3: Generate figures and report**
- Timeline figure
- Partition map figure
- Baseline comparison table
- Hardware counters table

**Step 4: Verify artifacts**
- Check logs, CSV/JSON outputs, and figures exist.
- Manually inspect at least one timeline/partition figure and one report.

**Step 5: Commit**
```bash
git add docs/plans ...
git commit -m "feat: add overlap hardware evidence tooling"
```
