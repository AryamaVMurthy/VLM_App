# Partition-Aware FastVLM System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the paper-facing LiteRT FastVLM system by validating runtime stage/backend observability, partition validation, and the mandatory GQA + COCO benchmarks on the real device path.

**Architecture:** Reuse the existing heterogeneous FastVLM runtime and extend it with explicit stage manifests, runtime validation, and benchmark harnesses. Keep pruning at the post-projection seam and keep all paper evidence grounded in structured runtime events and pinned compiled artifacts.

**Tech Stack:** C++, LiteRT-LM runtime, Bazel, Python 3, adb, external benchmark repo under `/home/aryamavmurthy/work/Liquid_benchmarking_image tokens pruning`.

---

### Task 1: Verify runtime stage/backend instrumentation

**Files:**
- Modify: `third_party/litert-lm/runtime/engine/litert_lm_overlap_main.cc`
- Test: `scripts/tests/test_fastvlm_host_edge_diff.py`
- Test: `scripts/tests/test_fastvlm_partition_validator.py`

**Step 1: Write or update failing parser tests**
- Ensure the validator/parser requires `STAGE_BACKEND` events for all required stages.

**Step 2: Run tests to verify red**
Run: `env -u PYTHONHOME -u PYTHONPATH python3 -m unittest scripts.tests.test_fastvlm_partition_validator scripts.tests.test_fastvlm_host_edge_diff`
Expected: fail if the new stage events are missing or malformed.

**Step 3: Implement minimal runtime event emission**
- Emit stage/backend manifest at overlap startup.
- Keep compile artifact IDs and notes explicit.

**Step 4: Re-run tests**
Run the same unittest command and confirm green.

### Task 2: Validate/build runtime targets

**Files:**
- Modify: `third_party/litert-lm/runtime/engine/BUILD`
- Test: `third_party/litert-lm/runtime/engine/visual_token_budget_controller_test.cc`

**Step 1: Run targeted Bazel tests/builds**
- Build/test the controller and overlap target.

**Step 2: Fix compilation issues only as needed**
- Keep changes minimal and explicit.

**Step 3: Re-run Bazel verification**
- Confirm the runtime changes build for host and Android targets.

### Task 3: Make the partition validator artifact-producing and fail-fast

**Files:**
- Create: `scripts/fastvlm_partition_validator.py`
- Test: `scripts/tests/test_fastvlm_partition_validator.py`

**Step 1: Write failing validator test**
- Cover required stage manifest extraction and config-deviation failure.

**Step 2: Implement validator**
- Emit `partition_validation.json` and `report.md`.
- Include subgraph/backend assignment, opaque-region limitations, transfer edges, compile artifact IDs, and deviations.

**Step 3: Re-run tests**
Run: `env -u PYTHONHOME -u PYTHONPATH python3 -m unittest scripts.tests.test_fastvlm_partition_validator`

### Task 4: Add COCO Karpathy LiteRT benchmark harness

**Files:**
- Create: `scripts/run_fastvlm_litert_coco_eval.py`
- Create: `scripts/tests/test_fastvlm_litert_coco_eval.py`

**Step 1: Write failing helper tests**
- Cover COCO sample loading, prediction row schema, and evaluator integration.

**Step 2: Implement the real LiteRT caption runner**
- Reuse the existing adb/NPU FastVLM path.
- Error out if cache or evaluator dependencies are missing.

**Step 3: Re-run tests**
Run: `env -u PYTHONHOME -u PYTHONPATH python3 -m unittest scripts.tests.test_fastvlm_litert_coco_eval`

### Task 5: Run smoke benchmarks and produce paper bundle

**Files:**
- Output under: `artifacts/analysis/`

**Step 1: Run a small overlap smoke benchmark**
- Verify runtime outputs, TTFT, stage manifests, and handoff events.

**Step 2: Run partition validation on the smoke log**
- Fail if runtime deviates from the expected NPU-prefill/CPU-decode config.

**Step 3: Run GQA and COCO smoke evaluations**
- Use the real LiteRT FastVLM paths only.
- Emit summaries and raw predictions.

**Step 4: Generate updated report/figures**
- Partition map figure
- Overlap timeline figure
- GQA + COCO summary table
- Validator report

**Step 5: Verify artifacts manually**
- Check that logs, JSON summaries, and figures exist and make sense.
