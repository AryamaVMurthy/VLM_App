# FastVLM Overlap Prepare Queue Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a bounded background request-preparation queue to the FastVLM overlap runtime so device-local image loading and preprocessing run ahead of prefill, then verify outputs and GQA accuracy on device.

**Architecture:** Introduce a small request-preparation helper in the LiteRT overlap runtime. A producer prepares the next `K` requests into real `InputData` payloads while the existing consumer loop keeps NPU prefill and CPU decode overlap intact. Preserve output semantics and add preparation observability events.

**Tech Stack:** C++, Bazel, Abseil synchronization/time, existing LiteRT-LM runtime components, Python unittest, adb device runner.

---

### Task 1: Add failing tests for the preparation queue helper

**Files:**
- Create: `third_party/litert-lm/runtime/engine/request_preparation_queue_test.cc`
- Modify: `third_party/litert-lm/runtime/engine/BUILD`

**Step 1: Write the failing test**

Add tests for:
- ordered delivery of prepared requests
- producer error propagation to consumer
- bounded queue behavior does not drop items

**Step 2: Run test to verify it fails**

Run: `cd third_party/litert-lm && bazel test //runtime/engine:request_preparation_queue_test --test_output=errors`
Expected: FAIL because the helper library/files do not exist yet.

**Step 3: Commit**

Do not commit yet. Continue into implementation after the failing test is confirmed.

### Task 2: Implement the queue helper minimally

**Files:**
- Create: `third_party/litert-lm/runtime/engine/request_preparation_queue.h`
- Create: `third_party/litert-lm/runtime/engine/request_preparation_queue.cc`
- Modify: `third_party/litert-lm/runtime/engine/BUILD`

**Step 1: Write minimal implementation**

Implement:
- request descriptor struct
- prepared request struct
- bounded producer-consumer queue with background producer thread
- explicit error propagation and ordered consumption

**Step 2: Run test to verify it passes**

Run: `cd third_party/litert-lm && bazel test //runtime/engine:request_preparation_queue_test --test_output=errors`
Expected: PASS

### Task 3: Integrate the queue into the overlap runtime

**Files:**
- Modify: `third_party/litert-lm/runtime/engine/litert_lm_overlap_main.cc`
- Modify: `third_party/litert-lm/runtime/engine/BUILD`

**Step 1: Write the failing integration test expectation**

Use existing C++/Python coverage to define the runtime contract:
- outputs still emitted through `OVERLAP_RESPONSE`
- runtime accepts `--prepare_queue_size`
- no fallback paths introduced

**Step 2: Implement minimal integration**

Replace inline `BuildRequestContents(...)` usage in the main request loop with queue consumption.

Add:
- `--prepare_queue_size` flag
- `PREPARE_START`, `PREPARE_DONE`, `PREPARE_QUEUE_WAIT`, `PREPARE_QUEUE_STATE` events

**Step 3: Run focused tests**

Run:
- `cd third_party/litert-lm && bazel test //runtime/engine:request_preparation_queue_test --test_output=errors`
- `env -u PYTHONHOME -u PYTHONPATH python3 -m unittest scripts.tests.test_fastvlm_litert_gqa_overlap_eval`
Expected: PASS

### Task 4: Plumb the queue size through the benchmark wrappers

**Files:**
- Modify: `scripts/run_fastvlm_litert_overlap_adb.sh`
- Modify: `scripts/run_fastvlm_litert_gqa_overlap_eval.py`
- Modify: `scripts/tests/test_fastvlm_litert_gqa_overlap_eval.py`

**Step 1: Write the failing Python test**

Assert the generated runner command includes `--prepare-queue-size` with the configured value.

**Step 2: Run test to verify it fails**

Run: `env -u PYTHONHOME -u PYTHONPATH python3 -m unittest scripts.tests.test_fastvlm_litert_gqa_overlap_eval`
Expected: FAIL because the argument is not wired yet.

**Step 3: Implement minimal wrapper changes**

Add the CLI option and pass it through to the adb runner and overlap binary.

**Step 4: Run test to verify it passes**

Run: `env -u PYTHONHOME -u PYTHONPATH python3 -m unittest scripts.tests.test_fastvlm_litert_gqa_overlap_eval`
Expected: PASS

### Task 5: Build and run on-device smoke validation

**Files:**
- No source changes required if prior tasks pass

**Step 1: Build overlap runtime**

Run: `bash scripts/run_fastvlm_litert_overlap_adb.sh --help >/dev/null`
Then run a 2-image smoke execution with `--skip-build 0` once.

**Step 2: Verify outputs**

Run a 2-image overlap smoke test and confirm:
- runtime exits `0`
- `OVERLAP_RESPONSE` lines are present
- `PREPARE_*` lines are present

**Step 3: Record evidence**

Capture the run log path and summarize preparation wait vs overlap timings.

### Task 6: Run GQA accuracy / throughput validation

**Files:**
- No source changes required if prior tasks pass

**Step 1: Run a small correctness pass**

Run overlap GQA on a small sample count first, confirm predictions are produced and scored.

**Step 2: Run the requested benchmark slice**

Run the overlap GQA eval with the new queue on-device, using the existing relaxed practical matcher.

**Step 3: Report results**

Report:
- wall clock
- outputs produced
- any observable prepare wait reduction
- accuracy summary
