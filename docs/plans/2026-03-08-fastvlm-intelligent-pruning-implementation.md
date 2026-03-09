# FastVLM Intelligent Pruning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace uniform FastVLM visual token pruning with prompt-conditioned, diversity-aware CPU-side pruning and add a discrete runtime-aware budget controller surface.

**Architecture:** Keep the current working post-projection seam. Implement token scoring and selection in `executor_data_util`, wire it through `SessionBasic` and `ExecutionManager`, and expose discrete controller decisions through runtime/orchestration logging.

**Tech Stack:** C++, Bazel tests, existing LiteRT-LM runtime utilities, Python benchmark harness.

---

### Task 1: Add intelligent-pruning config types and failing unit tests

**Files:**
- Modify: `third_party/litert-lm/runtime/util/executor_data_util.h`
- Modify: `third_party/litert-lm/runtime/util/executor_data_util_test.cc`

**Step 1: Write the failing tests**
- Add tests for:
  - prompt-conditioned scoring prefers prompt-aligned tokens
  - diversity-aware selection avoids keeping only duplicate high-score tokens
  - invalid strategy/config fails explicitly

**Step 2: Run test to verify it fails**
Run:
```bash
cd third_party/litert-lm
bazel test //runtime/util:executor_data_util_test
```
Expected:
- FAIL because the new config types / APIs do not exist yet

**Step 3: Write minimal implementation**
- Add strategy/config declarations only

**Step 4: Run test to verify it still fails for missing behavior**
Run the same command.

### Task 2: Implement prompt-conditioned token scoring in `executor_data_util`

**Files:**
- Modify: `third_party/litert-lm/runtime/util/executor_data_util.cc`
- Modify: `third_party/litert-lm/runtime/util/executor_data_util.h`
- Test: `third_party/litert-lm/runtime/util/executor_data_util_test.cc`

**Step 1: Write the failing tests**
- Add tests for prompt hash projection, cosine-style scoring, and deterministic ordering

**Step 2: Run test to verify it fails**
Run:
```bash
cd third_party/litert-lm
bazel test //runtime/util:executor_data_util_test --test_output=errors
```

**Step 3: Write minimal implementation**
- Implement prompt feature vector derivation from text token ids
- Implement token salience
- Implement redundancy-aware greedy selection
- Implement explicit result structure containing selected indices and scoring metadata

**Step 4: Run test to verify it passes**
Run the same command.

### Task 3: Wire intelligent pruning into `SessionBasic`

**Files:**
- Modify: `third_party/litert-lm/runtime/core/session_basic.cc`
- Modify: `third_party/litert-lm/runtime/core/session_basic_test.cc`

**Step 1: Write the failing tests**
- Add a session-level test showing prompt-conditioned pruning keeps the intended token positions and logs the strategy metadata

**Step 2: Run test to verify it fails**
Run:
```bash
cd third_party/litert-lm
bazel test //runtime/core:session_basic_test --test_filter=SessionBasicTest.ProcessAndCombineContentsImageRespectsVisualTokenBudget
```

**Step 3: Write minimal implementation**
- Capture combined prompt token ids before image pruning
- pass token ids and pruning config into the new scorer
- emit explicit logs with selected strategy and indices summary

**Step 4: Run test to verify it passes**
Run the same command.

### Task 4: Wire intelligent pruning into `ExecutionManager`

**Files:**
- Modify: `third_party/litert-lm/runtime/framework/resource_management/execution_manager.cc`
- Modify: `third_party/litert-lm/runtime/framework/resource_management/execution_manager_test.cc`

**Step 1: Write the failing tests**
- Add advanced/runtime-managed path test mirroring the basic session behavior

**Step 2: Run test to verify it fails**
Run:
```bash
cd third_party/litert-lm
bazel test //runtime/framework/resource_management:execution_manager_test --test_output=errors
```

**Step 3: Write minimal implementation**
- pass prompt token ids into the scorer in the advanced path
- emit the same explicit logs as `SessionBasic`

**Step 4: Run test to verify it passes**
Run the same command.

### Task 5: Add discrete budget-controller utility and tests

**Files:**
- Create: `third_party/litert-lm/runtime/util/visual_token_budget_controller.h`
- Create: `third_party/litert-lm/runtime/util/visual_token_budget_controller.cc`
- Create: `third_party/litert-lm/runtime/util/visual_token_budget_controller_test.cc`
- Modify: `third_party/litert-lm/runtime/util/BUILD`

**Step 1: Write the failing tests**
- Add tests for bucket selection from queue depth, queued-image count, and recent prefill/decode timings
- Add tests for invalid bucket sets and explicit errors

**Step 2: Run test to verify it fails**
Run:
```bash
cd third_party/litert-lm
bazel test //runtime/util:visual_token_budget_controller_test
```

**Step 3: Write minimal implementation**
- implement discrete bucket selection
- return reason codes alongside the chosen budget

**Step 4: Run test to verify it passes**
Run the same command.

### Task 6: Expose controller/scorer metadata to the benchmark harness

**Files:**
- Modify: `scripts/fastvlm_cases_benchmark.py`
- Modify: `scripts/tests/test_fastvlm_cases_benchmark.py`
- Optionally modify: `scripts/fastvlm_decode_profiles.py`

**Step 1: Write the failing tests**
- Add parser expectations for intelligent-pruning strategy fields, selected budget bucket, and reason codes

**Step 2: Run test to verify it fails**
Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest discover -s scripts/tests -p 'test_*.py'
```

**Step 3: Write minimal implementation**
- parse new log lines / structured metadata
- include them in CSV/JSON/HTML outputs

**Step 4: Run test to verify it passes**
Run the same command.

### Task 7: On-device verification and regression check

**Files:**
- Modify if needed: `scripts/run_fastvlm_litert_npu_adb.sh`
- Output: `artifacts/cases_benchmark/...`

**Step 1: Run one-image smoke with intelligent pruning**
Run:
```bash
scripts/run_fastvlm_litert_npu_adb.sh \
  --skip-build 1 \
  --image IMAGES/person.jpeg \
  --max-visual-tokens 96 \
  --benchmark 1 \
  --event-mode 1 \
  --max-num-tokens 128 \
  --max-output-tokens 64
```
Expected:
- PASS
- explicit intelligent-pruning log fields appear

**Step 2: Run multi-image budget sweep**
Run:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 scripts/fastvlm_cases_benchmark.py \
  --model artifacts/models/FastVLM-0.5B.qualcomm.sm8750.auxmaskrope_runtime.litertlm \
  --images IMAGES/person.jpeg IMAGES/download.jpeg IMAGES/images.jpeg \
  --budgets 96,160,256 \
  --warmup-runs 0 \
  --measured-runs 1 \
  --skip-build 1 \
  --max-num-tokens 384 \
  --answer-mode short \
  --output-dir artifacts/cases_benchmark/intelligent_pruning_$(date +%Y%m%d_%H%M%S)
```
Expected:
- PASS
- captions still make sense
- report includes scorer/controller metadata

### Task 8: Document results and update beads

**Files:**
- Modify: `docs/reports/2026-03-08-cases-fastvlm-progress.md`
- Create/Modify: `docs/reports/2026-03-08-fastvlm-intelligent-pruning.md`

**Step 1: Write the report**
- document algorithm, verification, and measured results

**Step 2: Update beads**
- append notes to `fvlm-zth.18` and `fvlm-zth.19`
- close only the issues that are fully verified
