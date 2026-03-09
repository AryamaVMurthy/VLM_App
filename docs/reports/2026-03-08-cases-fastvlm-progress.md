# CASES FastVLM Runtime Progress

Date: 2026-03-08

## Implemented And Verified

### 1. Prompt-conditioned post-projection visual token pruning

Primary code paths:
- `third_party/litert-lm/runtime/util/executor_data_util.h`
- `third_party/litert-lm/runtime/util/executor_data_util.cc`
- `third_party/litert-lm/runtime/core/session_basic.cc`
- `third_party/litert-lm/runtime/framework/resource_management/execution_manager.cc`
- `third_party/litert-lm/runtime/engine/engine_settings.h`
- `third_party/litert-lm/runtime/engine/engine_settings.cc`
- `third_party/litert-lm/runtime/engine/litert_lm_settings.h`
- `third_party/litert-lm/runtime/engine/litert_lm_settings_util.cc`
- `third_party/litert-lm/runtime/engine/litert_lm_advanced_main.cc`
- `third_party/litert-lm/runtime/engine/litert_lm_main.cc`

What is true now:
- `max_visual_tokens` is a real runtime control surface
- `visual_token_pruning_strategy` is a real runtime control surface
- pruning still executes after the vision adapter output and before multimodal prefill consumes the image token placeholders
- the working intelligent strategy is `prompt_conditioned_v1`
- runtime logs now expose the applied budget, strategy, mean prompt similarity, mean salience, and selected token indices summary explicitly on device

See:
- `docs/reports/2026-03-08-fastvlm-intelligent-pruning.md`

### 2. Deterministic benchmark / observability harness

New scripts:
- `scripts/fastvlm_cases_benchmark.py`
- `scripts/fastvlm_budget_controller.py`
- `scripts/fastvlm_decode_profiles.py`
- `scripts/fastvlm_partition_plan.py`

New script tests:
- `scripts/tests/test_fastvlm_budget_controller.py`
- `scripts/tests/test_fastvlm_cases_benchmark.py`
- `scripts/tests/test_fastvlm_decode_profiles.py`
- `scripts/tests/test_fastvlm_partition_plan.py`

What is true now:
- the harness emits CSV, JSON, and HTML reports
- it preserves raw runner stdout/stderr per run
- it records the raw device log path for every measured run
- it supports `short` and `long` answer-mode scaffolding
- it supports `fixed` and `adaptive_v1` visual-token budget policies
- the adaptive controller emits explicit reason codes per run

### 3. Explicit partition-plan artifacts

Artifacts:
- `artifacts/partition_plans/adb_npu_benchmark_20260307_162241.partition_plan.json`
- `artifacts/partition_plans/adb_npu_npu_run_20260308_150633.partition_plan.json`
- `artifacts/graph_inspect_auxmaskrope_runtime/`

What is true now:
- the working runtime bundle shows `TF_LITE_EMBEDDER`, `TF_LITE_AUX`, and `TF_LITE_VISION_ADAPTER` fully on Qualcomm Dispatch/NPU
- `TF_LITE_VISION_ENCODER` and `TF_LITE_PREFILL_DECODE` remain subgraph-level only because they are precompiled `DISPATCH_OP` sections
- per-op mapping is explicit where possible and explicitly limited where impossible

### 4. Working on-device runtime path

Canonical working model bundle:
- `artifacts/models/FastVLM-0.5B.qualcomm.sm8750.auxmaskrope_runtime.litertlm`

Canonical runner:
- `scripts/run_fastvlm_litert_npu_adb.sh`

Default runner model was updated to the working bundle so the default path in this branch is now reproducible.
The runner now also resolves the Android executable from `bazel-out/arm64-v8a-opt/bin/runtime/engine/litert_lm_advanced_main` and rejects host x86-64 binaries explicitly.

### 5. Fixed-budget sweep result

See:
- `docs/reports/2026-03-08-fastvlm-budget-sweep.md`
- `artifacts/cases_benchmark/budget_sweep_20260308/`

Headline result:
- budget `96` vs budget `256` gives a `63.9%` median TTFT reduction (`101.029 ms` vs `280.211 ms`) on the three-image set while prefill tokens/sec stays almost unchanged

### 6. Adaptive intelligent-pruning run

See:
- `docs/reports/2026-03-08-fastvlm-intelligent-pruning.md`
- `artifacts/cases_benchmark/intelligent_pruning_adaptive_20260308_1919/`

Headline result:
- adaptive controller selected budgets `160`, `96`, and `160` across the three-image queue
- all three captions remained semantically correct
- the `96` bucket delivered `98.177 ms` TTFT in the adaptive run

## Verification Evidence

### Bazel tests

```bash
cd third_party/litert-lm
bazel test //runtime/engine:litert_lm_settings_util_test
bazel test //runtime/engine:engine_settings_test
bazel test //runtime/util:executor_data_util_test
bazel test //runtime/core:session_basic_test \
  --test_filter='SessionBasicTest.ProcessAndCombineContentsImageRespectsVisualTokenBudget:SessionBasicTest.ProcessAndCombineContentsImageUsesPromptConditionedVisualPruning'
```

### Python tests

```bash
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest discover -s scripts/tests -p 'test_*.py'
```

### Shell validation

```bash
bash -n scripts/run_fastvlm_litert_npu_adb.sh
```

### On-device smoke

```bash
scripts/run_fastvlm_litert_npu_adb.sh \
  --image IMAGES/person.jpeg \
  --max-visual-tokens 96 \
  --benchmark 1 \
  --event-mode 1 \
  --max-num-tokens 128 \
  --max-output-tokens 64
```

## Explicit Blockers

### Overlap scheduler blocker

The paper-critical overlap claim is still blocked in the current shared advanced runtime.

See:
- `docs/reports/2026-03-08-fastvlm-overlap-blocker.md`

Current truth:
- shared executor access serializes LLM work in the advanced runtime
- honest CPU-decode / NPU-prefill overlap needs separate execution pools and explicit KV/state handoff

### Decode-mode switching blocker

Current truth:
- `short` and `long` modes are honest experiment-level controls today
- they do not yet select among separate decode compute variants
- full decode-mode switching still requires separate compatible decode exports

### Prefill regeneration blocker

Current truth:
- the shipped Qualcomm prefill section still cannot be regenerated from the public source bundle alone because the available source/export family does not match the shipped precompiled section contract

## What Should Happen Next

1. Close the intelligent-pruning and adaptive-budget-controller beads, because they are now implemented and verified.
2. Finish the partition-plan validation bead by making plan/runtime mismatch detection explicit.
3. Start a separate execution-pool design for honest CPU-decode / NPU-prefill overlap.
4. Run COCO and GQA subsets to lock safe budget bands by task quality floor.
5. Replace the hashed prompt-conditioning vector with a stronger scorer only if task-quality experiments justify it.
