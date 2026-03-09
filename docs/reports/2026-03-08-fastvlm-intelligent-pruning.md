# FastVLM Intelligent Pruning Report

Date: 2026-03-08

## Scope

This report covers the CPU-side prompt-conditioned FastVLM pruning implementation in this branch:
- prompt-conditioned token scoring after the vision adapter
- slight diversity-aware token selection
- discrete runtime-visible budget control in the benchmark harness
- explicit pruning observability in the runtime logs

## Implemented

### Runtime pruning path

Primary code:
- `third_party/litert-lm/runtime/util/executor_data_util.h`
- `third_party/litert-lm/runtime/util/executor_data_util.cc`
- `third_party/litert-lm/runtime/core/session_basic.cc`
- `third_party/litert-lm/runtime/framework/resource_management/execution_manager.cc`
- `third_party/litert-lm/runtime/engine/engine_settings.h`
- `third_party/litert-lm/runtime/engine/engine_settings.cc`
- `third_party/litert-lm/runtime/engine/litert_lm_settings.h`
- `third_party/litert-lm/runtime/engine/litert_lm_settings_util.cc`
- `third_party/litert-lm/runtime/engine/shared_flags.cc`
- `third_party/litert-lm/runtime/engine/litert_lm_main.cc`
- `third_party/litert-lm/runtime/engine/litert_lm_advanced_main.cc`

What changed:
- added explicit pruning strategies: `uniform` and `prompt_conditioned_v1`
- added deterministic prompt-conditioning vector construction from prompt token ids
- added prompt similarity + salience + redundancy-aware token selection
- added explicit runtime validation for invalid pruning strategies
- added explicit log emission for pruning decisions:
  - kept/original token counts
  - pruning strategy
  - mean prompt similarity
  - mean salience
  - selected token indices summary

### Budget controller

Primary code:
- `scripts/fastvlm_budget_controller.py`
- `scripts/fastvlm_cases_benchmark.py`
- `scripts/run_fastvlm_litert_npu_adb.sh`

What changed:
- added a discrete adaptive controller over buckets `{96,160,256}`
- controller uses:
  - queue depth
  - queued-image count
  - recent prefill latency
  - recent decode latency
  - answer mode
- added explicit controller reason codes per run
- runner now passes `--visual-token-pruning-strategy`

## Verification

### Unit and integration tests

```bash
cd third_party/litert-lm
bazel test //runtime/util:executor_data_util_test --test_output=errors
bazel test //runtime/engine:engine_settings_test --test_output=errors
bazel test //runtime/engine:litert_lm_settings_util_test --test_output=errors
bazel test //runtime/core:session_basic_test \
  --test_filter='SessionBasicTest.ProcessAndCombineContentsImageRespectsVisualTokenBudget:SessionBasicTest.ProcessAndCombineContentsImageUsesPromptConditionedVisualPruning' \
  --test_output=errors

env -u PYTHONHOME -u PYTHONPATH python3 -m unittest discover -s scripts/tests -p 'test_*.py'
bash -n scripts/run_fastvlm_litert_npu_adb.sh
```

### Trial image smoke

Command:

```bash
scripts/run_fastvlm_litert_npu_adb.sh \
  --image IMAGES/person.jpeg \
  --max-visual-tokens 96 \
  --benchmark 1 \
  --event-mode 1 \
  --max-num-tokens 128 \
  --max-output-tokens 64
```

Verified log:
- `artifacts/logs/adb_npu_npu_run_20260308_191111.log`

Manual check:
- caption output: `A man sits alone on a bench, lost in thought, with`
- this is truncated by the short decode cap, but the output is semantically correct for the image

Key runtime evidence:
- pruning decision logged at `artifacts/logs/adb_npu_npu_run_20260308_191111.log:2397`
- strategy is `prompt_conditioned_v1`
- selected token summary is emitted explicitly

### Images folder run

Command:

```bash
env -u PYTHONHOME -u PYTHONPATH python3 scripts/fastvlm_cases_benchmark.py \
  --model artifacts/models/FastVLM-0.5B.qualcomm.sm8750.auxmaskrope_runtime.litertlm \
  --images-dir IMAGES \
  --budgets 96,160,256 \
  --budget-policy adaptive_v1 \
  --warmup-runs 0 \
  --measured-runs 1 \
  --skip-build 1 \
  --max-num-tokens 384 \
  --answer-mode short \
  --output-dir artifacts/cases_benchmark/intelligent_pruning_adaptive_20260308_1919
```

Artifacts:
- `artifacts/cases_benchmark/intelligent_pruning_adaptive_20260308_1919/all_runs.json`
- `artifacts/cases_benchmark/intelligent_pruning_adaptive_20260308_1919/summary.json`
- `artifacts/cases_benchmark/intelligent_pruning_adaptive_20260308_1919/report.html`

Manual checks:
- `download.jpeg` -> `A forest with a dirt path winding through it.`
- `images.jpeg` -> `A bowl of assorted fruits including bananas, oranges, and a red pepper.`
- `person.jpeg` -> `A man sits alone on a bench, lost in thought, with a bare tree in the background.`

All three outputs make sense.

## Observed controller behavior

Adaptive runs selected:
- `download.jpeg` -> budget `160`, reasons `short_answer_mode`
- `images.jpeg` -> budget `96`, reasons `prefill_above_decode_window,short_answer_mode`
- `person.jpeg` -> budget `160`, reasons `prefill_below_decode_window,short_answer_mode`

Observed runtime scoring metadata:
- `download.jpeg` -> mean prompt similarity `-0.0130947`, mean salience `0.182225`
- `images.jpeg` -> mean prompt similarity `-0.015397`, mean salience `0.318177`
- `person.jpeg` -> mean prompt similarity `0.0014514`, mean salience `0.212162`

## Performance snapshot

Adaptive summary from `artifacts/cases_benchmark/intelligent_pruning_adaptive_20260308_1919/summary.json`:
- budget `96`: TTFT `98.177 ms`, prefill `1455.72 tok/s`, decode `98.1703 tok/s`
- budget `160`: median TTFT `189.316 ms`, prefill `1429.625 tok/s`, decode `97.8827 tok/s`

Comparison point:
- the earlier fixed-budget report already showed that `96` is the best TTFT bucket on this image set; the adaptive controller chose that bucket exactly when the recent prefill window exceeded the decode window

## Current limitations

- prompt conditioning is deterministic hashed projection from prompt token ids, not a learned semantic scorer
- real CPU-decode / NPU-prefill overlap is still blocked by the current runtime architecture
- this controller is honest about that limitation: it uses recent timings as a control signal but does not claim actual overlap
- host `//runtime/engine:litert_lm_main` and host `//runtime/engine:litert_lm_advanced_main` still hit the pre-existing Gemma constraint-provider linker issue in this repo; Android on-device builds and runs are verified
