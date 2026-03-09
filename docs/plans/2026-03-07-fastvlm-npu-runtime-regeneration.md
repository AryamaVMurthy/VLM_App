# FastVLM NPU Runtime And Regeneration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run FastVLM end to end so `TF_LITE_EMBEDDER` and `TF_LITE_VISION_ADAPTER` execute on Qualcomm DispatchDelegate/NPU instead of XNNPACK CPU, and regenerate the model only if runtime lowering still leaves CPU fallback.

**Architecture:** First remove the LiteRT-LM runtime's hard-coded CPU selection for the NPU executor paths and centralize Qualcomm NPU option setup in one tested helper. Then build and run the adb benchmark against the existing FastVLM artifact and inspect fresh logs. If the adapter or embedder still use CPU fallback after the runtime patch, regenerate a new `.litertlm` artifact that precompiles or lowers those sections for Qualcomm NPU and rerun the same benchmark flow.

**Tech Stack:** Bash, Bazel/C++, LiteRT, LiteRT-LM, Qualcomm DispatchDelegate/QNN, adb

---

### Task 1: Lock down the intended NPU options in a focused unit test

**Files:**
- Modify: `third_party/litert-lm/runtime/executor/qualcomm_npu_options.h`
- Modify: `third_party/litert-lm/runtime/executor/qualcomm_npu_options.cc`
- Modify: `third_party/litert-lm/runtime/executor/qualcomm_npu_options_test.cc`

**Step 1: Write the failing test**

Add tests that create LiteRT `Options`, apply the Qualcomm NPU helper, and assert:
- hardware accelerators are `kNpu`
- Qualcomm HTP performance mode is `kBurst`
- Qualcomm log level remains `kInfo`

**Step 2: Run test to verify it fails**

Run: `bazel test //third_party/litert-lm/runtime/executor:qualcomm_npu_options_test`

Expected: FAIL because the current helper does not configure LiteRT `Options` for `kNpu`.

**Step 3: Write minimal implementation**

Refactor the helper so executor code can configure a LiteRT `Options` object for Qualcomm NPU in one place instead of assembling partial state inline.

**Step 4: Run test to verify it passes**

Run: `bazel test //third_party/litert-lm/runtime/executor:qualcomm_npu_options_test`

Expected: PASS

### Task 2: Finish the runtime patch for FastVLM NPU executor paths

**Files:**
- Modify: `third_party/litert-lm/runtime/executor/llm_litert_npu_compiled_model_executor.cc`
- Modify: `third_party/litert-lm/runtime/executor/vision_litert_compiled_model_executor.cc`

**Step 1: Write the failing test**

Use the Task 1 helper contract as the regression gate: if the helper configures `kNpu`, all NPU executor call sites can consume that helper and stop hard-coding `kCpu`.

**Step 2: Run test to verify the pre-patch state**

Run: `rg -n "HwAccelerators::kCpu" third_party/litert-lm/runtime/executor/llm_litert_npu_compiled_model_executor.cc third_party/litert-lm/runtime/executor/vision_litert_compiled_model_executor.cc`

Expected: hits for embedder, auxiliary, and vision adapter/encoder NPU code paths.

**Step 3: Write minimal implementation**

Replace the hard-coded `kCpu` selections in:
- LLM NPU compiled-model options
- embedder compiled model creation
- per-layer embedder compiled model creation
- auxiliary compiled model creation
- vision encoder NPU backend
- vision adapter NPU backend

with the shared Qualcomm NPU helper.

**Step 4: Run tests to verify it passes**

Run:
- `bazel test //third_party/litert-lm/runtime/executor:qualcomm_npu_options_test`
- `bazel test //third_party/litert-lm/runtime/executor:vision_litert_compiled_model_executor_test`

Expected: PASS

### Task 3: Build and verify on device with the existing FastVLM artifact

**Files:**
- Modify only if needed for verification support: `scripts/run_fastvlm_litert_npu_adb.sh`

**Step 1: Build the runtime binary**

Run the repository’s build command for the LiteRT-LM advanced executable used by `scripts/run_fastvlm_litert_npu_adb.sh`.

**Step 2: Run the adb benchmark**

Run:
- `scripts/run_fastvlm_litert_npu_adb.sh`

Expected:
- fresh log under `artifacts/logs/`
- no XNNPACK delegation for `TF_LITE_EMBEDDER`
- no XNNPACK delegation for `TF_LITE_VISION_ADAPTER`
- Qualcomm dispatch / NPU compilation or execution evidence for both

**Step 3: Inspect the log**

Run focused checks for:
- `TF_LITE_EMBEDDER`
- `TF_LITE_VISION_ADAPTER`
- `TfLiteXNNPackDelegate`
- `DispatchDelegate`
- `NPU accelerator could not be loaded`

**Step 4: Decide**

If the log shows full NPU execution for both sections, stop here. If either section still falls back to CPU/XNNPACK, continue to Task 4.

### Task 4: Regenerate the FastVLM LiteRT-LM artifact if runtime lowering still falls back

**Files:**
- Modify: the regeneration/build scripts discovered during implementation
- Create: regenerated artifact under `artifacts/models/`

**Step 1: Identify the builder/regeneration path**

Use the vendored LiteRT-LM builder/compiler flow already present in the repo instead of inventing a new conversion path.

**Step 2: Regenerate the model**

Produce a new `.litertlm` artifact targeting the Qualcomm SM8750 runtime with the embedder and adapter sections lowered or precompiled for NPU dispatch.

**Step 3: Run the adb benchmark again**

Run:
- `scripts/run_fastvlm_litert_npu_adb.sh --model <regenerated-artifact>`

**Step 4: Verify**

Confirm from a fresh device log that `TF_LITE_EMBEDDER` and `TF_LITE_VISION_ADAPTER` no longer use CPU/XNNPACK and that the end-to-end FastVLM run succeeds.
