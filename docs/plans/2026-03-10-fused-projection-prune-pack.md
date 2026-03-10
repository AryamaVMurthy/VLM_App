# Fused Projection-Prune-Pack Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an explicit CPU-side fused projection-prune-pack path for FastVLM’s visible post-projection seam and measure the real device speedup.

**Architecture:** Keep the existing LiteRT vision encoder and adapter artifacts unchanged. Represent retained visual tokens as a sparse selected-token view over the adapter output, and extend multimodal embedding packing to consume that sparse view directly during prefill.

**Tech Stack:** C++, LiteRT-LM runtime, Bazel, adb, Python benchmark scripts.

---

### Task 1: Add failing sparse-view tests

**Files:**
- Modify: `third_party/litert-lm/runtime/util/executor_data_util_test.cc`
- Modify: `third_party/litert-lm/runtime/components/embedding_lookup/embedding_lookup_multi_modal_test.cc`
- Modify: `third_party/litert-lm/runtime/components/embedding_lookup/embedding_lookup_manager_test.cc`

**Step 1: Write failing pruning test**
- Assert fused mode returns a sparse selected-token view instead of a materialized sliced tensor.

**Step 2: Write failing embedding lookup test**
- Assert multimodal lookup packs only selected rows in order.

**Step 3: Run tests to verify red**
Run: `cd third_party/litert-lm && bazel test //runtime/util:executor_data_util_test //runtime/components/embedding_lookup:embedding_lookup_multi_modal_test //runtime/components/embedding_lookup:embedding_lookup_manager_test --test_output=errors`
Expected: failures because sparse fused support is missing.

### Task 2: Add sparse selected-token metadata plumbing

**Files:**
- Modify: `third_party/litert-lm/runtime/executor/llm_executor_io_types.h`
- Modify: `third_party/litert-lm/runtime/executor/llm_executor_io_types.cc`
- Modify: `third_party/litert-lm/runtime/framework/resource_management/resource_manager.cc`
- Modify: `third_party/litert-lm/runtime/util/executor_data_util.h`
- Modify: `third_party/litert-lm/runtime/util/executor_data_util.cc`

**Step 1: Add selected-token metadata to `ExecutorVisionData`**
- Getter/setter/print support
- Preserve metadata on duplication

**Step 2: Add explicit fused pruning config**
- Parse `LITERT_LM_PRUNING_FUSE_PROJECTION_PRUNE_PACK`
- Fail fast on invalid values

**Step 3: Implement sparse-view pruning path**
- Under fused mode, duplicate the original TensorBuffer handle and attach selected indices instead of materializing a sliced tensor.
- Error out if unsupported.

**Step 4: Re-run targeted tests**

### Task 3: Extend multimodal embedding packing to consume sparse views

**Files:**
- Modify: `third_party/litert-lm/runtime/components/embedding_lookup/embedding_lookup_multi_modal.h`
- Modify: `third_party/litert-lm/runtime/components/embedding_lookup/embedding_lookup_multi_modal.cc`
- Modify: `third_party/litert-lm/runtime/components/embedding_lookup/embedding_lookup_manager.cc`

**Step 1: Teach lookup creation to accept sparse vision selections**
- Keep legacy path unchanged when no selected-token metadata exists.

**Step 2: Pack selected rows directly into prefill embeddings**
- Copy only retained rows in token order.
- Track consumption count explicitly.

**Step 3: Re-run targeted tests**

### Task 4: Add observability and end-to-end toggle plumbing

**Files:**
- Modify: `third_party/litert-lm/runtime/core/session_basic.cc`
- Modify: `third_party/litert-lm/runtime/framework/resource_management/execution_manager.cc`
- Modify: `scripts/run_fastvlm_litert_gqa_eval.py`
- Modify: `scripts/run_fastvlm_litert_overlap_adb.sh`
- Modify: `scripts/fastvlm_overlap_stream_benchmark.py`

**Step 1: Log fused path usage / unsupported errors explicitly**
- No silent fallback.

**Step 2: Plumb explicit env override through runner scripts**
- Make A/B benchmarks reproducible.

**Step 3: Re-run unit tests and parser tests**

### Task 5: Measure speedup on device

**Files:**
- Output under: `artifacts/analysis/`

**Step 1: Run A/B on the same workload with fused mode off and on**
- Use identical prompt, token budget, image set, and pruning strategy.

**Step 2: Capture exact deltas**
- TTFT
- `vision_token_budget` time if visible
- total wall clock / event window

**Step 3: Write report**
- Include limitations and whether gains are CPU-side only.
