# FastVLM Real Text-Embedding Pruning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace hashed prompt conditioning with real FastVLM text embeddings computed early, cached once, reused for both visual-token pruning and later prefill.

**Architecture:** Build prompt text embeddings during preprocessing using the real FastVLM text embedder, store the cache on `ExecutorTextData`, pool that cache into the pruning prompt vector, and teach the compiled-model executors to reuse the cached embeddings instead of looking them up again for the same prompt tokens. No fallback path is allowed: if prompt-conditioned pruning is requested and the real embedder cache cannot be built, fail explicitly.

**Tech Stack:** LiteRT-LM runtime, FastVLM `.litertlm` bundle, Qualcomm NPU path, C++, Bazel, adb on-device verification.

---

### Task 1: Add cached text-embedding transport on executor inputs

**Files:**
- Modify: `third_party/litert-lm/runtime/executor/llm_executor_io_types.h`
- Modify: `third_party/litert-lm/runtime/executor/llm_executor_io_types.cc`

**Step 1: Add a cache container for real text embeddings**
- Add a small `CachedTextEmbeddings` value type with explicit `token_count`, `floats_per_token`, and `values`.
- Attach it to `ExecutorTextData` as an optional field.

**Step 2: Add getters/setters and validation helpers**
- Add accessors for reading and mutating the cache.
- Keep it explicit in logging; do not silently ignore malformed caches.

**Step 3: Build-check the io-types target**
Run: `cd third_party/litert-lm && bazel test //runtime/executor:llm_executor_io_types_test --test_output=errors`
Expected: pass.

### Task 2: Build the real prompt cache during preprocessing

**Files:**
- Modify: `third_party/litert-lm/runtime/core/session_basic.h`
- Modify: `third_party/litert-lm/runtime/core/session_basic.cc`
- Modify: `third_party/litert-lm/runtime/core/session_factory.cc`
- Modify: `third_party/litert-lm/runtime/core/engine_impl.cc`
- Modify: `third_party/litert-lm/runtime/framework/resource_management/execution_manager.h`
- Modify: `third_party/litert-lm/runtime/framework/resource_management/execution_manager.cc`
- Modify: `third_party/litert-lm/runtime/framework/resource_management/resource_manager.h`
- Modify: `third_party/litert-lm/runtime/framework/resource_management/resource_manager.cc`

**Step 1: Create a real FastVLM text-embedder lookup for preprocessing**
- Instantiate a dedicated `EmbeddingLookupManager` from the real FastVLM `TF_LITE_EMBEDDER` resource.
- Wire one instance into `SessionBasic` and one into `ExecutionManager` creation.

**Step 2: Cache prompt text embeddings while preprocessing text inputs**
- As text token ids are accumulated, look up their real embeddings once and append them into `CachedTextEmbeddings`.
- Store the completed cache on `ExecutorTextData` before returning `ExecutorInputs`.

**Step 3: Fail fast if prompt-conditioned pruning requests real embeddings but no cache can be produced**
- No hash proxy, no fallback.

### Task 3: Replace hashed prompt conditioning with real embedding pooling

**Files:**
- Modify: `third_party/litert-lm/runtime/util/executor_data_util.h`
- Modify: `third_party/litert-lm/runtime/util/executor_data_util.cc`

**Step 1: Add real-cache-based prompt pooling**
- Pool cached FastVLM text embeddings into a normalized prompt vector in the same space as the vision adapter output.

**Step 2: Update pruning to require the real cache for `prompt_conditioned_v1`**
- Use cosine similarity against the pooled real prompt vector.
- Preserve salience and slight redundancy penalty.
- Remove hashed-projection usage from the active prompt-conditioned path.

### Task 4: Reuse cached text embeddings during prefill

**Files:**
- Modify: `third_party/litert-lm/runtime/components/embedding_lookup/embedding_lookup_manager.h`
- Modify: `third_party/litert-lm/runtime/components/embedding_lookup/embedding_lookup_manager.cc`
- Modify: `third_party/litert-lm/runtime/executor/llm_litert_compiled_model_executor.cc`
- Modify: `third_party/litert-lm/runtime/executor/llm_litert_npu_compiled_model_executor.cc`

**Step 1: Add explicit APIs to populate prefill embeddings from a real cache plus multimodal tokens**
- Positive text tokens must come from the cache.
- Negative multimodal tokens must still come from multimodal embeddings.
- Error if the cache/token stream do not match.

**Step 2: Teach both compiled-model executor paths to consume the cache**
- CPU compiled-model executor path.
- Qualcomm NPU compiled-model executor path.
- Reuse the cached final prompt token for the pending-input-token embedding when possible.

### Task 5: Verify on the real FastVLM device path

**Files:**
- Modify if needed: `docs/reports/2026-03-08-fastvlm-intelligent-pruning.md`
- Modify if needed: `docs/reports/2026-03-08-cases-fastvlm-progress.md`

**Step 1: Build Android runtime targets**
Run: `cd third_party/litert-lm && bazel build --config=android_arm64 //runtime/engine:litert_lm_advanced_main --override_repository="litert=$(cd ../litert && pwd)"`
Expected: exit 0.

**Step 2: Run the real FastVLM bundle on device**
Run: `scripts/run_fastvlm_litert_npu_adb.sh --skip-build 1 --image IMAGES/person.jpeg --max-visual-tokens 96 --benchmark 1 --event-mode 1 --max-num-tokens 128 --max-output-tokens 64`
Expected: exit 0, pruning log present, caption makes sense.

**Step 3: Run the three-image verification set**
Run: `env -u PYTHONHOME -u PYTHONPATH python3 scripts/fastvlm_cases_benchmark.py --model artifacts/models/FastVLM-0.5B.qualcomm.sm8750.auxmaskrope_runtime.litertlm --images-dir IMAGES --budgets 96,160,256 --budget-policy adaptive_v1 --warmup-runs 0 --measured-runs 1 --skip-build 1 --max-num-tokens 384 --answer-mode short --output-dir artifacts/cases_benchmark/real_text_pruning_$(date +%Y%m%d_%H%M%S)`
Expected: exit 0, captions sane, logs show the real prompt-conditioning path.
