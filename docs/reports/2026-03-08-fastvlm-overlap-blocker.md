# FastVLM CPU-Decode / NPU-Prefill Overlap Blocker

Date: 2026-03-08

## Question

Can the current advanced FastVLM runtime overlap CPU decode for request `N`
with NPU prefill for request `N+1` inside one shared engine/session manager?

## Answer

No. The current shared advanced runtime serializes LLM executor access, so
prefill and decode do not overlap on the same engine path.

## Code Evidence

- `third_party/litert-lm/runtime/framework/resource_management/resource_manager.cc:626`
  returns a `LockedLlmExecutor` that owns `executor_mutex_`.
- `third_party/litert-lm/runtime/framework/resource_management/resource_manager.cc:644`
  does the same after context switching.
- `third_party/litert-lm/runtime/framework/resource_management/resource_manager.cc:729`
  also returns a `LockedLlmExecutor` while the same lock is held.
- `third_party/litert-lm/runtime/framework/resource_management/resource_manager.cc:167`
  documents that `LockedLlmExecutor` holds the mutex for the wrapper lifetime.
- `third_party/litert-lm/runtime/framework/resource_management/execution_manager.cc:701`
  acquires that locked executor for prefill.
- `third_party/litert-lm/runtime/framework/resource_management/execution_manager.cc:787`
  acquires the same locked executor for decode.
- `third_party/litert-lm/runtime/framework/resource_management/execution_manager.cc:966`
  does the same for scoring/session work.
- `third_party/litert-lm/runtime/framework/resource_management/execution_manager.h:263`
  hardcodes the execution and callback thread pools to a single worker by
  default for the current manager instance.

That means the advanced resource-managed path shares one LLM executor protected
by one mutex, and it also queues work through a single execution worker. Once
one request enters prefill or decode with the locked wrapper, another request
cannot run LLM work concurrently on that shared path.

## Additional System Constraint

The current FastVLM NPU path uses the Qualcomm compiled-model executor for both
prefill and decode. A paper claim of "CPU decode + NPU prefill" therefore
needs two things that do not exist yet in this branch:

1. a separate decode-capable CPU executor/session path, and
2. an explicit KV/state sharing contract between the NPU-prefill path and the
   CPU-decode path.

Without those two pieces, changing only the scheduler cannot create true
heterogeneous overlap.

## Required MVP Direction

The first viable overlap prototype should use separate execution pools:

- pool A: NPU-prefill sessions
- pool B: CPU-decode sessions

and must define an explicit handoff artifact for:

- prompt token state
- visual embeddings / pruned visual tokens
- mask / rope state
- KV cache ownership and layout

If that handoff cannot be made losslessly, the runtime must reject the overlap
configuration instead of silently serializing.
