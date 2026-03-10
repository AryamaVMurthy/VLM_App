# CPU Fused Projection-Prune-Pack Design

## Scope
Implement a CPU-side fused execution path for the visible FastVLM seam after the LiteRT vision adapter/projection output and before language-model prefill.

This is not a converter-level LiteRT fused custom op inside the packaged model. It is a runtime-level fused data path in LiteRT-LM that avoids materializing a pruned projection tensor when the adapter output is already host-visible.

## Why this path
The current FastVLM runtime does:
1. run vision encoder
2. run vision adapter/projection
3. build a pruning decision over projected tokens
4. materialize a new pruned TensorBuffer by slicing selected rows
5. later pack those pruned rows again into the prefill embedding lookup output

That creates an unnecessary CPU-side copy chain on the visible seam.

## Design

### 1. Sparse selected-token view in ExecutorVisionData
Add optional selected-token metadata to `ExecutorVisionData` so the runtime can represent:
- original projected embedding buffer
- monotonically increasing selected token indices

This view must preserve ownership of the original TensorBuffer through LiteRT handle duplication, not data copy.

### 2. Explicit fusion knob
Add an explicit environment-gated knob in the pruning config:
- `LITERT_LM_PRUNING_FUSE_PROJECTION_PRUNE_PACK`

Behavior:
- when unset/false: keep legacy materialized slicing path
- when true: use the fused sparse-view path
- if true but unsupported for the current input (for example per-layer vision embeddings), fail fast with an explicit error

No silent fallback.

### 3. Fused packing in multimodal embedding lookup
Extend the multimodal embedding lookup path so that when selected-token metadata is present it copies only the retained projected rows directly into the prefill embedding output tensor.

This fuses:
- projection output consumption
- retained-token selection
- prefill packing

without creating an intermediate pruned TensorBuffer.

### 4. Generality / correctness
- Single-image FastVLM path must work end to end.
- Multi-image combine path must remain correct; if sparse views are combined, materialize explicitly at combine time.
- Resource-manager duplication must preserve selected-token metadata.

### 5. Observability
Add explicit logs for:
- fused mode enabled/disabled
- fused path used for a request
- selected token count
- fail-fast unsupported reasons

## Expected benefit
The expected speedup is in the CPU-visible projection/prune/pack seam, not inside opaque NPU `DISPATCH_OP` regions. Gains will likely appear in:
- reduced `vision_token_budget` time
- slightly lower TTFT / wall-clock on small-budget runs
- reduced host-side copy work before prefill

## Validation plan
1. Unit-test sparse selected-token lookup semantics.
2. Unit-test pruning returns sparse views under fused mode.
3. Verify targeted Bazel tests.
4. Run device A/B with fused mode off vs on on the same request set.
5. Report exact deltas from artifacts only.
