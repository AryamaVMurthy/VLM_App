# FastVLM Overlap Request Preparation Queue Design

## Goal

Hide CPU-side image file I/O and image preprocessing behind the existing NPU-prefill / CPU-decode overlap pipeline so 500-image evaluation throughput improves without changing FastVLM model semantics.

## Current Bottleneck

The current overlap runtime reads each image from disk and runs `StbImagePreprocessor::Preprocess(...)` inside the main scheduling loop immediately before `RunPrefill(...)`. As a result, the pipeline overlaps only:

- CPU decode for request `i-1`
- NPU prefill for request `i`

It does **not** overlap preparation work for request `i+1`.

## Approved Approach

Add a bounded producer-consumer request-preparation queue inside the overlap runtime.

- A single background producer prepares upcoming requests in order.
- Preparation includes:
  - reading the device-local image bytes
  - preprocessing the image into the actual FastVLM input image payload
  - building the `std::vector<InputData>` that the prefill session consumes
- The existing main loop becomes a consumer:
  - pop the next prepared request
  - run NPU prefill
  - overlap with the previous CPU decode
  - emit the same `OVERLAP_RESPONSE` outputs as today

This preserves one-request inference semantics and keeps the current FastVLM/LiteRT session interfaces unchanged.

## Why This Approach

### Chosen

- Minimal change to FastVLM semantics
- Directly targets the real critical path
- Low risk compared with new artifact formats or true batched inference
- Works with current request manifest and output extraction path

### Rejected alternatives

- Precompute/cache preprocessed image tensors on device: faster steady state, but creates a new artifact format and invalidation problem.
- True batched inference: much higher risk and likely requires unsupported LiteRT/FastVLM interface changes.

## Runtime Behavior

### New data flow

1. Load request manifest
2. Start background preparation worker with bounded queue capacity `K`
3. Producer prepares requests `i, i+1, ...` sequentially and pushes ready items into the queue
4. Consumer pops the next prepared item in order
5. Consumer runs NPU prefill and existing CPU decode overlap logic
6. Runtime emits final `OVERLAP_RESPONSE` text per request exactly as before

### Ordering

Prepared requests remain strictly in manifest order. No out-of-order execution is introduced.

### Backpressure

If the queue is full, the producer blocks. This prevents unbounded memory growth across 500-image runs.

## Observability

Add structured `VLM_EVENT` logs for request preparation:

- `PREPARE_START`
- `PREPARE_DONE`
- `PREPARE_QUEUE_WAIT` when the consumer has to wait for the next prepared request
- `PREPARE_QUEUE_STATE` with queue depth after enqueue/dequeue

These events make it possible to determine whether preprocessing is being successfully hidden under decode/prefill overlap.

## Error Handling

No fallback behavior is allowed.

- If image read fails, preprocessing fails, or the producer throws, surface the exact error through the queue and terminate the run explicitly.
- If queue construction or startup is invalid, fail fast with actionable context.

## CLI Surface

Add a runtime flag for bounded prefetch depth:

- `--prepare_queue_size`

Expose the same setting through the adb wrapper and overlap GQA Python runner so benchmark sweeps can tune the queue depth explicitly.

## Verification

1. Unit tests for ordered producer-consumer behavior, backpressure-safe completion, and error propagation
2. Existing Python overlap runner tests remain green
3. Build overlap binary
4. Run on-device smoke test and verify printed outputs are still produced
5. Run GQA overlap accuracy on the new runtime path and compare throughput/accuracy with the previous implementation
