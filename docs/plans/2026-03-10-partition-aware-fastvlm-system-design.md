# Partition-Aware FastVLM System Design

## Scope
Build a paper-facing LiteRT FastVLM system with five explicit research components:
- partition-aware execution core
- post-projection pruning seam
- queue-aware overlap scheduler
- request-level observability layer
- partition validation tool

The system stays grounded in the existing packaged LiteRT FastVLM artifacts. It does not invent a new compiler. It exposes, validates, and optimizes the real heterogeneous execution path already present in the Qualcomm/LiteRT deployment.

## Design decisions

### 1. Partition-aware execution core
Use the packaged FastVLM LiteRT compiled artifacts as the execution truth and add explicit stage/backend declarations at runtime for:
- vision encoder
- vision adapter / projection
- pruning seam
- prefill
- decode
- aux / mask

The runtime must log both the declared experiment backend and the observed backend classification derived from the pinned artifact + runtime session config. Any mismatch between declared and observed configuration is a validation failure, not a silent warning.

### 2. Post-projection pruning seam
Keep pruning strictly after the vision adapter/projection and before prefill. Version 1 remains a cheap systems policy:
- salience from projected visual tokens
- optional diversity suppression / redundancy penalty
- top-K or thresholded keep mask
- packed retained-token transfer into prefill

The pruning policy must stay controllable and measurable. No learned side network is introduced.

### 3. Overlap scheduler
Keep the heterogeneous split explicit:
- NPU prefill for request i+1
- CPU decode for request i

The scheduler must be queue-aware and budget-aware. It chooses from a bounded set of token budgets using queue depth and recent prefill/decode windows, smooths per-request changes, and emits the chosen budget plus rationale per request.

### 4. Observability layer
Structured request-level events are mandatory. Every request should record:
- queue depth at admission
- chosen token budget
- retained-token count
- stage backend manifest
- stage timings for vision/projection/pruning/prefill/decode
- TTFT
- CPU active time
- NPU active time
- CPU idle waiting for NPU
- NPU idle waiting for CPU
- sync/copy counters where directly observable
- final answer / caption / quality outcome

### 5. Partition validation tool
For each run, emit a validation bundle containing:
- subgraph to backend assignment
- visible vs opaque regions
- expected transfer edges
- unsupported / unresolved mapping reasons
- compile artifact identities
- experiment-config deviation failures

This tool is the hardware-facing analysis layer that turns runtime evidence into a reproducible heterogeneous systems artifact.

## Benchmark set
Use exactly two evaluation tasks:
- GQA for grounded visual reasoning and short-answer quality
- COCO Karpathy captioning for free-form generation quality

## Paper story
The paper contribution is a partition-aware LiteRT deployment system for FastVLM on a heterogeneous mobile SoC:
1. explicit runtime/backend observability over real LiteRT partitions
2. a cheap post-projection token-budgeting seam
3. a queue-aware overlap scheduler that balances NPU prefill with CPU decode
4. a validation tool that makes opaque and visible regions explicit

## Success criteria
- Runtime emits stage/backend declarations and overlap TTFT directly from the heterogeneous path.
- Partition validator fails on config deviations and produces machine-readable bundles.
- GQA and COCO run on the real LiteRT FastVLM path without silent fallback.
- The paper bundle includes figures/tables for partition map, overlap timeline, baseline variants, and benchmark quality results.
