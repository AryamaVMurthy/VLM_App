# CASES 2026 FastVLM Paper Outline

## Working Title

Adaptive Visual Token Budgeting and Explicit Heterogeneous Execution for Edge FastVLM Inference

## One-Sentence Thesis

FastVLM on a phone-class CPU/NPU platform can reduce TTFT substantially when post-projection visual token budgeting is treated as a runtime systems control and paired with explicit heterogeneous partitioning and observability.

## Scope Statement

This paper is intentionally narrow:
- model family: FastVLM only
- hardware: one phone-class CPU/NPU target
- implemented contributions: visual token budgeting, partition-plan observability, deterministic benchmark harness
- explicit blocker: honest CPU-decode / NPU-prefill overlap is architecturally blocked in the current shared runtime and is presented as a next-step design rather than a false implementation claim

## Abstract Skeleton

Problem:
- Edge VLM inference is dominated by multimodal prefill cost and opaque heterogeneous execution behavior.

Observation:
- On-device FastVLM latency is sensitive to the visual token count that reaches prefill, but current runtimes do not expose this as a systems control variable or make backend placement explicit.

Approach:
- Insert a post-projection visual token budgeting layer in FastVLM.
- Expose the budget as a runtime control knob.
- Add explicit partition-plan extraction and deterministic benchmark tooling.
- Analyze the overlap path and surface the runtime architecture that blocks honest CPU-decode / NPU-prefill concurrency.

Key result:
- On the verified three-image set, reducing the visual budget from 256 to 96 cuts median TTFT from 280.211 ms to 101.029 ms, a 63.9% reduction, while prefill tokens/sec stays nearly unchanged.

Contribution framing:
- runtime-controlled visual token budgeting
- explicit heterogeneous placement/observability
- architecture-level overlap diagnosis and scheduler path definition

## Section Outline

### 1. Introduction

Points to make:
- Edge VLMs are latency-sensitive and backend-sensitive.
- Prefill dominates TTFT.
- Hidden fallback and opaque partitioning make systems claims weak.
- The paper contribution is a runtime view of visual-token budgeting, not just a pruning trick.

### 2. Background And Problem Statement

Cover:
- FastVLM execution stages: vision encoder, adapter, multimodal combine, prefill, decode
- CPU/NPU heterogeneity on edge SoCs
- Why TTFT is dominated by prefill work
- Why systems papers need explicit placement evidence

### 3. System Design

#### 3.1 Post-Projection Visual Token Budgeting
- insertion point after vision adapter output
- fixed-budget MVP in this branch
- future adaptive controller inputs: image complexity, queue depth, decode completion window

#### 3.2 Explicit Partition Plans And Observability
- runtime log capture
- graph-dump analysis
- partition-plan JSON artifacts
- no-hidden-fallback contract

#### 3.3 Overlap Scheduler Path
- desired pipeline: CPU decode for request N, NPU prefill for N+1
- current blocker in shared runtime
- chosen next-step architecture: separate execution pools plus explicit KV/state handoff

### 4. Implementation

Reference concrete files:
- pruning/runtime control:
  - `third_party/litert-lm/runtime/util/executor_data_util.cc`
  - `third_party/litert-lm/runtime/core/session_basic.cc`
  - `third_party/litert-lm/runtime/framework/resource_management/execution_manager.cc`
- settings/CLI:
  - `third_party/litert-lm/runtime/engine/engine_settings.cc`
  - `third_party/litert-lm/runtime/engine/litert_lm_settings_util.cc`
  - `scripts/run_fastvlm_litert_npu_adb.sh`
- benchmark harness:
  - `scripts/fastvlm_cases_benchmark.py`
  - `scripts/fastvlm_decode_profiles.py`
- partition plan:
  - `scripts/fastvlm_partition_plan.py`

### 5. Experimental Setup

Must include:
- target device description
- working model artifact:
  - `artifacts/models/FastVLM-0.5B.qualcomm.sm8750.auxmaskrope_runtime.litertlm`
- evaluation protocol:
  - deterministic warmup/measured runs
  - image set and prompts
  - answer mode definition
- metrics:
  - TTFT
  - prefill latency
  - prefill tokens/sec
  - decode tokens/sec
  - backend placement
  - caption correctness / quality metric once COCO and GQA are integrated

### 6. Results

#### 6.1 Fixed-Budget TTFT Reduction
Use:
- `docs/reports/2026-03-08-fastvlm-budget-sweep.md`
- `artifacts/cases_benchmark/budget_sweep_20260308/summary.csv`

Headline table:
- budget vs TTFT vs prefill latency vs prefill tok/s

#### 6.2 Discrete Prefill Bucket Behavior
Key insight:
- budgets 128-224 collapse into the same prefill token bucket
- budget 96 hits a smaller bucket and drives most of the TTFT win

#### 6.3 Partition Transparency
Use:
- `artifacts/partition_plans/adb_npu_npu_run_20260308_150633.partition_plan.json`
- `artifacts/graph_inspect_auxmaskrope_runtime/`

Make explicit:
- embedder, aux, and vision adapter are NPU in the working runtime bundle
- precompiled DISPATCH_OP sections are subgraph-only, not per-op mappable

#### 6.4 Overlap Blocker Analysis
Use:
- `docs/reports/2026-03-08-fastvlm-overlap-blocker.md`

State clearly:
- current runtime architecture prevents honest overlap claims
- the paper contribution is therefore a verified blocker analysis plus a concrete scheduler/execution-pool path, not a false speedup claim

### 7. Limitations

Be explicit:
- adaptive controller not yet implemented
- decode compute-variant switching not yet implemented
- overlap scheduler blocked by current runtime topology
- dataset-scale quality sweep still pending for COCO and GQA

### 8. Future Work

- adaptive controller using image complexity + queue depth + decode-time matching
- real overlap via separate execution pools and KV/state handoff
- decode variant switching at admission time
- broader dataset evaluation

## Figures To Produce

1. System diagram
   - FastVLM pipeline with pruning insertion point and CPU/NPU placement
2. TTFT vs budget curve
3. Prefill latency vs budget curve
4. Partition map figure
5. Overlap blocker / serialization diagram
6. Optional timeline figure once real overlap exists

## Tables To Produce

1. Main latency table by budget
2. Backend placement table per submodel/subgraph
3. Manual sanity-check table for repo images
4. Final COCO/GQA quality-vs-budget table

## Artifact Mapping

- working sweep report:
  - `artifacts/cases_benchmark/budget_sweep_20260308/report.html`
- long-mode spot check:
  - `artifacts/cases_benchmark/long_compare_20260308/report.html`
- working partition plan:
  - `artifacts/partition_plans/adb_npu_npu_run_20260308_150633.partition_plan.json`
- overlap blocker report:
  - `docs/reports/2026-03-08-fastvlm-overlap-blocker.md`
- progress report:
  - `docs/reports/2026-03-08-cases-fastvlm-progress.md`

## Writing Checklist

Before drafting the final paper body, make sure the following are true:
- no claim depends on hidden fallback
- every backend-placement claim has a log or partition artifact
- every latency claim has a raw CSV source
- every overlap claim is either measured or explicitly marked blocked
- every future-work item is separated from implemented contributions
