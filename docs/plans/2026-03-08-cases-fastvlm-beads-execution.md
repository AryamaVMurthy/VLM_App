# CASES FastVLM Beads Execution Plan

This is the concrete execution ladder for the current FastVLM-only paper branch. It is written as a beads-driven plan with explicit success and failure branches.

## Goal

Ship a defensible CASES submission around three contributions:
- adaptive post-projection visual token budgeting
- CPU-decode / NPU-prefill overlap scheduler
- explicit partition plan and observability

## Hard Rule

Do not claim overlap, backend placement, or decode-mode switching unless the runtime evidence is explicit in logs and artifacts.

## Canonical Runtime Artifact

Use this model bundle unless a task explicitly says otherwise:
- `artifacts/models/FastVLM-0.5B.qualcomm.sm8750.auxmaskrope_runtime.litertlm`

Use this runner unless a task explicitly says otherwise:
- `scripts/run_fastvlm_litert_npu_adb.sh`

## Phase 0: Lock The Contract

### `fvlm-zth.1` — Lock CASES paper contract and evaluation gates

Status:
- done

Source of record:
- https://esweek.org/cases/calls/call-for-papers/

What this bead locks:
- Track 5 as primary fit
- Track 1 as secondary fit
- paper dates
- required figures/tables
- artifact evaluation expectations

If anything in the CFP changes:
- update `docs/plans/2026-03-08-cases-fastvlm-edge-runtime-mvp.md`
- append a beads note to `fvlm-zth.1`
- do not continue with stale dates or stale track assumptions

## Phase 1: Keep The Working Baseline Reproducible

### `fvlm-zth.5` — Deterministic benchmark harness

Current state:
- partially done and usable now
- current implementation: `scripts/fastvlm_cases_benchmark.py`

What to run now:
```bash
env -u PYTHONHOME -u PYTHONPATH python3 scripts/fastvlm_cases_benchmark.py \
  --model artifacts/models/FastVLM-0.5B.qualcomm.sm8750.auxmaskrope_runtime.litertlm \
  --images IMAGES/person.jpeg IMAGES/download.jpeg IMAGES/images.jpeg \
  --budgets 96,128,160,192,224,256 \
  --warmup-runs 0 \
  --measured-runs 1 \
  --skip-build 1 \
  --max-num-tokens 384 \
  --answer-mode short \
  --output-dir artifacts/cases_benchmark/budget_sweep_20260308
```

Success means:
- CSV/JSON/HTML outputs exist
- raw runner stdout/stderr per run exist
- raw device log paths are captured

If this fails:
- first check `adb devices -l`
- then run the runner directly on one image with `--event-mode 1 --benchmark 1`
- do not continue to controller work until the baseline harness is reproducible

### `fvlm-zth.6` — Observability and runtime decision logging

Current state:
- partial but already useful

Already verified:
- visual budget is logged in runtime output
- structured `VLM_EVENT` lines are emitted
- benchmark latencies are emitted
- runner and harness preserve log paths

What still needs to be added before paper lock:
- queue-depth logging in the actual scheduler path
- overlap lifecycle/stall reasons
- explicit decode-variant identity when real decode variants exist

If new runtime work is added:
- it must emit structured logs
- if it cannot emit structured logs, stop and fix observability before continuing

### `fvlm-zth.7` — Explicit partition plan and validation

Current state:
- partition artifact generation works
- current tool: `scripts/fastvlm_partition_plan.py`

Canonical artifacts:
- baseline mixed-placement artifact:
  - `artifacts/partition_plans/adb_npu_benchmark_20260307_162241.partition_plan.json`
- working runtime artifact with reduced AUX bundle:
  - `artifacts/partition_plans/adb_npu_npu_run_20260308_150633.partition_plan.json`
- matching graph dump for the working bundle:
  - `artifacts/graph_inspect_auxmaskrope_runtime/`

What still needs to be finished:
- explicit plan/runtime mismatch validation instead of report-only output

If the plan tool sees ambiguous attribution:
- it must mark the subgraph unresolved or limited
- it must not silently guess

## Phase 2: Lock Contribution 1 — Post-Projection Pruning

### `fvlm-zth.3` — Probe pruning seam

Status:
- done

Decision already established:
- the clean seam is after vision adapter output is materialized on host and before multimodal embedding lookup consumes the vision token placeholders

### `fvlm-zth.9` — Choose pruning integration path

Decision:
- runtime-side interception is the chosen path for the paper MVP

Why:
- it preserves FastVLM tensor contracts for the fixed-budget MVP
- it avoids exporter/model-regeneration work for the first paper step

If runtime-side pruning later breaks correctness on a new model export:
- open a new bead for exporter-level pruning insertion
- do not silently change the claim for this branch

### `fvlm-zth.11` — Fixed-budget pruning MVP

Current state:
- working and verified

Evidence:
- runtime budget knob exists
- budget is externally configurable
- budget is logged on device
- benchmark artifacts quantify latency impact
- manual captions on the three repo images still make sense

What to do next:
- sweep more budgets and tasks
- choose safe bands by task/quality floor

### `fvlm-zth.12` — Choose safe quality bands

Next concrete ladder:
1. COCO caption subset with budgets `96,128,160,192,256`
2. GQA subset with the same budgets
3. record caption/VQA quality and TTFT jointly
4. define a quality floor per task
5. freeze the safe budget set used by the controller

If quality collapses at `96`:
- do not force it as the global default
- keep `96` as a throughput mode only if the paper explicitly frames the tradeoff

## Phase 3: Lock Contribution 2 — CPU/NPU Overlap

### `fvlm-zth.2` — Probe overlap feasibility

Current state:
- blocked with explicit evidence

Blocking report:
- `docs/reports/2026-03-08-fastvlm-overlap-blocker.md`

Current truth:
- the shared advanced runtime serializes LLM executor access
- one shared executor path cannot support honest CPU-decode / NPU-prefill overlap

### `fvlm-zth.8` — Choose overlap implementation path

Current decision path:
- do not fake overlap inside the current shared runtime
- implement overlap only through separate execution pools with explicit KV/state handoff

Concrete next steps:
1. split CPU decode and NPU prefill into separate execution pools or processes
2. define the state transfer boundary explicitly
3. log overlap windows, stalls, and admission decisions
4. prove real overlap with a timeline artifact before writing scheduler claims

If the split still serializes because of hidden shared resources:
- capture the exact layer causing it
- reduce the contribution claim to explicit serialization analysis plus the scheduler design, not a false implementation claim

### `fvlm-zth.13` — Implement overlap scheduler

Do this only after `fvlm-zth.8` is resolved.

Target behavior:
- CPU decodes request `N`
- NPU prefills request `N+1`
- scheduler adapts visual token budget to keep the NPU stage near the CPU decode window

Required logs:
- queue depth at admission
- chosen budget
- chosen answer mode
- overlap start/end timestamps
- stall reason if no overlap occurs

## Phase 4: Lock Contribution 3 — Decode Modes

### `fvlm-zth.4` — Probe decode switching

Status:
- done

Current truth:
- safe first implementation is admission-time selection among separate decode variants or session pools
- current branch only implements short/long prompt and `max_output_tokens` scaffolding

### `fvlm-zth.10` — Choose decode switching path

Decision:
- admission-time selection across separate decode variants is the paper path
- in-session variant switching is out of scope until KV/cache compatibility is proven

### `fvlm-zth.14` — Implement explicit runtime switching

Next ladder:
1. export two compatible decode variants
2. give each a stable identifier
3. select the variant at request admission
4. log the chosen variant in every run

If compatible decode exports are not available in time:
- keep short/long as an experiment-level mode only
- state clearly that full compute-variant switching is future work

## Phase 5: Adaptive Controller

### `fvlm-zth.15` — Adaptive token-budget controller

Only start after:
- fixed-budget bands are measured
- overlap path is real, not assumed

Controller order:
1. choose a base budget from image complexity
2. adjust for queue depth
3. refine toward recent decode completion time when the pipeline is stable

If the controller thrashes:
- add hysteresis
- cap per-request budget deltas
- prefer discrete budget buckets rather than dense continuous changes

## Phase 6: Final Paper Lock

### `fvlm-zth.16` — Integrated ablations and locked result set

Required ablations:
- no pruning
- fixed-budget pruning only
- overlap only
- adaptive controller on top of the real overlap path

If overlap is still blocked by submission time:
- make the paper scope honest
- present pruning + partitioning + observability as implemented
- present overlap as blocked by the current runtime architecture with measured evidence and the proposed split design

### `fvlm-zth.17` — Reproducible artifact and paper-ready figures

Must include:
- command lines
- model artifact name
- logs and report directories
- benchmark CSV/JSON/HTML outputs
- partition-plan JSON
- manual image sanity checks
- explicit failure cases and remediation
