# GraphPilot-Edge Runtime-First Expanded Design

## Goal

Extend the current audited GraphPilot-Edge prototype into the broader paper-grade system by taking a runtime-first path: strengthen the online scheduler, memory/KV controller, stream-edge handling, and predicted-cost schema first, while preserving the current support-safe deployed workflows and using them as the regression anchor for the later simulator, workload, baseline, calibration, and paper phases.

## Current Verified Anchor

- Workflow A is support-safe CPU-only and has live responder-to-TTS streaming with fresh audited evidence.
- Workflow B is support-safe with FastVLM on LiteRT NPU and the remaining stages on CPU.
- Workflow C is support-safe with FastVLM on LiteRT NPU, retrieval on CPU, and the remaining stages on CPU.
- Offline GraphPilot planner/simulation and script test surfaces are green under a clean Python environment.

This anchor is not optional. Every expanded runtime change must preserve it.

## Why Runtime-First

The current gap is not that GraphPilot lacks all abstractions. It already has offline planning, simulation, memory, KV, candidate-plan generation, and runtime scaffolding. The most visible weakness relative to the intended paper is that the online brain still behaves like a bounded FIFO admission queue wrapped around mostly workflow-specific execution logic.

The runtime-first path improves the real deployed system first:

1. it strengthens the online claim immediately,
2. it creates the correct execution model for later simulator calibration,
3. it avoids growing the offline side around a runtime that is too weak to validate the claim,
4. it preserves the current support-safe paths while making future heterogeneous growth measurable rather than aspirational.

## Design Principles

- Preserve FastVLM on LiteRT NPU as the primary VLM path.
- Never count silent fallback as success.
- Keep responder prefill and decode distinct in all models and schemas.
- Fail fast with explicit remediation when runtime plans request unsupported backends.
- Extend schemas in a backward-compatible way so old artifact evidence is still readable.
- Add runtime behavior in layers behind explicit GraphPilot types and tests.

## Runtime-First Architecture

### 1. Event Scheduler

Replace the current single-active FIFO scheduler with a backend-aware scheduler that still exposes the same `runWithAdmission` integration point to the coordinator.

The scheduler will track:

- request metadata,
- predicted stage costs,
- stage/backend assignments,
- queue depth,
- backend occupancy,
- admission pressure,
- deadline slack,
- first-output sensitivity.

The initial runtime realization will stay coarse enough to fit the current coordinator, but its scoring interface will match the paper formulas:

- upward-rank urgency,
- first-output bonus,
- age bonus,
- slack/lateness term,
- affinity term,
- copy penalty,
- memory-risk penalty,
- thermal-risk penalty.

This keeps the runtime deployable while making the scoring model paper-aligned.

### 2. Predicted-Cost and Objective Expansion

Current runtime plans only require:

- `stream_makespan_ms`
- `p95_queue_delay_ms`
- `deadline_miss_rate`
- `memory_mb`
- `peak_memory_bytes`
- `ttft_ms`
- `ttfs_ms`

The expanded schema must include the full 8-term objective inputs and carry them through:

- `p95_e2e_ms`
- `p95_ttfs_ms`
- `avg_energy_mj`
- `peak_memory_bytes`
- `copy_bytes`
- `quality_loss`
- `p95_queue_delay_ms`
- `deadline_miss_rate`

The runtime does not need to optimize all of these online immediately, but it must be able to load and log them so the offline and online sides share one truthful plan schema.

### 3. Memory/KV and Plan-Bank Strengthening

Current memory control is a workflow-heuristic admission controller with a degradation ladder. That is a useful base, but it must be extended with:

- explicit KV-sensitive policy fields,
- richer degradation outputs,
- decode stickiness metadata,
- state-aware plan-bank selection hooks,
- thermal state identifiers and hysteresis fields.

The runtime will still preserve the current admitted/degrade/reject flow, but it will be able to attach richer reasons and future switching inputs.

### 4. Stream-Edge Generalization

Current live streaming is strongest for responder-to-TTS. The next online extension is:

- chunked STT partials into planner where feasible,
- explicit runtime consumption of `stream_edges`,
- first-output aware stage release decisions,
- no regression to workflow A live streaming.

The first version can remain workflow-aware rather than fully generic, but the data model and tests must support the general stream-edge claim.

### 5. Simulator and Baseline Preparation

Even though this is a runtime-first plan, the runtime changes must prepare for the next phases:

- multi-resource hardware simulator,
- model-graph workload families,
- baseline policy classes,
- sim-to-real calibration.

That means the runtime types added now must not hard-code CPU/GPU/NPU as one global queue. They need to be resource-instance friendly.

## Code Mapping

Primary runtime files:

- `android-app/app/src/main/java/com/qidk/fastvlm/core/graphpilot/GraphPilotRuntimeScheduler.kt`
- `android-app/app/src/main/java/com/qidk/fastvlm/core/graphpilot/GraphPilotCoordinator.kt`
- `android-app/app/src/main/java/com/qidk/fastvlm/core/graphpilot/GraphPilotMemoryAdmissionController.kt`
- `android-app/app/src/main/java/com/qidk/fastvlm/core/graphpilot/GraphPilotPlanStore.kt`

Primary offline/schema files:

- `graphpilot_edge/cost_model.py`
- `graphpilot_edge/simulation.py`
- `graphpilot_edge/planner.py`
- `graphpilot_edge/plan_bank.py`
- `graphpilot_edge/models.py`
- candidate-plan and registry scripts under `scripts/`

Primary tests:

- `android-app/app/src/test/java/com/qidk/fastvlm/core/graphpilot/`
- `tests/graphpilot_edge/`
- `scripts/tests/`

## Execution Order

1. Revalidate the current audited runtime and tests.
2. Expand predicted-cost/objective schemas.
3. Upgrade runtime scheduler to backend-aware scoring and admission metadata.
4. Upgrade memory/KV and plan-bank policy surfaces.
5. Add stream-edge runtime extensions without regressing workflow A.
6. Only then move into the broader simulator, workload, and baseline phases.

## Success Conditions for This Design

- The current deployed support-safe flows still work.
- Runtime logs become richer and more paper-aligned.
- Offline and online plan schemas stop diverging.
- The next phases can build on real runtime behavior instead of placeholder orchestration.
