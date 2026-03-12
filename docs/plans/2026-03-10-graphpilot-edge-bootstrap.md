# GraphPilot-Edge Bootstrap Plan

## Scope
Bootstrap GraphPilot-Edge as a measurable heterogeneous runtime on the connected Snapdragon SM8750 device while preserving the existing FastVLM LiteRT NPU path as the primary VLM implementation.

## Immediate objectives
- recover reusable FastVLM runtime, benchmark, and artifact infrastructure already present in this repository
- recreate Beads task tracking around the new GraphPilot-Edge scope
- establish machine-readable registries and a persistent execution ledger
- verify the live device, host resource guardrails, and current FastVLM path before broader runtime expansion

## Phase breakdown

### Phase 1: Repository bootstrap and state recovery
- verify git state, adb connectivity, and host CPU budget
- create GraphPilot-Edge state ledger and empty registries
- document phase/task graph and ready work
- map existing FastVLM components to GraphPilot-Edge stage architecture

### Phase 2: Stage adapters and feasibility
- preserve FastVLM as the primary VLM path
- validate or add stage adapters for ASR, planner, retrieval, responder, and TTS
- build the backend feasibility matrix with explicit evidence and no silent fallback

### Phase 3: Profiling
- collect cold, warm, and steady-state latency
- separate prefill and decode
- record transfer, memory, thermal, and energy proxy data

### Phase 4: Core runtime algorithms
- candidate-plan enumerator
- discrete-event simulator
- fallback-aware planner
- dependency-aware scheduler
- interval buffer allocator
- KV-cache planner
- plan-bank logic

### Phase 5+: Runtime integration, baselines, evaluation, artifact pack, and final audit
- execute mandatory DAGs on device
- run baselines and top-K candidate validation
- produce plots, tables, traces, manifests, and final audit artifacts

## Guardrails
- no fake backend success; hidden fallback is a failure unless explicitly recorded as `fallback_reason`
- no host saturation; heavy local builds and tests must leave at least two host CPU threads free
- every reported metric must be tied to saved config, command, logs, and outputs

## Immediate ready tasks
1. populate the execution ledger and registry scaffolds
2. record device and host environment facts
3. map current FastVLM runtime pieces into GraphPilot-Edge stages
4. define missing stage-adapter gaps for ASR, planner, retrieval, responder, and TTS
