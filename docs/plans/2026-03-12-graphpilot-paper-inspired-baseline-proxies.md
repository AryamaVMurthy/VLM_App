# GraphPilot Paper-Inspired Baseline Proxies

This note documents the simplified baseline proxies used in the GraphPilot-Edge harness. These are not claims of exact paper reimplementation. They are bounded method-class reproductions inside the same simulator and artifact pipeline so that comparisons stay fair and support-safe.

## Scope

- All baselines run in the same GraphPilot simulator/runtime harness.
- All baselines are evaluated with the same downstream metrics.
- The proxy heuristic defines how each baseline chooses placement and edge behavior.
- GraphPilot still evaluates every resulting candidate with the same objective and trace machinery.

## Proxy definitions

- `band_like`
  - Whole-stage mobile multi-DNN heuristic.
  - Uses support-safe stage-greedy placement.
  - Forces all edges to `FULL`, so inter-stage streaming is disabled.
  - Captures the class of mobile heterogeneous schedulers that place stages but do not model assistant streaming edges.

- `adms_like`
  - Energy-biased mobile scheduler proxy.
  - Uses stage-local support-safe placement with an execution-plus-energy cost.
  - Forces all edges to `FULL`.
  - Captures dynamic mobile scheduling that trades latency against energy without GraphPilot stream/KV reasoning.

- `puzzle_like`
  - Static heterogeneous mapping proxy.
  - Searches support-safe stage placements over a no-pipeline workflow with all edges forced to `FULL`.
  - Captures static heterogeneous multi-DNN mapping without GraphPilot streaming-aware execution.

- `twill_like`
  - Compound-AI static scheduler proxy.
  - Preserves the compound workflow structure.
  - Uses the `no_memory_kv` ablated scenario and support-safe static mapping.
  - Captures compound-system scheduling without GraphPilot memory/KV control.

- `heteroinfer_like`
  - LLM-centric heterogeneity proxy.
  - Prefers accelerators for LLM/VLM prefill/decode-style heavy stages.
  - Keeps prompt assembly, postprocess, audio, and retrieval-style glue on CPU-first support-safe paths.
  - Captures single-LLM heterogeneity embedded into the GraphPilot workload universe.

- `agent_xpu_like`
  - Agentic SoC scheduling proxy.
  - Keeps VLM on NPU when feasible, retrieval on CPU, and prefers CPU for small first-output-sensitive glue stages when the latency gap is small.
  - Preserves streamable edges.
  - Captures agentic flow scheduling without GraphPilot thermal/memory/KV adaptation.

- `hero_like`
  - Mobile agentic-RAG orchestration proxy.
  - Keeps retrieval CPU-first, VLM NPU-first, and prefers accelerators for LLM-heavy stages when support-safe.
  - Preserves compound workflow overlap.
  - Captures mobile agentic RAG orchestration without GraphPilot fallback/KV/simulator coupling claims.

## Explicit limitations

- These are proxy baselines, not source-level reproductions of the original systems.
- They intentionally stay inside the GraphPilot harness so comparisons use one simulator, one workload universe, and one metric surface.
- They do not claim the full implementation details of the cited systems.
- Where the current support-safe feasible set is narrow, the proxies may collapse to similar assignments. That outcome is reported as evidence, not hidden.
