# CASES 2026 FastVLM Edge Runtime MVP

## Locked Contributions

This paper stays narrow. The main contributions are:

1. adaptive post-projection visual token budgeting for FastVLM,
2. a CPU-decode / NPU-prefill overlap scheduler for queued image requests,
3. explicit partition plans plus strong observability for heterogeneous execution.

Everything else is supporting infrastructure. The first paper uses FastVLM only.

## CASES Fit

Primary fit:
- CASES 2026 Track 5: Architectures, Compilers, and System-level Design.

Secondary fit:
- CASES 2026 Track 1: AI Systems and Applications of AI at Edge.

Official CFP constraints to design around:
- abstract deadline: May 22, 2026
- full paper deadline: May 29, 2026
- notification: July 28, 2026
- camera-ready: August 25, 2026
- conference: October 4-10, 2026, Seoul, South Korea
- accepted papers appear in IEEE TCAD
- artifact evaluation and badging are available

Source of record:
- https://esweek.org/cases/calls/call-for-papers/
- https://esweek.org/

## Paper Claim

The claim is not just pruning and not just heterogeneous execution.

The claim is that FastVLM throughput on an edge SoC improves when the runtime co-designs:
- visual token budget after the vision adapter,
- overlap between NPU prefill for image N+1 and CPU decode for image N,
- explicit backend partition plans and observability,

while keeping answer quality above a fixed floor.

## Hard Scope Boundaries

Included in the MVP:
- FastVLM only
- one phone-class CPU+NPU target
- queued independent-image workloads
- captioning and VQA evaluation
- short-answer and long-answer request modes
- explicit runtime decisions and no hidden fallback

Excluded from the first paper:
- multi-model orchestration in production form
- persistent memory / RAG / graph memory
- online training or adaptation
- broad cross-model generalization claims

## System Shape

Per request, the execution graph is:
- vision encoder
- vision adapter / projection
- visual token budgeting layer
- LLM prefill
- decode

Target backend assignment for the paper system:
- NPU: vision path, post-projection pruning-compatible processing, prefill, and any stable support graphs
- CPU: decode, scheduler, controller, queueing, and explicit orchestration

## Current Branch Status

Already working and verified in this branch:
- fixed-budget post-projection pruning after the vision adapter output
- end-to-end on-device execution with the working model bundle `artifacts/models/FastVLM-0.5B.qualcomm.sm8750.auxmaskrope_runtime.litertlm`
- deterministic benchmark harness with CSV/JSON/HTML outputs
- explicit partition-plan JSON generation from runtime logs + graph dumps
- manual caption verification on the three repo images in `IMAGES/`

Still blocked and must remain explicit in the paper:
- true CPU-decode / NPU-prefill overlap is not implemented in the current shared advanced runtime because executor access serializes; see `docs/reports/2026-03-08-fastvlm-overlap-blocker.md`
- short/long answer modes currently change prompt and `max_output_tokens`; they do not yet switch between separate decode compute variants
- regenerated Qualcomm prefill binaries are still blocked by missing source/export parity for the shipped precompiled section

## Near-Term Experimental Result To Build On

The verified fixed-budget sweep already shows the paper-worthy systems behavior:
- budget `96` gives median TTFT `101.029 ms`
- budget `256` gives median TTFT `280.211 ms`
- that is a `63.9%` TTFT reduction on the three-image set
- prefill tokens/sec stays almost flat (`1418.79` vs `1424.37`), so the TTFT gain is coming from doing less prefill work, not from a backend artifact
- budgets `128` through `224` all collapse to the same prefill token count (`256`), so the controller should treat those as one performance bucket unless quality differs materially

See `docs/reports/2026-03-08-fastvlm-budget-sweep.md` and `artifacts/cases_benchmark/budget_sweep_20260308/`.

## Required Figures / Tables

The paper package must include:
- TTFT vs visual token budget
- throughput vs quality floor
- CPU/NPU partition map figure
- overlap timeline figure or, if still blocked, an explicit serialization figure
- latency breakdown table by stage
- artifact reproducibility checklist
- no-hidden-fallback statement with explicit failure cases
