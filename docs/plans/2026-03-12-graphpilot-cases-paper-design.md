# GraphPilot-Edge CASES Paper Design

**Date:** 2026-03-12

## Context

GraphPilot-Edge already has a verified prototype scope with a canonical checkpoint, calibrated simulator, support-safe deployed workflows A/B/C, artifact pack generation, and a closed Beads graph for the prototype milestone. The next scope is a submission-grade IEEE ESWEEK CASES paper that must stay strictly inside the measured system boundary and the revision requirements captured in `Truth-docs/graphpilot_edge_revision_report.pdf`.

The paper must optimize for the strongest defensible claim, not the broadest possible claim. The verified deployed runtime remains support-safe but narrow: FastVLM on LiteRT NPU, CPU text/speech paths, explicit fallback accounting, calibrated simulation, and continuous assistant DAG scheduling. The CASES paper should make that system legible, rigorous, and review-resilient.

## Goal

Produce a solid 12-page IEEE ESWEEK CASES paper package from a single canonical evidence set by:
- keeping workflows A/B/C as the primary real-device results,
- using the broader workload universe for simulator characterization, continuous-stream studies, ablations, and baseline comparisons,
- implementing faithful method-class proxy baselines inside the same simulator/runtime environment,
- closing the remaining evidence gaps from the revision report,
- regenerating all reports, tables, figures, and audit outputs from one consistent checkpoint.

## Non-Goals

- Do not broaden the paper claim into unsupported “full CPU/GPU/NPU support across all model families”.
- Do not present simulator-side what-if studies as deployed runtime capabilities.
- Do not re-run arbitrary new experiments outside the revision report unless they directly strengthen the CASES submission.
- Do not create a new greenfield paper artifact path separate from the current GraphPilot artifact system.

## Chosen Strategy

### 1. Strongest Defensible Claim

The paper claim is:

> GraphPilot-Edge is a profiler-driven runtime and calibrated simulator for continuous multimodal assistant DAGs on Snapdragon SM8750 that performs support-safe placement, explicit fallback accounting, streaming-aware scheduling, and memory/KV-aware control, and validates those decisions in both simulation and on the real device.

This claim is intentionally narrower than “broad heterogeneous execution across every stage family”, because the latter is not yet supported by measured support-safe backend evidence.

### 2. Hybrid Evidence Layout

- **Primary real-device evidence:** Workflows A/B/C, support-safe backend feasibility, live streaming behavior, memory/KV admission behavior, and top-K sim-to-real validation.
- **Primary simulator evidence:** workload-family characterization, continuous-stream scheduling studies, faithful method-class baseline comparisons, and ablations.
- **Bridge between the two:** calibration and residual-error reporting, with explicit statements about what the simulator predicts well and where it remains approximate.

### 3. Revision-Driven Completion

`Truth-docs/graphpilot_edge_revision_report.pdf` is the hard scope boundary. Every required figure, experiment class, and narrative fix must trace back to that document.

## Architecture of the Paper Build

### Canonical Truth Surface

All paper-facing outputs must derive from one checkpoint manifest. That manifest pins:
- experiment summaries,
- calibration summaries,
- characterization summaries,
- baseline registries,
- workload registries,
- candidate plan registries,
- final audit inputs,
- figure/table sources.

No paper table or figure may be generated from a different “latest” artifact than the one referenced by the active checkpoint.

### Experiment Layers

1. **Real-device workflow layer**
   - workflows A/B/C
   - support-safe path only
   - TTFT, TTFS, end-to-end latency, peak memory, fallback counts, runtime traces
2. **Simulator characterization layer**
   - primitive/model-family workload characterization
   - calibration quality and residual error
   - backend affinity, launch overhead, transfer, thermal, batching, memory/KV curves
3. **Continuous-stream layer**
   - queue delay, miss rate, throughput, TTFS under overlapping workloads
4. **Comparison layer**
   - internal baselines
   - faithful method-class proxy baselines
   - component ablations

### Baseline Philosophy

Use faithful method-class proxies, not fake one-line caricatures and not impossible full reproductions. Each baseline class must be:
- implemented inside the same GraphPilot simulator/runtime environment,
- documented as a method-class proxy,
- evaluated on the same workloads,
- compared using the same metrics and checkpoint.

## Required Deliverables

1. **Single canonical CASES checkpoint**
   - consistent evidence set
   - no artifact drift
2. **Completed experiment matrix from revision report**
   - calibration
   - characterization
   - continuous-stream studies
   - baseline comparisons
   - ablations
   - fallback penalty studies
3. **Submission-grade CASES paper**
   - IEEE 2-column format
   - target 12 pages excluding references if allowed by venue rules, otherwise 12 pages total with compression pass
   - final abstract, intro, related work, system design, simulator, methodology, results, limitations, artifact statement
4. **Paper figures and tables**
   - architecture diagram
   - deployed workflow/backend map
   - sim-to-real calibration plot/table
   - primary workflow results
   - continuous-stream results
   - baseline comparison table
   - ablation table
   - fallback-cost illustration
5. **Final audit and artifact pack**
   - explicit scope statement
   - strongest surviving claim
   - reproducibility outputs

## Error Handling and Honesty Rules

- If a required method-class baseline is too ambiguous to reproduce exactly, implement the smallest faithful proxy, state the approximation, and keep going.
- If a claimed workload family still lacks sufficient evidence, either run the missing experiment or narrow the claim.
- If simulator and device differ materially, report the residual and either recalibrate or limit the conclusion.
- If the 12-page limit forces compression, compress breadth before overstating claims.

## Testing and Verification Strategy

- Keep the existing regression suite green throughout.
- Re-run Python, script, Android unit, and Android connected tests before claiming the submission package is complete.
- Regenerate the checkpoint, artifact pack, and paper outputs from one manifest and verify that all paper tables/figures point into that manifest.
- Treat any mismatch between paper numbers and checkpoint numbers as a release blocker.

## Execution Order

1. Create a new Beads epic for CASES paper completion.
2. Write a detailed implementation plan with small, testable tasks.
3. Execute evidence-closure tasks first: truth surface, experiment closure, baseline closure, ablation closure.
4. Generate the CASES paper only after the evidence set is clean.
5. Run final verification, commit, and push the submission-ready state.
