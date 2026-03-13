# GraphPilot-Edge Artifact Surface

This folder is the canonical evidence surface for GraphPilot-Edge.

## What Lives Here

- `analysis/`
  - derived summaries such as calibration, characterization, and focused investigations
- `checkpoints/`
  - checkpoint-pinned snapshots of the current repo truth surface
- `experiments/`
  - experiment batch outputs and measured workflow summaries
- `papers/`
  - generated CASES paper PDFs and build metadata
- `registries/`
  - machine-readable source-of-truth registries for baselines, candidate plans, feasibility, and workloads
- `reports/`
  - artifact packs, audits, paper tables, and report summaries

## Current Canonical Outputs

- Checkpoint: `checkpoints/graphpilot_checkpoint_20260312_195112/summary.json`
- Artifact pack: `reports/artifact_pack_20260312_195226/summary.json`
- CASES paper: `papers/graphpilot_cases_20260312_195225/main.pdf`

## Rules

- Do not hand-edit generated summaries unless the builder itself is wrong.
- Do not move timestamped outputs that are referenced by the current checkpoint.
- If a new checkpoint supersedes this one, update the root README and docs index rather than deleting historical evidence first.
