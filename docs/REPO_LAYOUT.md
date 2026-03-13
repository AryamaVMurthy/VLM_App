# Repository Layout

This repository is organized around stable evidence paths. Consolidation is documentation-first: files are indexed and explained, but canonical checkpoint and paper paths are not moved.

## Top-Level Map

- `android-app/`
  - deployed Android runtime, instrumentation tests, Gradle build, and app assets
- `graphpilot_edge/`
  - offline planner, simulator, calibration, scheduling, workload, memory, and baseline code
- `scripts/`
  - builders, calibration runs, experiment runners, packaging, provisioning, and paper-generation entry points
- `tests/`
  - Python tests for the `graphpilot_edge` package
- `scripts/tests/`
  - regression tests for repo scripts and build surfaces
- `artifacts/`
  - generated evidence, logs, experiments, reports, papers, and checkpoint outputs
- `configs/`
  - workload, optimization, and model configuration data
- `Truth-docs/`
  - scope and revision boundary documents used to constrain the paper claim
- `docs/`
  - repo-facing plans, indexes, and supporting notes
- `papers/`
  - LaTeX/IEEE template support used by the paper builders
- `native-bridge/`, `third_party/`, `apk/`, `IMAGES/`
  - supporting native code, vendored dependencies, installable outputs, and static assets

## Path-Stability Rules

Do not move or rename these surfaces casually:
- `artifacts/graphpilot_edge/checkpoints/`
- `artifacts/graphpilot_edge/reports/`
- `artifacts/graphpilot_edge/papers/`
- `Truth-docs/graphpilot_edge_revision_report.pdf`

These paths are referenced by tests, reports, builders, and the final paper. If they change, the evidence surface drifts and the audit stops being trustworthy.

## Current Canonical Outputs

- Checkpoint: `artifacts/graphpilot_edge/checkpoints/graphpilot_checkpoint_20260312_195112/summary.json`
- Artifact pack: `artifacts/graphpilot_edge/reports/artifact_pack_20260312_195226/summary.json`
- CASES paper: `artifacts/graphpilot_edge/papers/graphpilot_cases_20260312_195225/main.pdf`

## Working Conventions

- Use `env -u PYTHONHOME -u PYTHONPATH python3 ...` for Python commands in this repo.
- Treat `artifacts/` as generated evidence, not hand-edited source.
- Add new instructions close to the folder they describe rather than bloating the root README.
- Keep local-only junk out of git: `.venv/`, `tmp/`, caches, generated scratch files.

## Folder-Level Instructions

- `artifacts/graphpilot_edge/README.md`
- `graphpilot_edge/README.md`
- `scripts/README.md`
- `android-app/README.md`
- `Truth-docs/README.md`
