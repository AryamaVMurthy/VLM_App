# Artifact Retention Consolidation

Date: 2026-03-13

## Policy

- Source: `docs/ARTIFACT_RETENTION.md`
- Archive root: `artifacts/legacy_fastvlm/`
- Kept in place: `artifacts/graphpilot_edge/`, `artifacts/models/`

## Consolidation Result

- Manifest: `artifacts/legacy_fastvlm/consolidation_manifest.json`
- Moved top-level stale/generated directories:
  - `analysis` -> `artifacts/legacy_fastvlm/analysis`
  - `aux_experiments` -> `artifacts/legacy_fastvlm/aux_experiments`
  - `cases_benchmark` -> `artifacts/legacy_fastvlm/cases_benchmark`
  - `coco_eval` -> `artifacts/legacy_fastvlm/coco_eval`
  - `gqa_eval` -> `artifacts/legacy_fastvlm/gqa_eval`
  - `graph_inspect` -> `artifacts/legacy_fastvlm/graph_inspect`
  - `graph_inspect_auxmaskrope_runtime` -> `artifacts/legacy_fastvlm/graph_inspect_auxmaskrope_runtime`
  - `host_edge_diff` -> `artifacts/legacy_fastvlm/host_edge_diff`
  - `local_qairt_repo` -> `artifacts/legacy_fastvlm/local_qairt_repo`
  - `logs` -> `artifacts/legacy_fastvlm/logs`
  - `partition_plans` -> `artifacts/legacy_fastvlm/partition_plans`
  - `prefill_experiments` -> `artifacts/legacy_fastvlm/prefill_experiments`
  - `regeneration` -> `artifacts/legacy_fastvlm/regeneration`
  - `tmp` -> `artifacts/legacy_fastvlm/tmp`

## Post-Consolidation Top-Level Artifact Layout

- `artifacts/README.md`: `4.0K`
- `artifacts/graphpilot_edge`: `267M`
- `artifacts/models`: `4.2G`
- `artifacts/legacy_fastvlm`: `8.3G`

## Verification

- Dry run completed before move with `scripts/consolidate_artifacts.py --dry-run`.
- Consolidation executed with `scripts/consolidate_artifacts.py`.
- Legacy script defaults were updated to resolve archived outputs from `artifacts/legacy_fastvlm/`.
