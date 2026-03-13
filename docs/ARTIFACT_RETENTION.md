# Artifact Retention Policy

This repository keeps artifact paths stable by separating current GraphPilot evidence from older generated outputs.

## Retention Classes

### 1. Canonical GraphPilot evidence — retain in place

Keep these paths exactly where they are:
- `artifacts/graphpilot_edge/`
- `artifacts/graphpilot_edge/checkpoints/`
- `artifacts/graphpilot_edge/reports/`
- `artifacts/graphpilot_edge/papers/`

These directories define the current checkpoint-pinned truth surface for GraphPilot-Edge.

### 2. Shared operational cache — retain in place

Keep:
- `artifacts/models/`

This is large, shared model material used by multiple scripts. It is not archived with stale experiment outputs.

### 3. Legacy pre-GraphPilot generated outputs — consolidate under one archive root

Move these top-level directories under:
- `artifacts/legacy_fastvlm/`

Directories covered by this policy:
- `analysis/`
- `aux_experiments/`
- `cases_benchmark/`
- `coco_eval/`
- `gqa_eval/`
- `graph_inspect/`
- `graph_inspect_auxmaskrope_runtime/`
- `host_edge_diff/`
- `local_qairt_repo/`
- `logs/`
- `partition_plans/`
- `prefill_experiments/`
- `regeneration/`
- `tmp/`

These are retained for historical reproducibility, but they are no longer first-class repo-surface directories.

## Consolidation Rules

- Use `scripts/consolidate_artifacts.py` to archive stale top-level artifact directories.
- The script fails fast if a destination already exists.
- The script writes `artifacts/legacy_fastvlm/consolidation_manifest.json` so the local archive state is explicit.
- Do not move `artifacts/graphpilot_edge/` or `artifacts/models/`.

## Legacy Script Compatibility

Legacy FastVLM scripts now resolve their default artifact roots under `artifacts/legacy_fastvlm/`. Historical docs may still mention the old top-level paths; treat those as pre-consolidation references.

## Verification

```bash
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest scripts.tests.test_artifact_retention_policy -v
env -u PYTHONHOME -u PYTHONPATH python3 scripts/consolidate_artifacts.py --dry-run
```
