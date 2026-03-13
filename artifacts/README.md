# Artifacts Root

The `artifacts/` tree has two distinct roles:

- current GraphPilot-Edge evidence under `artifacts/graphpilot_edge/`
- archived legacy FastVLM generated outputs under `artifacts/legacy_fastvlm/`

Use `docs/ARTIFACT_RETENTION.md` for the retention policy and consolidation rules.

## Keep In Place

- `artifacts/graphpilot_edge/`
- `artifacts/models/`

## Archive Root

- `artifacts/legacy_fastvlm/`

Run the consolidator to archive stale top-level generated outputs:

```bash
env -u PYTHONHOME -u PYTHONPATH python3 scripts/consolidate_artifacts.py
```
