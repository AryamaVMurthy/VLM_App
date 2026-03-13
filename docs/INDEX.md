# Docs Index

This index is the fastest way to navigate the GraphPilot-Edge repository without guessing which file is current.

## Primary Entry Points

- Repo overview: `README.md`
- Repo layout and path-stability rules: `docs/REPO_LAYOUT.md`
- Artifact retention policy: `docs/ARTIFACT_RETENTION.md`
- Revision boundary document: `Truth-docs/graphpilot_edge_revision_report.pdf`

## Canonical Evidence Surface

- Artifact surface guide: `artifacts/graphpilot_edge/README.md`
- Current checkpoint summary: `artifacts/graphpilot_edge/checkpoints/graphpilot_checkpoint_20260312_195112/summary.json`
- Current artifact pack summary: `artifacts/graphpilot_edge/reports/artifact_pack_20260312_195226/summary.json`
- Current CASES paper PDF: `artifacts/graphpilot_edge/papers/graphpilot_cases_20260312_195225/main.pdf`

## Code Surfaces

- Offline planner/simulator package: `graphpilot_edge/README.md`
- Android deployed runtime: `android-app/README.md`
- Operational/build/evaluation scripts: `scripts/README.md`
- Unit/integration tests: `tests/graphpilot_edge`, `scripts/tests`

## Planning and Reports

- Active implementation plans: `docs/plans/`
- Historical paper/report notes: `docs/papers/`, `docs/reports/`

## Typical Workflows

### Verify the current repo state

```bash
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest discover -s tests/graphpilot_edge -v
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest discover -s scripts/tests -v
```

### Rebuild calibrated evidence and paper outputs

```bash
env -u PYTHONHOME -u PYTHONPATH python3 scripts/build_graphpilot_checkpoint.py
env -u PYTHONHOME -u PYTHONPATH python3 scripts/build_graphpilot_cases_figures.py
env -u PYTHONHOME -u PYTHONPATH python3 scripts/build_graphpilot_cases_paper.py
env -u PYTHONHOME -u PYTHONPATH python3 scripts/build_graphpilot_artifact_pack.py
```

### Re-run core experiment surfaces

```bash
env -u PYTHONHOME -u PYTHONPATH python3 scripts/run_graphpilot_experiments.py
env -u PYTHONHOME -u PYTHONPATH python3 scripts/run_graphpilot_characterization.py
env -u PYTHONHOME -u PYTHONPATH python3 scripts/calibrate_graphpilot_cost_model.py
```
