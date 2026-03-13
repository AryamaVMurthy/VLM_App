# Scripts

This directory contains the operational entry points for GraphPilot-Edge.

## High-Value Entry Points

### Artifact and paper builders
- `build_graphpilot_checkpoint.py`
- `build_graphpilot_cases_figures.py`
- `build_graphpilot_cases_paper.py`
- `build_graphpilot_artifact_pack.py`

### Experiment and calibration runners
- `run_graphpilot_experiments.py`
- `run_graphpilot_characterization.py`
- `run_graphpilot_stage_feasibility.py`
- `run_graphpilot_stage_profiler.py`
- `run_graphpilot_sustained_load.py`
- `calibrate_graphpilot_cost_model.py`
- `tune_graphpilot_hyperparameters.py`

### Registry generators
- `generate_graphpilot_candidate_plans.py`
- `generate_graphpilot_baseline_registry.py`
- `generate_graphpilot_workload_registry.py`

### Android/device helpers
- `start_root_vlm_daemon_adb.sh`
- `smoke_check_device.sh`
- `install_debug.sh`
- `provision_model_to_app.sh`

## Conventions

- Run Python scripts with `env -u PYTHONHOME -u PYTHONPATH python3 ...`.
- Builders should fail fast if required inputs are missing.
- Generated files belong under `artifacts/graphpilot_edge/`, not under `scripts/`.
