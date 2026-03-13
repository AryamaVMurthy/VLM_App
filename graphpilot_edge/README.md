# graphpilot_edge Python Package

This package contains the offline GraphPilot-Edge logic.

## Main Modules

- `cost_model.py`
  - objective scoring and calibrated cost terms
- `hardware_simulator.py`
  - multi-resource backend simulation primitives
- `model_graph_simulator.py`
  - workload-family and DAG-level scenario modeling
- `planner.py`, `enumeration.py`, `macro_regions.py`
  - candidate generation and planning logic
- `scheduler.py`, `plan_bank.py`
  - scheduling and policy-selection logic
- `memory.py`, `kv.py`
  - memory and KV accounting models
- `workflow.py`, `workload_universe.py`, `catalog.py`
  - workload and workflow definitions
- `baselines.py`
  - fair internal and proxy baseline constructions

## Verification

```bash
env -u PYTHONHOME -u PYTHONPATH python3 -m unittest discover -s tests/graphpilot_edge -v
```

## Design Rule

This package models explicit support-safe behavior. Unsupported ops, fallback partitions, and transfer costs must remain visible in code and outputs.
