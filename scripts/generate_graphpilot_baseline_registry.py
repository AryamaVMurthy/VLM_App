#!/usr/bin/env python3
"""Generate a machine-readable baseline registry for GraphPilot workloads."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from graphpilot_edge.baselines import build_baseline_candidate  # noqa: E402
from graphpilot_edge.hardware_topology import (  # noqa: E402
    DEFAULT_HARDWARE_TOPOLOGY_PATH,
    load_hardware_topology,
)
from graphpilot_edge.models import RequestSpec  # noqa: E402
from graphpilot_edge.simulation import simulate_request_stream  # noqa: E402
from graphpilot_edge.workload_universe import (  # noqa: E402
    DEFAULT_WORKLOAD_UNIVERSE_PATH,
    load_workload_universe,
)

DEFAULT_OUTPUT = ROOT_DIR / "artifacts" / "graphpilot_edge" / "registries" / "baseline_policy_registry.json"
DEFAULT_BASELINES = (
    "cpu_only",
    "gpu_only",
    "npu_only",
    "current_deployed_plan",
    "stage_greedy",
    "static_best_map",
    "no_pipeline",
    "no_fallback_aware",
    "no_memory_kv",
    "no_knob_tuning",
    "no_thermal_adaptation",
    "band_like",
    "adms_like",
    "puzzle_like",
    "twill_like",
    "heteroinfer_like",
    "agent_xpu_like",
    "hero_like",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-universe", type=Path, default=DEFAULT_WORKLOAD_UNIVERSE_PATH)
    parser.add_argument("--hardware-topology", type=Path, default=DEFAULT_HARDWARE_TOPOLOGY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workloads", nargs="+", required=True)
    parser.add_argument("--baselines", nargs="+", default=list(DEFAULT_BASELINES))
    return parser.parse_args(argv)


def build_registry(
    *,
    workload_ids: list[str],
    baseline_ids: list[str],
    workload_universe_path: Path,
    hardware_topology_path: Path,
) -> dict[str, object]:
    universe = load_workload_universe(workload_universe_path)
    topology = load_hardware_topology(hardware_topology_path)
    capacities: dict[str, int] = {}
    for resource_id in topology.simulator.resource_ids():
        backend = topology.simulator.backend_label(resource_id)
        capacities[backend] = capacities.get(backend, 0) + 1
    workloads = []
    for workload_id in workload_ids:
        spec = universe.specs[workload_id]
        baseline_rows = []
        for baseline_id in baseline_ids:
            try:
                if spec.category == "mixed_criticality_stream":
                    resource_assignments: dict[str, dict[str, str]] = {}
                    requests: list[RequestSpec] = []
                    for group in spec.request_groups:
                        base_workload_id = str(group["base_workload_id"])
                        scenario = universe.build_scenario(base_workload_id)
                        assignment = resource_assignments.get(base_workload_id)
                        if assignment is None:
                            candidate = build_baseline_candidate(
                                scenario,
                                simulator=topology.simulator,
                                baseline_id=baseline_id,
                            )
                            assignment = dict(candidate.resource_assignment)
                            resource_assignments[base_workload_id] = assignment
                        plan = scenario.instantiate_candidate_plan(
                            simulator=topology.simulator,
                            resource_assignment=assignment,
                        )
                        deadline_ms = group.get("deadline_ms", spec.deadline_ms)
                        criticality_class = str(group.get("criticality_class", "default"))
                        for request_index, arrival_ms in enumerate(group.get("arrivals_ms", ())):
                            requests.append(
                                RequestSpec(
                                    request_id=f"{workload_id}:{base_workload_id}:request_{request_index}",
                                    arrival_ms=int(arrival_ms),
                                    workflow=scenario.workflow,
                                    plan=plan,
                                    deadline_ms=int(deadline_ms) if deadline_ms is not None else None,
                                    criticality_class=criticality_class,
                                    source_workload_id=base_workload_id,
                                )
                            )
                    simulation = simulate_request_stream(tuple(requests), capacities)
                    baseline_rows.append(
                        {
                            "baseline_id": baseline_id,
                            "status": "ok",
                            "workflow_id": workload_id,
                            "plan_id": f"{baseline_id}:{workload_id}",
                            "score_ms": simulation.makespan_ms,
                            "resource_assignment": resource_assignments,
                            "stage_backends": {},
                        }
                    )
                    continue
                scenario = universe.build_scenario(workload_id)
                candidate = build_baseline_candidate(
                    scenario,
                    simulator=topology.simulator,
                    baseline_id=baseline_id,
                )
            except ValueError as exc:
                baseline_rows.append(
                    {
                        "baseline_id": baseline_id,
                        "status": "error",
                        "error": str(exc),
                    }
                )
                continue
            baseline_rows.append(
                {
                    "baseline_id": baseline_id,
                    "status": "ok",
                    "workflow_id": candidate.workflow.workflow_id,
                    "plan_id": candidate.plan.plan_id,
                    "score_ms": candidate.score_ms,
                    "resource_assignment": dict(candidate.resource_assignment),
                    "stage_backends": {
                        option.stage_id: option.backend for option in candidate.plan.stage_options
                    },
                }
            )
        workloads.append(
            {
                "workload_id": workload_id,
                "scenario_id": None if spec.category == "mixed_criticality_stream" else universe.build_scenario(workload_id).scenario_id,
                "baselines": baseline_rows,
            }
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_ids": baseline_ids,
        "hardware_topology": str(hardware_topology_path.resolve()),
        "workload_universe": str(workload_universe_path.resolve()),
        "workloads": workloads,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_registry(
        workload_ids=args.workloads,
        baseline_ids=args.baselines,
        workload_universe_path=args.workload_universe,
        hardware_topology_path=args.hardware_topology,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
