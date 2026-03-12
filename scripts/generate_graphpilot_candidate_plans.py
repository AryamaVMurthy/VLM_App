#!/usr/bin/env python3
"""Generate GraphPilot candidate plans from current registries and configs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from graphpilot_edge.catalog import load_stage_catalog, load_workflow_catalog, resolve_feasible_backends
from graphpilot_edge.cost_model import ObjectiveWeights, SimulationCalibration
from graphpilot_edge.errors import ValidationError
from graphpilot_edge.models import RequestSpec
from graphpilot_edge.plan_bank import load_plan_bank, select_plan_for_state
from graphpilot_edge.planner import build_stage_options, enumerate_and_rank_plans
from graphpilot_edge.profiles import load_profile_index
from graphpilot_edge.simulation import simulate_request_stream

CONFIG_DIR = ROOT_DIR / "configs" / "graphpilot_edge"
ARTIFACT_DIR = ROOT_DIR / "artifacts" / "graphpilot_edge"
ANALYSIS_DIR = ARTIFACT_DIR / "analysis"
BACKEND_MATRIX_PATH = ARTIFACT_DIR / "registries" / "backend_feasibility_matrix.json"
PROFILER_REGISTRY_PATH = ARTIFACT_DIR / "registries" / "profiler_registry.json"
CANDIDATE_REGISTRY_PATH = ARTIFACT_DIR / "registries" / "candidate_plan_registry.json"
WORKFLOW_PATH = CONFIG_DIR / "workflow_templates.json"
STAGE_PATH = CONFIG_DIR / "stage_catalog.json"
PLAN_BANK_PATH = CONFIG_DIR / "plan_bank_templates.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_latest_analysis_summary(root: Path, prefix: str) -> Path | None:
    candidates = sorted(root.glob(f"{prefix}_*/summary.json"))
    if not candidates:
        return None
    return candidates[-1]


def select_workflow_ids(workflows: dict[str, object], backend_matrix: dict[str, Any]) -> list[str]:
    try:
        retrieval_backends = resolve_feasible_backends(backend_matrix, "retrieval.embedder.primary")
    except KeyError:
        retrieval_backends = ()
    selected = []
    for workflow_id in sorted(workflows):
        if "retrieval" in workflow_id and not retrieval_backends:
            continue
        selected.append(workflow_id)
    return selected


def resolve_workflow_chunk_sizes(workflow_id: str, workflow: Any) -> dict[str, int]:
    chunk_sizes = {stage_id: size for stage_id, size in getattr(workflow, "chunk_sizes", ())}
    for edge in workflow.edges:
        if edge.stream_mode != "chunk":
            continue
        if edge.source_stage_id not in chunk_sizes:
            raise ValidationError(
                f"Workflow '{workflow_id}' declares chunk stream edge "
                f"'{edge.source_stage_id}->{edge.target_stage_id}' without chunk_sizes['{edge.source_stage_id}']."
            )
    return chunk_sizes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-path", type=Path, default=WORKFLOW_PATH)
    parser.add_argument("--stage-path", type=Path, default=STAGE_PATH)
    parser.add_argument("--plan-bank-path", type=Path, default=PLAN_BANK_PATH)
    parser.add_argument("--backend-matrix", type=Path, default=BACKEND_MATRIX_PATH)
    parser.add_argument("--profiler-registry", type=Path, default=PROFILER_REGISTRY_PATH)
    parser.add_argument("--candidate-registry", type=Path, default=CANDIDATE_REGISTRY_PATH)
    parser.add_argument("--optimization-config", type=Path, default=CONFIG_DIR / "optimization_defaults.json")
    parser.add_argument("--tuning-summary", type=Path, default=None)
    parser.add_argument("--calibration-summary", type=Path, default=None)
    return parser.parse_args(argv)


def resolve_objective_weights(
    tuning_summary: dict[str, Any] | None,
    optimization_config: dict[str, Any] | None = None,
) -> ObjectiveWeights:
    tuned = ((tuning_summary or {}).get("best_objective_weights") or {}).get("weights")
    if tuned:
        return ObjectiveWeights(**tuned)
    if optimization_config:
        grid = optimization_config.get("objective_weight_grid") or {}
        defaults: dict[str, float] = {}
        for key in ("alpha", "beta", "gamma", "delta", "eta", "zeta", "xi", "psi"):
            values = grid.get(key)
            if not values:
                raise ValueError(
                    f"Optimization config is missing objective weight '{key}'. "
                    "Remediation: provide configs/graphpilot_edge/optimization_defaults.json with all objective weights."
                )
            defaults[key] = float(values[0])
        return ObjectiveWeights(**defaults)
    return ObjectiveWeights(alpha=1.0, beta=1.0, gamma=0.0, delta=0.0, eta=0.0, zeta=0.0, xi=1.0, psi=1.0)


def resolve_simulation_calibration(
    workflow_id: str,
    calibration_summary: dict[str, Any] | None,
) -> SimulationCalibration | None:
    if not calibration_summary:
        return None
    workflow_thermal = (
        calibration_summary.get("thermal_scale_by_workflow", {}).get(workflow_id, {})
    )
    return SimulationCalibration(
        orchestration_overhead_ms=float(
            calibration_summary.get("global_orchestration_overhead_ms", 0.0)
        ),
        workflow_thermal_scale=float(workflow_thermal.get("warm_latency_scale") or 1.0),
    )


def resolve_stream_requests(
    workflow_id: str,
    workflow: object,
    plan: object,
    optimization_config: dict[str, Any] | None,
) -> tuple[RequestSpec, ...]:
    stream_workloads = (optimization_config or {}).get("stream_workloads") or {}
    config = stream_workloads.get(workflow_id)
    if not config:
        return ()
    arrivals = config.get("arrivals_ms")
    if not arrivals:
        raise ValueError(
            f"Stream workload for '{workflow_id}' is missing arrivals_ms. "
            "Remediation: provide an explicit arrivals_ms list in optimization_defaults.json."
        )
    deadline_ms = config.get("deadline_ms")
    return tuple(
        RequestSpec(
            request_id=f"{workflow_id}:req{index}",
            arrival_ms=int(arrival_ms),
            workflow=workflow,
            plan=plan,
            deadline_ms=int(deadline_ms) if deadline_ms is not None else None,
        )
        for index, arrival_ms in enumerate(arrivals)
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workflows = load_workflow_catalog(args.workflow_path)
    _stages = load_stage_catalog(args.stage_path)
    backend_matrix = load_json(args.backend_matrix)
    profiler_registry = load_json(args.profiler_registry)
    profile_index = load_profile_index(profiler_registry)
    bank = load_plan_bank(args.plan_bank_path)
    registry = load_json(args.candidate_registry)
    optimization_config = (
        load_json(args.optimization_config) if args.optimization_config is not None else None
    )
    tuning_summary_path = args.tuning_summary or resolve_latest_analysis_summary(
        ANALYSIS_DIR, "graphpilot_hparam_tuning"
    )
    calibration_summary_path = args.calibration_summary or resolve_latest_analysis_summary(
        ANALYSIS_DIR, "graphpilot_cost_calibration"
    )
    tuning_summary = load_json(tuning_summary_path) if tuning_summary_path else None
    calibration_summary = load_json(calibration_summary_path) if calibration_summary_path else None
    objective_weights = resolve_objective_weights(tuning_summary, optimization_config)

    selected_workflow_ids = select_workflow_ids(workflows, backend_matrix)
    plan_entries = []
    for workflow_id in selected_workflow_ids:
        workflow = workflows[workflow_id]
        stage_options = build_stage_options(workflow, backend_matrix, profile_index)
        if any(not options for options in stage_options.values()):
            continue
        ranked = enumerate_and_rank_plans(
            workflow,
            stage_options,
            resource_capacities={"cpu": 1, "npu": 1, "gpu": 1},
            objective_weights=objective_weights,
            simulation_calibration=resolve_simulation_calibration(workflow_id, calibration_summary),
        )
        if not ranked:
            continue
        for state_id in bank.state_ids:
            selected = select_plan_for_state(bank, state_id, [item.plan for item in ranked])
            selected_ranked = next(item for item in ranked if item.plan.plan_id == selected.plan_id)
            simulation_result = selected_ranked.simulation_result
            predicted_cost: dict[str, Any] = {
                "makespan_ms": selected_ranked.score_makespan_ms,
                "memory_mb": selected.total_memory_mb,
                "latency_sum_ms": selected.total_latency_ms,
            }
            if simulation_result is not None:
                predicted_cost.update(
                    {
                        "peak_memory_bytes": simulation_result.peak_memory_bytes,
                        "copy_bytes": simulation_result.copy_bytes,
                        "copy_time_ms": simulation_result.copy_time_ms,
                        "energy_mj": simulation_result.energy_mj,
                        "avg_energy_mj": simulation_result.avg_energy_mj,
                        "ttft_ms": simulation_result.ttft_ms,
                        "ttfs_ms": simulation_result.ttfs_ms,
                        "p95_e2e_ms": simulation_result.p95_e2e_ms,
                        "p95_ttfs_ms": simulation_result.p95_ttfs_ms,
                        "quality_loss": simulation_result.quality_loss,
                        "objective_score": simulation_result.objective_score,
                    }
                )
            stream_requests = resolve_stream_requests(
                workflow_id,
                workflow,
                selected,
                optimization_config,
            )
            if stream_requests:
                stream_result = simulate_request_stream(
                    stream_requests,
                    resource_capacities={"cpu": 1, "npu": 1, "gpu": 1},
                    objective_weights=objective_weights,
                    calibration=resolve_simulation_calibration(workflow_id, calibration_summary),
                )
                predicted_cost.update(
                    {
                        "stream_makespan_ms": stream_result.makespan_ms,
                        "p95_e2e_ms": stream_result.p95_e2e_ms,
                        "p95_ttfs_ms": stream_result.p95_ttfs_ms,
                        "avg_energy_mj": stream_result.avg_energy_mj,
                        "p95_queue_delay_ms": stream_result.p95_queue_delay_ms,
                        "deadline_miss_rate": stream_result.deadline_miss_rate,
                    }
                )
            plan_entries.append(
                {
                    "plan_id": f"{state_id}:{selected.plan_id}",
                    "workflow_template": workflow_id,
                    "state_id": state_id,
                    "stage_variant_map": {
                        option.stage_id: option.variant_id for option in selected.stage_options
                    },
                    "backend_map": {
                        option.stage_id: option.backend for option in selected.stage_options
                    },
                    "macro_region_map": {},
                    "stream_edges": [
                        {
                            "from": edge.source_stage_id,
                            "to": edge.target_stage_id,
                            "mode": edge.stream_mode,
                        }
                        for edge in workflow.edges
                    ],
                    "chunk_sizes": resolve_workflow_chunk_sizes(workflow_id, workflow),
                    "kv_policy": "sticky_decode",
                    "buffer_policy": "interval_reuse_best_fit",
                    "degradation_policy": "memory_guardrail_v1",
                    "predicted_cost": predicted_cost,
                }
            )

    registry["plans"] = plan_entries
    registry["objective_weights"] = {
        "alpha": objective_weights.alpha,
        "beta": objective_weights.beta,
        "gamma": objective_weights.gamma,
        "delta": objective_weights.delta,
        "eta": objective_weights.eta,
        "zeta": objective_weights.zeta,
        "xi": objective_weights.xi,
        "psi": objective_weights.psi,
    }
    registry["objective_weights_source"] = (
        str(tuning_summary_path.resolve())
        if tuning_summary_path
        else str(args.optimization_config.resolve()) if args.optimization_config else "builtin_defaults"
    )
    registry["calibration_summary"] = (
        str(calibration_summary_path.resolve()) if calibration_summary_path else None
    )
    registry["last_updated"] = datetime.now(timezone.utc).date().isoformat()
    write_json(args.candidate_registry, registry)
    print(args.candidate_registry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
