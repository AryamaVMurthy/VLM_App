from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .cost_model import ObjectiveWeights, SimulationCalibration
from .catalog import resolve_feasible_backends
from .enumeration import enumerate_candidate_plans
from .models import CandidatePlan, StageOption
from .profiles import StageProfile
from .simulation import simulate_candidate_plan
from .workflow import WorkflowDag


@dataclass(frozen=True)
class RankedPlan:
    plan: CandidatePlan
    score_makespan_ms: int
    objective_score: float | None = None
    simulation_result: object | None = None


def build_stage_options(
    workflow: WorkflowDag,
    backend_matrix: dict[str, object],
    profile_index: Mapping[tuple[str, str], StageProfile],
) -> dict[str, tuple[StageOption, ...]]:
    options: dict[str, tuple[StageOption, ...]] = {}
    for stage_id in workflow.stage_ids:
        feasible_backends = resolve_feasible_backends(backend_matrix, stage_id)
        stage_options = []
        for backend in feasible_backends:
            profile = profile_index.get((stage_id, backend))
            if profile is None or profile.warm_latency_ms is None:
                continue
            memory_mb = int((profile.peak_memory_bytes or 0) / (1024 * 1024))
            stage_options.append(
                StageOption(
                    stage_id=stage_id,
                    variant_id=profile.variant,
                    backend=backend,
                    latency_ms=profile.warm_latency_ms,
                    memory_mb=memory_mb,
                    output_bytes=profile.output_bytes or 0,
                    average_power_mw=profile.average_power_mw or 0.0,
                    quality_loss=profile.quality_loss or 0.0,
                    compile_cost_ms=profile.compile_cost_ms or 0,
                    ttft_ms=profile.ttft_ms,
                    tts_first_audio_ms=profile.tts_first_audio_ms,
                )
            )
        options[stage_id] = tuple(sorted(stage_options, key=lambda opt: (opt.backend, opt.variant_id)))
    return options


def enumerate_and_rank_plans(
    workflow: WorkflowDag,
    stage_options: dict[str, tuple[StageOption, ...]],
    resource_capacities: dict[str, int],
    *,
    objective_weights: ObjectiveWeights | None = None,
    bandwidth_bytes_per_ms: float = 1024.0 * 1024.0,
    transfer_fixed_overhead_ms: float = 0.0,
    transfer_layout_ms: float = 0.0,
    simulation_calibration: SimulationCalibration | None = None,
) -> list[RankedPlan]:
    ranked = []
    for plan in enumerate_candidate_plans(workflow, stage_options):
        result = simulate_candidate_plan(
            workflow,
            plan,
            resource_capacities,
            bandwidth_bytes_per_ms=bandwidth_bytes_per_ms,
            transfer_fixed_overhead_ms=transfer_fixed_overhead_ms,
            transfer_layout_ms=transfer_layout_ms,
            objective_weights=objective_weights,
            calibration=simulation_calibration,
        )
        ranked.append(
            RankedPlan(
                plan=plan,
                score_makespan_ms=result.makespan_ms,
                objective_score=result.objective_score,
                simulation_result=result,
            )
        )
    return sorted(
        ranked,
        key=lambda item: (
            item.objective_score if item.objective_score is not None else item.score_makespan_ms,
            item.score_makespan_ms,
            item.plan.total_memory_mb,
            item.plan.plan_id,
        ),
    )
