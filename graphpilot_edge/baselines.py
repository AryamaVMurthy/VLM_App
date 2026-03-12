from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping

from .hardware_simulator import HardwarePrediction, HardwareSimulator
from .model_graph_simulator import ModelGraphScenario
from .models import CandidatePlan
from .workflow import WorkflowDag, WorkflowEdge


@dataclass(frozen=True)
class BaselineCandidate:
    baseline_id: str
    workflow: WorkflowDag
    plan: CandidatePlan
    resource_assignment: Mapping[str, str]
    score_ms: float


def build_baseline_candidate(
    scenario: ModelGraphScenario,
    *,
    simulator: HardwareSimulator,
    baseline_id: str,
    batch_size: int = 1,
    current_temp_c: Mapping[str, float] | None = None,
    utilizations: Mapping[str, float] | None = None,
) -> BaselineCandidate:
    current_temp_c = current_temp_c or {}
    utilizations = utilizations or {}
    baseline_id = baseline_id.lower()

    if baseline_id in {"cpu_only", "gpu_only", "npu_only"}:
        backend = baseline_id.removesuffix("_only")
        assignment = _single_backend_assignment(
            scenario,
            simulator=simulator,
            backend=backend,
            batch_size=batch_size,
            current_temp_c=current_temp_c,
            utilizations=utilizations,
        )
        workflow = scenario.workflow
    elif baseline_id == "stage_greedy":
        assignment = _stage_greedy_assignment(
            scenario,
            simulator=simulator,
            batch_size=batch_size,
            current_temp_c=current_temp_c,
            utilizations=utilizations,
        )
        workflow = scenario.workflow
    elif baseline_id == "static_best_map":
        assignment = _static_best_map_assignment(
            scenario,
            simulator=simulator,
            batch_size=batch_size,
            current_temp_c=current_temp_c,
            utilizations=utilizations,
        )
        workflow = scenario.workflow
    elif baseline_id == "no_pipeline":
        assignment = _stage_greedy_assignment(
            scenario,
            simulator=simulator,
            batch_size=batch_size,
            current_temp_c=current_temp_c,
            utilizations=utilizations,
        )
        workflow = WorkflowDag(
            workflow_id=f"{scenario.workflow.workflow_id}.no_pipeline",
            stage_ids=scenario.workflow.stage_ids,
            edges=tuple(
                WorkflowEdge(edge.source_stage_id, edge.target_stage_id, "full")
                for edge in scenario.workflow.edges
            ),
            chunk_sizes=scenario.workflow.chunk_sizes,
        )
    else:
        raise ValueError(f"Unsupported baseline_id '{baseline_id}'.")

    plan = scenario.instantiate_candidate_plan(
        simulator=simulator,
        resource_assignment=assignment,
        batch_size=batch_size,
        current_temp_c=current_temp_c,
        utilizations=utilizations,
    )
    score_ms = _assignment_score(
        scenario,
        simulator=simulator,
        resource_assignment=assignment,
        batch_size=batch_size,
        current_temp_c=current_temp_c,
        utilizations=utilizations,
    )
    return BaselineCandidate(
        baseline_id=baseline_id,
        workflow=workflow,
        plan=plan,
        resource_assignment=assignment,
        score_ms=score_ms,
    )


def _single_backend_assignment(
    scenario: ModelGraphScenario,
    *,
    simulator: HardwareSimulator,
    backend: str,
    batch_size: int,
    current_temp_c: Mapping[str, float],
    utilizations: Mapping[str, float],
) -> dict[str, str]:
    resources = simulator.resources_for_backend(backend)
    if not resources:
        raise ValueError(f"No resources exist for backend '{backend}'.")
    resource_id = resources[0]
    assignment: dict[str, str] = {}
    for stage_id in scenario.workflow.topological_order():
        node = scenario.node_profiles[stage_id]
        try:
            simulator.predict_task(
                task=node.task_profile,
                resource_id=resource_id,
                batch_size=batch_size,
                current_temp_c=current_temp_c,
                utilizations=utilizations,
            )
        except ValueError as exc:
            raise ValueError(
                f"No support-safe resource for stage '{stage_id}' on backend '{backend}': {exc}"
            ) from exc
        assignment[stage_id] = resource_id
    return assignment


def _stage_greedy_assignment(
    scenario: ModelGraphScenario,
    *,
    simulator: HardwareSimulator,
    batch_size: int,
    current_temp_c: Mapping[str, float],
    utilizations: Mapping[str, float],
) -> dict[str, str]:
    assignment: dict[str, str] = {}
    for stage_id in scenario.workflow.topological_order():
        assignment[stage_id] = _best_resource_for_stage(
            scenario.node_profiles[stage_id],
            simulator=simulator,
            batch_size=batch_size,
            current_temp_c=current_temp_c,
            utilizations=utilizations,
        )[0]
    return assignment


def _static_best_map_assignment(
    scenario: ModelGraphScenario,
    *,
    simulator: HardwareSimulator,
    batch_size: int,
    current_temp_c: Mapping[str, float],
    utilizations: Mapping[str, float],
) -> dict[str, str]:
    stage_order = scenario.workflow.topological_order()
    feasible_resource_lists = [
        tuple(
            resource_id
            for resource_id, _ in _feasible_resources_for_stage(
                scenario.node_profiles[stage_id],
                simulator=simulator,
                batch_size=batch_size,
                current_temp_c=current_temp_c,
                utilizations=utilizations,
            )
        )
        for stage_id in stage_order
    ]
    if any(not resources for resources in feasible_resource_lists):
        raise ValueError(
            f"Scenario '{scenario.scenario_id}' has a stage without a support-safe resource."
        )
    best_assignment: dict[str, str] | None = None
    best_score: float | None = None
    for resource_tuple in product(*feasible_resource_lists):
        candidate = dict(zip(stage_order, resource_tuple, strict=True))
        score = _assignment_score(
            scenario,
            simulator=simulator,
            resource_assignment=candidate,
            batch_size=batch_size,
            current_temp_c=current_temp_c,
            utilizations=utilizations,
        )
        if best_score is None or score < best_score:
            best_assignment = candidate
            best_score = score
    if best_assignment is None:
        raise ValueError(f"Unable to find a support-safe assignment for '{scenario.scenario_id}'.")
    return best_assignment


def _assignment_score(
    scenario: ModelGraphScenario,
    *,
    simulator: HardwareSimulator,
    resource_assignment: Mapping[str, str],
    batch_size: int,
    current_temp_c: Mapping[str, float],
    utilizations: Mapping[str, float],
) -> float:
    predictions: dict[str, HardwarePrediction] = {}
    total_ms = 0.0
    for stage_id in scenario.workflow.topological_order():
        node = scenario.node_profiles[stage_id]
        prediction = simulator.predict_task(
            task=node.task_profile,
            resource_id=resource_assignment[stage_id],
            batch_size=batch_size,
            current_temp_c=current_temp_c,
            utilizations=utilizations,
        )
        predictions[stage_id] = prediction
        total_ms += prediction.execution_time_ms
    for edge in scenario.workflow.edges:
        src_id = resource_assignment[edge.source_stage_id]
        dst_id = resource_assignment[edge.target_stage_id]
        if src_id == dst_id:
            continue
        total_ms += simulator.compute_transfer_time_ms(
            size_bytes=scenario.node_profiles[edge.source_stage_id].task_profile.output_bytes,
            src_resource_id=src_id,
            dst_resource_id=dst_id,
        )
    return total_ms


def _best_resource_for_stage(
    node,
    *,
    simulator: HardwareSimulator,
    batch_size: int,
    current_temp_c: Mapping[str, float],
    utilizations: Mapping[str, float],
) -> tuple[str, HardwarePrediction]:
    feasible = _feasible_resources_for_stage(
        node,
        simulator=simulator,
        batch_size=batch_size,
        current_temp_c=current_temp_c,
        utilizations=utilizations,
    )
    if not feasible:
        raise ValueError(f"Stage '{node.stage_id}' has no support-safe resource.")
    return min(
        feasible,
        key=lambda item: (
            item[1].execution_time_ms,
            simulator.backend_label(item[0]),
            item[0],
        ),
    )


def _feasible_resources_for_stage(
    node,
    *,
    simulator: HardwareSimulator,
    batch_size: int,
    current_temp_c: Mapping[str, float],
    utilizations: Mapping[str, float],
) -> list[tuple[str, HardwarePrediction]]:
    feasible: list[tuple[str, HardwarePrediction]] = []
    for resource_id in simulator.resource_ids():
        try:
            prediction = simulator.predict_task(
                task=node.task_profile,
                resource_id=resource_id,
                batch_size=batch_size,
                current_temp_c=current_temp_c,
                utilizations=utilizations,
            )
        except ValueError:
            continue
        feasible.append((resource_id, prediction))
    return feasible


__all__ = [
    "BaselineCandidate",
    "build_baseline_candidate",
]
