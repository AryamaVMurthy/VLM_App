from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping

from .hardware_simulator import HardwarePrediction, HardwareSimulator, TaskProfile
from .model_graph_simulator import ModelGraphNodeProfile, ModelGraphScenario
from .models import CandidatePlan
from .simulation import simulate_candidate_plan
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

    effective_scenario = scenario

    if baseline_id in {"cpu_only", "gpu_only", "npu_only"}:
        backend = baseline_id.removesuffix("_only")
        assignment = _single_backend_assignment(
            effective_scenario,
            simulator=simulator,
            backend=backend,
            batch_size=batch_size,
            current_temp_c=current_temp_c,
            utilizations=utilizations,
        )
        workflow = effective_scenario.workflow
    elif baseline_id == "current_deployed_plan":
        assignment = _current_deployed_assignment(
            effective_scenario,
            simulator=simulator,
            batch_size=batch_size,
            current_temp_c=current_temp_c,
            utilizations=utilizations,
        )
        workflow = effective_scenario.workflow
    elif baseline_id == "stage_greedy":
        assignment = _stage_greedy_assignment(
            effective_scenario,
            simulator=simulator,
            batch_size=batch_size,
            current_temp_c=current_temp_c,
            utilizations=utilizations,
        )
        workflow = effective_scenario.workflow
    elif baseline_id == "static_best_map":
        assignment = _static_best_map_assignment(
            effective_scenario,
            simulator=simulator,
            batch_size=batch_size,
            current_temp_c=current_temp_c,
            utilizations=utilizations,
        )
        workflow = effective_scenario.workflow
    elif baseline_id == "no_pipeline":
        assignment = _stage_greedy_assignment(
            effective_scenario,
            simulator=simulator,
            batch_size=batch_size,
            current_temp_c=current_temp_c,
            utilizations=utilizations,
        )
        workflow = WorkflowDag(
            workflow_id=f"{effective_scenario.workflow.workflow_id}.no_pipeline",
            stage_ids=effective_scenario.workflow.stage_ids,
            edges=tuple(
                WorkflowEdge(edge.source_stage_id, edge.target_stage_id, "full")
                for edge in effective_scenario.workflow.edges
            ),
            chunk_sizes=effective_scenario.workflow.chunk_sizes,
        )
    elif baseline_id == "no_fallback_aware":
        assignment = _stage_greedy_assignment(
            effective_scenario,
            simulator=simulator,
            batch_size=batch_size,
            current_temp_c=current_temp_c,
            utilizations=utilizations,
            allow_fallback=True,
        )
        workflow = effective_scenario.workflow
    elif baseline_id == "no_memory_kv":
        effective_scenario = _scenario_scaled_for_ablation(
            effective_scenario,
            baseline_id=baseline_id,
        )
        assignment = _static_best_map_assignment(
            effective_scenario,
            simulator=simulator,
            batch_size=batch_size,
            current_temp_c=current_temp_c,
            utilizations=utilizations,
        )
        workflow = effective_scenario.workflow
    elif baseline_id == "no_knob_tuning":
        effective_scenario = _scenario_scaled_for_ablation(
            effective_scenario,
            baseline_id=baseline_id,
        )
        assignment = _static_best_map_assignment(
            effective_scenario,
            simulator=simulator,
            batch_size=batch_size,
            current_temp_c=current_temp_c,
            utilizations=utilizations,
        )
        workflow = effective_scenario.workflow
    elif baseline_id == "no_thermal_adaptation":
        hot_state = _hot_current_temperatures(simulator)
        assignment = _static_best_map_assignment(
            effective_scenario,
            simulator=simulator,
            batch_size=batch_size,
            current_temp_c=current_temp_c,
            utilizations=utilizations,
        )
        current_temp_c = hot_state
        workflow = effective_scenario.workflow
    else:
        raise ValueError(f"Unsupported baseline_id '{baseline_id}'.")

    plan = effective_scenario.instantiate_candidate_plan(
        simulator=simulator,
        resource_assignment=assignment,
        batch_size=batch_size,
        current_temp_c=current_temp_c,
        utilizations=utilizations,
    )
    score_ms = _assignment_score(
        workflow=workflow,
        plan=plan,
        simulator=simulator,
    )
    return BaselineCandidate(
        baseline_id=baseline_id,
        workflow=workflow,
        plan=plan,
        resource_assignment=assignment,
        score_ms=score_ms,
    )


def _hot_current_temperatures(simulator: HardwareSimulator) -> dict[str, float]:
    return {
        resource_id: simulator.resource(resource_id).thermal_threshold_c + 6.0
        for resource_id in simulator.resource_ids()
    }


def _scenario_scaled_for_ablation(
    scenario: ModelGraphScenario,
    *,
    baseline_id: str,
) -> ModelGraphScenario:
    family_scale = {
        "no_knob_tuning": {
            "llm": (1.18, 1.12),
            "vlm": (1.25, 1.2),
            "stt": (1.12, 1.08),
            "tts": (1.08, 1.05),
            "retrieval": (1.16, 1.1),
            "cnn": (1.08, 1.04),
            "vit": (1.12, 1.06),
        },
        "no_memory_kv": {
            "llm": (1.22, 1.35),
            "vlm": (1.14, 1.28),
            "retrieval": (1.05, 1.08),
            "tts": (1.02, 1.04),
        },
    }.get(baseline_id)
    if family_scale is None:
        raise ValueError(f"Unsupported ablation scaling baseline '{baseline_id}'.")

    node_profiles: dict[str, ModelGraphNodeProfile] = {}
    for stage_id, node in scenario.node_profiles.items():
        op_scale, memory_scale = family_scale.get(node.family, (1.0, 1.0))
        node_profiles[stage_id] = ModelGraphNodeProfile(
            stage_id=node.stage_id,
            family=node.family,
            variant_id=f"{node.variant_id}:{baseline_id}",
            task_profile=_scaled_task_profile(
                node.task_profile,
                op_scale=op_scale,
                memory_scale=memory_scale,
            ),
            knob_values=dict(node.knob_values),
            quality_loss=0.0 if baseline_id == "no_knob_tuning" else node.quality_loss,
            ttft_hint_ms=node.ttft_hint_ms,
            tts_first_audio_hint_ms=node.tts_first_audio_hint_ms,
        )
    return ModelGraphScenario(
        scenario_id=f"{scenario.scenario_id}.{baseline_id}",
        workflow=scenario.workflow,
        node_profiles=node_profiles,
    )


def _scaled_task_profile(task: TaskProfile, *, op_scale: float, memory_scale: float) -> TaskProfile:
    return TaskProfile(
        task_id=task.task_id,
        op_volume={op_class: volume * op_scale for op_class, volume in task.op_volume.items()},
        memory_bytes=max(1, int(round(task.memory_bytes * memory_scale))),
        input_bytes=max(1, int(round(task.input_bytes * op_scale))),
        output_bytes=max(1, int(round(task.output_bytes * memory_scale))),
        fallback_partitions=task.fallback_partitions,
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
            prediction = simulator.predict_task(
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
        if prediction.used_fallback:
            raise ValueError(
                f"No support-safe resource for stage '{stage_id}' on backend '{backend}': predicted execution requires explicit fallback partitions."
            )
        assignment[stage_id] = resource_id
    return assignment


def _current_deployed_assignment(
    scenario: ModelGraphScenario,
    *,
    simulator: HardwareSimulator,
    batch_size: int,
    current_temp_c: Mapping[str, float],
    utilizations: Mapping[str, float],
) -> dict[str, str]:
    supported_assistant_stages = {"asr.primary", "planner.primary", "responder.primary", "tts.primary", "vlm.fastvlm.primary", "retrieval.embedder.primary"}
    if not any(stage_id in supported_assistant_stages for stage_id in scenario.workflow.stage_ids):
        raise ValueError(
            "current_deployed_plan is only defined for assistant-style GraphPilot workflows."
        )
    cpu_resources = simulator.resources_for_backend("cpu")
    npu_resources = simulator.resources_for_backend("npu")
    if not cpu_resources:
        raise ValueError("No CPU resource exists for current_deployed_plan.")
    cpu_resource_id = cpu_resources[0]
    npu_resource_id = npu_resources[0] if npu_resources else None
    assignment: dict[str, str] = {}
    for stage_id in scenario.workflow.topological_order():
        resource_id = cpu_resource_id
        if stage_id == "vlm.fastvlm.primary":
            if npu_resource_id is None:
                raise ValueError("No NPU resource exists for current_deployed_plan VLM stage.")
            resource_id = npu_resource_id
        node = scenario.node_profiles[stage_id]
        try:
            prediction = simulator.predict_task(
                task=node.task_profile,
                resource_id=resource_id,
                batch_size=batch_size,
                current_temp_c=current_temp_c,
                utilizations=utilizations,
            )
        except ValueError as exc:
            raise ValueError(
                f"current_deployed_plan has no support-safe resource for stage '{stage_id}' on '{resource_id}': {exc}"
            ) from exc
        if prediction.used_fallback:
            raise ValueError(
                f"current_deployed_plan has no support-safe resource for stage '{stage_id}' on '{resource_id}': predicted execution requires explicit fallback partitions."
            )
        assignment[stage_id] = resource_id
    return assignment


def _stage_greedy_assignment(
    scenario: ModelGraphScenario,
    *,
    simulator: HardwareSimulator,
    batch_size: int,
    current_temp_c: Mapping[str, float],
    utilizations: Mapping[str, float],
    allow_fallback: bool = False,
) -> dict[str, str]:
    assignment: dict[str, str] = {}
    for stage_id in scenario.workflow.topological_order():
        assignment[stage_id] = _best_resource_for_stage(
            scenario.node_profiles[stage_id],
            simulator=simulator,
            batch_size=batch_size,
            current_temp_c=current_temp_c,
            utilizations=utilizations,
            allow_fallback=allow_fallback,
        )[0]
    return assignment


def _static_best_map_assignment(
    scenario: ModelGraphScenario,
    *,
    simulator: HardwareSimulator,
    batch_size: int,
    current_temp_c: Mapping[str, float],
    utilizations: Mapping[str, float],
    allow_fallback: bool = False,
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
                allow_fallback=allow_fallback,
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
        plan = scenario.instantiate_candidate_plan(
            simulator=simulator,
            resource_assignment=candidate,
            batch_size=batch_size,
            current_temp_c=current_temp_c,
            utilizations=utilizations,
        )
        score = _assignment_score(
            workflow=scenario.workflow,
            plan=plan,
            simulator=simulator,
        )
        if best_score is None or score < best_score:
            best_assignment = candidate
            best_score = score
    if best_assignment is None:
        raise ValueError(f"Unable to find a support-safe assignment for '{scenario.scenario_id}'.")
    return best_assignment


def _assignment_score(
    *,
    workflow: WorkflowDag,
    plan: CandidatePlan,
    simulator: HardwareSimulator,
) -> float:
    capacities = {
        backend: len(simulator.resources_for_backend(backend))
        for backend in sorted({simulator.backend_label(resource_id) for resource_id in simulator.resource_ids()})
    }
    simulation = simulate_candidate_plan(workflow, plan, capacities)
    return float(simulation.makespan_ms)


def _best_resource_for_stage(
    node,
    *,
    simulator: HardwareSimulator,
    batch_size: int,
    current_temp_c: Mapping[str, float],
    utilizations: Mapping[str, float],
    allow_fallback: bool = False,
) -> tuple[str, HardwarePrediction]:
    feasible = _feasible_resources_for_stage(
        node,
        simulator=simulator,
        batch_size=batch_size,
        current_temp_c=current_temp_c,
        utilizations=utilizations,
        allow_fallback=allow_fallback,
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
    allow_fallback: bool = False,
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
        if prediction.used_fallback and not allow_fallback:
            continue
        feasible.append((resource_id, prediction))
    return feasible


__all__ = [
    "BaselineCandidate",
    "build_baseline_candidate",
]
