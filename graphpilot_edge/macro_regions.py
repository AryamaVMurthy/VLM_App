from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .catalog import StageDefinition
from .models import CandidatePlan, StageOption
from .workflow import WorkflowDag, WorkflowEdge


def macro_region_stage_id(stage_id: str, region_id: str) -> str:
    return f"{stage_id}::{region_id}"


def expand_workflow_with_macro_regions(
    workflow: WorkflowDag,
    stage_catalog: Mapping[str, StageDefinition],
    opened_stage_ids: Iterable[str],
) -> WorkflowDag:
    opened = set(opened_stage_ids)
    expanded_stage_ids: list[str] = []
    replacement_map: dict[str, tuple[str, ...]] = {}

    for stage_id in workflow.stage_ids:
        definition = stage_catalog.get(stage_id)
        macro_regions = definition.macro_regions if definition is not None else ()
        if stage_id in opened and macro_regions:
            region_nodes = tuple(
                macro_region_stage_id(stage_id, region_id) for region_id in macro_regions
            )
            replacement_map[stage_id] = region_nodes
            expanded_stage_ids.extend(region_nodes)
        else:
            replacement_map[stage_id] = (stage_id,)
            expanded_stage_ids.append(stage_id)

    expanded_edges: list[WorkflowEdge] = []
    for stage_id in workflow.stage_ids:
        nodes = replacement_map[stage_id]
        if len(nodes) > 1:
            for src, dst in zip(nodes, nodes[1:]):
                expanded_edges.append(WorkflowEdge(src, dst, "full"))

    for edge in workflow.edges:
        source_nodes = replacement_map[edge.source_stage_id]
        target_nodes = replacement_map[edge.target_stage_id]
        expanded_edges.append(
            WorkflowEdge(source_nodes[-1], target_nodes[0], edge.stream_mode)
        )

    return WorkflowDag(
        workflow_id=f"{workflow.workflow_id}::macro",
        stage_ids=tuple(expanded_stage_ids),
        edges=tuple(expanded_edges),
    )


@dataclass(frozen=True)
class BeamSearchStats:
    expanded_partials: int
    completed_plans: int
    pruned_by_beam: int
    pruned_by_dominance: int


@dataclass(frozen=True)
class RankedMacroPlan:
    plan: CandidatePlan
    latency_lb_ms: int
    memory_lb_mb: int
    energy_lb_mj: float
    copy_lb_bytes: int
    quality_risk: float


@dataclass(frozen=True)
class _PartialPlan:
    assigned_options: tuple[StageOption, ...]
    next_index: int
    latency_committed_ms: int
    memory_committed_mb: int
    energy_committed_mj: float
    copy_committed_bytes: int
    quality_risk: float
    latency_lb_ms: int
    memory_lb_mb: int
    energy_lb_mj: float
    copy_lb_bytes: int


def _optimistic_remaining_sum(
    order: tuple[str, ...],
    stage_options: Mapping[str, tuple[StageOption, ...]],
    next_index: int,
    selector,
) -> float:
    total = 0.0
    for stage_id in order[next_index:]:
        total += min(selector(option) for option in stage_options[stage_id])
    return total


def _build_partial(
    workflow: WorkflowDag,
    order: tuple[str, ...],
    stage_options: Mapping[str, tuple[StageOption, ...]],
    partial: _PartialPlan | None,
    option: StageOption | None,
) -> _PartialPlan:
    assigned = partial.assigned_options if partial is not None else ()
    next_index = partial.next_index if partial is not None else 0
    latency_committed = partial.latency_committed_ms if partial is not None else 0
    memory_committed = partial.memory_committed_mb if partial is not None else 0
    energy_committed = partial.energy_committed_mj if partial is not None else 0.0
    copy_committed = partial.copy_committed_bytes if partial is not None else 0
    quality_risk = partial.quality_risk if partial is not None else 0.0

    if option is not None:
        assigned = assigned + (option,)
        latency_committed += option.latency_ms
        memory_committed += option.memory_mb
        energy_committed += (option.average_power_mw * option.latency_ms) / 1000.0
        quality_risk += option.quality_loss
        if len(assigned) > 1:
            prev = assigned[-2]
            if prev.backend != option.backend:
                copy_committed += prev.output_bytes
        next_index += 1

    latency_lb = int(
        round(
            latency_committed
            + _optimistic_remaining_sum(
                order, stage_options, next_index, lambda candidate: candidate.latency_ms
            )
        )
    )
    memory_lb = int(
        round(
            memory_committed
            + _optimistic_remaining_sum(
                order, stage_options, next_index, lambda candidate: candidate.memory_mb
            )
        )
    )
    energy_lb = energy_committed + _optimistic_remaining_sum(
        order,
        stage_options,
        next_index,
        lambda candidate: (candidate.average_power_mw * candidate.latency_ms) / 1000.0,
    )
    copy_lb = copy_committed

    return _PartialPlan(
        assigned_options=assigned,
        next_index=next_index,
        latency_committed_ms=latency_committed,
        memory_committed_mb=memory_committed,
        energy_committed_mj=energy_committed,
        copy_committed_bytes=copy_committed,
        quality_risk=quality_risk,
        latency_lb_ms=latency_lb,
        memory_lb_mb=memory_lb,
        energy_lb_mj=energy_lb,
        copy_lb_bytes=copy_lb,
    )


def _dominates(lhs: _PartialPlan, rhs: _PartialPlan) -> bool:
    lhs_metrics = (
        lhs.latency_lb_ms,
        lhs.memory_lb_mb,
        lhs.energy_lb_mj,
        lhs.copy_lb_bytes,
        lhs.quality_risk,
    )
    rhs_metrics = (
        rhs.latency_lb_ms,
        rhs.memory_lb_mb,
        rhs.energy_lb_mj,
        rhs.copy_lb_bytes,
        rhs.quality_risk,
    )
    no_worse = all(left <= right for left, right in zip(lhs_metrics, rhs_metrics))
    strictly_better = any(left < right for left, right in zip(lhs_metrics, rhs_metrics))
    return no_worse and strictly_better


def beam_search_macro_region_plans(
    workflow: WorkflowDag,
    stage_options: Mapping[str, tuple[StageOption, ...]],
    *,
    beam_width: int = 8,
) -> tuple[list[RankedMacroPlan], BeamSearchStats]:
    order = workflow.topological_order()
    if beam_width <= 0:
        raise ValueError("beam_width must be positive.")
    for stage_id in order:
        options = stage_options.get(stage_id)
        if not options:
            raise ValueError(
                f"Stage '{stage_id}' has no explicit macro-region options. "
                "Remediation: provide support-safe options for every expanded node."
            )

    initial = _build_partial(workflow, order, stage_options, None, None)
    beam = [initial]
    completed: list[_PartialPlan] = []
    stats = {
        "expanded_partials": 0,
        "completed_plans": 0,
        "pruned_by_beam": 0,
        "pruned_by_dominance": 0,
    }

    while beam:
        next_beam: list[_PartialPlan] = []
        for partial in beam:
            if partial.next_index == len(order):
                completed.append(partial)
                stats["completed_plans"] += 1
                continue
            stage_id = order[partial.next_index]
            for option in stage_options[stage_id]:
                next_partial = _build_partial(workflow, order, stage_options, partial, option)
                stats["expanded_partials"] += 1
                dominated = False
                survivors: list[_PartialPlan] = []
                for existing in next_beam:
                    if _dominates(existing, next_partial):
                        dominated = True
                        stats["pruned_by_dominance"] += 1
                        break
                    if _dominates(next_partial, existing):
                        stats["pruned_by_dominance"] += 1
                        continue
                    survivors.append(existing)
                if dominated:
                    continue
                survivors.append(next_partial)
                next_beam = survivors
        if not next_beam:
            break
        next_beam.sort(
            key=lambda item: (
                item.latency_lb_ms,
                item.memory_lb_mb,
                item.energy_lb_mj,
                item.copy_lb_bytes,
                item.quality_risk,
                tuple(option.backend for option in item.assigned_options),
            )
        )
        if len(next_beam) > beam_width:
            stats["pruned_by_beam"] += len(next_beam) - beam_width
            next_beam = next_beam[:beam_width]
        beam = next_beam

    ranked = [
        RankedMacroPlan(
            plan=CandidatePlan(workflow.workflow_id, partial.assigned_options),
            latency_lb_ms=partial.latency_lb_ms,
            memory_lb_mb=partial.memory_lb_mb,
            energy_lb_mj=partial.energy_lb_mj,
            copy_lb_bytes=partial.copy_lb_bytes,
            quality_risk=partial.quality_risk,
        )
        for partial in completed
    ]
    ranked.sort(
        key=lambda item: (
            item.latency_lb_ms,
            item.memory_lb_mb,
            item.energy_lb_mj,
            item.copy_lb_bytes,
            item.quality_risk,
            item.plan.plan_id,
        )
    )
    return ranked, BeamSearchStats(**stats)
