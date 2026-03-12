from __future__ import annotations

from typing import Mapping

from .workflow import WorkflowDag


def compute_upward_ranks(
    workflow: WorkflowDag,
    stage_costs: dict[str, int],
    communication_costs: dict[tuple[str, str], int],
) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for stage_id in reversed(workflow.topological_order()):
        successors = workflow.successors(stage_id)
        if not successors:
            ranks[stage_id] = stage_costs[stage_id]
            continue
        downstream = max(
            communication_costs.get((stage_id, successor), 0) + ranks[successor]
            for successor in successors
        )
        ranks[stage_id] = stage_costs[stage_id] + downstream
    return ranks


def compute_dynamic_priority(
    *,
    upward_rank: float,
    first_output_bonus: float,
    age_bonus: float,
    copy_penalty: float,
    memory_penalty: float,
    thermal_penalty: float,
    weights: Mapping[str, float],
) -> float:
    return (
        weights.get("rank", 1.0) * upward_rank
        + weights.get("first_output", 1.0) * first_output_bonus
        + weights.get("age", 1.0) * age_bonus
        - weights.get("copy", 1.0) * copy_penalty
        - weights.get("memory", 1.0) * memory_penalty
        - weights.get("thermal", 1.0) * thermal_penalty
    )
