from __future__ import annotations

from itertools import product

from .errors import EnumerationError
from .models import CandidatePlan, StageOption
from .workflow import WorkflowDag


def enumerate_candidate_plans(
    workflow: WorkflowDag, stage_options: dict[str, tuple[StageOption, ...]]
) -> list[CandidatePlan]:
    missing = [stage_id for stage_id in workflow.stage_ids if stage_id not in stage_options]
    if missing:
        raise EnumerationError(
            "Missing stage options for workflow stages: " + ", ".join(sorted(missing))
        )

    ordered_options = []
    for stage_id in workflow.stage_ids:
        options = stage_options[stage_id]
        if not options:
            raise EnumerationError(f"Stage '{stage_id}' has no explicit options.")
        for option in options:
            if option.stage_id != stage_id:
                raise EnumerationError(
                    f"Stage option stage mismatch: expected '{stage_id}', got '{option.stage_id}'."
                )
        ordered_options.append(options)

    return [
        CandidatePlan(workflow.workflow_id, tuple(option_combo))
        for option_combo in product(*ordered_options)
    ]

