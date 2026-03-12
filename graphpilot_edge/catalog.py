from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .workflow import WorkflowDag, WorkflowEdge


@dataclass(frozen=True)
class StageDefinition:
    stage_id: str
    role: str
    default_variant: str | None = None
    macro_regions: tuple[str, ...] = ()


def load_stage_catalog(path: Path) -> dict[str, StageDefinition]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        stage["stage_id"]: StageDefinition(
            stage_id=stage["stage_id"],
            role=stage["role"],
            default_variant=stage.get("default_variant"),
            macro_regions=tuple(stage.get("macro_regions", ())),
        )
        for stage in payload.get("stages", [])
    }


def load_workflow_catalog(path: Path) -> dict[str, WorkflowDag]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    workflows: dict[str, WorkflowDag] = {}
    for workflow in payload.get("workflows", []):
        workflows[workflow["workflow_id"]] = WorkflowDag(
            workflow_id=workflow["workflow_id"],
            stage_ids=tuple(workflow.get("nodes", ())),
            edges=tuple(
                WorkflowEdge(edge["from"], edge["to"], edge["stream_mode"])
                for edge in workflow.get("edges", ())
            ),
        )
    return workflows


def resolve_feasible_backends(matrix: dict[str, object], stage_id: str) -> tuple[str, ...]:
    for stage in matrix.get("stages", []):
        if stage["stage_id"] != stage_id:
            continue
        backends = []
        for backend_name, backend_state in stage["backends"].items():
            if backend_state.get("status") in {"feasible_smoke_pass", "known_working"}:
                backends.append(backend_name)
        return tuple(sorted(backends))
    raise KeyError(f"Unknown stage_id '{stage_id}' in backend matrix.")
