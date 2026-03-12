from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .model_graph_simulator import (
    DEFAULT_WORKFLOW_TEMPLATES_PATH,
    ModelGraphScenario,
    build_assistant_workflow_scenario,
    build_cnn_scenario,
    build_llm_scenario,
    build_primitive_operator_scenario,
    build_retrieval_scenario,
    build_sequential_glue_scenario,
    build_stt_scenario,
    build_tts_scenario,
    build_vit_scenario,
    build_vlm_scenario,
)
from .hardware_simulator import HardwareSimulator
from .models import RequestSpec

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WORKLOAD_UNIVERSE_PATH = ROOT_DIR / "configs" / "graphpilot_edge" / "workload_universe.json"


@dataclass(frozen=True)
class WorkloadSpec:
    workload_id: str
    category: str
    description: str
    builder: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    datasets: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    base_workload_id: str | None = None
    arrivals_ms: tuple[int, ...] = ()
    deadline_ms: int | None = None


@dataclass(frozen=True)
class WorkloadUniverse:
    specs: Mapping[str, WorkloadSpec]

    def __post_init__(self) -> None:
        if not self.specs:
            raise ValueError("Workload universe must contain at least one workload.")

    def build_scenario(self, workload_id: str) -> ModelGraphScenario:
        spec = self._require_spec(workload_id)
        if spec.category == "continuous_stream":
            if spec.base_workload_id is None:
                raise ValueError(
                    f"Continuous workload '{workload_id}' is missing base_workload_id."
                )
            return self.build_scenario(spec.base_workload_id)
        if spec.builder is None:
            raise ValueError(f"Workload '{workload_id}' is missing a scenario builder.")
        builder = _SCENARIO_BUILDERS.get(spec.builder)
        if builder is None:
            raise ValueError(
                f"Workload '{workload_id}' references unknown builder '{spec.builder}'."
            )
        return builder(scenario_id=workload_id, **dict(spec.params))

    def build_request_specs(
        self,
        workload_id: str,
        *,
        simulator: HardwareSimulator,
        resource_assignment: Mapping[str, str],
        batch_size: int = 1,
        current_temp_c: Mapping[str, float] | None = None,
        utilizations: Mapping[str, float] | None = None,
    ) -> tuple[RequestSpec, ...]:
        spec = self._require_spec(workload_id)
        base_workload_id = spec.base_workload_id or workload_id
        scenario = self.build_scenario(base_workload_id)
        plan = scenario.instantiate_candidate_plan(
            simulator=simulator,
            resource_assignment=resource_assignment,
            batch_size=batch_size,
            current_temp_c=current_temp_c,
            utilizations=utilizations,
        )
        arrivals_ms = spec.arrivals_ms or (0,)
        return tuple(
            RequestSpec(
                request_id=f"{workload_id}:request_{index}",
                arrival_ms=arrival_ms,
                workflow=scenario.workflow,
                plan=plan,
                deadline_ms=spec.deadline_ms,
            )
            for index, arrival_ms in enumerate(arrivals_ms)
        )

    def _require_spec(self, workload_id: str) -> WorkloadSpec:
        spec = self.specs.get(workload_id)
        if spec is None:
            raise KeyError(f"Unknown workload '{workload_id}'.")
        return spec


def load_workload_universe(path: Path = DEFAULT_WORKLOAD_UNIVERSE_PATH) -> WorkloadUniverse:
    payload = json.loads(path.read_text(encoding="utf-8"))
    specs: dict[str, WorkloadSpec] = {}
    for entry in payload.get("workloads", ()):
        workload_id = entry["workload_id"]
        if workload_id in specs:
            raise ValueError(f"Duplicate workload_id '{workload_id}' in workload universe.")
        spec = WorkloadSpec(
            workload_id=workload_id,
            category=entry["category"],
            description=entry.get("description", ""),
            builder=entry.get("builder"),
            params=dict(entry.get("params", {})),
            datasets=tuple(entry.get("datasets", ())),
            tags=tuple(entry.get("tags", ())),
            base_workload_id=entry.get("base_workload_id"),
            arrivals_ms=tuple(entry.get("arrivals_ms", ())),
            deadline_ms=entry.get("deadline_ms"),
        )
        if spec.category == "continuous_stream":
            if spec.base_workload_id is None:
                raise ValueError(
                    f"Continuous workload '{workload_id}' must declare base_workload_id."
                )
            if not spec.arrivals_ms:
                raise ValueError(
                    f"Continuous workload '{workload_id}' must declare arrivals_ms."
                )
        elif spec.builder is None:
            raise ValueError(
                f"Non-continuous workload '{workload_id}' must declare a builder."
            )
        specs[workload_id] = spec

    universe = WorkloadUniverse(specs=specs)
    for spec in universe.specs.values():
        if spec.base_workload_id is not None and spec.base_workload_id not in universe.specs:
            raise ValueError(
                f"Workload '{spec.workload_id}' references missing base workload '{spec.base_workload_id}'."
            )
        if spec.category != "continuous_stream":
            universe.build_scenario(spec.workload_id)
    return universe


_SCENARIO_BUILDERS = {
    "assistant_workflow": build_assistant_workflow_scenario,
    "cnn": build_cnn_scenario,
    "llm": build_llm_scenario,
    "primitive_operator": build_primitive_operator_scenario,
    "retrieval": build_retrieval_scenario,
    "sequential_glue": build_sequential_glue_scenario,
    "stt": build_stt_scenario,
    "tts": build_tts_scenario,
    "vit": build_vit_scenario,
    "vlm": build_vlm_scenario,
}


__all__ = [
    "DEFAULT_WORKLOAD_UNIVERSE_PATH",
    "DEFAULT_WORKFLOW_TEMPLATES_PATH",
    "WorkloadSpec",
    "WorkloadUniverse",
    "load_workload_universe",
]
