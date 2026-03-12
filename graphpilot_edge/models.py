from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StageOption:
    stage_id: str
    variant_id: str
    backend: str
    latency_ms: int
    memory_mb: int
    output_bytes: int = 0
    average_power_mw: float = 0.0
    quality_loss: float = 0.0
    compile_cost_ms: int = 0
    ttft_ms: int | None = None
    tts_first_audio_ms: int | None = None


@dataclass(frozen=True)
class CandidatePlan:
    workflow_id: str
    stage_options: tuple[StageOption, ...]
    plan_id: str = field(init=False)
    total_latency_ms: int = field(init=False)
    total_memory_mb: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "plan_id",
            f"{self.workflow_id}:"
            + "|".join(
                f"{option.stage_id}={option.variant_id}" for option in self.stage_options
            ),
        )
        object.__setattr__(
            self,
            "total_latency_ms",
            sum(option.latency_ms for option in self.stage_options),
        )
        object.__setattr__(
            self,
            "total_memory_mb",
            sum(option.memory_mb for option in self.stage_options),
        )

    def option_for_stage(self, stage_id: str) -> StageOption:
        for option in self.stage_options:
            if option.stage_id == stage_id:
                return option
        raise KeyError(f"Missing stage option for '{stage_id}' in plan '{self.plan_id}'")


@dataclass(frozen=True)
class SimulationEvent:
    stage_id: str
    backend: str
    event_type: str
    timestamp_ms: int
    request_id: str | None = None


@dataclass(frozen=True)
class StageTiming:
    stage_id: str
    backend: str
    start_ms: int
    finish_ms: int
    request_id: str | None = None


@dataclass(frozen=True)
class SimulationResult:
    workflow_id: str
    plan_id: str
    makespan_ms: int
    events: tuple[SimulationEvent, ...]
    stage_timings: dict[str, StageTiming]
    copy_bytes: int = 0
    copy_time_ms: float = 0.0
    peak_memory_bytes: int = 0
    energy_mj: float = 0.0
    ttft_ms: int | None = None
    ttfs_ms: int | None = None
    quality_loss: float = 0.0
    p95_e2e_ms: float = 0.0
    p95_ttfs_ms: float = 0.0
    avg_energy_mj: float = 0.0
    p95_queue_delay_ms: float = 0.0
    deadline_miss_rate: float = 0.0
    objective_score: float | None = None


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    arrival_ms: int
    workflow: "WorkflowDag"
    plan: CandidatePlan
    deadline_ms: int | None = None
    criticality_class: str = "default"
    source_workload_id: str | None = None


@dataclass(frozen=True)
class RequestSimulationResult:
    request_id: str
    workflow_id: str
    plan_id: str
    arrival_ms: int
    start_ms: int
    finish_ms: int
    makespan_ms: int
    queue_delay_ms: int
    ttft_ms: int | None = None
    ttfs_ms: int | None = None
    deadline_ms: int | None = None
    deadline_missed: bool = False


@dataclass(frozen=True)
class StreamSimulationResult:
    makespan_ms: int
    request_results: dict[str, RequestSimulationResult]
    events: tuple[SimulationEvent, ...]
    copy_bytes: int = 0
    copy_time_ms: float = 0.0
    peak_memory_bytes: int = 0
    energy_mj: float = 0.0
    quality_loss: float = 0.0
    p95_e2e_ms: float = 0.0
    p95_ttfs_ms: float = 0.0
    avg_energy_mj: float = 0.0
    objective_score: float | None = None
    p95_queue_delay_ms: float = 0.0
    deadline_miss_rate: float = 0.0
