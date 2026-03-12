from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class ObjectiveWeights:
    alpha: float
    beta: float
    gamma: float
    delta: float
    eta: float
    zeta: float
    xi: float = 0.0
    psi: float = 0.0


@dataclass(frozen=True)
class SimulationCalibration:
    orchestration_overhead_ms: float = 0.0
    workflow_thermal_scale: float = 1.0
    backend_thermal_factors: Mapping[str, float] = field(default_factory=dict)
    contention_sensitivities: Mapping[tuple[str, str], float] = field(default_factory=dict)
    backend_utilizations: Mapping[str, float] = field(default_factory=dict)
    stage_latency_scales: Mapping[tuple[str, str], float] = field(default_factory=dict)
    family_latency_scales: Mapping[tuple[str, str], float] = field(default_factory=dict)
    stage_residual_bias_ms: Mapping[tuple[str, str], float] = field(default_factory=dict)
    family_residual_bias_ms: Mapping[tuple[str, str], float] = field(default_factory=dict)
    backend_launch_overheads_ms: Mapping[str, float] = field(default_factory=dict)
    backend_transfer_bias_ms: Mapping[str, float] = field(default_factory=dict)
    backend_contention_scales: Mapping[str, float] = field(default_factory=dict)


def compute_execution_time_ms(
    *,
    base_latency_ms: float,
    thermal_factor: float = 1.0,
    contention_factor: float = 1.0,
    compile_cost_ms: float = 0.0,
    cold: bool = False,
) -> float:
    return base_latency_ms * thermal_factor * contention_factor + (compile_cost_ms if cold else 0.0)


def compute_transfer_cost_ms(
    *,
    size_bytes: int,
    src_backend: str,
    dst_backend: str,
    bandwidth_bytes_per_ms: float,
    fixed_overhead_ms: float = 0.0,
    layout_ms: float = 0.0,
    same_memory: bool = False,
    zero_copy: bool = False,
    map_cost_ms: float = 0.0,
) -> float:
    if src_backend == dst_backend and same_memory:
        return 0.0
    if zero_copy:
        return map_cost_ms
    return fixed_overhead_ms + (size_bytes / max(bandwidth_bytes_per_ms, 1e-9)) + layout_ms


def compute_contention_factor(
    *,
    backend: str,
    sensitivities: Mapping[tuple[str, str], float],
    utilizations: Mapping[str, float],
) -> float:
    return 1.0 + sum(
        sensitivity * utilizations.get(other_backend, 0.0)
        for (target_backend, other_backend), sensitivity in sensitivities.items()
        if target_backend == backend
    )


def compute_energy_mj(*, avg_power_mw: float, time_ms: float, idle_power_mw: float = 0.0, wall_time_ms: float = 0.0) -> float:
    return ((avg_power_mw * time_ms) + (idle_power_mw * wall_time_ms)) / 1000.0


def kv_cache_size_bytes(
    *,
    layers: int,
    kv_heads: int,
    head_dim: int,
    cached_tokens: int,
    bytes_per_element: int,
) -> int:
    return 2 * layers * kv_heads * head_dim * cached_tokens * bytes_per_element


def evaluate_hybrid_prefill_decode(
    *,
    prefill_npu_ms: float,
    prefill_alt_ms: float,
    decode_npu_per_token_ms: float,
    decode_alt_per_token_ms: float,
    output_tokens: int,
    kv_migration_ms: float,
) -> dict[str, bool]:
    return {
        "beats_all_alt": (prefill_alt_ms - prefill_npu_ms) > kv_migration_ms,
        "beats_all_npu": (output_tokens * (decode_npu_per_token_ms - decode_alt_per_token_ms)) > kv_migration_ms,
    }


def compute_objective_score(
    *,
    p95_e2e_ms: float,
    p95_ttfs_ms: float,
    avg_energy_mj: float,
    peak_memory_bytes: int,
    copy_bytes: int,
    quality_loss: float,
    p95_queue_delay_ms: float = 0.0,
    deadline_miss_rate: float = 0.0,
    weights: ObjectiveWeights,
) -> float:
    return (
        weights.alpha * p95_e2e_ms
        + weights.beta * p95_ttfs_ms
        + weights.gamma * avg_energy_mj
        + weights.delta * peak_memory_bytes
        + weights.eta * copy_bytes
        + weights.zeta * quality_loss
        + weights.xi * p95_queue_delay_ms
        + weights.psi * deadline_miss_rate
    )
