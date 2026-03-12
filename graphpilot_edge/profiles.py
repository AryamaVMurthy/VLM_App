from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StageProfile:
    stage_id: str
    backend: str
    variant: str
    warm_latency_ms: int | None
    cold_latency_ms: int | None
    steady_state_latency_ms: int | None
    peak_memory_bytes: int | None
    output_bytes: int | None = None
    average_power_mw: float | None = None
    quality_loss: float | None = None
    compile_cost_ms: int | None = None
    ttft_ms: int | None = None
    tts_first_audio_ms: int | None = None
    transfer_bytes: int | None = None
    transfer_time_ms: int | None = None
    thermal_bin: str | None = None
    hidden_fallback_status: str | None = None


def load_profile_index(registry: dict[str, Any]) -> dict[tuple[str, str], StageProfile]:
    index: dict[tuple[str, str], StageProfile] = {}
    for entry in registry.get("entries", []):
        metrics = entry.get("metrics", {})
        profile = StageProfile(
            stage_id=entry["stage_id"],
            backend=entry["backend"],
            variant=entry["variant"],
            warm_latency_ms=metrics.get("warm_latency_ms"),
            cold_latency_ms=metrics.get("cold_latency_ms"),
            steady_state_latency_ms=metrics.get("steady_state_latency_ms"),
            peak_memory_bytes=metrics.get("peak_memory_bytes"),
            output_bytes=metrics.get("output_bytes"),
            average_power_mw=metrics.get("average_power_mw"),
            quality_loss=metrics.get("quality_loss"),
            compile_cost_ms=metrics.get("compile_cost_ms"),
            ttft_ms=metrics.get("ttft_ms"),
            tts_first_audio_ms=metrics.get("tts_first_audio_ms"),
            transfer_bytes=metrics.get("transfer_bytes"),
            transfer_time_ms=metrics.get("transfer_time_ms"),
            thermal_bin=metrics.get("thermal_bin"),
            hidden_fallback_status=metrics.get("hidden_fallback_status"),
        )
        index[(profile.stage_id, profile.backend)] = profile
    return index
