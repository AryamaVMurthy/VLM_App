from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .hardware_simulator import HardwareInstance, HardwareSimulator

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_HARDWARE_TOPOLOGY_PATH = (
    ROOT_DIR / "configs" / "graphpilot_edge" / "hardware_topology_sm8750.json"
)


@dataclass(frozen=True)
class LoadedHardwareTopology:
    topology_id: str
    simulator: HardwareSimulator
    public_anchors: tuple[str, ...] = ()
    surrogate_parameter_notes: Mapping[str, str] = field(default_factory=dict)
    raw_payload: Mapping[str, Any] = field(default_factory=dict)


def load_hardware_topology(path: Path = DEFAULT_HARDWARE_TOPOLOGY_PATH) -> LoadedHardwareTopology:
    payload = json.loads(path.read_text(encoding="utf-8"))
    topology_id = payload.get("topology_id")
    if not topology_id:
        raise ValueError("Hardware topology file is missing topology_id.")

    resources = tuple(
        HardwareInstance(
            resource_id=resource["resource_id"],
            resource_type=resource["resource_type"],
            supported_op_classes=tuple(resource["supported_op_classes"]),
            service_rates={key: float(value) for key, value in resource["service_rates"].items()},
            memory_bandwidth_bytes_per_ms=float(resource["memory_bandwidth_bytes_per_ms"]),
            launch_overhead_ms=float(resource["launch_overhead_ms"]),
            busy_power_mw=float(resource["busy_power_mw"]),
            idle_power_mw=float(resource["idle_power_mw"]),
            queue_depth=int(resource["queue_depth"]),
            batching_beta=float(resource["batching_beta"]),
            batching_tau=float(resource["batching_tau"]),
            thermal_time_constant_ms=float(resource["thermal_time_constant_ms"]),
            thermal_resistance_c_per_mw=float(resource["thermal_resistance_c_per_mw"]),
            thermal_threshold_c=float(resource["thermal_threshold_c"]),
            thermal_gamma=float(resource["thermal_gamma"]),
            contention_domain=resource["contention_domain"],
        )
        for resource in payload.get("resources", ())
    )
    simulator = HardwareSimulator(
        resources=resources,
        transfer_bandwidth_bytes_per_ms=_load_pair_float_map(
            payload.get("transfer_bandwidth_bytes_per_ms", ())
        ),
        transfer_fixed_overhead_ms=_load_pair_float_map(
            payload.get("transfer_fixed_overhead_ms", ())
        ),
        transfer_layout_overhead_ms=_load_pair_float_map(
            payload.get("transfer_layout_overhead_ms", ())
        ),
        contention_sensitivities=_load_pair_float_map(
            payload.get("contention_sensitivities", ())
        ),
        ambient_temperature_c=float(payload.get("ambient_temperature_c", 30.0)),
    )
    return LoadedHardwareTopology(
        topology_id=topology_id,
        simulator=simulator,
        public_anchors=tuple(payload.get("public_anchors", ())),
        surrogate_parameter_notes=dict(payload.get("surrogate_parameter_notes", {})),
        raw_payload=payload,
    )


def _load_pair_float_map(entries: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> dict[tuple[str, str], float]:
    mapping: dict[tuple[str, str], float] = {}
    for entry in entries:
        src = entry["src"]
        dst = entry["dst"]
        mapping[(src, dst)] = float(entry["value"])
    return mapping


__all__ = [
    "DEFAULT_HARDWARE_TOPOLOGY_PATH",
    "LoadedHardwareTopology",
    "load_hardware_topology",
]
