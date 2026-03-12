#!/usr/bin/env python3
"""Export the GraphPilot workload universe into a machine-readable registry."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from graphpilot_edge.workload_universe import (  # noqa: E402
    DEFAULT_WORKLOAD_UNIVERSE_PATH,
    load_workload_universe,
)

DEFAULT_OUTPUT = ROOT_DIR / "artifacts" / "graphpilot_edge" / "registries" / "workload_universe_registry.json"


def scenario_to_dict(workload: Any) -> dict[str, Any]:
    scenario = workload
    return {
        "workflow_id": scenario.workflow.workflow_id,
        "stage_ids": list(scenario.workflow.stage_ids),
        "edges": [
            {
                "from": edge.source_stage_id,
                "to": edge.target_stage_id,
                "stream_mode": edge.stream_mode,
            }
            for edge in scenario.workflow.edges
        ],
        "chunk_sizes": {
            stage_id: chunk_size for stage_id, chunk_size in scenario.workflow.chunk_sizes
        },
        "node_profiles": {
            stage_id: {
                "family": node.family,
                "variant_id": node.variant_id,
                "task_id": node.task_profile.task_id,
                "op_volume": dict(node.task_profile.op_volume),
                "memory_bytes": node.task_profile.memory_bytes,
                "input_bytes": node.task_profile.input_bytes,
                "output_bytes": node.task_profile.output_bytes,
                "knob_values": dict(node.knob_values),
                "quality_loss": node.quality_loss,
                "ttft_hint_ms": node.ttft_hint_ms,
                "tts_first_audio_hint_ms": node.tts_first_audio_hint_ms,
            }
            for stage_id, node in scenario.node_profiles.items()
        },
    }


def build_registry(workload_universe_path: Path) -> dict[str, Any]:
    universe = load_workload_universe(workload_universe_path)
    workloads = []
    for spec in universe.specs.values():
        entry = {
            "workload_id": spec.workload_id,
            "category": spec.category,
            "description": spec.description,
            "builder": spec.builder,
            "datasets": list(spec.datasets),
            "tags": list(spec.tags),
            "base_workload_id": spec.base_workload_id,
            "arrivals_ms": list(spec.arrivals_ms),
            "deadline_ms": spec.deadline_ms,
            "params": dict(spec.params),
        }
        if spec.category != "continuous_stream":
            entry["scenario"] = scenario_to_dict(universe.build_scenario(spec.workload_id))
        workloads.append(entry)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(workload_universe_path.resolve()),
        "workloads": workloads,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload-universe",
        type=Path,
        default=DEFAULT_WORKLOAD_UNIVERSE_PATH,
        help="Path to the workload_universe.json configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output registry path.",
    )
    args = parser.parse_args(argv)

    payload = build_registry(args.workload_universe)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
