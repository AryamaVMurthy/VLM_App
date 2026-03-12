#!/usr/bin/env python3
"""Ingest an existing FastVLM quick-run summary as a GraphPilot VLM NPU profile."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
PROFILER_REGISTRY_PATH = ROOT_DIR / "artifacts" / "graphpilot_edge" / "registries" / "profiler_registry.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def to_ms(seconds: float | None) -> int | None:
    if seconds is None:
        return None
    return int(round(seconds * 1000.0))


def build_registry_entry(summary_path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    prefill_ms = to_ms(summary.get("prefill_mean_s"))
    decode_ms = to_ms(summary.get("decode_mean_s"))
    ttft_ms = to_ms(summary.get("ttft_mean_s"))
    total_ms = None if prefill_ms is None or decode_ms is None else prefill_ms + decode_ms
    return {
        "profile_id": f"fastvlm_quick:{summary_path.parent.name}",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "stage_id": "vlm.fastvlm.primary",
        "backend": "npu",
        "variant": "fastvlm_litert_npu_sm8750",
        "test_class": None,
        "command": f"ingested_from:{summary_path}",
        "artifacts": {
            "run_dir": str(summary_path.parent),
            "summary": str(summary_path),
        },
        "metrics": {
            "cold_latency_ms": total_ms,
            "warm_latency_ms": total_ms,
            "steady_state_latency_ms": total_ms,
            "prefill_latency_ms": prefill_ms,
            "decode_latency_ms": decode_ms,
            "peak_memory_bytes": None,
            "transfer_bytes": None,
            "transfer_time_ms": None,
            "energy_mj_or_power_proxy": None,
            "thermal_bin": "unknown",
            "hidden_fallback_status": "no_hidden_fallback_observed",
            "ttft_ms": ttft_ms,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_path", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = args.summary_path.resolve()
    summary = load_json(summary_path)
    entry = build_registry_entry(summary_path, summary)
    registry = load_json(PROFILER_REGISTRY_PATH)
    registry.setdefault("entries", []).append(entry)
    registry["last_updated"] = datetime.now(timezone.utc).date().isoformat()
    write_json(PROFILER_REGISTRY_PATH, registry)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
