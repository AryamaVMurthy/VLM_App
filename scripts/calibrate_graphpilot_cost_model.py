#!/usr/bin/env python3
"""Calibrate GraphPilot cost-model factors from measured experiment artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def calibrate(experiment_summary: dict[str, Any], sustained_summary: dict[str, Any]) -> dict[str, Any]:
    comparisons = experiment_summary.get("comparisons", [])
    if not comparisons:
        raise ValueError("Experiment summary has no comparisons. Remediation: run scripts/run_graphpilot_experiments.py first.")
    orchestration_overhead_ms = {
        row["workflow_id"]: float(row["latency_delta_ms"])
        for row in comparisons
    }
    thermal_scale_by_workflow: dict[str, Any] = {}
    for workflow_id, workflow in sustained_summary.get("workflow_summary", {}).items():
        warm = workflow.get("warm_latency_ms") or {}
        cpu = (workflow.get("thermal") or {}).get("cpu_c") or {}
        skin = (workflow.get("thermal") or {}).get("skin_c") or {}
        first = warm.get("first")
        last = warm.get("last")
        scale = (last / first) if first not in (None, 0) and last is not None else None
        thermal_scale_by_workflow[workflow_id] = {
            "warm_latency_scale": scale,
            "cpu_temp_drift_c": cpu.get("drift"),
            "skin_temp_drift_c": skin.get("drift"),
        }
    return {
        "orchestration_overhead_ms": orchestration_overhead_ms,
        "global_orchestration_overhead_ms": statistics.mean(orchestration_overhead_ms.values()),
        "thermal_scale_by_workflow": thermal_scale_by_workflow,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-summary", type=Path, required=True)
    parser.add_argument("--sustained-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    experiment_summary = load_json(args.experiment_summary)
    sustained_summary = load_json(args.sustained_summary)
    payload = calibrate(experiment_summary, sustained_summary)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"graphpilot_cost_calibration_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_summary": str(args.experiment_summary.resolve()),
        "sustained_summary": str(args.sustained_summary.resolve()),
        **payload,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(
        "# GraphPilot Cost Calibration\n\n"
        f"- global_orchestration_overhead_ms={summary['global_orchestration_overhead_ms']:.2f}\n",
        encoding="utf-8",
    )
    print(output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
