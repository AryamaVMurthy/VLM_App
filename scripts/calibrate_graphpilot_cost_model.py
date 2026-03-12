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
    raw_latency_deltas_ms = {
        row["workflow_id"]: float(row["latency_delta_ms"])
        for row in comparisons
    }
    orchestration_overhead_ms = {
        workflow_id: max(0.0, delta_ms)
        for workflow_id, delta_ms in raw_latency_deltas_ms.items()
    }
    residual_bias_ms = {
        workflow_id: delta_ms - orchestration_overhead_ms[workflow_id]
        for workflow_id, delta_ms in raw_latency_deltas_ms.items()
    }
    thermal_scale_by_workflow: dict[str, Any] = {}
    backend_thermal_factors_by_workflow: dict[str, Any] = {}
    backend_utilizations_by_workflow: dict[str, Any] = {}
    actual_workflows = experiment_summary.get("actual_workflows", {})
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
        stage_backends = (actual_workflows.get(workflow_id) or {}).get("stage_backends") or {}
        backend_counts: dict[str, int] = {}
        for backend in stage_backends.values():
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
        total_backend_count = sum(backend_counts.values())
        backend_utilizations_by_workflow[workflow_id] = (
            {
                backend: count / total_backend_count
                for backend, count in sorted(backend_counts.items())
            }
            if total_backend_count
            else {}
        )
        backend_thermal_factors_by_workflow[workflow_id] = (
            {
                backend: scale
                for backend in sorted(backend_counts)
            }
            if scale is not None
            else {}
        )
    return {
        "orchestration_overhead_ms": orchestration_overhead_ms,
        "global_orchestration_overhead_ms": statistics.mean(orchestration_overhead_ms.values()),
        "residual_bias_ms": residual_bias_ms,
        "thermal_scale_by_workflow": thermal_scale_by_workflow,
        "backend_thermal_factors_by_workflow": backend_thermal_factors_by_workflow,
        "backend_utilizations_by_workflow": backend_utilizations_by_workflow,
        "surrogate_parameter_notes": {
            "backend_thermal_factors_by_workflow": "Workflow-level warm-latency slowdown assigned to participating backends as a surrogate thermal factor, because per-backend slowdown is not directly measured yet.",
            "backend_utilizations_by_workflow": "Backend utilization proxy derived from the fraction of stages assigned to each backend in the support-safe deployed plan.",
            "residual_bias_ms": "Signed simulator-vs-device residual after removing non-negative orchestration overhead. Negative values mean the simulator overpredicted latency."
        },
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
