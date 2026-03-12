#!/usr/bin/env python3
"""Calibrate GraphPilot cost-model factors from measured experiment artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
ARTIFACT_ROOT = ROOT_DIR / "artifacts" / "graphpilot_edge"
DEFAULT_EXPERIMENT_SUMMARY = ARTIFACT_ROOT / "experiments" / "graphpilot_experiment_batch_20260312_065529" / "summary.json"
DEFAULT_SUSTAINED_SUMMARY = ARTIFACT_ROOT / "experiments" / "graphpilot_sustained_load_20260311_064317" / "summary.json"
DEFAULT_PROFILER_REGISTRY = ARTIFACT_ROOT / "registries" / "profiler_registry.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _median(values: Iterable[float]) -> float:
    values = tuple(float(value) for value in values)
    if not values:
        return 0.0
    return float(statistics.median(values))


def _mean(values: Iterable[float]) -> float:
    values = tuple(float(value) for value in values)
    if not values:
        return 0.0
    return float(statistics.mean(values))


def _safe_ratio(numerator: float, denominator: float, *, default: float = 1.0) -> float:
    if denominator == 0:
        return default
    return float(numerator / denominator)


def _stage_family(stage_id: str) -> str:
    return stage_id.split(".", 1)[0]


def _metric_float(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key, 0.0)
    if value is None:
        return 0.0
    return float(value)


def _checkpoint_paths(path: Path | None) -> dict[str, Path]:
    if path is None:
        return {}
    payload = load_json(path)
    canonical = payload.get("canonical_evidence_paths") or {}
    if not isinstance(canonical, dict):
        raise ValueError("Checkpoint manifest canonical_evidence_paths must be a mapping.")
    return {
        key: Path(value)
        for key, value in canonical.items()
        if value is not None
    }


def _build_stage_observations(profiler_registry: dict[str, Any]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    stage_backend_observations: dict[tuple[str, str], dict[str, Any]] = {}
    latest_workflow_stage_observations: dict[tuple[str, str, str], dict[str, Any]] = {}
    for entry in profiler_registry.get("entries", []):
        stage_id = str(entry.get("stage_id") or "")
        backend = str(entry.get("backend") or "")
        recorded_at = str(entry.get("recorded_at") or "")
        metrics = entry.get("metrics") or {}
        if not stage_id or not backend or not isinstance(metrics, dict):
            continue
        if "stage_timings_ms" in metrics and "stage_backends" in metrics:
            for observed_stage_id, timing_ms in (metrics.get("stage_timings_ms") or {}).items():
                observed_backend = (metrics.get("stage_backends") or {}).get(observed_stage_id)
                if observed_backend is None:
                    continue
                key = (stage_id, str(observed_stage_id), str(observed_backend))
                current = latest_workflow_stage_observations.get(key)
                if current is None or recorded_at >= current["recorded_at"]:
                    latest_workflow_stage_observations[key] = {
                        "recorded_at": recorded_at,
                        "workflow_id": stage_id,
                        "backend": str(observed_backend),
                        "timing_ms": float(timing_ms),
                    }
            continue
        if backend == "mixed":
            continue
        key = (stage_id, backend)
        bucket = stage_backend_observations.setdefault(
            key,
            {
                "family": _stage_family(stage_id),
                "warm_latency_ms": [],
                "prefill_latency_ms": [],
                "decode_latency_ms": [],
                "transfer_time_ms": [],
                "peak_memory_bytes": [],
                "launch_overhead_ms": [],
                "hidden_fallback_count": 0,
                "sample_count": 0,
            },
        )
        bucket["sample_count"] += 1
        bucket["warm_latency_ms"].append(_metric_float(metrics, "warm_latency_ms"))
        bucket["prefill_latency_ms"].append(_metric_float(metrics, "prefill_latency_ms"))
        bucket["decode_latency_ms"].append(_metric_float(metrics, "decode_latency_ms"))
        bucket["transfer_time_ms"].append(_metric_float(metrics, "transfer_time_ms"))
        bucket["peak_memory_bytes"].append(_metric_float(metrics, "peak_memory_bytes"))
        cold_latency = _metric_float(metrics, "cold_latency_ms") or _metric_float(metrics, "warm_latency_ms")
        warm_latency = _metric_float(metrics, "warm_latency_ms")
        bucket["launch_overhead_ms"].append(max(0.0, cold_latency - warm_latency))
        if metrics.get("hidden_fallback_status") not in (None, "none", "support_safe"):
            bucket["hidden_fallback_count"] += 1
    workflow_stage_observations: dict[str, list[dict[str, Any]]] = {}
    for (_workflow_id, observed_stage_id, _backend), sample in latest_workflow_stage_observations.items():
        workflow_stage_observations.setdefault(observed_stage_id, []).append(
            {
                "workflow_id": sample["workflow_id"],
                "backend": sample["backend"],
                "timing_ms": sample["timing_ms"],
            }
        )
    return stage_backend_observations, workflow_stage_observations


def _build_stage_backend_calibration(
    stage_backend_observations: dict[tuple[str, str], dict[str, Any]],
    workflow_stage_observations: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any], dict[str, float], dict[str, float], dict[str, float]]:
    stage_backend_calibration: dict[str, dict[str, Any]] = {}
    family_backend_calibration: dict[str, dict[str, dict[str, Any]]] = {}
    family_quality_samples: dict[str, list[float]] = {}
    launch_overhead_by_backend: dict[str, list[float]] = {}
    transfer_bias_by_backend: dict[str, list[float]] = {}
    contention_scale_by_backend: dict[str, list[float]] = {}

    for (stage_id, backend), observed in sorted(stage_backend_observations.items()):
        family = str(observed["family"])
        predicted_warm_ms = _median(observed["warm_latency_ms"])
        actual_stage_times = [
            sample["timing_ms"]
            for sample in workflow_stage_observations.get(stage_id, [])
            if sample["backend"] == backend
        ]
        actual_warm_ms = _mean(actual_stage_times) if actual_stage_times else predicted_warm_ms
        latency_scale = _safe_ratio(actual_warm_ms, predicted_warm_ms)
        residual_bias_ms = actual_warm_ms - predicted_warm_ms
        sample_count = int(observed["sample_count"])
        hidden_fallback_rate = _safe_ratio(float(observed["hidden_fallback_count"]), float(sample_count), default=0.0)
        calibration_row = {
            "family": family,
            "sample_count": sample_count,
            "warm_latency_ms": predicted_warm_ms,
            "actual_stage_timing_ms": actual_warm_ms,
            "prefill_latency_ms": _median(observed["prefill_latency_ms"]),
            "decode_latency_ms": _median(observed["decode_latency_ms"]),
            "transfer_time_ms": _median(observed["transfer_time_ms"]),
            "peak_memory_bytes": _median(observed["peak_memory_bytes"]),
            "launch_overhead_ms": _median(observed["launch_overhead_ms"]),
            "latency_scale": latency_scale,
            "residual_bias_ms": residual_bias_ms,
            "hidden_fallback_rate": hidden_fallback_rate,
        }
        stage_backend_calibration.setdefault(stage_id, {})[backend] = calibration_row
        family_backend_calibration.setdefault(family, {}).setdefault(backend, []).append(calibration_row)
        family_quality_samples.setdefault(family, []).append(abs(residual_bias_ms))
        launch_overhead_by_backend.setdefault(backend, []).append(calibration_row["launch_overhead_ms"])
        transfer_bias_by_backend.setdefault(backend, []).append(calibration_row["transfer_time_ms"])
        contention_scale_by_backend.setdefault(backend, []).append(max(1.0, latency_scale))

    reduced_family_backend: dict[str, dict[str, Any]] = {}
    calibration_quality_by_family: dict[str, Any] = {}
    for family, backend_rows in sorted(family_backend_calibration.items()):
        reduced_family_backend[family] = {}
        for backend, rows in sorted(backend_rows.items()):
            reduced_family_backend[family][backend] = {
                "sample_count": sum(int(row["sample_count"]) for row in rows),
                "warm_latency_ms": _mean(row["warm_latency_ms"] for row in rows),
                "actual_stage_timing_ms": _mean(row["actual_stage_timing_ms"] for row in rows),
                "prefill_latency_ms": _mean(row["prefill_latency_ms"] for row in rows),
                "decode_latency_ms": _mean(row["decode_latency_ms"] for row in rows),
                "transfer_time_ms": _mean(row["transfer_time_ms"] for row in rows),
                "peak_memory_bytes": _mean(row["peak_memory_bytes"] for row in rows),
                "launch_overhead_ms": _mean(row["launch_overhead_ms"] for row in rows),
                "latency_scale": _mean(row["latency_scale"] for row in rows),
                "residual_bias_ms": _mean(row["residual_bias_ms"] for row in rows),
                "hidden_fallback_rate": _mean(row["hidden_fallback_rate"] for row in rows),
            }
        residuals = tuple(family_quality_samples.get(family, ()))
        calibration_quality_by_family[family] = {
            "sample_count": len(residuals),
            "mean_absolute_error_ms": _mean(residuals),
            "max_absolute_error_ms": max(residuals, default=0.0),
        }

    return (
        stage_backend_calibration,
        reduced_family_backend,
        calibration_quality_by_family,
        {backend: _mean(values) for backend, values in sorted(launch_overhead_by_backend.items())},
        {backend: _mean(values) for backend, values in sorted(transfer_bias_by_backend.items())},
        {backend: _mean(values) for backend, values in sorted(contention_scale_by_backend.items())},
    )


def calibrate(
    experiment_summary: dict[str, Any],
    sustained_summary: dict[str, Any],
    profiler_registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
        raw_scale = (last / first) if first not in (None, 0) and last is not None else None
        scale = max(1.0, raw_scale) if raw_scale is not None else None
        thermal_scale_by_workflow[workflow_id] = {
            "warm_latency_scale": scale,
            "cpu_temp_drift_c": cpu.get("drift"),
            "skin_temp_drift_c": skin.get("drift"),
        }
        stage_backends = (actual_workflows.get(workflow_id) or {}).get("stage_backends") or {}
        backend_counts: dict[str, int] = {}
        for backend in stage_backends.values():
            backend_counts[str(backend)] = backend_counts.get(str(backend), 0) + 1
        total_backend_count = sum(backend_counts.values())
        backend_utilizations_by_workflow[workflow_id] = (
            {backend: count / total_backend_count for backend, count in sorted(backend_counts.items())}
            if total_backend_count
            else {}
        )
        backend_thermal_factors_by_workflow[workflow_id] = (
            {backend: scale for backend in sorted(backend_counts)}
            if scale is not None
            else {}
        )

    profiler_registry = profiler_registry or {"entries": []}
    stage_backend_observations, workflow_stage_observations = _build_stage_observations(profiler_registry)
    (
        stage_backend_calibration,
        family_backend_calibration,
        calibration_quality_by_family,
        launch_overhead_ms_by_backend,
        transfer_bias_ms_by_backend,
        contention_scale_by_backend,
    ) = _build_stage_backend_calibration(stage_backend_observations, workflow_stage_observations)

    return {
        "orchestration_overhead_ms": orchestration_overhead_ms,
        "global_orchestration_overhead_ms": statistics.mean(orchestration_overhead_ms.values()),
        "residual_bias_ms": residual_bias_ms,
        "thermal_scale_by_workflow": thermal_scale_by_workflow,
        "backend_thermal_factors_by_workflow": backend_thermal_factors_by_workflow,
        "backend_utilizations_by_workflow": backend_utilizations_by_workflow,
        "stage_backend_calibration": stage_backend_calibration,
        "family_backend_calibration": family_backend_calibration,
        "calibration_quality_by_family": calibration_quality_by_family,
        "launch_overhead_ms_by_backend": launch_overhead_ms_by_backend,
        "transfer_bias_ms_by_backend": transfer_bias_ms_by_backend,
        "contention_scale_by_backend": contention_scale_by_backend,
        "surrogate_parameter_notes": {
            "backend_thermal_factors_by_workflow": "Workflow-level warm-latency slowdown assigned to participating backends as a surrogate thermal factor, because per-backend slowdown is not directly measured yet.",
            "backend_utilizations_by_workflow": "Backend utilization proxy derived from the fraction of stages assigned to each backend in the support-safe deployed plan.",
            "residual_bias_ms": "Signed simulator-vs-device residual after removing non-negative orchestration overhead. Negative values mean the simulator overpredicted latency.",
            "family_backend_calibration": "Median per-family/per-backend surrogates fitted from profiler stage measurements and workflow stage timings.",
            "stage_backend_calibration": "Per-stage/per-backend surrogates fitted from measured stage latency, transfer, and workflow stage timing evidence.",
            "launch_overhead_ms_by_backend": "Cold-vs-warm latency delta surrogate per backend.",
            "transfer_bias_ms_by_backend": "Median measured transfer-time bias per backend from profiler artifacts.",
            "contention_scale_by_backend": "Family-calibration-derived contention scale surrogate per backend.",
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-summary", type=Path, default=DEFAULT_EXPERIMENT_SUMMARY)
    parser.add_argument("--sustained-summary", type=Path, default=DEFAULT_SUSTAINED_SUMMARY)
    parser.add_argument("--profiler-registry", type=Path, default=DEFAULT_PROFILER_REGISTRY)
    parser.add_argument("--checkpoint-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint_paths = _checkpoint_paths(args.checkpoint_manifest)
    experiment_summary_path = Path(checkpoint_paths.get("experiment_summary", args.experiment_summary))
    sustained_summary_path = Path(checkpoint_paths.get("sustained_summary", args.sustained_summary))
    profiler_registry_path = Path(checkpoint_paths.get("profiler_registry", args.profiler_registry))

    experiment_summary = load_json(experiment_summary_path)
    sustained_summary = load_json(sustained_summary_path)
    profiler_registry = load_json(profiler_registry_path)
    payload = calibrate(experiment_summary, sustained_summary, profiler_registry)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"graphpilot_cost_calibration_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_summary": str(experiment_summary_path.resolve()),
        "sustained_summary": str(sustained_summary_path.resolve()),
        "profiler_registry": str(profiler_registry_path.resolve()),
        "checkpoint_manifest": str(args.checkpoint_manifest.resolve()) if args.checkpoint_manifest else None,
        **payload,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(
        "# GraphPilot Cost Calibration\n\n"
        f"- global_orchestration_overhead_ms={summary['global_orchestration_overhead_ms']:.2f}\n"
        f"- calibrated_families={sorted(summary['family_backend_calibration'])}\n",
        encoding="utf-8",
    )
    print(output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
