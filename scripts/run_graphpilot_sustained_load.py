#!/usr/bin/env python3
"""Run sustained GraphPilot workflow loops on device and record drift evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
ARTIFACT_ROOT = ROOT_DIR / "artifacts" / "graphpilot_edge"
DEFAULT_OUTPUT_ROOT = ARTIFACT_ROOT / "experiments"
DEFAULT_EXPERIMENT_REGISTRY = ARTIFACT_ROOT / "registries" / "experiment_registry.json"
DEFAULT_PROFILES = (
    "graphpilot_workflow_a",
    "graphpilot_workflow_b",
    "graphpilot_workflow_c",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, env=env, check=False, capture_output=True, text=True)


def cleaned_python_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    return env


def adb_serial() -> str:
    result = run(["adb", "get-serialno"], cwd=ROOT_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to query adb serial: {result.stderr.strip()}")
    serial = result.stdout.strip()
    if not serial or serial == "unknown":
        raise RuntimeError("adb returned an empty or unknown serial. Remediation: reconnect the device and rerun.")
    return serial


def git_commit() -> str:
    result = run(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to resolve git commit: {result.stderr.strip()}")
    return result.stdout.strip()


def capture_adb_dump(command: list[str], failure_hint: str) -> str:
    result = run(["adb", "shell", *command], cwd=ROOT_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"{failure_hint}: {result.stderr.strip()}")
    return result.stdout


def capture_thermalservice() -> str:
    return capture_adb_dump(["dumpsys", "thermalservice"], "Failed to capture dumpsys thermalservice")


def capture_batterystats() -> str:
    return capture_adb_dump(["dumpsys", "batterystats", "--charged"], "Failed to capture dumpsys batterystats --charged")


def parse_thermalservice_output(text: str) -> dict[str, Any]:
    status_match = re.search(r"Thermal Status:\s*(?P<status>\d+)", text)
    if not status_match:
        raise ValueError("Failed to parse thermal status from dumpsys thermalservice output.")
    section_anchor = "Current temperatures from HAL:"
    start = text.find(section_anchor)
    if start < 0:
        section_anchor = "Cached temperatures:"
        start = text.find(section_anchor)
    if start < 0:
        raise ValueError("Failed to find temperature section in dumpsys thermalservice output.")
    section = text[start:]
    entries = re.findall(
        r"Temperature\{mValue=(?P<value>[-+0-9.]+), mType=(?P<type>\d+), mName=(?P<name>[^,]+), mStatus=(?P<status>\d+)\}",
        section,
    )
    if not entries:
        raise ValueError("Failed to parse any temperature entries from dumpsys thermalservice output.")
    categorized: dict[str, list[float]] = {
        "cpu": [],
        "gpu": [],
        "npu": [],
        "skin": [],
        "battery": [],
    }
    for value, sensor_type, name, _status in entries:
        temp = float(value)
        if math.isclose(temp, 0.0):
            continue
        sensor_name = name.lower()
        if sensor_type == "0" or sensor_name.startswith("cpu"):
            categorized["cpu"].append(temp)
        elif sensor_type == "1" or sensor_name.startswith("gpu"):
            categorized["gpu"].append(temp)
        elif sensor_type == "9" or sensor_name.startswith("nsp") or "npu" in sensor_name:
            categorized["npu"].append(temp)
        elif sensor_type == "3" or sensor_name == "skin":
            categorized["skin"].append(temp)
        elif sensor_type == "2" or sensor_name == "battery":
            categorized["battery"].append(temp)
    return {
        "thermal_status": int(status_match.group("status")),
        "max_cpu_c": max(categorized["cpu"]) if categorized["cpu"] else None,
        "max_gpu_c": max(categorized["gpu"]) if categorized["gpu"] else None,
        "max_npu_c": max(categorized["npu"]) if categorized["npu"] else None,
        "skin_c": max(categorized["skin"]) if categorized["skin"] else None,
        "battery_c": max(categorized["battery"]) if categorized["battery"] else None,
    }


def parse_batterystats_summary(text: str) -> dict[str, Any]:
    on_battery_match = re.search(r"currently on battery:\s*(?P<flag>true|false)", text)
    drain_match = re.search(r"Computed drain:\s*(?P<drain>[-+0-9.]+)", text)
    return {
        "currently_on_battery": on_battery_match.group("flag") == "true" if on_battery_match else None,
        "computed_drain_mah": float(drain_match.group("drain")) if drain_match else None,
    }


def run_profile_once(profile_key: str, max_workers: int, skip_install: bool) -> tuple[dict[str, Any], Path]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_graphpilot_stage_profiler.py"),
        "--max-workers",
        str(max_workers),
        "--profiles",
        profile_key,
    ]
    if skip_install:
        cmd.append("--skip-install")
    result = run(cmd, cwd=ROOT_DIR, env=cleaned_python_env())
    if result.returncode != 0:
        raise RuntimeError(
            "GraphPilot sustained-load profile failed.\n"
            f"profile={profile_key}\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(
            f"Sustained-load profile '{profile_key}' produced no summary path. Remediation: inspect profiler stdout."
        )
    summary_path = Path(lines[-1]).resolve()
    payload = load_json(summary_path)
    profiles = payload.get("profiles", [])
    if len(profiles) != 1:
        raise RuntimeError(
            f"Sustained-load profile '{profile_key}' returned {len(profiles)} profile rows; expected exactly 1."
        )
    return profiles[0], summary_path


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(sample["workflow_id"], []).append(sample)

    def summarize_metric(values: list[float | int | None]) -> dict[str, float] | None:
        filtered = [float(value) for value in values if value is not None]
        if not filtered:
            return None
        return {
            "count": float(len(filtered)),
            "mean": statistics.mean(filtered),
            "stddev": statistics.pstdev(filtered) if len(filtered) > 1 else 0.0,
            "min": min(filtered),
            "max": max(filtered),
            "first": filtered[0],
            "last": filtered[-1],
            "drift": filtered[-1] - filtered[0],
        }

    workflow_summary: dict[str, Any] = {}
    for workflow_id, workflow_samples in grouped.items():
        workflow_summary[workflow_id] = {
            "count": len(workflow_samples),
            "variant": workflow_samples[-1]["variant"],
            "warm_latency_ms": summarize_metric([sample["warm_latency_ms"] for sample in workflow_samples]),
            "ttft_ms": summarize_metric([sample.get("ttft_ms") for sample in workflow_samples]),
            "tts_first_audio_ms": summarize_metric(
                [sample.get("tts_first_audio_ms") for sample in workflow_samples]
            ),
            "thermal": {
                "status": summarize_metric([sample["thermal_after"]["thermal_status"] for sample in workflow_samples]),
                "cpu_c": summarize_metric([sample["thermal_after"].get("max_cpu_c") for sample in workflow_samples]),
                "gpu_c": summarize_metric([sample["thermal_after"].get("max_gpu_c") for sample in workflow_samples]),
                "npu_c": summarize_metric([sample["thermal_after"].get("max_npu_c") for sample in workflow_samples]),
                "skin_c": summarize_metric([sample["thermal_after"].get("skin_c") for sample in workflow_samples]),
            },
        }
    global_cpu = [sample["thermal_after"].get("max_cpu_c") for sample in samples if sample["thermal_after"].get("max_cpu_c") is not None]
    global_npu = [sample["thermal_after"].get("max_npu_c") for sample in samples if sample["thermal_after"].get("max_npu_c") is not None]
    global_skin = [sample["thermal_after"].get("skin_c") for sample in samples if sample["thermal_after"].get("skin_c") is not None]
    return {
        "workflow_summary": workflow_summary,
        "global_summary": {
            "sample_count": len(samples),
            "thermal_status_max": max((sample["thermal_after"]["thermal_status"] for sample in samples), default=None),
            "max_cpu_c": max(global_cpu) if global_cpu else None,
            "max_npu_c": max(global_npu) if global_npu else None,
            "max_skin_c": max(global_skin) if global_skin else None,
        },
    }


def upsert_experiment_registry(
    registry_path: Path,
    summary_path: Path,
    payload: dict[str, Any],
    commit: str,
    serial: str,
) -> None:
    registry = load_json(registry_path)
    experiments = registry.setdefault("experiments", [])
    entry = {
        "experiment_id": payload["experiment_id"],
        "workflow_template": "graphpilot_phase7_sustained_load",
        "plan_id": "graphpilot_sustained_load",
        "command": "env -u PYTHONHOME -u PYTHONPATH python3 scripts/run_graphpilot_sustained_load.py",
        "commit": commit,
        "device_state": {"adb_serial": serial},
        "input_manifest": str(summary_path),
        "output_artifacts": [payload["output_dir"]],
        "metrics_path": str(summary_path),
        "logs_path": payload["output_dir"],
        "trace_path": None,
        "verdict": "pass",
        "recorded_at": utc_now(),
    }
    index = next((i for i, item in enumerate(experiments) if item["experiment_id"] == entry["experiment_id"]), None)
    if index is None:
        experiments.append(entry)
    else:
        experiments[index] = entry
    registry["last_updated"] = datetime.now(timezone.utc).date().isoformat()
    write_json(registry_path, registry)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# GraphPilot Sustained Load Report",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Duration seconds: `{payload['target_duration_seconds']}`",
        f"- Actual elapsed seconds: `{payload['actual_elapsed_seconds']:.1f}`",
        f"- Profiles: `{', '.join(payload['profiles'])}`",
        "",
        "## Workflow summary",
        "",
    ]
    for workflow_id, summary in sorted(payload["workflow_summary"].items()):
        warm = summary["warm_latency_ms"]
        ttft = summary["ttft_ms"]
        ttfs = summary["tts_first_audio_ms"]
        thermal = summary["thermal"]
        lines.append(
            f"- `{workflow_id}` count={summary['count']} warm_first_ms={warm['first']:.1f} "
            f"warm_last_ms={warm['last']:.1f} warm_drift_ms={warm['drift']:.1f}"
            + (f" ttft_drift_ms={ttft['drift']:.1f}" if ttft else "")
            + (f" ttfs_drift_ms={ttfs['drift']:.1f}" if ttfs else "")
            + (f" skin_drift_c={thermal['skin_c']['drift']:.2f}" if thermal.get("skin_c") else "")
        )
    lines.extend(
        [
            "",
            "## Power proxy",
            "",
            f"- `currently_on_battery`: {payload['battery_summary_before'].get('currently_on_battery')}",
            f"- `computed_drain_mah_before`: {payload['battery_summary_before'].get('computed_drain_mah')}",
            f"- `computed_drain_mah_after`: {payload['battery_summary_after'].get('computed_drain_mah')}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", default=list(DEFAULT_PROFILES))
    parser.add_argument("--duration-minutes", type=float, default=20.0)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiment-registry", type=Path, default=DEFAULT_EXPERIMENT_REGISTRY)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    serial = adb_serial()
    commit = git_commit()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"graphpilot_sustained_load_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    battery_before_raw = capture_batterystats()
    thermal_bootstrap_raw = capture_thermalservice()
    (output_dir / "batterystats.before.log").write_text(battery_before_raw, encoding="utf-8")
    (output_dir / "thermal.bootstrap.log").write_text(thermal_bootstrap_raw, encoding="utf-8")
    battery_summary_before = parse_batterystats_summary(battery_before_raw)

    target_seconds = max(args.duration_minutes, 0.0) * 60.0
    samples: list[dict[str, Any]] = []
    start_monotonic = time.monotonic()
    cycle_index = 0
    iteration_index = 0
    while True:
        for profile_key in args.profiles:
            if args.max_iterations is not None and iteration_index >= args.max_iterations:
                break
            sample_dir = output_dir / f"sample_{iteration_index:03d}_{profile_key}"
            sample_dir.mkdir(parents=True, exist_ok=False)
            thermal_before_raw = capture_thermalservice()
            (sample_dir / "thermal.before.log").write_text(thermal_before_raw, encoding="utf-8")
            thermal_before = parse_thermalservice_output(thermal_before_raw)
            wall_start = time.monotonic()
            entry, profiler_summary_path = run_profile_once(
                profile_key=profile_key,
                max_workers=args.max_workers,
                skip_install=args.skip_install,
            )
            wall_elapsed_ms = int(round((time.monotonic() - wall_start) * 1000.0))
            thermal_after_raw = capture_thermalservice()
            (sample_dir / "thermal.after.log").write_text(thermal_after_raw, encoding="utf-8")
            thermal_after = parse_thermalservice_output(thermal_after_raw)
            metrics = entry["metrics"]
            workflow_id = metrics.get("workflow_id", entry["stage_id"])
            sample = {
                "sample_index": iteration_index,
                "cycle_index": cycle_index,
                "profile_key": profile_key,
                "workflow_id": workflow_id,
                "variant": entry["variant"],
                "recorded_at": utc_now(),
                "warm_latency_ms": metrics.get("warm_latency_ms"),
                "cold_latency_ms": metrics.get("cold_latency_ms"),
                "ttft_ms": metrics.get("ttft_ms"),
                "tts_first_audio_ms": metrics.get("tts_first_audio_ms"),
                "stage_backends": metrics.get("stage_backends"),
                "wall_elapsed_ms": wall_elapsed_ms,
                "profiler_summary": str(profiler_summary_path),
                "profiler_run_dir": entry["artifacts"]["run_dir"],
                "thermal_before": thermal_before,
                "thermal_after": thermal_after,
                "artifacts": {
                    "sample_dir": str(sample_dir),
                    "thermal_before_log": str(sample_dir / "thermal.before.log"),
                    "thermal_after_log": str(sample_dir / "thermal.after.log"),
                },
            }
            write_json(sample_dir / "sample.json", sample)
            samples.append(sample)
            iteration_index += 1
        cycle_index += 1
        elapsed = time.monotonic() - start_monotonic
        duration_done = elapsed >= target_seconds if target_seconds > 0 else True
        iteration_done = args.max_iterations is not None and iteration_index >= args.max_iterations
        if duration_done or iteration_done:
            break

    battery_after_raw = capture_batterystats()
    thermal_final_raw = capture_thermalservice()
    (output_dir / "batterystats.after.log").write_text(battery_after_raw, encoding="utf-8")
    (output_dir / "thermal.final.log").write_text(thermal_final_raw, encoding="utf-8")
    battery_summary_after = parse_batterystats_summary(battery_after_raw)
    sustained_summary = summarize_samples(samples)

    payload = {
        "experiment_id": output_dir.name,
        "generated_at": utc_now(),
        "device_serial": serial,
        "commit": commit,
        "output_dir": str(output_dir.resolve()),
        "profiles": list(args.profiles),
        "target_duration_seconds": target_seconds,
        "actual_elapsed_seconds": time.monotonic() - start_monotonic,
        "battery_summary_before": battery_summary_before,
        "battery_summary_after": battery_summary_after,
        "samples": samples,
        "workflow_summary": sustained_summary["workflow_summary"],
        "global_summary": sustained_summary["global_summary"],
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, payload)
    write_report(output_dir / "report.md", payload)
    upsert_experiment_registry(args.experiment_registry, summary_path, payload, commit, serial)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
