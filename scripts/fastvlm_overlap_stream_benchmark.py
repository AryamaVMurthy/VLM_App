#!/usr/bin/env python3
"""Run a small FastVLM image-stream benchmark for the paper hardware story."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fastvlm_cases_benchmark import extract_run_log_path, parse_run_log
from fastvlm_host_edge_diff import parse_event_stream

DEFAULT_OVERLAP_RUNNER = SCRIPT_DIR / "run_fastvlm_litert_overlap_adb.sh"
DEFAULT_SEQUENTIAL_RUNNER = SCRIPT_DIR / "run_fastvlm_litert_npu_adb.sh"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR.parent / "artifacts" / "analysis"
DEFAULT_IMAGES = [
    "IMAGES/download.jpeg",
    "IMAGES/images.jpeg",
    "IMAGES/person.jpeg",
]
DEFAULT_PROMPT = "Describe this image in three short sentences with concrete visual details."
DEFAULT_SMART_ENV = {
    "LITERT_LM_PRUNING_PROMPT_ATTENTION_TOP_K": "4",
    "LITERT_LM_PRUNING_PROMPT_ATTENTION_LOGIT_SCALE": "6.0",
    "LITERT_LM_PRUNING_LOCAL_REFINEMENT_MIN_PROMPT_GAIN": "0.10",
    "LITERT_LM_PRUNING_MAX_LOCAL_REFINEMENT_FRACTION": "0.015625",
    "LITERT_LM_PRUNING_MAX_LOCAL_REFINEMENT_SALIENCE_DROP": "0.0",
}


@dataclass(frozen=True)
class VariantConfig:
    name: str
    mode: str
    strategy: str
    max_visual_tokens: int
    max_output_tokens: int
    adaptive_budgets: str = ""
    budget_controller_answer_mode: str = "none"
    forward_env: dict[str, str] | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_battery_dumpsys(text: str) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = key.strip().lower().replace(" ", "_")
        state[normalized_key] = value.strip()
    ac_powered = state.get("ac_powered")
    charge_counter = state.get("charge_counter")
    voltage = state.get("voltage")
    available = (
        ac_powered == "false"
        and charge_counter is not None
        and charge_counter not in {"", "0"}
        and voltage is not None
        and voltage not in {"", "0"}
    )
    if available:
        state["power_estimate_available"] = True
    else:
        reason = "missing_charge_counter_or_voltage"
        if ac_powered == "true":
            reason = "device_ac_powered"
        state["power_estimate_available"] = False
        state["power_estimate_unavailable_reason"] = reason
    return state


def adb_shell(command: str) -> str:
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        ["adb", "shell", command],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"adb shell failed: cmd={command!r} rc={result.returncode} stderr={result.stderr.strip()}"
        )
    return result.stdout


def capture_battery_state() -> dict[str, Any]:
    return parse_battery_dumpsys(adb_shell("dumpsys battery"))


def capture_proc_stat_snapshot() -> dict[str, int]:
    output = adb_shell("cat /proc/stat | head -n 1")
    fields = output.strip().split()
    if len(fields) < 8 or fields[0] != "cpu":
        raise RuntimeError(f"Unexpected /proc/stat format: {output!r}")
    values = [int(field) for field in fields[1:8]]
    user, nice, system, idle, iowait, irq, softirq = values
    busy = user + nice + system + irq + softirq
    total = busy + idle + iowait
    return {
        "busy": busy,
        "idle": idle,
        "iowait": iowait,
        "total": total,
    }


def summarize_proc_stat_delta(
    start: dict[str, int], end: dict[str, int]
) -> dict[str, Any]:
    busy_delta = end["busy"] - start["busy"]
    total_delta = end["total"] - start["total"]
    idle_delta = end["idle"] - start["idle"]
    iowait_delta = end["iowait"] - start["iowait"]
    if total_delta <= 0:
        return {
            "available": False,
            "reason": "non_positive_total_delta",
            "busy_ticks": busy_delta,
            "total_ticks": total_delta,
            "idle_ticks": idle_delta,
            "iowait_ticks": iowait_delta,
        }
    return {
        "available": True,
        "busy_ticks": busy_delta,
        "total_ticks": total_delta,
        "idle_ticks": idle_delta,
        "iowait_ticks": iowait_delta,
        "busy_percent": 100.0 * busy_delta / total_delta,
        "idle_percent": 100.0 * idle_delta / total_delta,
        "iowait_percent": 100.0 * iowait_delta / total_delta,
    }


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.mean(values))


def median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def parse_overlap_metrics(log_path: Path) -> dict[str, Any]:
    parsed = parse_event_stream(log_path)
    events = parsed.events
    by_type: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_type.setdefault(str(event.get("type", "")), []).append(event)

    def durations(event_type: str, key: str = "duration_ms") -> list[float]:
        values: list[float] = []
        for event in by_type.get(event_type, []):
            value = event.get(key)
            if value is not None:
                values.append(float(value))
        return values

    prefill_start_ns = [
        int(event["timestamp_unix_nanos"])
        for event in by_type.get("PREFILL_START", [])
        if event.get("timestamp_unix_nanos") is not None
    ]
    decode_done_ns = [
        int(event["timestamp_unix_nanos"])
        for event in by_type.get("DECODE_DONE", [])
        if event.get("timestamp_unix_nanos") is not None
    ]
    event_window_s = None
    if prefill_start_ns and decode_done_ns:
        event_window_s = (max(decode_done_ns) - min(prefill_start_ns)) / 1e9

    first_token_ttft_ms = [
        float(event["ttft_ms"])
        for event in by_type.get("FIRST_TOKEN", [])
        if event.get("ttft_ms") is not None
    ]
    first_token_decode_latency_ms = [
        float(event["decode_first_token_latency_ms"])
        for event in by_type.get("FIRST_TOKEN", [])
        if event.get("decode_first_token_latency_ms") is not None
    ]
    stall_ms = [
        float(event["cpu_stall_before_next_decode_ms"])
        for event in by_type.get("OVERLAP_ANALYSIS", [])
        if event.get("cpu_stall_before_next_decode_ms") is not None
    ]
    budget_counts: dict[str, int] = {}
    for event in by_type.get("BUDGET_DECISION", []):
        budget_key = str(event.get("selected_budget"))
        budget_counts[budget_key] = budget_counts.get(budget_key, 0) + 1
    handoff_export_buffers = [
        int(event["kv_cache_buffer_count"])
        for event in by_type.get("HANDOFF_EXPORT", [])
        if event.get("kv_cache_buffer_count") is not None
    ]
    handoff_import_buffers = [
        int(event["kv_cache_buffer_count"])
        for event in by_type.get("HANDOFF_IMPORT", [])
        if event.get("kv_cache_buffer_count") is not None
    ]
    handoff_export_bytes = [
        int(event["kv_cache_total_bytes"])
        for event in by_type.get("HANDOFF_EXPORT", [])
        if event.get("kv_cache_total_bytes") is not None
    ]
    handoff_import_bytes = [
        int(event["kv_cache_total_bytes"])
        for event in by_type.get("HANDOFF_IMPORT", [])
        if event.get("kv_cache_total_bytes") is not None
    ]
    processed_token_counts = [
        int(event["processed_token_count"])
        for event in by_type.get("HANDOFF_EXPORT", [])
        if event.get("processed_token_count") is not None
    ]
    cpu_summary = (by_type.get("PROCESS_CPU_SUMMARY") or [None])[-1]
    responses = [event for event in by_type.get("OVERLAP_RESPONSE", [])]
    unique_outputs: list[str] = []
    seen_outputs: set[str] = set()
    for event in responses:
        text = str(event.get("text", ""))
        if text in seen_outputs:
            continue
        seen_outputs.add(text)
        unique_outputs.append(text)

    total_prefill_ms = sum(durations("PREFILL_DONE"))
    total_decode_ms = sum(durations("DECODE_DONE"))
    serial_stage_sum_s = (total_prefill_ms + total_decode_ms) / 1000.0

    return {
        "log_path": str(log_path),
        "recovered_split_events": parsed.recovered_split_events,
        "event_window_s": event_window_s,
        "response_count": len(responses),
        "prepare_ms_mean": mean_or_none(durations("PREPARE_DONE")),
        "prepare_queue_wait_ms_mean": mean_or_none(durations("PREPARE_QUEUE_WAIT")),
        "prefill_ms_mean": mean_or_none(durations("PREFILL_DONE")),
        "decode_ms_mean": mean_or_none(durations("DECODE_DONE")),
        "first_token_ttft_ms_mean": mean_or_none(first_token_ttft_ms),
        "first_token_ttft_ms_median": median_or_none(first_token_ttft_ms),
        "first_token_decode_latency_ms_mean": mean_or_none(
            first_token_decode_latency_ms
        ),
        "sync_wait_ms_total": sum(stall_ms) + sum(durations("PREPARE_QUEUE_WAIT")),
        "cpu_stall_ms_total": sum(stall_ms),
        "npu_active_ratio": (
            sum(durations("PREFILL_DONE")) / (event_window_s * 1000.0)
            if event_window_s and event_window_s > 0.0
            else None
        ),
        "npu_idle_ratio": (
            1.0 - (sum(durations("PREFILL_DONE")) / (event_window_s * 1000.0))
            if event_window_s and event_window_s > 0.0
            else None
        ),
        "serial_stage_sum_s": serial_stage_sum_s,
        "overlap_pipeline_speedup": (
            serial_stage_sum_s / event_window_s
            if event_window_s and event_window_s > 0.0
            else None
        ),
        "handoff_export_event_count": len(by_type.get("HANDOFF_EXPORT", [])),
        "handoff_import_event_count": len(by_type.get("HANDOFF_IMPORT", [])),
        "handoff_export_buffer_count_total": sum(handoff_export_buffers),
        "handoff_export_buffer_count_mean": mean_or_none(handoff_export_buffers),
        "handoff_import_buffer_count_total": sum(handoff_import_buffers),
        "handoff_import_buffer_count_mean": mean_or_none(handoff_import_buffers),
        "handoff_export_total_bytes": sum(handoff_export_bytes),
        "handoff_import_total_bytes": sum(handoff_import_bytes),
        "processed_token_count_total": sum(processed_token_counts),
        "budget_counts": budget_counts,
        "cpu_summary": cpu_summary,
        "example_outputs": unique_outputs[:3],
    }


def run_subprocess(command: list[str], *, env_overrides: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        command,
        cwd=SCRIPT_DIR.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def build_variants(fixed_budget: int, max_output_tokens: int) -> list[VariantConfig]:
    return [
        VariantConfig(
            name="sequential_npu_npu_uniform",
            mode="sequential",
            strategy="uniform",
            max_visual_tokens=fixed_budget,
            max_output_tokens=max_output_tokens,
        ),
        VariantConfig(
            name="overlap_cpu_decode_npu_prefill_uniform",
            mode="overlap",
            strategy="uniform",
            max_visual_tokens=fixed_budget,
            max_output_tokens=max_output_tokens,
        ),
        VariantConfig(
            name="overlap_cpu_decode_npu_prefill_prompt_conditioned_v2",
            mode="overlap",
            strategy="prompt_conditioned_v2",
            max_visual_tokens=fixed_budget,
            max_output_tokens=max_output_tokens,
            forward_env=DEFAULT_SMART_ENV,
        ),
        VariantConfig(
            name="overlap_cpu_decode_npu_prefill_adaptive_prompt_conditioned_v2",
            mode="overlap",
            strategy="prompt_conditioned_v2",
            max_visual_tokens=fixed_budget,
            max_output_tokens=max_output_tokens,
            adaptive_budgets="32,64,128",
            budget_controller_answer_mode="long",
            forward_env=DEFAULT_SMART_ENV,
        ),
    ]


def run_overlap_variant(
    config: VariantConfig,
    *,
    sequence: list[str],
    prompt: str,
    runner: Path,
    skip_push: bool,
) -> dict[str, Any]:
    cpu_start = capture_proc_stat_snapshot()
    command = ["bash", str(runner)]
    for image in sequence:
        command.extend(["--image", image])
    command.extend(
        [
            "--prompt",
            prompt,
            "--max-output-tokens",
            str(config.max_output_tokens),
            "--max-visual-tokens",
            str(config.max_visual_tokens),
            "--visual-token-pruning-strategy",
            config.strategy,
            "--prepare-queue-size",
            "4",
            "--skip-build",
            "1",
            "--skip-push",
            "1" if skip_push else "0",
            "--skip-input-sync",
            "0",
            "--event-mode",
            "1",
        ]
    )
    if config.adaptive_budgets:
        command.extend(
            [
                "--adaptive-visual-token-budgets",
                config.adaptive_budgets,
                "--budget-controller-answer-mode",
                config.budget_controller_answer_mode,
            ]
        )
    start = time.monotonic()
    result = run_subprocess(command, env_overrides=config.forward_env)
    wall_clock_s = time.monotonic() - start
    cpu_end = capture_proc_stat_snapshot()
    if result.returncode != 0:
        raise RuntimeError(
            f"Overlap variant failed: variant={config.name} rc={result.returncode} stdout={result.stdout.strip()} stderr={result.stderr.strip()}"
        )
    log_path = extract_run_log_path(result.stdout)
    metrics = parse_overlap_metrics(log_path)
    metrics["host_wall_clock_s"] = wall_clock_s
    metrics["device_runtime_s"] = metrics["event_window_s"]
    metrics["throughput_img_per_s_host"] = len(sequence) / wall_clock_s
    metrics["throughput_img_per_s_event_window"] = (
        len(sequence) / metrics["event_window_s"]
        if metrics["event_window_s"]
        else None
    )
    metrics["device_cpu_summary"] = summarize_proc_stat_delta(cpu_start, cpu_end)
    return metrics


def run_sequential_variant(
    config: VariantConfig,
    *,
    sequence: list[str],
    prompt: str,
    runner: Path,
) -> dict[str, Any]:
    cpu_start = capture_proc_stat_snapshot()
    rows: list[dict[str, Any]] = []
    start = time.monotonic()
    sequential_push_done = False
    for image in sequence:
        push_result = run_subprocess(
            ["adb", "push", str((SCRIPT_DIR.parent / image).resolve()), "/data/local/tmp/vlm_phase1/image.jpg"]
        )
        if push_result.returncode != 0:
            raise RuntimeError(
                f"adb push failed for {image}: rc={push_result.returncode} stderr={push_result.stderr.strip()}"
            )
        command = [
            "bash",
            str(runner),
            "--image",
            image,
            "--prompt",
            prompt,
            "--max-output-tokens",
            str(config.max_output_tokens),
            "--max-visual-tokens",
            str(config.max_visual_tokens),
            "--visual-token-pruning-strategy",
            config.strategy,
            "--benchmark",
            "1",
            "--event-mode",
            "1",
            "--skip-build",
            "1",
            "--skip-push",
            "1" if sequential_push_done else "0",
        ]
        if config.forward_env:
            for key, value in config.forward_env.items():
                command.extend(["--forward-env", f"{key}={value}"])
        run_start = time.monotonic()
        result = run_subprocess(command)
        row_wall_clock_s = time.monotonic() - run_start
        if result.returncode != 0:
            raise RuntimeError(
                f"Sequential variant failed: variant={config.name} image={image} rc={result.returncode} stdout={result.stdout.strip()} stderr={result.stderr.strip()}"
            )
        sequential_push_done = True
        log_path = extract_run_log_path(result.stdout)
        metrics = parse_run_log(log_path)
        rows.append(
            {
                "image": image,
                "wall_clock_s": row_wall_clock_s,
                "metrics": metrics,
            }
        )
    total_wall_clock_s = time.monotonic() - start
    cpu_end = capture_proc_stat_snapshot()
    prefill_ms = [
        float(row["metrics"].get("prefill_latency_us") or 0.0) / 1000.0
        for row in rows
    ]
    decode_ms = [
        float(row["metrics"].get("decode_latency_us") or 0.0) / 1000.0
        for row in rows
    ]
    ttft_ms = [float(row["metrics"].get("ttft_ms") or 0.0) for row in rows]
    example_outputs: list[str] = []
    seen_outputs: set[str] = set()
    for row in rows:
        text = str(row["metrics"].get("caption") or "")
        if text in seen_outputs:
            continue
        seen_outputs.add(text)
        example_outputs.append(text)
    return {
        "runs": rows,
        "host_wall_clock_s": total_wall_clock_s,
        "device_runtime_s": sum(prefill_ms) / 1000.0 + sum(decode_ms) / 1000.0,
        "throughput_img_per_s_host": len(sequence) / total_wall_clock_s,
        "prefill_ms_mean": mean_or_none(prefill_ms),
        "decode_ms_mean": mean_or_none(decode_ms),
        "first_token_ttft_ms_mean": mean_or_none(ttft_ms),
        "device_cpu_summary": summarize_proc_stat_delta(cpu_start, cpu_end),
        "example_outputs": example_outputs[:3],
    }


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# FastVLM Stream Hardware Story",
        "",
        f"- Timestamp: {summary['timestamp_utc']}",
        f"- Prompt: {summary['prompt']}",
        f"- Requests: {summary['num_requests']}",
        f"- Sequence: {', '.join(summary['sequence'])}",
        "",
        "## Variants",
        "",
    ]
    for variant in summary["variants"]:
        lines.append(f"### {variant['name']}")
        lines.append("")
        lines.append(f"- Mode: {variant['mode']}")
        lines.append(f"- Strategy: {variant['strategy']}")
        lines.append(f"- Host wall clock: {variant['metrics']['host_wall_clock_s']:.3f} s")
        device_runtime_s = variant["metrics"].get("device_runtime_s")
        if device_runtime_s is not None:
            lines.append(f"- Device runtime window: {device_runtime_s:.3f} s")
        if variant["mode"] == "overlap":
            metrics = variant["metrics"]
            lines.append(f"- Event window: {metrics['event_window_s']:.3f} s")
            if metrics.get("serial_stage_sum_s") is not None:
                lines.append(
                    f"- Same-backend serial stage sum: {metrics['serial_stage_sum_s']:.3f} s"
                )
            if metrics.get("overlap_pipeline_speedup") is not None:
                lines.append(
                    f"- Pipeline speedup vs non-overlapped same-backend schedule: {metrics['overlap_pipeline_speedup']:.3f}x"
                )
            lines.append(
                f"- Mean TTFT: {metrics['first_token_ttft_ms_mean']:.3f} ms"
            )
            lines.append(
                f"- CPU stall total: {metrics['cpu_stall_ms_total']:.3f} ms"
            )
            lines.append(
                f"- NPU active ratio: {metrics['npu_active_ratio']:.3f}"
                if metrics["npu_active_ratio"] is not None
                else "- NPU active ratio: unavailable"
            )
            cpu_summary = metrics.get("cpu_summary") or {}
            if cpu_summary.get("cpu_summary_available"):
                lines.append(
                    "- Avg CPU util (total capacity): "
                    f"{cpu_summary['avg_cpu_util_percent_total_capacity']:.3f}%"
                )
            else:
                lines.append(
                    "- Avg CPU util (total capacity): unavailable"
                )
            lines.append(
                f"- Handoff export bytes: {metrics['handoff_export_total_bytes']}"
            )
            lines.append(
                f"- Handoff exports/imports: {metrics['handoff_export_event_count']}/{metrics['handoff_import_event_count']}"
            )
            lines.append(
                f"- Handoff processed token count: {metrics['processed_token_count_total']}"
            )
            lines.append(
                f"- Budget counts: {json.dumps(metrics['budget_counts'], sort_keys=True)}"
            )
        else:
            lines.append(
                f"- Mean TTFT: {variant['metrics']['first_token_ttft_ms_mean']:.3f} ms"
            )
        device_cpu_summary = variant["metrics"].get("device_cpu_summary") or {}
        if device_cpu_summary.get("available"):
            lines.append(
                "- Device CPU busy/iowait: "
                f"{device_cpu_summary['busy_percent']:.3f}% / {device_cpu_summary['iowait_percent']:.3f}%"
            )
        else:
            lines.append("- Device CPU busy/iowait: unavailable")
        for index, output in enumerate(variant["metrics"]["example_outputs"]):
            lines.append(f"- Output {index}: {output}")
        lines.append("")
    lines.extend(
        [
            "## Power availability",
            "",
            f"- Start battery snapshot: {json.dumps(summary['battery_start'], sort_keys=True)}",
            f"- End battery snapshot: {json.dumps(summary['battery_end'], sort_keys=True)}",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlap-runner", type=Path, default=DEFAULT_OVERLAP_RUNNER)
    parser.add_argument("--sequential-runner", type=Path, default=DEFAULT_SEQUENTIAL_RUNNER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--fixed-budget", type=int, default=64)
    parser.add_argument("--max-output-tokens", type=int, default=48)
    parser.add_argument("--stream-repeats", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sequence = DEFAULT_IMAGES * args.stream_repeats
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"overlap_hardware_story_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    battery_start = capture_battery_state()
    variants: list[dict[str, Any]] = []
    overlap_push_done = False
    for config in build_variants(args.fixed_budget, args.max_output_tokens):
        if config.mode == "overlap":
            metrics = run_overlap_variant(
                config,
                sequence=sequence,
                prompt=args.prompt,
                runner=args.overlap_runner,
                skip_push=overlap_push_done,
            )
            overlap_push_done = True
        else:
            metrics = run_sequential_variant(
                config,
                sequence=sequence,
                prompt=args.prompt,
                runner=args.sequential_runner,
            )
        variants.append(
            {
                "name": config.name,
                "mode": config.mode,
                "strategy": config.strategy,
                "adaptive_budgets": config.adaptive_budgets,
                "metrics": metrics,
            }
        )
    battery_end = capture_battery_state()

    summary = {
        "timestamp_utc": utc_now(),
        "prompt": args.prompt,
        "num_requests": len(sequence),
        "sequence": sequence,
        "battery_start": battery_start,
        "battery_end": battery_end,
        "variants": variants,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
