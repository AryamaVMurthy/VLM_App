#!/usr/bin/env python3
"""Generate paper-facing SVG figures from the FastVLM stream benchmark bundle."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fastvlm_host_edge_diff import parse_event_stream
from legacy_fastvlm_paths import LEGACY_ANALYSIS_ROOT

DEFAULT_PARTITION_JSON = (
    LEGACY_ANALYSIS_ROOT
    / "cases_paper_bundle_20260310_030627/heterogeneous_partition_mapping.json"
)

CPU_COLOR = "#d95f02"
CPU_PARTIAL_COLOR = "#fdb863"
CPU_UNKNOWN_COLOR = "#fee0b6"
NPU_COLOR = "#1b9e77"
TEXT_COLOR = "#222222"
GRID_COLOR = "#cccccc"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        f'text {{ font-family: Arial, sans-serif; fill: {TEXT_COLOR}; }}',
        '</style>',
    ]


def svg_footer() -> list[str]:
    return ["</svg>"]


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_variant_comparison_svg(summary: dict[str, Any]) -> str:
    variants = summary["variants"]
    width = 980
    height = 420
    left = 220
    top = 60
    chart_width = 700
    row_height = 70
    max_runtime = max(
        float(
            variant["metrics"].get("device_runtime_s")
            or variant["metrics"]["host_wall_clock_s"]
        )
        for variant in variants
    )
    lines = svg_header(width, height)
    lines.append(
        f'<text x="{width // 2}" y="28" font-size="20" text-anchor="middle">Stream benchmark comparison</text>'
    )
    lines.append(
        f'<text x="{left}" y="{top - 18}" font-size="12">Device runtime window (s)</text>'
    )
    for index, variant in enumerate(variants):
        y = top + index * row_height
        runtime_s = float(
            variant["metrics"].get("device_runtime_s")
            or variant["metrics"]["host_wall_clock_s"]
        )
        wall = float(variant["metrics"]["host_wall_clock_s"])
        throughput = float(variant["metrics"].get("throughput_img_per_s_host") or 0.0)
        ttft = variant["metrics"].get("first_token_ttft_ms_mean")
        bar_width = 0 if max_runtime <= 0 else chart_width * (runtime_s / max_runtime)
        color = NPU_COLOR if variant["mode"] == "overlap" else CPU_COLOR
        lines.append(
            f'<text x="12" y="{y + 22}" font-size="13">{escape_xml(variant["name"])}</text>'
        )
        lines.append(
            f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="22" fill="{color}" rx="3" />'
        )
        lines.append(
            f'<text x="{left + bar_width + 8:.1f}" y="{y + 16}" font-size="12">{runtime_s:.3f} s</text>'
        )
        lines.append(
            f'<text x="{left}" y="{y + 42}" font-size="11">host wall={wall:.3f}s throughput={throughput:.3f} img/s</text>'
        )
        if ttft is not None:
            lines.append(
                f'<text x="{left + 210}" y="{y + 42}" font-size="11">mean TTFT={float(ttft):.3f} ms</text>'
            )
    lines.extend(svg_footer())
    return "\n".join(lines) + "\n"


def build_hardware_counters_svg(summary: dict[str, Any]) -> str:
    overlap_variants = [
        variant for variant in summary["variants"] if variant["mode"] == "overlap"
    ]
    width = 980
    height = 420
    left = 210
    top = 70
    row_height = 90
    chart_width = 700
    max_wait = max(
        float(variant["metrics"].get("sync_wait_ms_total") or 0.0)
        for variant in overlap_variants
    ) or 1.0
    lines = svg_header(width, height)
    lines.append(
        f'<text x="{width // 2}" y="28" font-size="20" text-anchor="middle">Overlap hardware counters</text>'
    )
    for index, variant in enumerate(overlap_variants):
        metrics = variant["metrics"]
        y = top + index * row_height
        wait_total = float(metrics.get("sync_wait_ms_total") or 0.0)
        wait_width = chart_width * (wait_total / max_wait)
        cpu_summary = metrics.get("cpu_summary") or {}
        process_cpu_util = cpu_summary.get("avg_cpu_util_percent_total_capacity")
        device_cpu_summary = metrics.get("device_cpu_summary") or {}
        device_cpu_util = device_cpu_summary.get("busy_percent")
        npu_active = metrics.get("npu_active_ratio")
        lines.append(
            f'<text x="12" y="{y + 16}" font-size="13">{escape_xml(variant["name"])}</text>'
        )
        lines.append(
            f'<rect x="{left}" y="{y}" width="{wait_width:.1f}" height="18" fill="{CPU_COLOR}" rx="3" />'
        )
        lines.append(
            f'<text x="{left + wait_width + 8:.1f}" y="{y + 14}" font-size="11">sync/wait={wait_total:.1f} ms</text>'
        )
        lines.append(
            f'<text x="{left}" y="{y + 40}" font-size="11">proc cpu={process_cpu_util if process_cpu_util is not None else "n/a"}%</text>'
        )
        lines.append(
            f'<text x="{left + 210}" y="{y + 40}" font-size="11">device cpu={device_cpu_util if device_cpu_util is not None else "n/a"}%</text>'
        )
        lines.append(
            f'<text x="{left + 420}" y="{y + 40}" font-size="11">npu active={npu_active if npu_active is not None else "n/a"}</text>'
        )
        lines.append(
            f'<text x="{left}" y="{y + 58}" font-size="11">handoff bytes={int(metrics.get("handoff_export_total_bytes") or 0)}</text>'
        )
        lines.append(
            f'<text x="{left + 210}" y="{y + 58}" font-size="11">copies={int(metrics.get("handoff_export_event_count") or 0)}/{int(metrics.get("handoff_import_event_count") or 0)}</text>'
        )
        lines.append(
            f'<text x="{left + 420}" y="{y + 58}" font-size="11">budget counts={escape_xml(json.dumps(metrics.get("budget_counts") or {}, sort_keys=True))}</text>'
        )
    lines.extend(svg_footer())
    return "\n".join(lines) + "\n"


def build_partition_map_svg(partition_map: dict[str, Any]) -> str:
    width = 1120
    row_height = 52
    top = 70
    left = 20
    label_width = 260
    total_subgraphs = sum(len(entry.get("subgraphs", [])) for entry in partition_map.values())
    chart_width = 800
    height = top + max(1, total_subgraphs) * row_height + 40
    max_subgraphs = max(len(entry.get("subgraphs", [])) for entry in partition_map.values())
    lines = svg_header(width, height)
    lines.append(
        f'<text x="{width // 2}" y="28" font-size="20" text-anchor="middle">Corrected FastVLM partition map</text>'
    )
    y = top
    for submodel, entry in partition_map.items():
        subgraphs = entry.get("subgraphs", [])
        lines.append(
            f'<text x="{left}" y="{y + 18}" font-size="13">{escape_xml(submodel)}</text>'
        )
        subgraph_width = chart_width / max(1, max_subgraphs)
        for index, subgraph in enumerate(subgraphs):
            backend = str(subgraph.get("runtime_backend", ""))
            color = CPU_UNKNOWN_COLOR
            if backend.startswith("NPU"):
                color = NPU_COLOR
            elif backend.startswith("CPU_PARTIAL"):
                color = CPU_PARTIAL_COLOR
            elif backend.startswith("CPU"):
                color = CPU_COLOR
            x = left + label_width + index * subgraph_width
            lines.append(
                f'<rect x="{x:.1f}" y="{y}" width="{subgraph_width - 12:.1f}" height="24" fill="{color}" rx="4" />'
            )
            lines.append(
                f'<text x="{x + 6:.1f}" y="{y + 16}" font-size="11">sg{index}: {escape_xml(backend)}</text>'
            )
            limitation = str(subgraph.get("mapping_limitation", ""))
            if limitation:
                lines.append(
                    f'<text x="{x + 6:.1f}" y="{y + 40}" font-size="9">{escape_xml(limitation[:68])}</text>'
                )
        y += row_height
    lines.extend(svg_footer())
    return "\n".join(lines) + "\n"


def build_timeline_svg(stream_summary: dict[str, Any], overlap_log: Path) -> str:
    parsed = parse_event_stream(overlap_log)
    events = parsed.events
    request_ids = []
    requests_by_id: dict[str, dict[str, Any]] = {}
    for event in events:
        request_id = event.get("request_id")
        if not request_id:
            continue
        request_id = str(request_id)
        record = requests_by_id.setdefault(request_id, {"request_id": request_id})
        if request_id not in request_ids:
            request_ids.append(request_id)
        event_type = event.get("type")
        if event_type in {
            "PREPARE_START",
            "PREFILL_START",
            "DECODE_START",
            "DECODE_DONE",
            "FIRST_TOKEN",
        }:
            record[event_type] = int(event.get("timestamp_unix_nanos", 0))
        if event_type == "PREPARE_DONE":
            record["PREPARE_DONE"] = int(event.get("timestamp_unix_nanos", 0))
        if event_type == "PREFILL_DONE":
            record["PREFILL_DONE"] = int(event.get("timestamp_unix_nanos", 0))
        if event_type == "DECODE_DONE":
            record["DECODE_DURATION_MS"] = float(event.get("duration_ms", 0.0))
    timeline_start = min(
        int(record["PREFILL_START"])
        for record in requests_by_id.values()
        if "PREFILL_START" in record
    )
    timeline_end = max(
        int(record["DECODE_DONE"])
        for record in requests_by_id.values()
        if "DECODE_DONE" in record
    )
    duration_ms = (timeline_end - timeline_start) / 1e6
    width = 1180
    left = 140
    top = 60
    chart_width = 980
    row_height = 48
    height = top + row_height * len(request_ids) + 50
    lines = svg_header(width, height)
    lines.append(
        f'<text x="{width // 2}" y="28" font-size="20" text-anchor="middle">CPU/NPU overlap timeline</text>'
    )
    for request_index, request_id in enumerate(request_ids):
        record = requests_by_id[request_id]
        y = top + request_index * row_height
        lines.append(
            f'<text x="12" y="{y + 18}" font-size="12">{escape_xml(request_id)}</text>'
        )
        lines.append(
            f'<line x1="{left}" y1="{y + 32}" x2="{left + chart_width}" y2="{y + 32}" stroke="{GRID_COLOR}" stroke-width="1" />'
        )
        for label, color in (
            ("PREPARE", "#7570b3"),
            ("PREFILL", NPU_COLOR),
            ("DECODE", CPU_COLOR),
        ):
            start_key = f"{label}_START"
            end_key = f"{label}_DONE"
            if start_key not in record or end_key not in record:
                continue
            start_ns = int(record[start_key])
            end_ns = int(record[end_key])
            start_x = left + chart_width * ((start_ns - timeline_start) / (timeline_end - timeline_start))
            width_x = chart_width * ((end_ns - start_ns) / (timeline_end - timeline_start))
            lines.append(
                f'<rect x="{start_x:.1f}" y="{y}" width="{max(width_x, 1.0):.1f}" height="18" fill="{color}" rx="3" />'
            )
        if "FIRST_TOKEN" in record:
            x = left + chart_width * ((int(record["FIRST_TOKEN"]) - timeline_start) / (timeline_end - timeline_start))
            lines.append(
                f'<line x1="{x:.1f}" y1="{y - 2}" x2="{x:.1f}" y2="{y + 24}" stroke="#000000" stroke-width="2" />'
            )
    lines.append(
        f'<text x="{left}" y="{height - 16}" font-size="11">total window={duration_ms:.1f} ms</text>'
    )
    lines.extend(svg_footer())
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-summary", type=Path, required=True)
    parser.add_argument("--partition-json", type=Path, default=DEFAULT_PARTITION_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = load_json(args.stream_summary)
    output_dir = args.stream_summary.parent
    partition_map = load_json(args.partition_json)

    (output_dir / "variant_comparison.svg").write_text(
        build_variant_comparison_svg(summary), encoding="utf-8"
    )
    (output_dir / "hardware_counters.svg").write_text(
        build_hardware_counters_svg(summary), encoding="utf-8"
    )
    (output_dir / "partition_map.svg").write_text(
        build_partition_map_svg(partition_map), encoding="utf-8"
    )

    adaptive_variant = next(
        (
            variant
            for variant in summary["variants"]
            if variant["mode"] == "overlap" and "adaptive" in variant["name"]
        ),
        None,
    )
    if adaptive_variant is None:
        raise RuntimeError("Missing adaptive overlap variant in stream summary.")
    overlap_log = Path(adaptive_variant["metrics"]["log_path"])
    (output_dir / "overlap_timeline.svg").write_text(
        build_timeline_svg(summary, overlap_log), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
