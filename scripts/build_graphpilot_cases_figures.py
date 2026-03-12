#!/usr/bin/env python3
"""Build checkpoint-pinned figures and LaTeX tables for the GraphPilot CASES paper."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
ARTIFACT_ROOT = ROOT_DIR / "artifacts" / "graphpilot_edge"
DEFAULT_OUTPUT_ROOT = ARTIFACT_ROOT / "papers"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def require_checkpoint_paths(manifest: dict[str, Any]) -> dict[str, Path]:
    canonical = manifest.get("canonical_evidence_paths")
    if not isinstance(canonical, dict):
        raise ValueError("Checkpoint manifest is missing canonical_evidence_paths.")
    required = (
        "artifact_pack_summary",
        "calibration_summary",
        "characterization_summary",
        "experiment_summary",
        "workload_registry",
    )
    resolved: dict[str, Path] = {}
    for key in required:
        raw = canonical.get(key)
        if not raw:
            raise FileNotFoundError(
                f"Checkpoint manifest is missing canonical_evidence_paths.{key}. "
                f"Remediation: rebuild the checkpoint with {key} pinned."
            )
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint path for {key} does not exist: {path}. "
                "Remediation: regenerate the canonical evidence set."
            )
        resolved[key] = path
    return resolved


def load_optional_json_from_manifest(manifest: dict[str, Any], key: str) -> dict[str, Any] | None:
    canonical = manifest.get("canonical_evidence_paths") or {}
    raw = canonical.get(key)
    if not raw:
        return None
    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint path for {key} does not exist: {path}. "
            "Remediation: regenerate the canonical checkpoint."
        )
    return load_json(path)


def write_bar_svg(path: Path, title: str, rows: list[tuple[str, float]], x_label: str) -> None:
    width = 980
    height = 120 + 52 * max(len(rows), 1)
    left = 260
    max_value = max((value for _, value in rows), default=1.0)
    scale = 620.0 / max(max_value, 1.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="24" y="36" font-size="24" font-family="monospace">{title}</text>',
        f'<text x="{left}" y="{height - 18}" font-size="14" font-family="monospace">{x_label}</text>',
    ]
    y = 70
    for label, value in rows:
        bar_width = value * scale
        parts.append(f'<text x="20" y="{y + 18}" font-size="14" font-family="monospace">{label}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="24" fill="#4c78a8" />')
        parts.append(
            f'<text x="{left + bar_width + 10:.1f}" y="{y + 18}" font-size="14" font-family="monospace">{value:.1f}</text>'
        )
        y += 46
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_bar_png(path: Path, title: str, rows: list[tuple[str, float]], x_label: str) -> None:
    width = 1100
    height = 140 + 60 * max(len(rows), 1)
    left = 320
    max_value = max((value for _, value in rows), default=1.0)
    scale = 680.0 / max(max_value, 1.0)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((24, 24), title, fill="black", font=font)
    draw.text((left, height - 28), x_label, fill="black", font=font)
    y = 72
    for label, value in rows:
        bar_width = int(round(value * scale))
        draw.text((20, y + 8), label, fill="black", font=font)
        draw.rectangle((left, y, left + bar_width, y + 24), fill="#4c78a8", outline="#1f3552")
        draw.text((left + bar_width + 12, y + 8), f"{value:.1f}", fill="black", font=font)
        y += 52
    image.save(path)


def write_architecture_svg(path: Path, workflow_ids: list[str]) -> None:
    width = 1200
    height = 280
    boxes = []
    x = 60
    stages = ["ASR", "Planner", "VLM / Retrieval", "Responder", "TTS"]
    for label in stages:
        boxes.append(
            f'<rect x="{x}" y="90" width="180" height="70" rx="12" fill="#dce9f9" stroke="#4c78a8" stroke-width="2" />'
        )
        boxes.append(
            f'<text x="{x + 90}" y="132" font-size="18" font-family="monospace" text-anchor="middle">{label}</text>'
        )
        x += 220
    arrows = []
    for arrow_x in (240, 460, 680, 900):
        arrows.append(f'<line x1="{arrow_x}" y1="125" x2="{arrow_x + 40}" y2="125" stroke="#333" stroke-width="3" />')
        arrows.append(f'<polygon points="{arrow_x + 40},125 {arrow_x + 30},119 {arrow_x + 30},131" fill="#333" />')
    workflow_text = ", ".join(workflow_ids) if workflow_ids else "no workflow ids"
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
                '<text x="24" y="34" font-size="24" font-family="monospace">GraphPilot-Edge deployed workflow architecture</text>',
                '<text x="24" y="64" font-size="14" font-family="monospace">Primary real-device workflows: '
                + workflow_text
                + "</text>",
                *boxes,
                *arrows,
                "</svg>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_architecture_png(path: Path, workflow_ids: list[str]) -> None:
    width = 1400
    height = 340
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title = "GraphPilot-Edge deployed workflow architecture"
    subtitle = "Primary real-device workflows: " + (", ".join(workflow_ids) if workflow_ids else "none")
    draw.text((24, 24), title, fill="black", font=font)
    draw.text((24, 48), subtitle, fill="black", font=font)
    x = 60
    stages = ["ASR", "Planner", "VLM / Retrieval", "Responder", "TTS"]
    for index, label in enumerate(stages):
        draw.rounded_rectangle((x, 110, x + 200, 180), radius=14, fill="#dce9f9", outline="#4c78a8", width=2)
        draw.text((x + 40, 138), label, fill="black", font=font)
        if index < len(stages) - 1:
            draw.line((x + 200, 145, x + 240, 145), fill="black", width=3)
            draw.polygon([(x + 240, 145), (x + 228, 139), (x + 228, 151)], fill="black")
        x += 260
    image.save(path)


def write_offline_online_split_svg(path: Path) -> None:
    width = 1200
    height = 360
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<text x="24" y="34" font-size="24" font-family="monospace">GraphPilot offline/online split</text>',
        '<rect x="60" y="90" width="460" height="210" rx="16" fill="#e8f0fe" stroke="#4c78a8" stroke-width="2"/>',
        '<text x="90" y="122" font-size="20" font-family="monospace">Offline brain</text>',
        '<rect x="96" y="146" width="170" height="44" rx="10" fill="#dce9f9" stroke="#4c78a8"/>',
        '<text x="181" y="173" font-size="14" text-anchor="middle" font-family="monospace">Profiler</text>',
        '<rect x="300" y="146" width="170" height="44" rx="10" fill="#dce9f9" stroke="#4c78a8"/>',
        '<text x="385" y="173" font-size="14" text-anchor="middle" font-family="monospace">Simulator</text>',
        '<rect x="96" y="214" width="170" height="44" rx="10" fill="#dce9f9" stroke="#4c78a8"/>',
        '<text x="181" y="241" font-size="14" text-anchor="middle" font-family="monospace">Planner</text>',
        '<rect x="300" y="214" width="170" height="44" rx="10" fill="#dce9f9" stroke="#4c78a8"/>',
        '<text x="385" y="241" font-size="14" text-anchor="middle" font-family="monospace">Plan bank</text>',
        '<line x1="520" y1="195" x2="660" y2="195" stroke="#333" stroke-width="4"/>',
        '<polygon points="660,195 646,188 646,202" fill="#333"/>',
        '<rect x="700" y="90" width="440" height="210" rx="16" fill="#f7ead8" stroke="#d17c28" stroke-width="2"/>',
        '<text x="730" y="122" font-size="20" font-family="monospace">Online brain</text>',
        '<rect x="732" y="146" width="170" height="44" rx="10" fill="#fde6c7" stroke="#d17c28"/>',
        '<text x="817" y="173" font-size="14" text-anchor="middle" font-family="monospace">Runtime scheduler</text>',
        '<rect x="936" y="146" width="170" height="44" rx="10" fill="#fde6c7" stroke="#d17c28"/>',
        '<text x="1021" y="173" font-size="14" text-anchor="middle" font-family="monospace">Streaming edges</text>',
        '<rect x="732" y="214" width="170" height="44" rx="10" fill="#fde6c7" stroke="#d17c28"/>',
        '<text x="817" y="241" font-size="14" text-anchor="middle" font-family="monospace">Memory / KV</text>',
        '<rect x="936" y="214" width="170" height="44" rx="10" fill="#fde6c7" stroke="#d17c28"/>',
        '<text x="1021" y="241" font-size="14" text-anchor="middle" font-family="monospace">Thermal switching</text>',
        '</svg>',
    ]
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_offline_online_split_png(path: Path) -> None:
    width = 1400
    height = 420
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((24, 24), "GraphPilot offline/online split", fill="black", font=font)
    draw.rounded_rectangle((60, 100, 560, 330), radius=16, fill="#e8f0fe", outline="#4c78a8", width=2)
    draw.text((88, 118), "Offline brain", fill="black", font=font)
    offline_boxes = [("Profiler", 96, 152), ("Simulator", 314, 152), ("Planner", 96, 228), ("Plan bank", 314, 228)]
    for label, x, y in offline_boxes:
        draw.rounded_rectangle((x, y, x + 180, y + 46), radius=10, fill="#dce9f9", outline="#4c78a8", width=2)
        draw.text((x + 50, y + 15), label, fill="black", font=font)
    draw.line((560, 215, 700, 215), fill="black", width=4)
    draw.polygon([(700, 215), (686, 208), (686, 222)], fill="black")
    draw.rounded_rectangle((740, 100, 1280, 330), radius=16, fill="#f7ead8", outline="#d17c28", width=2)
    draw.text((768, 118), "Online brain", fill="black", font=font)
    online_boxes = [
        ("Runtime scheduler", 772, 152),
        ("Streaming edges", 1010, 152),
        ("Memory / KV", 772, 228),
        ("Thermal switching", 1010, 228),
    ]
    for label, x, y in online_boxes:
        draw.rounded_rectangle((x, y, x + 200, y + 46), radius=10, fill="#fde6c7", outline="#d17c28", width=2)
        draw.text((x + 34, y + 15), label, fill="black", font=font)
    image.save(path)


def write_line_svg(path: Path, title: str, series_map: dict[str, list[tuple[float, float]]], x_label: str, y_label: str) -> None:
    width = 980
    height = 520
    left = 110
    right = 40
    top = 60
    bottom = 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_values = [x for series in series_map.values() for x, _ in series] or [0.0, 1.0]
    y_values = [y for series in series_map.values() for _, y in series] or [0.0, 1.0]
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)
    if max_x == min_x:
        max_x += 1.0
    if max_y == min_y:
        max_y += 1.0
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="24" y="32" font-size="24" font-family="monospace">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="2"/>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 20}" font-size="14" text-anchor="middle" font-family="monospace">{x_label}</text>',
        f'<text x="22" y="{top + plot_h / 2:.1f}" transform="rotate(-90 22,{top + plot_h / 2:.1f})" font-size="14" text-anchor="middle" font-family="monospace">{y_label}</text>',
    ]
    for index, (label, series) in enumerate(series_map.items()):
        color = colors[index % len(colors)]
        points = []
        for x, y in series:
            px = left + ((x - min_x) / (max_x - min_x)) * plot_w
            py = top + plot_h - ((y - min_y) / (max_y - min_y)) * plot_h
            points.append((px, py))
        polyline = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{polyline}"/>')
        lx = left + plot_w - 180
        ly = top + 18 + index * 20
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 18}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx + 26}" y="{ly + 4}" font-size="14" font-family="monospace">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_line_png(path: Path, title: str, series_map: dict[str, list[tuple[float, float]]], x_label: str, y_label: str) -> None:
    width = 1100
    height = 580
    left = 120
    right = 40
    top = 70
    bottom = 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_values = [x for series in series_map.values() for x, _ in series] or [0.0, 1.0]
    y_values = [y for series in series_map.values() for _, y in series] or [0.0, 1.0]
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)
    if max_x == min_x:
        max_x += 1.0
    if max_y == min_y:
        max_y += 1.0
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((24, 24), title, fill="black", font=font)
    draw.line((left, top, left, top + plot_h), fill="black", width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="black", width=2)
    draw.text((left + plot_w // 2 - 40, height - 28), x_label, fill="black", font=font)
    draw.text((20, top + plot_h // 2), y_label, fill="black", font=font)
    for index, (label, series) in enumerate(series_map.items()):
        color = colors[index % len(colors)]
        points = []
        for x, y in series:
            px = left + int(round(((x - min_x) / (max_x - min_x)) * plot_w))
            py = top + plot_h - int(round(((y - min_y) / (max_y - min_y)) * plot_h))
            points.append((px, py))
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        for px, py in points:
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color, outline=color)
        lx = left + plot_w - 210
        ly = top + 18 + index * 22
        draw.line((lx, ly, lx + 20, ly), fill=color, width=3)
        draw.text((lx + 28, ly - 6), label, fill="black", font=font)
    image.save(path)


def write_tables_tex(path: Path, markdown_source: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "% Auto-generated from the canonical artifact-pack paper tables",
                "\\begin{verbatim}",
                markdown_source.read_text(encoding="utf-8"),
                "\\end{verbatim}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_json(args.checkpoint_manifest)
    canonical = require_checkpoint_paths(manifest)
    pack_summary = load_json(canonical["artifact_pack_summary"])
    calibration_summary = load_json(canonical["calibration_summary"])
    characterization_summary = load_json(canonical["characterization_summary"])
    experiment_summary = load_json(canonical["experiment_summary"])
    workload_registry = load_json(canonical["workload_registry"])
    tuning_summary = load_optional_json_from_manifest(manifest, "tuning_summary")

    paper_tables = pack_summary.get("paper_tables")
    if not paper_tables:
        raise FileNotFoundError(
            "Artifact pack summary is missing paper_tables. Remediation: rebuild the artifact pack."
        )
    paper_tables_path = Path(paper_tables)
    if not paper_tables_path.exists():
        raise FileNotFoundError(
            f"Artifact-pack tables file not found: {paper_tables_path}. "
            "Remediation: rebuild the artifact pack."
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"graphpilot_cases_figures_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    workflow_ids = sorted(experiment_summary.get("actual_workflows", {}).keys())
    write_architecture_svg(output_dir / "architecture_overview.svg", workflow_ids)
    write_architecture_png(output_dir / "architecture_overview.png", workflow_ids)
    write_offline_online_split_svg(output_dir / "offline_online_split.svg")
    write_offline_online_split_png(output_dir / "offline_online_split.png")

    workload_counts: dict[str, int] = {}
    for workload in workload_registry.get("workloads", []):
        category = str(workload.get("category", "unknown"))
        workload_counts[category] = workload_counts.get(category, 0) + 1
    workload_rows = sorted(
        ((category, float(count)) for category, count in workload_counts.items()),
        key=lambda item: (-item[1], item[0]),
    )
    write_bar_svg(
        output_dir / "workload_universe_coverage.svg",
        "Checkpoint-pinned workload-universe coverage",
        workload_rows or [("no-data", 0.0)],
        "workload count",
    )
    write_bar_png(
        output_dir / "workload_universe_coverage.png",
        "Checkpoint-pinned workload-universe coverage",
        workload_rows or [("no-data", 0.0)],
        "workload count",
    )

    comparison_rows = [
        (
            item["workflow_id"],
            float(item["latency_delta_ms"]),
        )
        for item in experiment_summary.get("comparisons", [])
        if item.get("latency_delta_ms") is not None
    ]
    write_bar_svg(
        output_dir / "sim_real_calibration.svg",
        "Sim-to-real latency deltas",
        comparison_rows or [("no-data", 0.0)],
        "delta ms",
    )
    write_bar_png(
        output_dir / "sim_real_calibration.png",
        "Sim-to-real latency deltas",
        comparison_rows or [("no-data", 0.0)],
        "delta ms",
    )

    workflow_rows = [
        (workflow_id, float(metrics["warm_latency_ms"]))
        for workflow_id, metrics in sorted(experiment_summary.get("actual_workflows", {}).items())
        if metrics.get("warm_latency_ms") is not None
    ]
    write_bar_svg(
        output_dir / "workflow_primary_results.svg",
        "Primary workflow warm latencies",
        workflow_rows or [("no-data", 0.0)],
        "latency ms",
    )
    write_bar_png(
        output_dir / "workflow_primary_results.png",
        "Primary workflow warm latencies",
        workflow_rows or [("no-data", 0.0)],
        "latency ms",
    )

    continuous_rows = []
    for entry in characterization_summary.get("baseline_comparisons", {}).get("continuous_workloads", []):
        value = entry.get("graphpilot_score_ms")
        if value is not None:
            continuous_rows.append((entry["workload_id"], float(value)))
    write_bar_svg(
        output_dir / "continuous_stream_results.svg",
        "Continuous-stream GraphPilot scores",
        continuous_rows or [("no-data", 0.0)],
        "score ms",
    )
    write_bar_png(
        output_dir / "continuous_stream_results.png",
        "Continuous-stream GraphPilot scores",
        continuous_rows or [("no-data", 0.0)],
        "score ms",
    )

    baseline_rows = []
    for entry in characterization_summary.get("baseline_comparisons", {}).get("compound_workloads", []):
        value = entry.get("graphpilot_margin_vs_best_other_ms")
        if value is not None:
            baseline_rows.append((entry["workload_id"], float(value)))
    write_bar_svg(
        output_dir / "baseline_comparison.svg",
        "GraphPilot margin versus best other baseline",
        baseline_rows or [("no-data", 0.0)],
        "margin ms",
    )
    write_bar_png(
        output_dir / "baseline_comparison.png",
        "GraphPilot margin versus best other baseline",
        baseline_rows or [("no-data", 0.0)],
        "margin ms",
    )

    ablation_rows = [
        (entry["workload_id"], float(entry.get("delta_ms", 0.0)))
        for entry in characterization_summary.get("ablations", {}).get("pipeline", [])
    ]
    write_bar_svg(
        output_dir / "ablation_breakdown.svg",
        "Pipeline ablation delta by workload",
        ablation_rows or [("no-data", 0.0)],
        "delta ms",
    )
    write_bar_png(
        output_dir / "ablation_breakdown.png",
        "Pipeline ablation delta by workload",
        ablation_rows or [("no-data", 0.0)],
        "delta ms",
    )

    fallback_rows = []
    for entry in experiment_summary.get("comparisons", []):
        fallback_rows.append((entry["workflow_id"], float(entry.get("latency_delta_ms", 0.0))))
    write_bar_svg(
        output_dir / "fallback_penalty.svg",
        "Fallback-sensitive delta proxy",
        fallback_rows or [("no-data", 0.0)],
        "delta ms",
    )
    write_bar_png(
        output_dir / "fallback_penalty.png",
        "Fallback-sensitive delta proxy",
        fallback_rows or [("no-data", 0.0)],
        "delta ms",
    )

    thermal_series = {}
    for resource_id, rows in characterization_summary.get("thermal_curves", {}).items():
        thermal_series[resource_id] = [
            (float(entry.get("temp_c", 0.0)), float(entry.get("slowdown", 0.0)))
            for entry in rows
        ]
    write_line_svg(
        output_dir / "thermal_plan_bank.svg",
        "Thermal slowdown curves by resource",
        thermal_series or {"no-data": [(0.0, 0.0), (1.0, 0.0)]},
        "temperature C",
        "slowdown",
    )
    write_line_png(
        output_dir / "thermal_plan_bank.png",
        "Thermal slowdown curves by resource",
        thermal_series or {"no-data": [(0.0, 0.0), (1.0, 0.0)]},
        "temperature C",
        "slowdown",
    )

    objective_rows = []
    if tuning_summary:
        evaluations = tuning_summary.get("objective_evaluations", [])
        ranked = sorted(
            (float(entry.get("total_error", 0.0)) for entry in evaluations if entry.get("total_error") is not None)
        )
        objective_rows = [(f"candidate_{index + 1}", value) for index, value in enumerate(ranked[:10])]
    write_bar_svg(
        output_dir / "objective_sensitivity.svg",
        "Objective sensitivity candidates",
        objective_rows or [("no-data", 0.0)],
        "total error",
    )
    write_bar_png(
        output_dir / "objective_sensitivity.png",
        "Objective sensitivity candidates",
        objective_rows or [("no-data", 0.0)],
        "total error",
    )

    write_tables_tex(output_dir / "tables.tex", paper_tables_path)

    summary = {
        "checkpoint_manifest": str(args.checkpoint_manifest.resolve()),
        "artifact_pack_summary": str(canonical["artifact_pack_summary"].resolve()),
        "calibration_summary": str(canonical["calibration_summary"].resolve()),
        "characterization_summary": str(canonical["characterization_summary"].resolve()),
        "experiment_summary": str(canonical["experiment_summary"].resolve()),
        "output_dir": str(output_dir.resolve()),
        "outputs": [
            str((output_dir / name).resolve())
            for name in (
                "architecture_overview.svg",
                "architecture_overview.png",
                "offline_online_split.svg",
                "offline_online_split.png",
                "workload_universe_coverage.svg",
                "workload_universe_coverage.png",
                "sim_real_calibration.svg",
                "sim_real_calibration.png",
                "workflow_primary_results.svg",
                "workflow_primary_results.png",
                "continuous_stream_results.svg",
                "continuous_stream_results.png",
                "baseline_comparison.svg",
                "baseline_comparison.png",
                "ablation_breakdown.svg",
                "ablation_breakdown.png",
                "fallback_penalty.svg",
                "fallback_penalty.png",
                "thermal_plan_bank.svg",
                "thermal_plan_bank.png",
                "objective_sensitivity.svg",
                "objective_sensitivity.png",
                "tables.tex",
            )
        ],
        "plot_outputs": pack_summary.get("plot_outputs", []),
        "calibration_family_count": len(calibration_summary.get("calibration_quality_by_family", {})),
    }
    write_json(output_dir / "summary.json", summary)
    print(output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
