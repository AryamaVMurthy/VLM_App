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
SANS_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
MONO_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")


def load_font(size: int, *, mono: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidate = MONO_FONT_PATH if mono else SANS_FONT_PATH
    if candidate.exists():
        try:
            return ImageFont.truetype(str(candidate), size=size)
        except OSError:
            pass
    return ImageFont.load_default()


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
        "sustained_summary",
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
    width = 900
    height = 96 + 36 * max(len(rows), 1)
    left = 220
    values = [value for _, value in rows]
    min_value = min(values + [0.0])
    max_value = max(values + [0.0])
    span = max(max_value - min_value, 1.0)
    scale = 560.0 / span
    zero_x = left + (-min_value * scale)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="20" y="28" font-size="20" font-family="monospace">{title}</text>',
        f'<text x="{left}" y="{height - 14}" font-size="12" font-family="monospace">{x_label}</text>',
        f'<line x1="{zero_x:.1f}" y1="44" x2="{zero_x:.1f}" y2="{height - 30}" stroke="#333" stroke-width="2" />',
    ]
    y = 52
    for label, value in rows:
        bar_start = min(zero_x, zero_x + value * scale)
        bar_end = max(zero_x, zero_x + value * scale)
        parts.append(f'<text x="16" y="{y + 14}" font-size="11" font-family="monospace">{label}</text>')
        parts.append(
            f'<rect x="{bar_start:.1f}" y="{y}" width="{max(bar_end - bar_start, 1.0):.1f}" height="18" fill="#4c78a8" />'
        )
        label_x = (bar_end + 10.0) if value >= 0 else max(bar_start - 58.0, 20.0)
        parts.append(
            f'<text x="{label_x:.1f}" y="{y + 14}" font-size="11" font-family="monospace">{value:.1f}</text>'
        )
        y += 28
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_bar_png(path: Path, title: str, rows: list[tuple[str, float]], x_label: str) -> None:
    width = 1200
    height = 140 + 50 * max(len(rows), 1)
    left = 360
    values = [value for _, value in rows]
    min_value = min(values + [0.0])
    max_value = max(values + [0.0])
    span = max(max_value - min_value, 1.0)
    scale = 720.0 / span
    zero_x = int(round(left + (-min_value * scale)))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(24)
    label_font = load_font(16)
    mono_font = load_font(16, mono=True)
    draw.text((24, 18), title, fill="black", font=title_font)
    draw.text((left, height - 28), x_label, fill="black", font=label_font)
    draw.line((zero_x, 42, zero_x, height - 28), fill="#333333", width=2)
    y = 58
    for label, value in rows:
        bar_extent = int(round(value * scale))
        bar_start = min(zero_x, zero_x + bar_extent)
        bar_end = max(zero_x, zero_x + bar_extent)
        draw.text((18, y + 6), label, fill="black", font=mono_font)
        draw.rectangle((bar_start, y, max(bar_end, bar_start + 1), y + 24), fill="#4c78a8", outline="#1f3552")
        label_x = (bar_end + 14) if value >= 0 else max(bar_start - 72, 20)
        draw.text((label_x, y + 6), f"{value:.1f}", fill="black", font=mono_font)
        y += 38
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
    width = 1600
    height = 380
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(24)
    label_font = load_font(18)
    box_font = load_font(18, mono=True)
    title = "GraphPilot-Edge deployed workflow architecture"
    subtitle = "Primary real-device workflows: " + (", ".join(workflow_ids) if workflow_ids else "none")
    draw.text((24, 20), title, fill="black", font=title_font)
    draw.text((24, 52), subtitle, fill="black", font=label_font)
    x = 60
    stages = ["ASR", "Planner", "VLM / Retrieval", "Responder", "TTS"]
    for index, label in enumerate(stages):
        draw.rounded_rectangle((x, 110, x + 200, 180), radius=14, fill="#dce9f9", outline="#4c78a8", width=2)
        draw.text((x + 34, 136), label, fill="black", font=box_font)
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
    width = 1600
    height = 460
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(24)
    label_font = load_font(18)
    box_font = load_font(17, mono=True)
    draw.text((24, 24), "GraphPilot offline/online split", fill="black", font=title_font)
    draw.rounded_rectangle((60, 100, 560, 330), radius=16, fill="#e8f0fe", outline="#4c78a8", width=2)
    draw.text((88, 118), "Offline brain", fill="black", font=label_font)
    offline_boxes = [("Profiler", 96, 152), ("Simulator", 314, 152), ("Planner", 96, 228), ("Plan bank", 314, 228)]
    for label, x, y in offline_boxes:
        draw.rounded_rectangle((x, y, x + 180, y + 46), radius=10, fill="#dce9f9", outline="#4c78a8", width=2)
        draw.text((x + 42, y + 14), label, fill="black", font=box_font)
    draw.line((560, 215, 700, 215), fill="black", width=4)
    draw.polygon([(700, 215), (686, 208), (686, 222)], fill="black")
    draw.rounded_rectangle((740, 100, 1280, 330), radius=16, fill="#f7ead8", outline="#d17c28", width=2)
    draw.text((768, 118), "Online brain", fill="black", font=label_font)
    online_boxes = [
        ("Runtime scheduler", 772, 152),
        ("Streaming edges", 1010, 152),
        ("Memory / KV", 772, 228),
        ("Thermal switching", 1010, 228),
    ]
    for label, x, y in online_boxes:
        draw.rounded_rectangle((x, y, x + 200, y + 46), radius=10, fill="#fde6c7", outline="#d17c28", width=2)
        draw.text((x + 24, y + 14), label, fill="black", font=box_font)
    image.save(path)


def write_line_svg(path: Path, title: str, series_map: dict[str, list[tuple[float, float]]], x_label: str, y_label: str) -> None:
    width = 900
    height = 380
    left = 90
    right = 40
    top = 42
    bottom = 58
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
        f'<text x="20" y="24" font-size="20" font-family="monospace">{title}</text>',
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
        lx = left + plot_w - 170
        ly = top + 12 + index * 18
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 18}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx + 26}" y="{ly + 4}" font-size="14" font-family="monospace">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_line_png(path: Path, title: str, series_map: dict[str, list[tuple[float, float]]], x_label: str, y_label: str) -> None:
    width = 1180
    height = 500
    left = 112
    right = 40
    top = 50
    bottom = 62
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
    title_font = load_font(24)
    label_font = load_font(16)
    mono_font = load_font(14, mono=True)
    draw.text((20, 16), title, fill="black", font=title_font)
    draw.line((left, top, left, top + plot_h), fill="black", width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="black", width=2)
    draw.text((left + plot_w // 2 - 40, height - 28), x_label, fill="black", font=label_font)
    draw.text((20, top + plot_h // 2), y_label, fill="black", font=label_font)
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
        lx = left + plot_w - 190
        ly = top + 12 + index * 20
        draw.line((lx, ly, lx + 20, ly), fill=color, width=3)
        draw.text((lx + 28, ly - 6), label, fill="black", font=mono_font)
    image.save(path)


def select_top_abs_rows(rows: list[tuple[str, float]], limit: int) -> list[tuple[str, float]]:
    ranked = sorted(rows, key=lambda item: (abs(item[1]), item[0]), reverse=True)
    return ranked[:limit]


def bytes_to_mib(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024.0 * 1024.0)


def build_baseline_delta_rows(characterization_summary: dict[str, Any]) -> list[tuple[str, float]]:
    preferred_baselines = (
        "current_deployed_plan",
        "stage_greedy",
        "static_best_map",
        "no_pipeline",
        "no_fallback_aware",
        "no_memory_kv",
        "no_knob_tuning",
        "no_thermal_adaptation",
    )
    rows: list[tuple[str, float]] = []
    groups = characterization_summary.get("baseline_comparisons", {})
    for entries in groups.values():
        for entry in entries:
            graphpilot_score = entry.get("graphpilot_score_ms")
            if graphpilot_score is None:
                continue
            for baseline in entry.get("baselines", []):
                baseline_id = baseline.get("baseline_id")
                if baseline_id not in preferred_baselines or baseline.get("status") != "ok":
                    continue
                baseline_score = baseline.get("score_ms")
                if baseline_score is None:
                    continue
                rows.append(
                    (
                        f"{entry['workload_id']} :: {baseline_id}",
                        float(baseline_score) - float(graphpilot_score),
                    )
                )
    return select_top_abs_rows(rows, 12)


def build_proxy_baseline_rows(characterization_summary: dict[str, Any]) -> list[tuple[str, float]]:
    proxy_baselines = (
        "band_like",
        "adms_like",
        "puzzle_like",
        "twill_like",
        "heteroinfer_like",
        "agent_xpu_like",
        "hero_like",
    )
    rows: list[tuple[str, float]] = []
    groups = characterization_summary.get("baseline_comparisons", {})
    for entries in groups.values():
        for entry in entries:
            graphpilot_score = entry.get("graphpilot_score_ms")
            if graphpilot_score is None:
                continue
            for baseline in entry.get("baselines", []):
                baseline_id = baseline.get("baseline_id")
                if baseline_id not in proxy_baselines or baseline.get("status") != "ok":
                    continue
                baseline_score = baseline.get("score_ms")
                if baseline_score is None:
                    continue
                rows.append(
                    (
                        f"{entry['workload_id']} :: {baseline_id}",
                        float(baseline_score) - float(graphpilot_score),
                    )
                )
    return select_top_abs_rows(rows, 12)


def build_ablation_delta_rows(characterization_summary: dict[str, Any]) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for entry in characterization_summary.get("ablations", {}).get("pipeline", []):
        rows.append((f"{entry['workload_id']} :: no_pipeline", float(entry.get("delta_ms", 0.0))))
    for entry in characterization_summary.get("ablations", {}).get("single_backend", []):
        rows.append((f"{entry['workload_id']} :: single_backend", float(entry.get("delta_ms", 0.0))))
    retrieval_backend = characterization_summary.get("ablations", {}).get("retrieval_backend", {})
    if retrieval_backend:
        rows.append(("workflow_c retrieval cpu-vs-gpu", float(retrieval_backend.get("delta_ms", 0.0))))
    return select_top_abs_rows(rows, 12)


def build_knob_frontier_panels(characterization_summary: dict[str, Any]) -> list[tuple[str, dict[str, list[tuple[float, float]]]]]:
    panels: list[tuple[str, dict[str, list[tuple[float, float]]]]] = []
    for family, entries in sorted(characterization_summary.get("knob_frontiers", {}).items()):
        points = []
        for entry in entries:
            score_ms = entry.get("score_ms")
            quality_loss = entry.get("quality_proxy_loss")
            if score_ms is None or quality_loss is None:
                continue
            points.append((float(score_ms), float(quality_loss)))
        if not points:
            continue
        points.sort(key=lambda item: item[0])
        panels.append((family, {family: points}))
    return panels


def write_composite_png(path: Path, title: str, image_paths: list[tuple[str, Path]], columns: int = 2) -> None:
    opened = [(label, Image.open(image_path).convert("RGB")) for label, image_path in image_paths]
    if not opened:
        raise ValueError("Composite figure requires at least one source image.")
    thumb_size = (420, 260)
    label_height = 22
    tile_w = thumb_size[0] + 20
    tile_h = thumb_size[1] + label_height + 20
    rows = (len(opened) + columns - 1) // columns
    width = columns * tile_w + 28
    height = rows * tile_h + 58
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(22)
    label_font = load_font(14, mono=True)
    draw.text((14, 12), title, fill="black", font=title_font)
    for index, (label, source) in enumerate(opened):
        row = index // columns
        col = index % columns
        ox = 14 + col * tile_w
        oy = 38 + row * tile_h
        tile = source.copy()
        tile.thumbnail(thumb_size)
        image.paste(tile, (ox + (thumb_size[0] - tile.width) // 2, oy + label_height))
        draw.rectangle((ox, oy, ox + thumb_size[0], oy + tile_h - 20), outline="#b0b0b0", width=1)
        draw.text((ox + 8, oy + 6), label, fill="black", font=label_font)
    image.save(path)


def write_composite_svg(path: Path, title: str, image_paths: list[tuple[str, Path]], columns: int = 2) -> None:
    if not image_paths:
        raise ValueError("Composite figure requires at least one source image.")
    thumb_w = 420
    thumb_h = 260
    label_h = 22
    tile_w = thumb_w + 20
    tile_h = thumb_h + label_h + 20
    rows = (len(image_paths) + columns - 1) // columns
    width = columns * tile_w + 28
    height = rows * tile_h + 58
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{width}" height="{height}">',
        f'<text x="14" y="20" font-size="18" font-family="monospace">{title}</text>',
    ]
    for index, (label, image_path) in enumerate(image_paths):
        row = index // columns
        col = index % columns
        ox = 14 + col * tile_w
        oy = 38 + row * tile_h
        parts.append(f'<rect x="{ox}" y="{oy}" width="{thumb_w}" height="{tile_h - 20}" fill="white" stroke="#b0b0b0" stroke-width="1"/>')
        parts.append(f'<text x="{ox + 8}" y="{oy + 14}" font-size="11" font-family="monospace">{label}</text>')
        parts.append(
            f'<image x="{ox}" y="{oy + label_h}" width="{thumb_w}" height="{thumb_h}" href="{image_path.name}" preserveAspectRatio="xMidYMid meet"/>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_value_matrix_svg(
    path: Path,
    title: str,
    row_labels: list[str],
    col_labels: list[str],
    values: list[list[float | None]],
) -> None:
    cell_w = 120
    cell_h = 32
    left = 170
    top = 56
    width = left + cell_w * len(col_labels) + 24
    height = top + cell_h * len(row_labels) + 28
    flat = [value for row in values for value in row if value is not None]
    min_value = min(flat) if flat else 0.0
    max_value = max(flat) if flat else 1.0
    span = max(max_value - min_value, 1.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="16" y="24" font-size="18" font-family="monospace">{title}</text>',
    ]
    for index, label in enumerate(col_labels):
        x = left + index * cell_w + cell_w / 2
        parts.append(f'<text x="{x:.1f}" y="46" text-anchor="middle" font-size="12" font-family="monospace">{label}</text>')
    for row_index, row_label in enumerate(row_labels):
        y = top + row_index * cell_h
        parts.append(f'<text x="12" y="{y + 20}" font-size="11" font-family="monospace">{row_label}</text>')
        row_values = values[row_index]
        numeric_row = [value for value in row_values if value is not None]
        best_value = min(numeric_row) if numeric_row else None
        for col_index, value in enumerate(row_values):
            x = left + col_index * cell_w
            if value is None:
                fill = "#f0f0f0"
                label = "N/A"
            else:
                normalized = (value - min_value) / span
                shade = int(235 - normalized * 120)
                fill = f"rgb({shade},{shade},{255})"
                label = f"{value:.0f}"
            stroke = "#2f5aa8" if value is not None and best_value is not None and value == best_value else "#b0b0b0"
            stroke_width = 2 if stroke == "#2f5aa8" else 1
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>')
            parts.append(f'<text x="{x + cell_w/2:.1f}" y="{y + 20}" text-anchor="middle" font-size="11" font-family="monospace">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_value_matrix_png(
    path: Path,
    title: str,
    row_labels: list[str],
    col_labels: list[str],
    values: list[list[float | None]],
) -> None:
    cell_w = 120
    cell_h = 32
    left = 170
    top = 56
    width = left + cell_w * len(col_labels) + 24
    height = top + cell_h * len(row_labels) + 28
    flat = [value for row in values for value in row if value is not None]
    min_value = min(flat) if flat else 0.0
    max_value = max(flat) if flat else 1.0
    span = max(max_value - min_value, 1.0)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(20)
    label_font = load_font(14)
    mono_font = load_font(13, mono=True)
    draw.text((16, 12), title, fill="black", font=title_font)
    for index, label in enumerate(col_labels):
        x = left + index * cell_w + 10
        draw.text((x, 38), label, fill="black", font=label_font)
    for row_index, row_label in enumerate(row_labels):
        y = top + row_index * cell_h
        draw.text((12, y + 10), row_label, fill="black", font=mono_font)
        row_values = values[row_index]
        numeric_row = [value for value in row_values if value is not None]
        best_value = min(numeric_row) if numeric_row else None
        for col_index, value in enumerate(row_values):
            x = left + col_index * cell_w
            if value is None:
                fill = (240, 240, 240)
                label = "N/A"
            else:
                normalized = (value - min_value) / span
                shade = int(235 - normalized * 120)
                fill = (shade, shade, 255)
                label = f"{value:.0f}"
            outline = "#2f5aa8" if value is not None and best_value is not None and value == best_value else "#b0b0b0"
            width_px = 2 if outline == "#2f5aa8" else 1
            draw.rectangle((x, y, x + cell_w, y + cell_h), fill=fill, outline=outline, width=width_px)
            draw.text((x + 36, y + 10), label, fill="black", font=mono_font)
    image.save(path)


def write_status_matrix_svg(
    path: Path,
    title: str,
    row_labels: list[str],
    col_labels: list[str],
    values: list[list[str]],
) -> None:
    color_map = {
        "feasible_smoke_pass": "#d7f0d8",
        "known_working": "#cde7ff",
        "source_inspection": "#fff0c9",
        "infeasible_no_backend_adapter": "#f7d4d4",
        "missing": "#f0f0f0",
    }
    cell_w = 120
    cell_h = 32
    left = 170
    top = 56
    width = left + cell_w * len(col_labels) + 24
    height = top + cell_h * len(row_labels) + 28
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="16" y="24" font-size="18" font-family="monospace">{title}</text>',
    ]
    for index, label in enumerate(col_labels):
        x = left + index * cell_w + cell_w / 2
        parts.append(f'<text x="{x:.1f}" y="46" text-anchor="middle" font-size="12" font-family="monospace">{label}</text>')
    for row_index, row_label in enumerate(row_labels):
        y = top + row_index * cell_h
        parts.append(f'<text x="12" y="{y + 20}" font-size="11" font-family="monospace">{row_label}</text>')
        for col_index, status in enumerate(values[row_index]):
            x = left + col_index * cell_w
            fill = color_map.get(status, "#f0f0f0")
            short = status.replace("_", " ")
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{fill}" stroke="#b0b0b0" stroke-width="1"/>')
            parts.append(f'<text x="{x + cell_w/2:.1f}" y="{y + 20}" text-anchor="middle" font-size="9" font-family="monospace">{short}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_status_matrix_png(
    path: Path,
    title: str,
    row_labels: list[str],
    col_labels: list[str],
    values: list[list[str]],
) -> None:
    color_map = {
        "feasible_smoke_pass": (215, 240, 216),
        "known_working": (205, 231, 255),
        "source_inspection": (255, 240, 201),
        "infeasible_no_backend_adapter": (247, 212, 212),
        "missing": (240, 240, 240),
    }
    cell_w = 120
    cell_h = 32
    left = 170
    top = 56
    width = left + cell_w * len(col_labels) + 24
    height = top + cell_h * len(row_labels) + 28
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(20)
    label_font = load_font(14)
    mono_font = load_font(12, mono=True)
    draw.text((16, 12), title, fill="black", font=title_font)
    for index, label in enumerate(col_labels):
        x = left + index * cell_w + 12
        draw.text((x, 38), label, fill="black", font=label_font)
    for row_index, row_label in enumerate(row_labels):
        y = top + row_index * cell_h
        draw.text((12, y + 10), row_label, fill="black", font=mono_font)
        for col_index, status in enumerate(values[row_index]):
            x = left + col_index * cell_w
            draw.rectangle((x, y, x + cell_w, y + cell_h), fill=color_map.get(status, (240, 240, 240)), outline="#b0b0b0", width=1)
            draw.text((x + 6, y + 10), status.replace("_", " "), fill="black", font=mono_font)
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
    sustained_summary = load_json(canonical["sustained_summary"])
    workload_registry = load_json(canonical["workload_registry"])
    backend_matrix = load_optional_json_from_manifest(manifest, "backend_matrix")
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

    family_mae_rows = [
        (family, float(entry.get("mean_absolute_error_ms", 0.0)))
        for family, entry in sorted(calibration_summary.get("calibration_quality_by_family", {}).items())
    ]
    write_bar_svg(
        output_dir / "calibration_family_mae.svg",
        "Family-level calibration MAE",
        family_mae_rows or [("no-data", 0.0)],
        "MAE ms",
    )
    write_bar_png(
        output_dir / "calibration_family_mae.png",
        "Family-level calibration MAE",
        family_mae_rows or [("no-data", 0.0)],
        "MAE ms",
    )

    launch_overhead_rows = [
        (backend, float(value))
        for backend, value in sorted((calibration_summary.get("launch_overhead_ms_by_backend") or {}).items())
    ]
    write_bar_svg(
        output_dir / "backend_launch_overhead.svg",
        "Backend launch-overhead surrogates",
        launch_overhead_rows or [("no-data", 0.0)],
        "launch overhead ms",
    )
    write_bar_png(
        output_dir / "backend_launch_overhead.png",
        "Backend launch-overhead surrogates",
        launch_overhead_rows or [("no-data", 0.0)],
        "launch overhead ms",
    )

    contention_rows = [
        (backend, float(value))
        for backend, value in sorted((calibration_summary.get("contention_scale_by_backend") or {}).items())
    ]
    write_bar_svg(
        output_dir / "backend_contention_scale.svg",
        "Backend contention-scale surrogates",
        contention_rows or [("no-data", 0.0)],
        "contention scale",
    )
    write_bar_png(
        output_dir / "backend_contention_scale.png",
        "Backend contention-scale surrogates",
        contention_rows or [("no-data", 0.0)],
        "contention scale",
    )

    family_backend_mean_ms = characterization_summary.get("backend_affinity", {}).get("family_backend_mean_ms", {})
    affinity_row_labels = sorted(family_backend_mean_ms)
    affinity_col_labels = ["cpu", "gpu", "npu"]
    affinity_values = [
        [family_backend_mean_ms.get(family, {}).get(backend) for backend in affinity_col_labels]
        for family in affinity_row_labels
    ]
    write_value_matrix_svg(
        output_dir / "backend_affinity_matrix.svg",
        "Family/backend mean latency matrix",
        affinity_row_labels or ["no-data"],
        affinity_col_labels,
        affinity_values or [[0.0, None, None]],
    )
    write_value_matrix_png(
        output_dir / "backend_affinity_matrix.png",
        "Family/backend mean latency matrix",
        affinity_row_labels or ["no-data"],
        affinity_col_labels,
        affinity_values or [[0.0, None, None]],
    )

    feasibility_rows = []
    feasibility_values = []
    for stage in (backend_matrix or {}).get("stages", []):
        feasibility_rows.append(stage.get("stage_id", "unknown"))
        stage_backends = stage.get("backends", {})
        feasibility_values.append(
            [
                str(stage_backends.get("cpu", {}).get("status", "missing")),
                str(stage_backends.get("gpu", {}).get("status", "missing")),
                str(stage_backends.get("npu", {}).get("status", "missing")),
            ]
        )
    write_status_matrix_svg(
        output_dir / "support_safe_feasibility.svg",
        "Support-safe backend feasibility matrix",
        feasibility_rows or ["no-data"],
        affinity_col_labels,
        feasibility_values or [["missing", "missing", "missing"]],
    )
    write_status_matrix_png(
        output_dir / "support_safe_feasibility.png",
        "Support-safe backend feasibility matrix",
        feasibility_rows or ["no-data"],
        affinity_col_labels,
        feasibility_values or [["missing", "missing", "missing"]],
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

    baseline_rows = build_baseline_delta_rows(characterization_summary)
    write_bar_svg(
        output_dir / "baseline_comparison.svg",
        "Baseline deltas versus GraphPilot",
        baseline_rows or [("no-data", 0.0)],
        "score delta ms",
    )
    write_bar_png(
        output_dir / "baseline_comparison.png",
        "Baseline deltas versus GraphPilot",
        baseline_rows or [("no-data", 0.0)],
        "score delta ms",
    )

    proxy_baseline_rows = build_proxy_baseline_rows(characterization_summary)
    write_bar_svg(
        output_dir / "proxy_baseline_comparison.svg",
        "Method-class proxy deltas versus GraphPilot",
        proxy_baseline_rows or [("no-data", 0.0)],
        "score delta ms",
    )
    write_bar_png(
        output_dir / "proxy_baseline_comparison.png",
        "Method-class proxy deltas versus GraphPilot",
        proxy_baseline_rows or [("no-data", 0.0)],
        "score delta ms",
    )

    ablation_rows = build_ablation_delta_rows(characterization_summary)
    write_bar_svg(
        output_dir / "ablation_breakdown.svg",
        "Ablation deltas across selected workloads",
        ablation_rows or [("no-data", 0.0)],
        "delta ms",
    )
    write_bar_png(
        output_dir / "ablation_breakdown.png",
        "Ablation deltas across selected workloads",
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

    sustained_latency_series: dict[str, list[tuple[float, float]]] = {}
    sustained_thermal_series: dict[str, list[tuple[float, float]]] = {
        "cpu": [],
        "npu": [],
        "skin": [],
    }
    for sample in sustained_summary.get("samples", []):
        sample_index = float(sample.get("sample_index", 0.0))
        workflow_id = str(sample.get("workflow_id", "unknown"))
        warm_latency_ms = sample.get("warm_latency_ms")
        if warm_latency_ms is not None:
            sustained_latency_series.setdefault(workflow_id, []).append((sample_index, float(warm_latency_ms)))
        thermal_after = sample.get("thermal_after", {})
        cpu_temp = thermal_after.get("max_cpu_c")
        npu_temp = thermal_after.get("max_npu_c")
        skin_temp = thermal_after.get("skin_c")
        if cpu_temp is not None:
            sustained_thermal_series["cpu"].append((sample_index, float(cpu_temp)))
        if npu_temp is not None:
            sustained_thermal_series["npu"].append((sample_index, float(npu_temp)))
        if skin_temp is not None:
            sustained_thermal_series["skin"].append((sample_index, float(skin_temp)))
    sustained_thermal_series = {key: value for key, value in sustained_thermal_series.items() if value}

    write_line_svg(
        output_dir / "sustained_latency_drift.svg",
        "Sustained workflow warm latency drift",
        sustained_latency_series or {"no-data": [(0.0, 0.0), (1.0, 0.0)]},
        "sample index",
        "warm latency ms",
    )
    write_line_png(
        output_dir / "sustained_latency_drift.png",
        "Sustained workflow warm latency drift",
        sustained_latency_series or {"no-data": [(0.0, 0.0), (1.0, 0.0)]},
        "sample index",
        "warm latency ms",
    )
    write_line_svg(
        output_dir / "sustained_thermal_drift.svg",
        "Sustained thermal traces",
        sustained_thermal_series or {"no-data": [(0.0, 0.0), (1.0, 0.0)]},
        "sample index",
        "temperature C",
    )
    write_line_png(
        output_dir / "sustained_thermal_drift.png",
        "Sustained thermal traces",
        sustained_thermal_series or {"no-data": [(0.0, 0.0), (1.0, 0.0)]},
        "sample index",
        "temperature C",
    )
    write_composite_png(
        output_dir / "sustained_overview.png",
        "Sustained runtime behavior",
        [
            ("Warm latency drift", output_dir / "sustained_latency_drift.png"),
            ("Thermal traces", output_dir / "sustained_thermal_drift.png"),
        ],
    )
    write_composite_svg(
        output_dir / "sustained_overview.svg",
        "Sustained runtime behavior",
        [
            ("Warm latency drift", output_dir / "sustained_latency_drift.png"),
            ("Thermal traces", output_dir / "sustained_thermal_drift.png"),
        ],
    )
    write_composite_png(
        output_dir / "sustained_detail.png",
        "Detailed sustained runtime panels",
        [
            ("Warm latency drift", output_dir / "sustained_latency_drift.png"),
            ("Thermal traces", output_dir / "sustained_thermal_drift.png"),
        ],
        columns=1,
    )
    write_composite_svg(
        output_dir / "sustained_detail.svg",
        "Detailed sustained runtime panels",
        [
            ("Warm latency drift", output_dir / "sustained_latency_drift.png"),
            ("Thermal traces", output_dir / "sustained_thermal_drift.png"),
        ],
        columns=1,
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

    kv_curve = characterization_summary.get("memory_kv_curves", {}).get("kv_curve", [])
    kv_series = {
        "KV bytes": [
            (float(entry.get("cached_tokens", 0.0)), bytes_to_mib(entry.get("kv_bytes")) or 0.0)
            for entry in kv_curve
        ]
    }
    write_line_svg(
        output_dir / "kv_cache_curve.svg",
        "KV-cache growth",
        kv_series or {"no-data": [(0.0, 0.0), (1.0, 0.0)]},
        "cached tokens",
        "memory MiB",
    )
    write_line_png(
        output_dir / "kv_cache_curve.png",
        "KV-cache growth",
        kv_series or {"no-data": [(0.0, 0.0), (1.0, 0.0)]},
        "cached tokens",
        "memory MiB",
    )

    context_curve = characterization_summary.get("memory_kv_curves", {}).get("llm_context_curve", [])
    context_series = {
        "Context bytes": [
            (float(entry.get("context_tokens", 0.0)), bytes_to_mib(entry.get("aggregate_memory_bytes")) or 0.0)
            for entry in context_curve
        ]
    }
    write_line_svg(
        output_dir / "context_memory_curve.svg",
        "Context-memory growth",
        context_series or {"no-data": [(0.0, 0.0), (1.0, 0.0)]},
        "context tokens",
        "memory MiB",
    )
    write_line_png(
        output_dir / "context_memory_curve.png",
        "Context-memory growth",
        context_series or {"no-data": [(0.0, 0.0), (1.0, 0.0)]},
        "context tokens",
        "memory MiB",
    )

    write_composite_png(
        output_dir / "memory_kv_overview.png",
        "Memory and KV growth overview",
        [
            ("KV-cache curve", output_dir / "kv_cache_curve.png"),
            ("Context-memory curve", output_dir / "context_memory_curve.png"),
        ],
    )
    write_composite_svg(
        output_dir / "memory_kv_overview.svg",
        "Memory and KV growth overview",
        [
            ("KV-cache curve", output_dir / "kv_cache_curve.png"),
            ("Context-memory curve", output_dir / "context_memory_curve.png"),
        ],
    )

    knob_panels = build_knob_frontier_panels(characterization_summary)
    knob_panel_pngs: list[tuple[str, Path]] = []
    for family, series in knob_panels:
        svg_path = output_dir / f"knob_frontier_{family}.svg"
        png_path = output_dir / f"knob_frontier_{family}.png"
        write_line_svg(svg_path, f"{family} knob frontier", series, "score ms", "quality loss")
        write_line_png(png_path, f"{family} knob frontier", series, "score ms", "quality loss")
        knob_panel_pngs.append((family, png_path))
    if not knob_panel_pngs:
        knob_panel_pngs = [("no-data", output_dir / "workflow_primary_results.png")]
    write_composite_png(
        output_dir / "knob_frontier_overview.png",
        "Stage-knob frontier overview",
        knob_panel_pngs,
    )
    write_composite_svg(
        output_dir / "knob_frontier_overview.svg",
        "Stage-knob frontier overview",
        knob_panel_pngs,
    )

    write_composite_png(
        output_dir / "calibration_overview.png",
        "Calibration overview",
        [
            ("Sim-to-real deltas", output_dir / "sim_real_calibration.png"),
            ("Family MAE", output_dir / "calibration_family_mae.png"),
            ("Launch overhead", output_dir / "backend_launch_overhead.png"),
            ("Contention scale", output_dir / "backend_contention_scale.png"),
        ],
    )
    write_composite_svg(
        output_dir / "calibration_overview.svg",
        "Calibration overview",
        [
            ("Sim-to-real deltas", output_dir / "sim_real_calibration.png"),
            ("Family MAE", output_dir / "calibration_family_mae.png"),
            ("Launch overhead", output_dir / "backend_launch_overhead.png"),
            ("Contention scale", output_dir / "backend_contention_scale.png"),
        ],
    )

    write_composite_png(
        output_dir / "evaluation_overview.png",
        "Evaluation overview",
        [
            ("Calibration deltas", output_dir / "sim_real_calibration.png"),
            ("Primary workflows", output_dir / "workflow_primary_results.png"),
            ("Continuous streams", output_dir / "continuous_stream_results.png"),
            ("Baseline margins", output_dir / "baseline_comparison.png"),
        ],
    )
    write_composite_svg(
        output_dir / "evaluation_overview.svg",
        "Evaluation overview",
        [
            ("Calibration deltas", output_dir / "sim_real_calibration.png"),
            ("Primary workflows", output_dir / "workflow_primary_results.png"),
            ("Continuous streams", output_dir / "continuous_stream_results.png"),
            ("Baseline deltas", output_dir / "baseline_comparison.png"),
        ],
    )
    write_composite_png(
        output_dir / "sensitivity_overview.png",
        "Sensitivity and robustness overview",
        [
            ("Fallback penalty", output_dir / "fallback_penalty.png"),
            ("Pipeline ablation", output_dir / "ablation_breakdown.png"),
            ("Thermal plan bank", output_dir / "thermal_plan_bank.png"),
            ("Objective sensitivity", output_dir / "objective_sensitivity.png"),
        ],
    )
    write_composite_svg(
        output_dir / "sensitivity_overview.svg",
        "Sensitivity and robustness overview",
        [
            ("Fallback penalty", output_dir / "fallback_penalty.png"),
            ("Ablation delta", output_dir / "ablation_breakdown.png"),
            ("Thermal plan bank", output_dir / "thermal_plan_bank.png"),
            ("Objective sensitivity", output_dir / "objective_sensitivity.png"),
        ],
    )

    write_tables_tex(output_dir / "tables.tex", paper_tables_path)

    summary = {
        "checkpoint_manifest": str(args.checkpoint_manifest.resolve()),
        "artifact_pack_summary": str(canonical["artifact_pack_summary"].resolve()),
        "calibration_summary": str(canonical["calibration_summary"].resolve()),
        "characterization_summary": str(canonical["characterization_summary"].resolve()),
        "experiment_summary": str(canonical["experiment_summary"].resolve()),
        "sustained_summary": str(canonical["sustained_summary"].resolve()),
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
                "calibration_family_mae.svg",
                "calibration_family_mae.png",
                "backend_launch_overhead.svg",
                "backend_launch_overhead.png",
                "backend_contention_scale.svg",
                "backend_contention_scale.png",
                "backend_affinity_matrix.svg",
                "backend_affinity_matrix.png",
                "support_safe_feasibility.svg",
                "support_safe_feasibility.png",
                "calibration_overview.svg",
                "calibration_overview.png",
                "evaluation_overview.svg",
                "evaluation_overview.png",
                "proxy_baseline_comparison.svg",
                "proxy_baseline_comparison.png",
                "sensitivity_overview.svg",
                "sensitivity_overview.png",
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
                "memory_kv_overview.svg",
                "memory_kv_overview.png",
                "fallback_penalty.svg",
                "fallback_penalty.png",
                "knob_frontier_overview.svg",
                "knob_frontier_overview.png",
                "thermal_plan_bank.svg",
                "thermal_plan_bank.png",
                "objective_sensitivity.svg",
                "objective_sensitivity.png",
                "sustained_latency_drift.svg",
                "sustained_latency_drift.png",
                "sustained_thermal_drift.svg",
                "sustained_thermal_drift.png",
                "sustained_detail.svg",
                "sustained_detail.png",
                "sustained_overview.svg",
                "sustained_overview.png",
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
