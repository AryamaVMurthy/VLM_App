#!/usr/bin/env python3
"""Build checkpoint-pinned figures and LaTeX tables for the GraphPilot CASES paper."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

    fallback_rows = []
    for entry in experiment_summary.get("comparisons", []):
        fallback_rows.append((entry["workflow_id"], float(entry.get("latency_delta_ms", 0.0))))
    write_bar_svg(
        output_dir / "fallback_penalty.svg",
        "Fallback-sensitive delta proxy",
        fallback_rows or [("no-data", 0.0)],
        "delta ms",
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
                "sim_real_calibration.svg",
                "workflow_primary_results.svg",
                "continuous_stream_results.svg",
                "baseline_comparison.svg",
                "ablation_breakdown.svg",
                "fallback_penalty.svg",
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
