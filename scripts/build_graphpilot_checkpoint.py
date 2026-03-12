#!/usr/bin/env python3
"""Freeze a canonical GraphPilot checkpoint manifest from one explicit evidence set."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
ARTIFACT_ROOT = ROOT_DIR / "artifacts" / "graphpilot_edge"
DEFAULT_BACKEND_MATRIX = ARTIFACT_ROOT / "registries" / "backend_feasibility_matrix.json"
DEFAULT_PROFILER_REGISTRY = ARTIFACT_ROOT / "registries" / "profiler_registry.json"
DEFAULT_CANDIDATE_PLANS = ARTIFACT_ROOT / "registries" / "candidate_plan_registry.json"
DEFAULT_BASELINE_REGISTRY = ARTIFACT_ROOT / "registries" / "baseline_policy_registry.json"
DEFAULT_WORKLOAD_REGISTRY = ARTIFACT_ROOT / "registries" / "workload_universe_registry.json"
DEFAULT_EXPERIMENT_REGISTRY = ARTIFACT_ROOT / "registries" / "experiment_registry.json"
DEFAULT_PLOT_REGISTRY = ARTIFACT_ROOT / "registries" / "plot_registry.json"
DEFAULT_STATE_LEDGER = ARTIFACT_ROOT / "state" / "state_ledger.json"
DEFAULT_ANALYSIS_ROOT = ARTIFACT_ROOT / "analysis"
DEFAULT_OUTPUT_ROOT = ARTIFACT_ROOT / "checkpoints"
DEFAULT_TRUTH_SOURCE_PDF = ROOT_DIR / "Truth-docs" / "graphpilot_edge_revision_report.pdf"
DEFAULT_REQUIRED_BASELINES = (
    "cpu_only",
    "gpu_only",
    "npu_only",
    "current_deployed_plan",
    "stage_greedy",
    "static_best_map",
    "no_pipeline",
    "no_fallback_aware",
    "no_memory_kv",
    "no_knob_tuning",
    "no_thermal_adaptation",
    "band_like",
    "adms_like",
    "puzzle_like",
    "twill_like",
    "heteroinfer_like",
    "agent_xpu_like",
    "hero_like",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def latest_experiment_summary(experiment_registry: dict[str, Any]) -> Path:
    candidates = [
        item["metrics_path"]
        for item in experiment_registry.get("experiments", [])
        if item.get("workflow_template") == "graphpilot_phase6_7"
    ]
    if not candidates:
        raise FileNotFoundError(
            "No GraphPilot phase6_7 experiment summary found. Remediation: run scripts/run_graphpilot_experiments.py first."
        )
    return Path(sorted(candidates)[-1])


def latest_sustained_summary(experiment_registry: dict[str, Any]) -> Path | None:
    candidates = [
        item["metrics_path"]
        for item in experiment_registry.get("experiments", [])
        if item.get("workflow_template") == "graphpilot_phase7_sustained_load"
    ]
    if not candidates:
        return None
    return Path(sorted(candidates)[-1])


def latest_analysis_summary(root: Path, prefix: str) -> Path | None:
    candidates = sorted(root.glob(f"{prefix}_*/summary.json"))
    if not candidates:
        return None
    return candidates[-1]


def load_beads_snapshot(json_path: Path | None, *, status: str) -> list[dict[str, Any]]:
    if json_path is not None:
        payload = load_json(json_path)
        if not isinstance(payload, list):
            raise ValueError(f"Expected list payload in '{json_path}'.")
        return payload
    result = subprocess.run(
        ["bd", "list", "--json", "--status", status, "-n", "200"],
        cwd=ROOT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to query beads status '{status}'. stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
    payload = json.loads(result.stdout)
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected beads payload for status '{status}'.")
    return payload


def build_checkpoint_manifest(args: argparse.Namespace) -> dict[str, Any]:
    experiment_registry = load_json(args.experiment_registry)
    canonical_paths = {
        "backend_matrix": args.backend_matrix.resolve(),
        "profiler_registry": args.profiler_registry.resolve(),
        "candidate_plans": args.candidate_plans.resolve(),
        "baseline_registry": args.baseline_registry.resolve(),
        "workload_registry": args.workload_registry.resolve(),
        "experiment_registry": args.experiment_registry.resolve(),
        "plot_registry": args.plot_registry.resolve(),
        "state_ledger": args.state_ledger.resolve(),
        "experiment_summary": (args.experiment_summary or latest_experiment_summary(experiment_registry)).resolve(),
        "sustained_summary": None,
        "calibration_summary": None,
        "tuning_summary": None,
        "characterization_summary": None,
        "memory_admission_summary": None,
        "artifact_pack_summary": args.artifact_pack_summary.resolve() if args.artifact_pack_summary else None,
    }
    sustained = args.sustained_summary or latest_sustained_summary(experiment_registry)
    if sustained is not None:
        canonical_paths["sustained_summary"] = sustained.resolve()
    for name, prefix in (
        ("calibration_summary", "graphpilot_cost_calibration"),
        ("tuning_summary", "graphpilot_hparam_tuning"),
        ("characterization_summary", "graphpilot_characterization"),
        ("memory_admission_summary", "graphpilot_memory_admission"),
    ):
        explicit = getattr(args, name)
        resolved = explicit or latest_analysis_summary(DEFAULT_ANALYSIS_ROOT, prefix)
        canonical_paths[name] = resolved.resolve() if resolved is not None else None

    missing = [name for name, path in canonical_paths.items() if name not in {"sustained_summary", "calibration_summary", "tuning_summary", "characterization_summary", "memory_admission_summary"} and path is None]
    if missing:
        raise ValueError(f"Missing required canonical paths: {missing}")
    for name, path in canonical_paths.items():
        if path is not None and not Path(path).exists():
            raise FileNotFoundError(f"Canonical evidence path '{name}' does not exist: {path}")
    if not args.truth_source_pdf.exists():
        raise FileNotFoundError(
            f"Truth-source PDF is missing: {args.truth_source_pdf}. Remediation: restore Truth-docs/graphpilot_edge_revision_report.pdf."
        )

    open_beads = load_beads_snapshot(args.open_beads_json, status="open")
    closed_beads = load_beads_snapshot(args.closed_beads_json, status="closed")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "truth_source_pdf": str(args.truth_source_pdf.resolve()),
        "canonical_evidence_paths": {
            name: (str(path) if path is not None else None)
            for name, path in canonical_paths.items()
        },
        "required_baseline_ids": list(args.required_baselines),
        "open_beads": open_beads,
        "closed_beads": closed_beads,
        "verification_commands": [
            "env -u PYTHONHOME -u PYTHONPATH python3 -m unittest discover -s tests/graphpilot_edge -v",
            "env -u PYTHONHOME -u PYTHONPATH python3 -m unittest discover -s scripts/tests -v",
        ],
    }


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# GraphPilot Canonical Checkpoint",
        "",
        f"- Generated at: `{manifest['generated_at']}`",
        f"- Truth-source PDF: `{manifest['truth_source_pdf']}`",
        f"- Open beads: {len(manifest.get('open_beads', []))}",
        f"- Closed beads: {len(manifest.get('closed_beads', []))}",
        "",
        "## Canonical evidence paths",
        "",
    ]
    for name, value in sorted(manifest["canonical_evidence_paths"].items()):
        lines.append(f"- `{name}`: `{value}`")
    lines.extend(["", "## Required baseline IDs", ""])
    for baseline_id in manifest.get("required_baseline_ids", []):
        lines.append(f"- `{baseline_id}`")
    lines.extend(["", "## Open Beads", ""])
    for bead in manifest.get("open_beads", []):
        lines.append(f"- `{bead['id']}`: {bead.get('title')}")
    if not manifest.get("open_beads"):
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-matrix", type=Path, default=DEFAULT_BACKEND_MATRIX)
    parser.add_argument("--profiler-registry", type=Path, default=DEFAULT_PROFILER_REGISTRY)
    parser.add_argument("--candidate-plans", type=Path, default=DEFAULT_CANDIDATE_PLANS)
    parser.add_argument("--baseline-registry", type=Path, default=DEFAULT_BASELINE_REGISTRY)
    parser.add_argument("--workload-registry", type=Path, default=DEFAULT_WORKLOAD_REGISTRY)
    parser.add_argument("--experiment-registry", type=Path, default=DEFAULT_EXPERIMENT_REGISTRY)
    parser.add_argument("--plot-registry", type=Path, default=DEFAULT_PLOT_REGISTRY)
    parser.add_argument("--state-ledger", type=Path, default=DEFAULT_STATE_LEDGER)
    parser.add_argument("--experiment-summary", type=Path, default=None)
    parser.add_argument("--sustained-summary", type=Path, default=None)
    parser.add_argument("--calibration-summary", type=Path, default=None)
    parser.add_argument("--tuning-summary", type=Path, default=None)
    parser.add_argument("--characterization-summary", type=Path, default=None)
    parser.add_argument("--memory-admission-summary", type=Path, default=None)
    parser.add_argument("--artifact-pack-summary", type=Path, default=None)
    parser.add_argument("--truth-source-pdf", type=Path, default=DEFAULT_TRUTH_SOURCE_PDF)
    parser.add_argument("--open-beads-json", type=Path, default=None)
    parser.add_argument("--closed-beads-json", type=Path, default=None)
    parser.add_argument("--required-baselines", nargs="+", default=list(DEFAULT_REQUIRED_BASELINES))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_checkpoint_manifest(args)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"graphpilot_checkpoint_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "summary.json"
    manifest["checkpoint_manifest_path"] = str(summary_path.resolve())
    write_json(summary_path, manifest)
    write_report(output_dir / "report.md", manifest)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
