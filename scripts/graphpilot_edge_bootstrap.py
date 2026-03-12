#!/usr/bin/env python3
"""Validate and initialize GraphPilot-Edge workspace scaffolding."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "artifacts" / "graphpilot_edge"
DEFAULT_CONFIG_DIR = ROOT_DIR / "configs" / "graphpilot_edge"

REQUIRED_SUBDIRS = (
    "state",
    "registries",
    "manifests",
    "logs",
    "metrics",
    "plots",
    "traces",
    "reports",
)

REQUIRED_JSON_PATHS = (
    "state/state_ledger.json",
    "state/repo_component_inventory.json",
    "registries/backend_feasibility_matrix.json",
    "registries/profiler_registry.json",
    "registries/candidate_plan_registry.json",
    "registries/experiment_registry.json",
    "registries/plot_registry.json",
    "manifests/device_environment.json",
)

REQUIRED_CONFIG_PATHS = (
    "workflow_templates.json",
    "stage_catalog.json",
    "plan_bank_templates.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Missing required JSON file '{path}'. Remediation: initialize the GraphPilot-Edge scaffolding first."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in '{path}': {exc}") from exc


def ensure_dirs(output_root: Path) -> list[str]:
    created: list[str] = []
    for relative in REQUIRED_SUBDIRS:
        target = output_root / relative
        if not target.exists():
            target.mkdir(parents=True, exist_ok=False)
            created.append(str(target))
    return created


def adb_devices() -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            ["adb", "devices", "-l"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "adb is not available on PATH. Remediation: install Android platform-tools or fix PATH."
        ) from exc
    devices: list[dict[str, str]] = []
    lines = result.stdout.splitlines()
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        record = {"serial": parts[0], "state": parts[1], "raw": stripped}
        for token in parts[2:]:
            if ":" not in token:
                continue
            key, value = token.split(":", 1)
            record[key] = value
        devices.append(record)
    return devices


def build_summary(output_root: Path, config_dir: Path, probe_adb: bool) -> dict[str, Any]:
    for relative in REQUIRED_JSON_PATHS:
        load_json(output_root / relative)
    config_payload = {}
    for relative in REQUIRED_CONFIG_PATHS:
        config_payload[relative] = load_json(config_dir / relative)

    summary: dict[str, Any] = {
        "generated_at": utc_now(),
        "output_root": str(output_root),
        "config_dir": str(config_dir),
        "required_subdirs": [str(output_root / item) for item in REQUIRED_SUBDIRS],
        "required_json_files": [str(output_root / item) for item in REQUIRED_JSON_PATHS],
        "required_config_files": [str(config_dir / item) for item in REQUIRED_CONFIG_PATHS],
        "workflow_count": len(config_payload["workflow_templates.json"]["workflows"]),
        "stage_count": len(config_payload["stage_catalog.json"]["stages"]),
        "plan_bank_states": [
            entry["state_id"]
            for entry in config_payload["plan_bank_templates.json"]["plan_bank_states"]
        ],
        "host": {
            "logical_cpus": os.cpu_count(),
        },
    }
    if probe_adb:
        summary["adb_devices"] = adb_devices()
    return summary


def write_summary(summary: dict[str, Any], output_root: Path) -> Path:
    report_path = output_root / "reports" / "bootstrap_summary.json"
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="GraphPilot-Edge artifact root",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="GraphPilot-Edge config directory",
    )
    parser.add_argument(
        "--ensure-dirs",
        action="store_true",
        help="Create missing required subdirectories under the output root.",
    )
    parser.add_argument(
        "--probe-adb",
        action="store_true",
        help="Query adb devices and include the result in the bootstrap summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output_root = args.output_root.resolve()
    config_dir = args.config_dir.resolve()
    if args.ensure_dirs:
        ensure_dirs(output_root)
    summary = build_summary(output_root, config_dir, probe_adb=args.probe_adb)
    report_path = write_summary(summary, output_root)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
