#!/usr/bin/env python3
"""Summarize GraphPilot profiler coverage and update ledger/report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
ARTIFACT_ROOT = ROOT_DIR / "artifacts" / "graphpilot_edge"
STATE_LEDGER_PATH = ARTIFACT_ROOT / "state" / "state_ledger.json"
PHASE3_REPORT_PATH = ARTIFACT_ROOT / "reports" / "phase3_profiler_status.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summarize_profile_coverage(summary: dict[str, Any]) -> dict[str, Any]:
    covered_pairs = sorted(
        f"{profile['stage_id']}@{profile['backend']}" for profile in summary.get("profiles", [])
    )
    return {
        "profile_count": len(covered_pairs),
        "covered_pairs": covered_pairs,
    }


def update_state_ledger(ledger: dict[str, Any], summary_path: Path, coverage: dict[str, Any]) -> None:
    ledger["current_phase"] = "Phase 3: Profiling Infrastructure and Data Collection"
    ledger["next_immediate_action"] = (
        "Use the recorded stage/workflow profiles to drive Phase 4 planning, scheduling, memory, and KV policy implementation."
    )
    completed = ledger.setdefault("completed_tasks", [])
    title = "Collected GraphPilot stage and workflow profiler evidence"
    if not any(task.get("title") == title for task in completed):
        completed.append(
            {
                "id": "phase3-001",
                "title": title,
                "evidence": {
                    "command": "env -u PYTHONHOME -u PYTHONPATH python3 scripts/run_graphpilot_stage_profiler.py ...",
                    "result": f"Profiler summary {summary_path} captured {coverage['profile_count']} stage/workflow profile entries.",
                },
            }
        )


def write_phase3_report(summary_path: Path, coverage: dict[str, Any]) -> None:
    lines = [
        "# Phase 3 Profiler Status",
        "",
        f"Date: {datetime.now(timezone.utc).date().isoformat()}",
        "",
        f"- Summary: `{summary_path}`",
        f"- Profile count: `{coverage['profile_count']}`",
        "",
        "## Covered stage/backend pairs",
        "",
    ]
    for pair in coverage["covered_pairs"]:
        lines.append(f"- `{pair}`")
    PHASE3_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_path", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = args.summary_path.resolve()
    summary = load_json(summary_path)
    coverage = summarize_profile_coverage(summary)
    ledger = load_json(STATE_LEDGER_PATH)
    update_state_ledger(ledger, summary_path, coverage)
    write_phase3_report(summary_path, coverage)
    ledger["date"] = datetime.now(timezone.utc).date().isoformat()
    write_json(STATE_LEDGER_PATH, ledger)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
