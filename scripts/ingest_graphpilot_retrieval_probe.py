#!/usr/bin/env python3
"""Ingest retrieval-artifact probe evidence into GraphPilot registries."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
ARTIFACT_ROOT = ROOT_DIR / "artifacts" / "graphpilot_edge"
BACKEND_MATRIX_PATH = ARTIFACT_ROOT / "registries" / "backend_feasibility_matrix.json"
STATE_LEDGER_PATH = ARTIFACT_ROOT / "state" / "state_ledger.json"
PHASE2_REPORT_PATH = ARTIFACT_ROOT / "reports" / "phase2_stage_feasibility_status.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_unique(strings: list[str], value: str) -> None:
    if value not in strings:
        strings.append(value)


def update_backend_matrix(matrix: dict[str, Any], summary_path: Path, summary: dict[str, Any]) -> None:
    target = next(
        stage for stage in matrix["stages"] if stage["stage_id"] == summary["expected_stage"]
    )
    for backend_name, backend_state in target["backends"].items():
        backend_state["status"] = "infeasible_missing_artifact"
        backend_state["last_verdict"] = summary["verdict"]
        backend_state["last_verified_at"] = summary["generated_at"]
        backend_state["last_summary"] = summary["remediation"]
        append_unique(backend_state.setdefault("evidence", []), str(summary_path))
        append_unique(backend_state.setdefault("evidence", []), summary["remediation"])


def update_state_ledger(ledger: dict[str, Any], summary_path: Path, summary: dict[str, Any]) -> None:
    blockers = ledger.setdefault("active_blockers", [])
    blocker = (
        "Retrieval workflow is blocked by missing EmbeddingGemma-300m-compatible on-device artifact; "
        f"see {summary_path}"
    )
    if blocker not in blockers:
        blockers.append(blocker)
    ledger["next_immediate_action"] = (
        "Advance Phase 3/4 on the feasible ASR/planner/VLM/responder/TTS paths while treating retrieval as explicitly infeasible until a real embedder artifact is staged."
    )


def write_phase2_report(summary_path: Path, summary: dict[str, Any], backend_matrix: dict[str, Any]) -> None:
    lines = [
        "# Phase 2 Stage Feasibility Status",
        "",
        f"Date: {datetime.now(timezone.utc).date().isoformat()}",
        "",
        "## Retrieval artifact probe",
        "",
        f"- Summary: `{summary_path}`",
        f"- Stage: `{summary['expected_stage']}`",
        f"- Verdict: `{summary['verdict']}`",
        f"- Remediation: {summary['remediation']}",
        "",
        "## Aggregated backend feasibility",
        "",
    ]
    for stage in backend_matrix["stages"]:
        lines.append(f"- `{stage['stage_id']}`")
        for backend_name, backend_state in stage["backends"].items():
            lines.append(f"  - `{backend_name}`: `{backend_state['status']}`")
    lines.extend(
        [
            "",
            "## Retrieval blocker",
            "",
            "- `retrieval.embedder.primary` is currently blocked by a missing real on-device EmbeddingGemma-300m-compatible artifact.",
            "- Workflow C must remain explicitly infeasible until that artifact exists and passes a device smoke test.",
            "",
        ]
    )
    PHASE2_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_path", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = args.summary_path.resolve()
    summary = load_json(summary_path)
    backend_matrix = load_json(BACKEND_MATRIX_PATH)
    state_ledger = load_json(STATE_LEDGER_PATH)

    update_backend_matrix(backend_matrix, summary_path, summary)
    update_state_ledger(state_ledger, summary_path, summary)
    write_phase2_report(summary_path, summary, backend_matrix)

    backend_matrix["last_updated"] = datetime.now(timezone.utc).date().isoformat()
    state_ledger["date"] = datetime.now(timezone.utc).date().isoformat()

    write_json(BACKEND_MATRIX_PATH, backend_matrix)
    write_json(STATE_LEDGER_PATH, state_ledger)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
