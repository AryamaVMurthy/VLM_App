#!/usr/bin/env python3
"""Close explicit GraphPilot backend-feasibility gaps using source/build evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
ARTIFACT_ROOT = ROOT_DIR / "artifacts" / "graphpilot_edge"
DEFAULT_BACKEND_MATRIX = ARTIFACT_ROOT / "registries" / "backend_feasibility_matrix.json"
DEFAULT_STATE_LEDGER = ARTIFACT_ROOT / "state" / "state_ledger.json"
DEFAULT_PHASE2_REPORT = ARTIFACT_ROOT / "reports" / "phase2_stage_feasibility_status.md"

GAP_CLOSURES: dict[tuple[str, str], dict[str, Any]] = {
    ("asr.primary", "gpu"): {
        "status": "infeasible_no_backend_adapter",
        "summary": "WhisperSttEngine wraps whisper.cpp CPU execution only and exposes no GPU selector in the current Android build.",
        "evidence": [
            "android-app/app/src/main/java/com/qidk/fastvlm/core/speech/WhisperSttEngine.kt:1",
        ],
    },
    ("asr.primary", "npu"): {
        "status": "infeasible_no_backend_adapter",
        "summary": "WhisperSttEngine wraps whisper.cpp CPU execution only and exposes no NPU selector in the current Android build.",
        "evidence": [
            "android-app/app/src/main/java/com/qidk/fastvlm/core/speech/WhisperSttEngine.kt:1",
        ],
    },
    ("tts.primary", "gpu"): {
        "status": "infeasible_no_backend_adapter",
        "summary": "AndroidTtsSpeaker delegates to the Android TextToSpeech service; no GPU-executable TTS path exists in the current build.",
        "evidence": [
            "android-app/app/src/main/java/com/qidk/fastvlm/core/speech/AndroidTtsSpeaker.kt:1",
        ],
    },
    ("tts.primary", "npu"): {
        "status": "infeasible_no_backend_adapter",
        "summary": "AndroidTtsSpeaker delegates to the Android TextToSpeech service; no NPU-executable TTS path exists in the current build.",
        "evidence": [
            "android-app/app/src/main/java/com/qidk/fastvlm/core/speech/AndroidTtsSpeaker.kt:1",
        ],
    },
    ("vlm.fastvlm.primary", "gpu"): {
        "status": "infeasible_no_backend_adapter",
        "summary": "FastVlmNativeBridge exposes only CPU/NPU targets in BackendTarget and rejects any non-CPU request in the in-app bridge path.",
        "evidence": [
            "android-app/app/src/main/java/com/qidk/fastvlm/core/model/Contracts.kt:7",
            "android-app/app/src/main/java/com/qidk/fastvlm/core/bridge/FastVlmNativeBridge.kt:272",
        ],
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def apply_gap_closures(matrix: dict[str, Any]) -> list[tuple[str, str]]:
    updated: list[tuple[str, str]] = []
    stages = {stage["stage_id"]: stage for stage in matrix.get("stages", [])}
    for key, closure in GAP_CLOSURES.items():
        stage_id, backend = key
        stage = stages.get(stage_id)
        if stage is None:
            continue
        backend_state = stage["backends"][backend]
        if backend_state.get("status") == closure["status"]:
            continue
        backend_state["status"] = closure["status"]
        backend_state["last_verified_at"] = datetime.now(timezone.utc).isoformat()
        backend_state["last_verdict"] = "source_inspection"
        backend_state["last_summary"] = closure["summary"]
        backend_state["notes"] = closure["summary"]
        for evidence in closure["evidence"]:
            append_unique(backend_state.setdefault("evidence", []), evidence)
        updated.append(key)
    vlm_stage = stages.get("vlm.fastvlm.primary")
    if vlm_stage is not None:
        npu_state = vlm_stage["backends"]["npu"]
        npu_state["notes"] = (
            "Known-working FastVLM NPU evidence is available both from the preserved external adb runner path "
            "and from GraphPilotCoordinator workflow B/C instrumentation."
        )
        for evidence in (
            "scripts/run_fastvlm_litert_npu_adb.sh:1",
            "scripts/run_fastvlm_litert_overlap_adb.sh:1",
            "android-app/app/src/main/java/com/qidk/fastvlm/core/graphpilot/GraphPilotCoordinator.kt:1",
        ):
            append_unique(npu_state.setdefault("evidence", []), evidence)
    return updated


def update_state_ledger(ledger: dict[str, Any], updated: list[tuple[str, str]]) -> None:
    if not updated:
        return
    completed = ledger.setdefault("completed_tasks", [])
    if not any(item.get("id") == "phase2-closure-001" for item in completed):
        completed.append(
            {
                "id": "phase2-closure-001",
                "title": "Closed GraphPilot backend-feasibility gaps for unsupported ASR/TTS/VLM adapter paths",
                "evidence": {
                    "command": "env -u PYTHONHOME -u PYTHONPATH python3 scripts/close_graphpilot_backend_gaps.py",
                    "result": "Explicit infeasible backend paths recorded for ASR/TTS GPU+NPU and VLM GPU using source/build evidence.",
                },
            }
        )
    ledger["current_phase"] = "Phase 2: Stage Adapters and Backend Feasibility Bring-Up"
    ledger["next_immediate_action"] = (
        "Run GraphPilot experiment baselines and candidate-plan comparisons now that backend feasibility coverage is explicit."
    )
    ledger["date"] = datetime.now(timezone.utc).date().isoformat()


def append_phase2_report(report_path: Path, updated: list[tuple[str, str]], matrix: dict[str, Any]) -> None:
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else "# Phase 2 Stage Feasibility Status\n"
    lines = [existing.rstrip(), "", "## Backend gap closure", ""]
    if not updated:
        lines.append("- No additional gap closures were required in this run.")
    else:
        for stage_id, backend in updated:
            state = next(stage for stage in matrix["stages"] if stage["stage_id"] == stage_id)["backends"][backend]
            lines.append(f"- `{stage_id}` / `{backend}` -> `{state['status']}`")
            lines.append(f"  - Reason: {state['last_summary']}")
            for evidence in state.get("evidence", []):
                lines.append(f"  - Evidence: `{evidence}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-matrix", type=Path, default=DEFAULT_BACKEND_MATRIX)
    parser.add_argument("--state-ledger", type=Path, default=DEFAULT_STATE_LEDGER)
    parser.add_argument("--phase2-report", type=Path, default=DEFAULT_PHASE2_REPORT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    matrix = load_json(args.backend_matrix)
    ledger = load_json(args.state_ledger)

    updated = apply_gap_closures(matrix)
    matrix["last_updated"] = datetime.now(timezone.utc).date().isoformat()
    update_state_ledger(ledger, updated)
    append_phase2_report(args.phase2_report, updated, matrix)

    write_json(args.backend_matrix, matrix)
    write_json(args.state_ledger, ledger)
    print(json.dumps({"updated": updated}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
