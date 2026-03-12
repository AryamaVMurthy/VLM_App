#!/usr/bin/env python3
"""Ingest a GraphPilot stage-feasibility summary into the registries and ledger."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
ARTIFACT_ROOT = ROOT_DIR / "artifacts" / "graphpilot_edge"
BACKEND_MATRIX_PATH = ARTIFACT_ROOT / "registries" / "backend_feasibility_matrix.json"
EXPERIMENT_REGISTRY_PATH = ARTIFACT_ROOT / "registries" / "experiment_registry.json"
STATE_LEDGER_PATH = ARTIFACT_ROOT / "state" / "state_ledger.json"
PHASE2_REPORT_PATH = ARTIFACT_ROOT / "reports" / "phase2_stage_feasibility_status.md"


@dataclass(frozen=True)
class StageBackendBinding:
    stage_id: str
    backend: str


STAGE_BACKEND_BINDINGS = {
    "asr": StageBackendBinding("asr.primary", "cpu"),
    "planner_cpu": StageBackendBinding("planner.primary", "cpu"),
    "planner_gpu": StageBackendBinding("planner.primary", "gpu"),
    "planner_npu": StageBackendBinding("planner.primary", "npu"),
    "responder_cpu": StageBackendBinding("responder.primary", "cpu"),
    "responder_gpu": StageBackendBinding("responder.primary", "gpu"),
    "responder_npu": StageBackendBinding("responder.primary", "npu"),
    "tts": StageBackendBinding("tts.primary", "cpu"),
    "retrieval_cpu": StageBackendBinding("retrieval.embedder.primary", "cpu"),
    "retrieval_gpu": StageBackendBinding("retrieval.embedder.primary", "gpu"),
    "retrieval_npu": StageBackendBinding("retrieval.embedder.primary", "npu"),
    "vlm_cpu": StageBackendBinding("vlm.fastvlm.primary", "cpu"),
}


FAILURE_NOTES_BY_STAGE_KEY = {
    "planner_gpu": (
        "LiteRT text-engine creation failed after GPU accelerator auto-registration failed on device."
    ),
    "planner_npu": (
        "LiteRT text-engine creation failed after NPU accelerator registration failed on device."
    ),
    "responder_gpu": (
        "LiteRT text-engine creation failed after GPU accelerator auto-registration failed on device."
    ),
    "responder_npu": (
        "LiteRT text-engine creation failed after NPU accelerator registration failed on device."
    ),
    "retrieval_gpu": (
        "EmbeddingGemma retrieval GPU init failed because LiteRT GPU accelerator registration/compile failed on device."
    ),
    "retrieval_npu": (
        "EmbeddingGemma retrieval NPU init failed because the Qualcomm LiteRT compiler-plugin/runtime path is ABI-incompatible on device."
    ),
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to resolve git commit: {result.stderr.strip()}")
    return result.stdout.strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_unique(strings: list[str], value: str) -> None:
    if value not in strings:
        strings.append(value)


def update_backend_matrix(matrix: dict[str, Any], summary_path: Path, summary: dict[str, Any]) -> None:
    stage_records = {stage["stage_id"]: stage for stage in matrix["stages"]}
    for run in summary["stage_runs"]:
        binding = STAGE_BACKEND_BINDINGS.get(run["stage_key"])
        if binding is None:
            continue
        stage = stage_records[binding.stage_id]
        backend_state = stage["backends"][binding.backend]
        backend_state["status"] = (
            "feasible_smoke_pass" if run["verdict"] == "pass" else "infeasible_smoke_fail"
        )
        for evidence in (
            str(summary_path),
            run["stdout_log"],
            run["stderr_log"],
            run.get("logcat_log"),
            run["test_class"],
        ):
            if evidence:
                append_unique(backend_state.setdefault("evidence", []), evidence)
        backend_state["last_verified_at"] = summary["generated_at"]
        backend_state["last_verdict"] = run["verdict"]
        backend_state["last_summary"] = run["summary"]
        if run["verdict"] != "pass":
            note = FAILURE_NOTES_BY_STAGE_KEY.get(run["stage_key"])
            if note:
                backend_state["notes"] = note


def upsert_experiment(registry: dict[str, Any], summary_path: Path, summary: dict[str, Any]) -> None:
    experiment_id = Path(summary["output_dir"]).name
    device_manifest = load_json(ARTIFACT_ROOT / "manifests" / "device_environment.json")
    experiment = {
        "experiment_id": experiment_id,
        "workflow_template": "stage_feasibility",
        "plan_id": "phase2_stage_feasibility",
        "command": "env -u PYTHONHOME -u PYTHONPATH python3 scripts/run_graphpilot_stage_feasibility.py",
        "commit": git_commit(),
        "device_state": {
            "adb_serial": device_manifest.get("device", {}).get("adb_serial", "unknown"),
            "install": summary.get("install", {}),
        },
        "input_manifest": str(summary_path),
        "output_artifacts": [summary["output_dir"]],
        "metrics_path": str(summary_path),
        "logs_path": summary["output_dir"],
        "trace_path": None,
        "verdict": "pass" if all(run["verdict"] == "pass" for run in summary["stage_runs"]) else "fail",
        "stage_runs": summary["stage_runs"],
        "recorded_at": utc_now(),
    }
    experiments = registry.setdefault("experiments", [])
    existing_index = next(
        (idx for idx, item in enumerate(experiments) if item["experiment_id"] == experiment_id),
        None,
    )
    if existing_index is None:
        experiments.append(experiment)
    else:
        experiments[existing_index] = experiment
    registry["last_updated"] = datetime.now(timezone.utc).date().isoformat()


def update_state_ledger(ledger: dict[str, Any], summary_path: Path, summary: dict[str, Any]) -> None:
    ledger["current_phase"] = "Phase 2: Stage Adapters and Backend Feasibility Bring-Up"
    completed = ledger.setdefault("completed_tasks", [])
    task_titles = {task["title"] for task in completed}
    new_entries = [
        {
            "id": "phase2-001",
            "title": "Executed Android stage-feasibility smokes for ASR, FastVLM CPU bridge, and voice pipeline",
            "evidence": {
                "command": "env -u PYTHONHOME -u PYTHONPATH python3 scripts/run_graphpilot_stage_feasibility.py --max-workers 6",
                "result": f"Stage-feasibility summary recorded at {summary_path}",
            },
        },
        {
            "id": "phase2-002",
            "title": "Fixed Android TTS package visibility and verified isolated TTS plus voice pipeline execution",
            "evidence": {
                "command": "env -u PYTHONHOME -u PYTHONPATH python3 scripts/run_graphpilot_stage_feasibility.py --max-workers 6 --stages tts voice_pipeline",
                "result": f"TTS and voice pipeline passed on device in {summary_path}",
            },
        },
    ]
    for entry in new_entries:
        if entry["title"] not in task_titles:
            completed.append(entry)
    ledger["active_ready_tasks"] = [
        "fvlm-5fw.2",
        "fvlm-5fw.10",
        "fvlm-5fw.11",
        "fvlm-5fw.12",
    ]
    ledger["next_immediate_action"] = (
        "Regenerate candidate plans using the updated backend matrix and stage them to device for calibrated Phase 3 workflow profiling."
    )


def write_phase2_report(
    summary_path: Path, summary: dict[str, Any], backend_matrix: dict[str, Any]
) -> None:
    lines = [
        "# Phase 2 Stage Feasibility Status",
        "",
        f"Date: {datetime.now(timezone.utc).date().isoformat()}",
        "",
        "## Latest verified batch",
        "",
        f"- Summary: `{summary_path}`",
        f"- Output directory: `{summary['output_dir']}`",
        "",
        "## Latest batch stage verdicts",
        "",
    ]
    for run in summary["stage_runs"]:
        lines.extend(
            [
                f"- `{run['stage_id']}` / `{run['stage_key']}`: `{run['verdict']}`",
                f"  - Test: `{run['test_class']}`",
                f"  - Stdout: `{run['stdout_log']}`",
                f"  - Stderr: `{run['stderr_log']}`",
                f"  - Logcat: `{run.get('logcat_log', 'missing')}`",
                f"  - Summary: `{run['summary']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Aggregated backend feasibility",
            "",
        ]
    )
    for stage in backend_matrix["stages"]:
        lines.append(f"- `{stage['stage_id']}`")
        for backend_name, backend_state in stage["backends"].items():
            lines.append(
                f"  - `{backend_name}`: `{backend_state['status']}`"
                + (
                    f" (last_verdict={backend_state.get('last_verdict')})"
                    if backend_state.get("last_verdict")
                    else ""
                )
            )
            if backend_state.get("notes"):
                lines.append(f"    - Notes: {backend_state['notes']}")

    lines.extend(
        [
            "",
            "## Root causes observed",
            "",
            "- Planner/Responder GPU: LiteRT GPU accelerator auto-registration failed during engine creation on device.",
            "- Planner/Responder NPU: LiteRT NPU accelerator registration failed during engine creation on device.",
            "- Retrieval GPU: EmbeddingGemma retrieval compile failed after GPU accelerator registration failed on device.",
            "- Retrieval NPU: EmbeddingGemma retrieval compile failed on the Qualcomm compiler-plugin/runtime ABI path.",
            "",
            "## Next step",
            "",
            "1. Regenerate candidate plans using only support-safe backends from the updated matrix.",
            "2. Stage candidate_plan_registry.json to device and run calibrated GraphPilot workflow profiling.",
            "3. Keep the preserved FastVLM LiteRT NPU path intact while the online coordinator executes plan-selected stages.",
            "",
        ]
    )
    PHASE2_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_path", type=Path, help="Path to stage-feasibility summary.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = args.summary_path.resolve()
    summary = load_json(summary_path)
    backend_matrix = load_json(BACKEND_MATRIX_PATH)
    experiment_registry = load_json(EXPERIMENT_REGISTRY_PATH)
    state_ledger = load_json(STATE_LEDGER_PATH)

    update_backend_matrix(backend_matrix, summary_path, summary)
    upsert_experiment(experiment_registry, summary_path, summary)
    update_state_ledger(state_ledger, summary_path, summary)
    write_phase2_report(summary_path, summary, backend_matrix)

    backend_matrix["last_updated"] = datetime.now(timezone.utc).date().isoformat()
    experiment_registry["last_updated"] = datetime.now(timezone.utc).date().isoformat()
    state_ledger["date"] = datetime.now(timezone.utc).date().isoformat()

    write_json(BACKEND_MATRIX_PATH, backend_matrix)
    write_json(EXPERIMENT_REGISTRY_PATH, experiment_registry)
    write_json(STATE_LEDGER_PATH, state_ledger)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
