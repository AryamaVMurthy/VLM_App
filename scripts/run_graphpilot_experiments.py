#!/usr/bin/env python3
"""Run GraphPilot experiment aggregation and optional actual workflow re-profiling."""

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
ARTIFACT_ROOT = ROOT_DIR / "artifacts" / "graphpilot_edge"
DEFAULT_PROFILER_REGISTRY = ARTIFACT_ROOT / "registries" / "profiler_registry.json"
DEFAULT_CANDIDATE_PLANS = ARTIFACT_ROOT / "registries" / "candidate_plan_registry.json"
DEFAULT_BACKEND_MATRIX = ARTIFACT_ROOT / "registries" / "backend_feasibility_matrix.json"
DEFAULT_EXPERIMENT_REGISTRY = ARTIFACT_ROOT / "registries" / "experiment_registry.json"
DEFAULT_OUTPUT_ROOT = ARTIFACT_ROOT / "experiments"
DEFAULT_RERUN_PROFILES = (
    "graphpilot_workflow_a",
    "graphpilot_workflow_b",
    "graphpilot_workflow_c",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def latest_entries_by_variant(registry: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in registry.get("entries", []):
        key = (entry["stage_id"], entry["variant"])
        previous = latest.get(key)
        if previous is None or entry.get("recorded_at", "") > previous.get("recorded_at", ""):
            latest[key] = entry
    return latest


def workflow_variant_preference(workflow_id: str, variant: str) -> int | None:
    if workflow_id == "workflow_a_voice_only":
        if variant == "graphpilot_cpu_stack":
            return 0
        if variant == "voice_pipeline_cpu":
            return 1
        return None
    if workflow_id == "workflow_b_voice_vision":
        if variant.startswith("graphpilot_vlm_"):
            return 0
        return None
    if workflow_id == "workflow_c_voice_vision_retrieval":
        if variant.startswith("graphpilot_vlm_") and "_retrieval_" in variant:
            return 0
        return None
    return None


def latest_actual_workflow_entries(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for entry in registry.get("entries", []):
        workflow_id = entry["stage_id"]
        preference = workflow_variant_preference(workflow_id, entry.get("variant", ""))
        if preference is None:
            continue
        grouped.setdefault(workflow_id, []).append((preference, entry))
    selected: dict[str, dict[str, Any]] = {}
    for workflow_id, entries in grouped.items():
        entries.sort(key=lambda item: (item[0], item[1].get("recorded_at", "")))
        selected[workflow_id] = entries[-1][1]
        best_pref = min(item[0] for item in entries)
        same_pref = [item[1] for item in entries if item[0] == best_pref]
        same_pref.sort(key=lambda item: item.get("recorded_at", ""))
        selected[workflow_id] = same_pref[-1]
    return selected


def stage_has_feasible_backend(backend_matrix: dict[str, Any], stage_id: str) -> bool:
    for stage in backend_matrix.get("stages", []):
        if stage["stage_id"] != stage_id:
            continue
        return any(
            state.get("status") in {"feasible_smoke_pass", "known_working"}
            for state in stage["backends"].values()
        )
    return False


def summarize_results(
    profiler_registry: dict[str, Any],
    candidate_plans: dict[str, Any],
    backend_matrix: dict[str, Any],
) -> dict[str, Any]:
    latest = latest_actual_workflow_entries(profiler_registry)

    actual_workflows: dict[str, Any] = {}
    for workflow_id in (
        "workflow_a_voice_only",
        "workflow_b_voice_vision",
        "workflow_c_voice_vision_retrieval",
    ):
        entry = latest.get(workflow_id)
        if entry is None:
            continue
        key = entry["metrics"].get("workflow_id", workflow_id)
        actual_workflows[key] = {
            "variant": entry["variant"],
            "recorded_at": entry["recorded_at"],
            "plan_id": entry["metrics"].get("plan_id"),
            "state_id": entry["metrics"].get("state_id"),
            "warm_latency_ms": entry["metrics"].get("warm_latency_ms"),
            "cold_latency_ms": entry["metrics"].get("cold_latency_ms"),
            "ttft_ms": entry["metrics"].get("ttft_ms"),
            "tts_first_chunk_queued_ms": entry["metrics"].get("tts_first_chunk_queued_ms"),
            "tts_first_audio_ms": entry["metrics"].get("tts_first_audio_ms"),
            "stage_backends": entry["metrics"].get("stage_backends", {}),
            "artifacts": entry.get("artifacts", {}),
        }

    candidate_summary: dict[str, list[dict[str, Any]]] = {}
    for plan in candidate_plans.get("plans", []):
        workflow_id = plan["workflow_template"]
        candidate_summary.setdefault(workflow_id, []).append(
            {
                "plan_id": plan["plan_id"],
                "state_id": plan["state_id"],
                "backend_map": plan["backend_map"],
                "predicted_makespan_ms": plan["predicted_cost"]["makespan_ms"],
                "predicted_latency_sum_ms": plan["predicted_cost"]["latency_sum_ms"],
                "predicted_objective_score": plan["predicted_cost"].get("objective_score"),
            }
        )
    for plans in candidate_summary.values():
        plans.sort(
            key=lambda item: (
                item["predicted_objective_score"]
                if item["predicted_objective_score"] is not None
                else item["predicted_makespan_ms"],
                item["predicted_makespan_ms"],
                item["plan_id"],
            )
        )

    comparisons = []
    for workflow_id, workflow_actual in actual_workflows.items():
        candidates = candidate_summary.get(workflow_id, [])
        if not candidates:
            continue
        best = candidates[0]
        comparisons.append(
            {
                "workflow_id": workflow_id,
                "actual_variant": workflow_actual["variant"],
                "actual_warm_latency_ms": workflow_actual["warm_latency_ms"],
                "candidate_plan_id": best["plan_id"],
                "candidate_predicted_makespan_ms": best["predicted_makespan_ms"],
                "candidate_predicted_objective_score": best.get("predicted_objective_score"),
                "latency_delta_ms": workflow_actual["warm_latency_ms"] - best["predicted_makespan_ms"],
                "comparison_scope": "predicted_candidate_vs_latest_actual",
            }
        )

    blocked_workflows = []
    if not stage_has_feasible_backend(backend_matrix, "retrieval.embedder.primary"):
        blocked_workflows.append(
            {
                "workflow_id": "workflow_c_voice_vision_retrieval",
                "reason": "retrieval_infeasible",
                "detail": "Retrieval stage has no feasible backend in the current backend matrix.",
            }
        )
    elif "workflow_c_voice_vision_retrieval" not in actual_workflows:
        blocked_workflows.append(
            {
                "workflow_id": "workflow_c_voice_vision_retrieval",
                "reason": "workflow_not_profiled",
                "detail": "Retrieval is feasible, but workflow C has not been profiled in the current experiment batch.",
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "actual_workflows": actual_workflows,
        "candidate_workflows": candidate_summary,
        "comparisons": comparisons,
        "blocked_workflows": blocked_workflows,
    }


def adb_serial() -> str:
    result = subprocess.run(
        ["adb", "get-serialno"],
        cwd=ROOT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


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


def rerun_actual_profiles(profiles: tuple[str, ...], repeat: int, max_workers: int) -> list[Path]:
    summary_paths: list[Path] = []
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    for _ in range(repeat):
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "run_graphpilot_stage_profiler.py"),
            "--max-workers",
            str(max_workers),
            "--profiles",
            *profiles,
        ]
        result = subprocess.run(
            cmd,
            cwd=ROOT_DIR,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "GraphPilot stage profiler rerun failed.\n"
                f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
            )
        summary_paths.append(Path(result.stdout.strip().splitlines()[-1]).resolve())
    return summary_paths


def upsert_experiment_registry(
    registry: dict[str, Any],
    summary_path: Path,
    payload: dict[str, Any],
    commit: str,
    adb_serial: str,
) -> None:
    experiments = registry.setdefault("experiments", [])
    entry = {
        "experiment_id": payload["experiment_id"],
        "workflow_template": "graphpilot_phase6_7",
        "plan_id": "graphpilot_baseline_and_candidate_evaluation",
        "command": "env -u PYTHONHOME -u PYTHONPATH python3 scripts/run_graphpilot_experiments.py",
        "commit": commit,
        "device_state": {"adb_serial": adb_serial},
        "input_manifest": str(summary_path),
        "output_artifacts": [payload["output_dir"]],
        "metrics_path": str(summary_path),
        "logs_path": payload["output_dir"],
        "trace_path": None,
        "verdict": "pass",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    index = next((i for i, item in enumerate(experiments) if item["experiment_id"] == entry["experiment_id"]), None)
    if index is None:
        experiments.append(entry)
    else:
        experiments[index] = entry
    registry["last_updated"] = datetime.now(timezone.utc).date().isoformat()


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# GraphPilot Experiment Batch",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        "",
        "## Actual workflows",
        "",
    ]
    for workflow_id, workflow in sorted(summary["actual_workflows"].items()):
        lines.append(
            f"- `{workflow_id}` via `{workflow['variant']}` state=`{workflow.get('state_id')}` "
            f"plan=`{workflow.get('plan_id')}`: warm_latency_ms={workflow['warm_latency_ms']} "
            f"ttft_ms={workflow['ttft_ms']} "
            f"tts_first_chunk_queued_ms={workflow.get('tts_first_chunk_queued_ms')} "
            f"tts_first_audio_ms={workflow['tts_first_audio_ms']}"
        )
    lines.extend(["", "## Candidate workflows", ""])
    for workflow_id, plans in sorted(summary["candidate_workflows"].items()):
        lines.append(f"- `{workflow_id}`")
        for plan in plans[:4]:
            lines.append(
                f"  - `{plan['state_id']}` `{plan['plan_id']}` predicted_makespan_ms={plan['predicted_makespan_ms']} "
                f"predicted_objective_score={plan.get('predicted_objective_score')}"
            )
    lines.extend(["", "## Comparisons", ""])
    for row in summary["comparisons"]:
        lines.append(
            f"- `{row['workflow_id']}` actual={row['actual_warm_latency_ms']}ms vs candidate={row['candidate_predicted_makespan_ms']}ms "
            f"delta_ms={row['latency_delta_ms']} candidate_objective={row.get('candidate_predicted_objective_score')}"
        )
    lines.extend(["", "## Blocked workflows", ""])
    for row in summary["blocked_workflows"]:
        lines.append(f"- `{row['workflow_id']}` blocked: {row['detail']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiler-registry", type=Path, default=DEFAULT_PROFILER_REGISTRY)
    parser.add_argument("--candidate-plans", type=Path, default=DEFAULT_CANDIDATE_PLANS)
    parser.add_argument("--backend-matrix", type=Path, default=DEFAULT_BACKEND_MATRIX)
    parser.add_argument("--experiment-registry", type=Path, default=DEFAULT_EXPERIMENT_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rerun-actual", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(DEFAULT_RERUN_PROFILES),
        help="Profiler profiles to rerun when --rerun-actual is set.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rerun_actual:
        rerun_actual_profiles(tuple(args.profiles), args.repeat, args.max_workers)

    profiler_registry = load_json(args.profiler_registry)
    candidate_plans = load_json(args.candidate_plans)
    backend_matrix = load_json(args.backend_matrix)
    experiment_registry = load_json(args.experiment_registry)

    summary = summarize_results(profiler_registry, candidate_plans, backend_matrix)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"graphpilot_experiment_batch_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    payload = {
        "experiment_id": output_dir.name,
        "output_dir": str(output_dir),
        **summary,
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, payload)
    write_report(output_dir / "report.md", payload)

    upsert_experiment_registry(
        registry=experiment_registry,
        summary_path=summary_path,
        payload=payload,
        commit=git_commit(),
        adb_serial=adb_serial(),
    )
    write_json(args.experiment_registry, experiment_registry)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
