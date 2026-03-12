#!/usr/bin/env python3
"""Tune GraphPilot objective and scheduler weights against measured artifacts."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from graphpilot_edge.cost_model import ObjectiveWeights, compute_objective_score


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def expand_weight_grid(grid: dict[str, list[float]]) -> list[dict[str, float]]:
    keys = sorted(grid)
    return [
        {key: values[index] for index, key in enumerate(keys)}
        for values in itertools.product(*(grid[key] for key in keys))
    ]


def candidate_objective(plan: dict[str, Any], weights: ObjectiveWeights) -> float:
    predicted = plan.get("predicted_cost", {})
    return compute_objective_score(
        p95_e2e_ms=float(predicted.get("p95_e2e_ms", predicted.get("makespan_ms", 0.0))),
        p95_ttfs_ms=float(predicted.get("p95_ttfs_ms", predicted.get("ttfs_ms", predicted.get("makespan_ms", 0.0)))),
        avg_energy_mj=float(predicted.get("avg_energy_mj", predicted.get("energy_mj", 0.0))),
        peak_memory_bytes=int(predicted.get("peak_memory_bytes", float(predicted.get("memory_mb", 0.0)) * 1024 * 1024)),
        copy_bytes=int(predicted.get("copy_bytes", 0)),
        quality_loss=float(predicted.get("quality_loss", 0.0)),
        p95_queue_delay_ms=float(predicted.get("p95_queue_delay_ms", 0.0)),
        deadline_miss_rate=float(predicted.get("deadline_miss_rate", 0.0)),
        weights=weights,
    )


def actual_objective(workflow: dict[str, Any], weights: ObjectiveWeights) -> float:
    return compute_objective_score(
        p95_e2e_ms=float(workflow.get("warm_latency_ms", 0.0)),
        p95_ttfs_ms=float(workflow.get("tts_first_audio_ms", workflow.get("warm_latency_ms", 0.0))),
        avg_energy_mj=0.0,
        peak_memory_bytes=0,
        copy_bytes=0,
        quality_loss=0.0,
        p95_queue_delay_ms=float(workflow.get("p95_queue_delay_ms", 0.0)),
        deadline_miss_rate=float(workflow.get("deadline_miss_rate", 0.0)),
        weights=weights,
    )


def evaluate_objective_weights(
    candidate_plans: dict[str, Any],
    experiment_summary: dict[str, Any],
    weight_sets: Iterable[dict[str, float]],
) -> dict[str, Any]:
    actual_workflows = experiment_summary.get("actual_workflows", {})
    best: dict[str, Any] | None = None
    evaluations = []
    for raw in weight_sets:
        weights = ObjectiveWeights(**raw)
        total_error = 0.0
        per_workflow = {}
        for workflow_id, workflow in actual_workflows.items():
            candidates = [plan for plan in candidate_plans.get("plans", []) if plan.get("workflow_template") == workflow_id]
            if not candidates:
                continue
            best_candidate = min(candidates, key=lambda plan: candidate_objective(plan, weights))
            predicted_score = candidate_objective(best_candidate, weights)
            actual_score = actual_objective(workflow, weights)
            error = abs(predicted_score - actual_score)
            total_error += error
            per_workflow[workflow_id] = {
                "selected_plan_id": best_candidate["plan_id"],
                "predicted_score": predicted_score,
                "actual_score": actual_score,
                "absolute_error": error,
            }
        evaluation = {
            "weights": raw,
            "total_error": total_error,
            "per_workflow": per_workflow,
        }
        evaluations.append(evaluation)
        if best is None or evaluation["total_error"] < best["total_error"]:
            best = evaluation
    if best is None:
        raise ValueError("No objective-weight evaluations were produced. Remediation: ensure candidate plans and actual workflows are present.")
    return {"best": best, "evaluations": evaluations}


def evaluate_scheduler_weights(
    scheduler_weight_sets: Iterable[dict[str, float]],
    objective_best: dict[str, Any],
) -> dict[str, Any]:
    # Transparent heuristic: favor higher first-output weight while keeping copy/memory/thermal penalties bounded.
    best: dict[str, Any] | None = None
    evaluations = []
    for weights in scheduler_weight_sets:
        score = (
            10.0 * weights.get("first_output", 0.0)
            + 2.0 * weights.get("rank", 0.0)
            + weights.get("age", 0.0)
            - weights.get("copy", 0.0)
            - weights.get("memory", 0.0)
            - weights.get("thermal", 0.0)
            - 0.001 * objective_best["total_error"]
        )
        evaluation = {"weights": weights, "score": score}
        evaluations.append(evaluation)
        if best is None or evaluation["score"] > best["score"]:
            best = evaluation
    if best is None:
        raise ValueError("No scheduler-weight evaluations were produced.")
    return {"best": best, "evaluations": evaluations}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-plans", type=Path, required=True)
    parser.add_argument("--experiment-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_json(args.config)
    candidate_plans = load_json(args.candidate_plans)
    experiment_summary = load_json(args.experiment_summary)
    objective_eval = evaluate_objective_weights(
        candidate_plans,
        experiment_summary,
        expand_weight_grid(config["objective_weight_grid"]),
    )
    scheduler_eval = evaluate_scheduler_weights(
        expand_weight_grid(config["scheduler_weight_grid"]),
        objective_eval["best"],
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"graphpilot_hparam_tuning_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config.resolve()),
        "candidate_plans": str(args.candidate_plans.resolve()),
        "experiment_summary": str(args.experiment_summary.resolve()),
        "best_objective_weights": objective_eval["best"],
        "best_scheduler_weights": scheduler_eval["best"],
        "objective_evaluations": objective_eval["evaluations"],
        "scheduler_evaluations": scheduler_eval["evaluations"],
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(
        "# GraphPilot Hyperparameter Tuning\n\n"
        f"- best_objective_total_error={objective_eval['best']['total_error']:.3f}\n"
        f"- best_scheduler_score={scheduler_eval['best']['score']:.3f}\n",
        encoding="utf-8",
    )
    print(output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
