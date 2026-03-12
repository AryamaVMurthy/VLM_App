#!/usr/bin/env python3
"""Generate GraphPilot characterization, ablation, and baseline-comparison artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from graphpilot_edge.baselines import build_baseline_candidate  # noqa: E402
from graphpilot_edge.cost_model import kv_cache_size_bytes  # noqa: E402
from graphpilot_edge.hardware_simulator import (  # noqa: E402
    compute_batching_gain,
    compute_thermal_slowdown,
)
from graphpilot_edge.hardware_topology import (  # noqa: E402
    DEFAULT_HARDWARE_TOPOLOGY_PATH,
    load_hardware_topology,
)
from graphpilot_edge.model_graph_simulator import (  # noqa: E402
    build_llm_scenario,
    build_retrieval_scenario,
    build_stt_scenario,
    build_tts_scenario,
    build_vlm_scenario,
)
from graphpilot_edge.models import RequestSpec  # noqa: E402
from graphpilot_edge.simulation import simulate_candidate_plan, simulate_request_stream  # noqa: E402
from graphpilot_edge.workload_universe import (  # noqa: E402
    DEFAULT_WORKLOAD_UNIVERSE_PATH,
    WorkloadUniverse,
    load_workload_universe,
)

ARTIFACT_ROOT = ROOT_DIR / "artifacts" / "graphpilot_edge"
DEFAULT_OUTPUT_ROOT = ARTIFACT_ROOT / "analysis"
DEFAULT_PROFILER_REGISTRY = ARTIFACT_ROOT / "registries" / "profiler_registry.json"
DEFAULT_CALIBRATION_SUMMARY = ARTIFACT_ROOT / "analysis" / "graphpilot_cost_calibration_latest.json"
DEFAULT_BASELINES = (
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
MEBIBYTE = 1024 * 1024


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-universe", type=Path, default=DEFAULT_WORKLOAD_UNIVERSE_PATH)
    parser.add_argument("--hardware-topology", type=Path, default=DEFAULT_HARDWARE_TOPOLOGY_PATH)
    parser.add_argument("--profiler-registry", type=Path, default=DEFAULT_PROFILER_REGISTRY)
    parser.add_argument("--calibration-summary", type=Path, default=None)
    parser.add_argument("--checkpoint-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_checkpoint_inputs(manifest_path: Path | None) -> dict[str, Path]:
    if manifest_path is None:
        return {}
    manifest = load_json(manifest_path)
    canonical = manifest.get("canonical_evidence_paths") or {}
    if not isinstance(canonical, dict):
        raise ValueError("Checkpoint manifest canonical_evidence_paths must be a mapping.")
    return {
        key: Path(value)
        for key, value in canonical.items()
        if value is not None
    }


def resource_capacities_from_topology(simulator) -> dict[str, int]:
    capacities: dict[str, int] = {}
    for resource_id in simulator.resource_ids():
        backend = simulator.backend_label(resource_id)
        capacities[backend] = capacities.get(backend, 0) + 1
    return capacities


def _non_continuous_specs(universe: WorkloadUniverse) -> list[Any]:
    return [
        spec
        for spec in universe.specs.values()
        if spec.category not in {"continuous_stream", "mixed_criticality_stream"}
    ]


def _stage_predictions(node_profile, simulator) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for resource_id in simulator.resource_ids():
        try:
            prediction = simulator.predict_task(
                task=node_profile.task_profile,
                resource_id=resource_id,
                batch_size=1,
                current_temp_c={},
                utilizations={},
            )
        except ValueError:
            continue
        predictions.append(
            {
                "resource_id": resource_id,
                "backend": simulator.backend_label(resource_id),
                "execution_time_ms": prediction.execution_time_ms,
                "energy_mj": prediction.energy_mj,
                "used_fallback": prediction.used_fallback,
            }
        )
    predictions.sort(key=lambda item: (item["execution_time_ms"], item["resource_id"]))
    return predictions


def build_backend_affinity(universe: WorkloadUniverse, simulator) -> dict[str, Any]:
    resource_win_counts = {backend: 0 for backend in ("cpu", "gpu", "npu")}
    family_backend_latencies: dict[str, dict[str, list[float]]] = {}
    stage_rows: list[dict[str, Any]] = []

    for spec in _non_continuous_specs(universe):
        scenario = universe.build_scenario(spec.workload_id)
        for stage_id in scenario.workflow.topological_order():
            node = scenario.node_profiles[stage_id]
            predictions = _stage_predictions(node, simulator)
            if not predictions:
                continue
            best = predictions[0]
            resource_win_counts[best["backend"]] = resource_win_counts.get(best["backend"], 0) + 1
            family_backend_latencies.setdefault(node.family, {})
            for prediction in predictions:
                family_backend_latencies[node.family].setdefault(prediction["backend"], []).append(
                    prediction["execution_time_ms"]
                )
            stage_rows.append(
                {
                    "workload_id": spec.workload_id,
                    "stage_id": stage_id,
                    "family": node.family,
                    "best_backend": best["backend"],
                    "best_resource_id": best["resource_id"],
                    "predictions": predictions,
                }
            )

    family_backend_mean_ms = {
        family: {
            backend: round(statistics.mean(values), 4)
            for backend, values in sorted(backend_rows.items())
        }
        for family, backend_rows in sorted(family_backend_latencies.items())
    }
    return {
        "resource_win_counts": resource_win_counts,
        "family_backend_mean_ms": family_backend_mean_ms,
        "stage_rows": stage_rows,
    }


def _simulate_baseline(universe: WorkloadUniverse, spec, baseline_id: str, simulator, capacities) -> dict[str, Any]:
    if spec.category == "continuous_stream":
        scenario = universe.build_scenario(spec.base_workload_id or spec.workload_id)
        candidate = build_baseline_candidate(
            scenario,
            simulator=simulator,
            baseline_id=baseline_id,
        )
        requests = universe.build_request_specs(
            spec.workload_id,
            simulator=simulator,
            resource_assignment=candidate.resource_assignment,
        )
        simulation = simulate_request_stream(requests, capacities)
        return {
            "baseline_id": baseline_id,
            "status": "ok",
            "score_ms": float(simulation.makespan_ms),
            "makespan_ms": simulation.makespan_ms,
            "p95_queue_delay_ms": simulation.p95_queue_delay_ms,
            "deadline_miss_rate": simulation.deadline_miss_rate,
            "ttfs_ms": simulation.p95_ttfs_ms,
            "resource_assignment": dict(candidate.resource_assignment),
        }
    if spec.category == "mixed_criticality_stream":
        resource_assignments: dict[str, dict[str, str]] = {}
        requests: list[RequestSpec] = []
        for group in spec.request_groups:
            base_workload_id = str(group["base_workload_id"])
            assignment = resource_assignments.get(base_workload_id)
            if assignment is None:
                scenario = universe.build_scenario(base_workload_id)
                candidate = build_baseline_candidate(
                    scenario,
                    simulator=simulator,
                    baseline_id=baseline_id,
                )
                assignment = dict(candidate.resource_assignment)
                resource_assignments[base_workload_id] = assignment
            scenario = universe.build_scenario(base_workload_id)
            plan = scenario.instantiate_candidate_plan(
                simulator=simulator,
                resource_assignment=assignment,
            )
            deadline_ms = group.get("deadline_ms", spec.deadline_ms)
            criticality_class = str(group.get("criticality_class", "default"))
            for request_index, arrival_ms in enumerate(group.get("arrivals_ms", ())):
                requests.append(
                    RequestSpec(
                        request_id=f"{spec.workload_id}:{base_workload_id}:request_{request_index}",
                        arrival_ms=int(arrival_ms),
                        workflow=scenario.workflow,
                        plan=plan,
                        deadline_ms=int(deadline_ms) if deadline_ms is not None else None,
                        criticality_class=criticality_class,
                        source_workload_id=base_workload_id,
                    )
                )
        simulation = simulate_request_stream(tuple(requests), capacities)
        return {
            "baseline_id": baseline_id,
            "status": "ok",
            "score_ms": float(simulation.makespan_ms),
            "makespan_ms": simulation.makespan_ms,
            "p95_queue_delay_ms": simulation.p95_queue_delay_ms,
            "deadline_miss_rate": simulation.deadline_miss_rate,
            "ttfs_ms": simulation.p95_ttfs_ms,
            "resource_assignment": resource_assignments,
        }
    scenario = universe.build_scenario(spec.base_workload_id or spec.workload_id)
    candidate = build_baseline_candidate(
        scenario,
        simulator=simulator,
        baseline_id=baseline_id,
    )
    simulation = simulate_candidate_plan(scenario.workflow, candidate.plan, capacities)
    return {
        "baseline_id": baseline_id,
        "status": "ok",
        "score_ms": float(simulation.makespan_ms),
        "makespan_ms": simulation.makespan_ms,
        "p95_queue_delay_ms": simulation.p95_queue_delay_ms,
        "deadline_miss_rate": simulation.deadline_miss_rate,
        "ttfs_ms": simulation.ttfs_ms,
        "resource_assignment": dict(candidate.resource_assignment),
    }


def build_baseline_comparisons(universe: WorkloadUniverse, simulator) -> dict[str, Any]:
    capacities = resource_capacities_from_topology(simulator)
    grouped: dict[str, list[dict[str, Any]]] = {
        "model_family_workloads": [],
        "compound_workloads": [],
        "continuous_workloads": [],
        "stress_workloads": [],
    }
    for spec in universe.specs.values():
        category_key = {
            "model_family": "model_family_workloads",
            "compound_assistant": "compound_workloads",
            "continuous_stream": "continuous_workloads",
            "mixed_criticality_stream": "continuous_workloads",
            "stress_failure": "stress_workloads",
            "primitive_operator": "model_family_workloads",
        }.get(spec.category)
        if category_key is None:
            continue
        baseline_rows = []
        for baseline_id in DEFAULT_BASELINES:
            try:
                baseline_rows.append(
                    _simulate_baseline(universe, spec, baseline_id, simulator, capacities)
                )
            except ValueError as exc:
                baseline_rows.append(
                    {
                        "baseline_id": baseline_id,
                        "status": "error",
                        "error": str(exc),
                    }
                )
        ok_rows = [row for row in baseline_rows if row["status"] == "ok"]
        graphpilot_row = next((row for row in ok_rows if row["baseline_id"] == "static_best_map"), None)
        if graphpilot_row is None:
            raise ValueError(
                f"Workload '{spec.workload_id}' has no support-safe static_best_map baseline for characterization."
            )
        best_other = min(
            (row for row in ok_rows if row["baseline_id"] != "static_best_map"),
            key=lambda row: row["score_ms"],
            default=None,
        )
        grouped[category_key].append(
            {
                "workload_id": spec.workload_id,
                "description": spec.description,
                "graphpilot_policy": "static_best_map",
                "graphpilot_score_ms": graphpilot_row["score_ms"],
                "graphpilot_metrics": graphpilot_row,
                "best_other_baseline_id": best_other["baseline_id"] if best_other else None,
                "best_other_score_ms": best_other["score_ms"] if best_other else None,
                "graphpilot_margin_vs_best_other_ms": (
                    best_other["score_ms"] - graphpilot_row["score_ms"] if best_other else None
                ),
                "baselines": baseline_rows,
            }
        )
    return grouped


def build_ablations(
    baseline_comparisons: dict[str, Any],
    profiler_registry: dict[str, Any] | None,
) -> dict[str, Any]:
    pipeline_rows: list[dict[str, Any]] = []
    single_backend_rows: list[dict[str, Any]] = []
    for rows in baseline_comparisons.values():
        for row in rows:
            baselines = {item["baseline_id"]: item for item in row["baselines"] if item["status"] == "ok"}
            graphpilot_score = row["graphpilot_score_ms"]
            no_pipeline = baselines.get("no_pipeline")
            if no_pipeline is not None:
                pipeline_rows.append(
                    {
                        "workload_id": row["workload_id"],
                        "graphpilot_score_ms": graphpilot_score,
                        "ablation_score_ms": no_pipeline["score_ms"],
                        "delta_ms": no_pipeline["score_ms"] - graphpilot_score,
                        "speedup_vs_ablation": (
                            no_pipeline["score_ms"] / graphpilot_score if graphpilot_score else None
                        ),
                    }
                )
            single_backend_candidates = [
                baselines[baseline_id]
                for baseline_id in ("cpu_only", "gpu_only", "npu_only")
                if baseline_id in baselines
            ]
            if single_backend_candidates:
                best_single = min(single_backend_candidates, key=lambda item: item["score_ms"])
                single_backend_rows.append(
                    {
                        "workload_id": row["workload_id"],
                        "graphpilot_score_ms": graphpilot_score,
                        "best_single_backend_id": best_single["baseline_id"],
                        "best_single_backend_score_ms": best_single["score_ms"],
                        "delta_ms": best_single["score_ms"] - graphpilot_score,
                    }
                )

    retrieval_backend: dict[str, Any] | None = None
    if profiler_registry is not None:
        cpu_entry = _latest_profiler_entry(
            profiler_registry,
            stage_id="workflow_c_voice_vision_retrieval",
            variant="graphpilot_vlm_npu_retrieval_cpu",
        )
        gpu_entry = _latest_profiler_entry(
            profiler_registry,
            stage_id="workflow_c_voice_vision_retrieval",
            variant="graphpilot_vlm_npu_retrieval_gpu",
        )
        if cpu_entry and gpu_entry:
            retrieval_backend = {
                "cpu_warm_latency_ms": cpu_entry["metrics"].get("warm_latency_ms"),
                "gpu_warm_latency_ms": gpu_entry["metrics"].get("warm_latency_ms"),
                "cpu_ttft_ms": cpu_entry["metrics"].get("ttft_ms"),
                "gpu_ttft_ms": gpu_entry["metrics"].get("ttft_ms"),
                "delta_ms": gpu_entry["metrics"].get("warm_latency_ms", 0)
                - cpu_entry["metrics"].get("warm_latency_ms", 0),
            }
    return {
        "pipeline": pipeline_rows,
        "single_backend": single_backend_rows,
        "retrieval_backend": retrieval_backend,
    }


def _latest_profiler_entry(
    profiler_registry: dict[str, Any],
    *,
    stage_id: str,
    variant: str,
) -> dict[str, Any] | None:
    matches = [
        entry
        for entry in profiler_registry.get("entries", [])
        if entry.get("stage_id") == stage_id and entry.get("variant") == variant
    ]
    if not matches:
        return None
    return sorted(matches, key=lambda item: item.get("recorded_at", ""))[-1]


def build_transfer_matrix(simulator) -> list[dict[str, Any]]:
    rows = []
    for src_resource_id in simulator.resource_ids():
        for dst_resource_id in simulator.resource_ids():
            if src_resource_id == dst_resource_id:
                continue
            rows.append(
                {
                    "src_resource_id": src_resource_id,
                    "dst_resource_id": dst_resource_id,
                    "transfer_time_ms_per_mib": round(
                        simulator.compute_transfer_time_ms(
                            size_bytes=MEBIBYTE,
                            src_resource_id=src_resource_id,
                            dst_resource_id=dst_resource_id,
                        ),
                        6,
                    ),
                }
            )
    return rows


def build_batching_curves(simulator) -> dict[str, list[dict[str, float]]]:
    curves: dict[str, list[dict[str, float]]] = {}
    for resource_id in simulator.resource_ids():
        resource = simulator.resource(resource_id)
        curves[resource_id] = [
            {
                "batch_size": float(batch_size),
                "gain": round(
                    compute_batching_gain(batch_size, resource.batching_beta, resource.batching_tau), 6
                ),
            }
            for batch_size in (1, 2, 4, 8)
        ]
    return curves


def build_thermal_curves(simulator) -> dict[str, list[dict[str, float]]]:
    curves: dict[str, list[dict[str, float]]] = {}
    for resource_id in simulator.resource_ids():
        resource = simulator.resource(resource_id)
        curves[resource_id] = [
            {
                "temp_c": float(temp_c),
                "slowdown": round(
                    compute_thermal_slowdown(temp_c, resource.thermal_threshold_c, resource.thermal_gamma),
                    6,
                ),
            }
            for temp_c in (30, 35, 40, 45, 50)
        ]
    return curves


def build_memory_kv_curves() -> dict[str, Any]:
    kv_curve = [
        {
            "cached_tokens": float(tokens),
            "kv_bytes": float(
                kv_cache_size_bytes(
                    layers=18,
                    kv_heads=8,
                    head_dim=256,
                    cached_tokens=tokens,
                    bytes_per_element=2,
                )
            ),
        }
        for tokens in (256, 512, 1024, 2048, 4096)
    ]
    llm_curve = []
    for context_tokens in (256, 512, 1024, 2048):
        scenario = build_llm_scenario(
            scenario_id=f"char.llm.context_{context_tokens}",
            prompt_tokens=256,
            output_tokens=128,
            context_tokens=context_tokens,
            context_cap=context_tokens,
            max_output_tokens=128,
        )
        total_memory_bytes = sum(
            node.task_profile.memory_bytes for node in scenario.node_profiles.values()
        )
        llm_curve.append(
            {
                "context_tokens": float(context_tokens),
                "aggregate_memory_bytes": float(total_memory_bytes),
            }
        )
    return {
        "kv_curve": kv_curve,
        "llm_context_curve": llm_curve,
    }


def _best_score_for_scenario(scenario, simulator) -> float:
    candidate = build_baseline_candidate(
        scenario,
        simulator=simulator,
        baseline_id="static_best_map",
    )
    return candidate.score_ms


def build_knob_frontiers(simulator) -> dict[str, list[dict[str, Any]]]:
    frontiers = {
        "stt": [],
        "vlm": [],
        "retrieval": [],
        "llm": [],
        "tts": [],
    }
    for chunk_size_ms in (400, 800, 1600):
        for beam_size in (2, 4, 8):
            scenario = build_stt_scenario(
                scenario_id=f"char.stt.{chunk_size_ms}.{beam_size}",
                audio_duration_ms=3200,
                chunk_size_ms=chunk_size_ms,
                beam_size=beam_size,
            )
            frontiers["stt"].append(
                {
                    "chunk_size_ms": chunk_size_ms,
                    "beam_size": beam_size,
                    "score_ms": _best_score_for_scenario(scenario, simulator),
                    "quality_proxy_loss": round(max(0.0, (4 - min(beam_size, 4)) * 0.08), 4),
                }
            )
    for visual_token_budget in (128, 256, 384):
        for image_resolution in (512, 768, 896):
            scenario = build_vlm_scenario(
                scenario_id=f"char.vlm.{image_resolution}.{visual_token_budget}",
                prompt_tokens=96,
                output_tokens=48,
                image_resolution=image_resolution,
                visual_token_budget=visual_token_budget,
            )
            frontiers["vlm"].append(
                {
                    "image_resolution": image_resolution,
                    "visual_token_budget": visual_token_budget,
                    "score_ms": _best_score_for_scenario(scenario, simulator),
                    "quality_proxy_loss": round(
                        max(0.0, (256 - min(visual_token_budget, 256)) / 256.0 * 0.25), 4
                    ),
                }
            )
    for top_k in (2, 4, 8):
        for rerank in (False, True):
            scenario = build_retrieval_scenario(
                scenario_id=f"char.retrieval.{top_k}.{int(rerank)}",
                query_tokens=32,
                corpus_size=1000,
                top_k=top_k,
                rerank=rerank,
            )
            frontiers["retrieval"].append(
                {
                    "top_k": top_k,
                    "rerank": rerank,
                    "score_ms": _best_score_for_scenario(scenario, simulator),
                    "quality_proxy_loss": round(max(0.0, (4 - min(top_k, 4)) * 0.05), 4),
                }
            )
    for context_cap in (1024, 2048, 4096):
        for max_output_tokens in (64, 128, 192):
            scenario = build_llm_scenario(
                scenario_id=f"char.llm.{context_cap}.{max_output_tokens}",
                prompt_tokens=384,
                output_tokens=192,
                context_tokens=2048,
                context_cap=context_cap,
                max_output_tokens=max_output_tokens,
            )
            quality_loss = sum(node.quality_loss for node in scenario.node_profiles.values())
            frontiers["llm"].append(
                {
                    "context_cap": context_cap,
                    "max_output_tokens": max_output_tokens,
                    "score_ms": _best_score_for_scenario(scenario, simulator),
                    "quality_proxy_loss": round(quality_loss, 4),
                }
            )
    for chunk_size_chars in (48, 80, 128):
        for quality_mode in ("fast", "balanced", "high"):
            scenario = build_tts_scenario(
                scenario_id=f"char.tts.{chunk_size_chars}.{quality_mode}",
                text_tokens=96,
                chunk_size_chars=chunk_size_chars,
                quality_mode=quality_mode,
            )
            frontiers["tts"].append(
                {
                    "chunk_size_chars": chunk_size_chars,
                    "quality_mode": quality_mode,
                    "score_ms": _best_score_for_scenario(scenario, simulator),
                    "quality_proxy_loss": {"fast": 0.18, "balanced": 0.06, "high": 0.0}[quality_mode],
                }
            )
    for rows in frontiers.values():
        rows.sort(key=lambda item: (item["score_ms"], item["quality_proxy_loss"]))
    return frontiers


def build_characterization_summary(
    *,
    workload_universe_path: Path = DEFAULT_WORKLOAD_UNIVERSE_PATH,
    hardware_topology_path: Path = DEFAULT_HARDWARE_TOPOLOGY_PATH,
    profiler_registry_path: Path = DEFAULT_PROFILER_REGISTRY,
    calibration_summary_path: Path | None = None,
    checkpoint_manifest_path: Path | None = None,
) -> dict[str, Any]:
    universe = load_workload_universe(workload_universe_path)
    topology = load_hardware_topology(hardware_topology_path)
    profiler_registry = (
        load_json(profiler_registry_path) if profiler_registry_path.exists() else None
    )
    calibration_summary = (
        load_json(calibration_summary_path)
        if calibration_summary_path is not None and calibration_summary_path.exists()
        else None
    )
    simulator = topology.simulator
    baseline_comparisons = build_baseline_comparisons(universe, simulator)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workload_universe": str(workload_universe_path.resolve()),
        "hardware_topology": str(hardware_topology_path.resolve()),
        "profiler_registry": str(profiler_registry_path.resolve()),
        "calibration_summary": str(calibration_summary_path.resolve()) if calibration_summary_path is not None else None,
        "checkpoint_manifest": str(checkpoint_manifest_path.resolve()) if checkpoint_manifest_path is not None else None,
        "surrogate_parameter_notes": dict(topology.surrogate_parameter_notes),
        "public_anchors": list(topology.public_anchors),
        "calibration_quality_by_family": (calibration_summary or {}).get("calibration_quality_by_family", {}),
        "backend_affinity": build_backend_affinity(universe, simulator),
        "baseline_comparisons": baseline_comparisons,
        "ablations": build_ablations(baseline_comparisons, profiler_registry),
        "transfer_matrix_ms_per_mib": build_transfer_matrix(simulator),
        "batching_curves": build_batching_curves(simulator),
        "thermal_curves": build_thermal_curves(simulator),
        "memory_kv_curves": build_memory_kv_curves(),
        "knob_frontiers": build_knob_frontiers(simulator),
        "graphpilot_policy_note": (
            "Simulator-side baseline comparisons use static_best_map as the current support-safe "
            "GraphPilot policy proxy over the available feasible set."
        ),
        "quality_proxy_note": (
            "Knob frontier quality values are explicit surrogate quality-loss proxies used for "
            "simulator-side tradeoff studies, not measured model-evaluation scores."
        ),
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# GraphPilot Characterization and Ablations",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Workload universe: `{summary['workload_universe']}`",
        f"- Hardware topology: `{summary['hardware_topology']}`",
        f"- Calibration summary: `{summary['calibration_summary']}`",
        f"- Checkpoint manifest: `{summary['checkpoint_manifest']}`",
        "",
        "## Backend affinity",
        "",
        f"- Resource win counts: `{summary['backend_affinity']['resource_win_counts']}`",
        "",
        "## Baseline comparisons",
        "",
        f"- Policy note: {summary['graphpilot_policy_note']}",
    ]
    for section_name, rows in summary["baseline_comparisons"].items():
        lines.extend(["", f"### {section_name}", ""])
        for row in rows:
            lines.append(
                f"- `{row['workload_id']}` graphpilot_policy=`{row['graphpilot_policy']}` "
                f"graphpilot_score_ms={row['graphpilot_score_ms']} "
                f"best_other_baseline=`{row['best_other_baseline_id']}` "
                f"margin_ms={row['graphpilot_margin_vs_best_other_ms']}"
            )
    lines.extend(["", "## Ablations", ""])
    for row in summary["ablations"]["pipeline"]:
        lines.append(
            f"- pipeline `{row['workload_id']}` graphpilot_ms={row['graphpilot_score_ms']} "
            f"no_pipeline_ms={row['ablation_score_ms']} delta_ms={row['delta_ms']}"
        )
    for row in summary["ablations"]["single_backend"]:
        lines.append(
            f"- single-backend `{row['workload_id']}` best_single=`{row['best_single_backend_id']}` "
            f"delta_ms={row['delta_ms']}"
        )
    retrieval_backend = summary["ablations"].get("retrieval_backend")
    if retrieval_backend:
        lines.append(
            f"- retrieval backend actual delta_ms={retrieval_backend['delta_ms']} "
            f"cpu_warm_ms={retrieval_backend['cpu_warm_latency_ms']} "
            f"gpu_warm_ms={retrieval_backend['gpu_warm_latency_ms']}"
        )
    lines.extend(
        [
            "",
            "## Characterization primitives",
            "",
            f"- Transfer matrix entries: {len(summary['transfer_matrix_ms_per_mib'])}",
            f"- Batching resources: {sorted(summary['batching_curves'])}",
            f"- Thermal resources: {sorted(summary['thermal_curves'])}",
            f"- Memory/KV curves: {sorted(summary['memory_kv_curves'])}",
            "",
            "## Knob frontiers",
            "",
            f"- Quality proxy note: {summary['quality_proxy_note']}",
        ]
    )
    if summary["calibration_quality_by_family"]:
        lines.extend(["", "## Calibration quality by family", ""])
        for family, payload in sorted(summary["calibration_quality_by_family"].items()):
            lines.append(
                f"- `{family}` sample_count={payload.get('sample_count')} "
                f"mean_absolute_error_ms={payload.get('mean_absolute_error_ms')}"
            )
    for family, rows in sorted(summary["knob_frontiers"].items()):
        best = rows[0] if rows else None
        if best is None:
            continue
        lines.append(f"- `{family}` best surrogate point: `{best}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint_inputs = resolve_checkpoint_inputs(args.checkpoint_manifest)
    summary = build_characterization_summary(
        workload_universe_path=args.workload_universe,
        hardware_topology_path=args.hardware_topology,
        profiler_registry_path=args.profiler_registry,
        calibration_summary_path=args.calibration_summary or checkpoint_inputs.get("calibration_summary"),
        checkpoint_manifest_path=args.checkpoint_manifest,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"graphpilot_characterization_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "report.md"
    write_report(report_path, summary)
    payload = {
        **summary,
        "report": str(report_path.resolve()),
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, payload)
    print(summary_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
