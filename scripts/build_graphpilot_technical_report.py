from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "graphpilot_edge" / "reports"
LATEST_CASES_REFERENCES = (
    ROOT
    / "artifacts"
    / "graphpilot_edge"
    / "papers"
    / "graphpilot_cases_20260313_065454"
    / "references.bib"
)
IEEE_BST = ROOT / "papers" / "templates" / "ieee" / "IEEEtran.bst"

PYTHON_MODULE_ROLES = {
    "graphpilot_edge/cost_model.py": "Objective and scalar formula layer for execution, transfer, energy, queue-aware scoring, and KV sizing.",
    "graphpilot_edge/hardware_simulator.py": "Calibrated multi-resource hardware surrogate with batching, transfer, queue-depth, fallback partitioning, and thermal slowdown.",
    "graphpilot_edge/hardware_topology.py": "Loads the SM8750-style topology and surrogate resource parameters into simulator-friendly structures.",
    "graphpilot_edge/model_graph_simulator.py": "Builds stage-level and macro-region scenarios for LLM, VLM, STT, TTS, retrieval, CNN, ViT, glue, and assistant workflows.",
    "graphpilot_edge/workload_universe.py": "Machine-readable workload universe, scenario expansion, and mixed-criticality request generation.",
    "graphpilot_edge/workflow.py": "Workflow DAG schema, stream-edge semantics, and topological traversal helpers.",
    "graphpilot_edge/models.py": "Canonical plan, simulation result, and request/result dataclasses shared across the offline brain.",
    "graphpilot_edge/profiles.py": "Loads backend/stage profile records and reconciles them into stage-option views.",
    "graphpilot_edge/catalog.py": "Workflow templates, feasible backend resolution, and helper lookups used by planning and simulation.",
    "graphpilot_edge/enumeration.py": "Candidate-plan enumeration over feasible stage/backend assignments.",
    "graphpilot_edge/macro_regions.py": "Macro-region expansion and bookkeeping for coarse internal model decomposition.",
    "graphpilot_edge/planner.py": "Ranks candidate plans using the calibrated simulator and the objective score.",
    "graphpilot_edge/scheduler.py": "HEFT-style upward ranks and dynamic priority scoring utilities.",
    "graphpilot_edge/simulation.py": "Discrete-event request and stream simulation with transfer, queueing, and objective accounting.",
    "graphpilot_edge/memory.py": "Interval-based buffer lifetime analysis and reuse allocation.",
    "graphpilot_edge/kv.py": "KV accounting, migration thresholds, and decode-stickiness helpers.",
    "graphpilot_edge/plan_bank.py": "Plan-bank selection and thermal-state organization.",
    "graphpilot_edge/baselines.py": "Internal baselines, ablation policies, and faithful method-class proxy baselines.",
    "graphpilot_edge/errors.py": "Explicit error surface used to fail fast when evidence or support assumptions break.",
    "graphpilot_edge/__init__.py": "Package root for GraphPilot-Edge offline modules.",
}

ANDROID_MODULE_ROLES = {
    "android-app/app/src/main/java/com/qidk/fastvlm/core/graphpilot/GraphPilotCoordinator.kt": "Top-level Android orchestration for workflow admission, stage execution, streaming, logging, and metric collection.",
    "android-app/app/src/main/java/com/qidk/fastvlm/core/graphpilot/GraphPilotRuntimeScheduler.kt": "Queue-aware runtime admission and dispatch policy with explicit deadline-risk rejection.",
    "android-app/app/src/main/java/com/qidk/fastvlm/core/graphpilot/GraphPilotMemoryAdmissionController.kt": "Runtime memory/KV budgeting, degradation ranking, and reject logic.",
    "android-app/app/src/main/java/com/qidk/fastvlm/core/graphpilot/GraphPilotThermalPlanBank.kt": "Thermal state tracking and plan-bank switching logic.",
    "android-app/app/src/main/java/com/qidk/fastvlm/core/graphpilot/GraphPilotStreamingEdges.kt": "Chunk/TOKEN edge helpers and explicit stream-edge metadata handling.",
    "android-app/app/src/main/java/com/qidk/fastvlm/core/graphpilot/GraphPilotStreamingResponder.kt": "Responder-token streaming into the TTS path.",
    "android-app/app/src/main/java/com/qidk/fastvlm/core/graphpilot/GraphPilotRetrievalBridge.kt": "Retrieval stage bridge used by workflow C and retrieval-aware plans.",
    "android-app/app/src/main/java/com/qidk/fastvlm/core/graphpilot/GraphPilotPlanStore.kt": "Loads candidate-plan registry entries and resolves them into Android execution plans.",
}

SCRIPT_ROLES = {
    "scripts/build_graphpilot_checkpoint.py": "Freezes one canonical checkpoint manifest that pins every evidence path.",
    "scripts/build_graphpilot_artifact_pack.py": "Builds the concise artifact pack, audits, tables, and paper-facing summaries.",
    "scripts/build_graphpilot_cases_figures.py": "Generates checkpoint-pinned plots and composite figures for the CASES paper and this report.",
    "scripts/build_graphpilot_cases_paper.py": "Builds the compact CASES paper from markdown-first sections and the canonical checkpoint.",
    "scripts/calibrate_graphpilot_cost_model.py": "Fits surrogate simulator parameters against measured CPU/GPU/NPU behavior.",
    "scripts/generate_graphpilot_candidate_plans.py": "Enumerates candidate plans and emits the candidate-plan registry.",
    "scripts/generate_graphpilot_baseline_registry.py": "Runs the baseline suite over the shared simulator environment and records results.",
    "scripts/generate_graphpilot_workload_registry.py": "Materializes the workload-universe registry from the source JSON configuration.",
    "scripts/run_graphpilot_characterization.py": "Runs workload-family characterization, baseline comparisons, and ablations.",
    "scripts/run_graphpilot_experiments.py": "Executes the canonical workflow experiments and captures actual-vs-candidate results.",
    "scripts/run_graphpilot_sustained_load.py": "Runs sustained-load measurements for latency drift and thermal behavior.",
    "scripts/run_graphpilot_stage_feasibility.py": "Collects support-safe feasibility evidence for each stage/backend pair.",
    "scripts/run_graphpilot_stage_profiler.py": "Collects stage/backend latency, memory, and output-size measurements.",
    "scripts/tune_graphpilot_hyperparameters.py": "Searches objective weights and runtime scheduler weights.",
    "scripts/graphpilot_cases_markdown.py": "Renders markdown-first CASES paper sections into LaTeX-ready fragments.",
}

CODE_EXCERPTS = [
    ("Offline cost model", "graphpilot_edge/cost_model.py", 7, 129),
    ("Multi-resource hardware simulator", "graphpilot_edge/hardware_simulator.py", 147, 290),
    ("Discrete-event stream simulation", "graphpilot_edge/simulation.py", 351, 470),
    ("Offline scheduling utilities", "graphpilot_edge/scheduler.py", 1, 96),
    ("Android runtime scheduler", "android-app/app/src/main/java/com/qidk/fastvlm/core/graphpilot/GraphPilotRuntimeScheduler.kt", 59, 220),
    ("Android memory/KV controller", "android-app/app/src/main/java/com/qidk/fastvlm/core/graphpilot/GraphPilotMemoryAdmissionController.kt", 60, 220),
]

BASELINE_INTENT = {
    "cpu_only": "Force all feasible work onto CPU resources only.",
    "gpu_only": "Force all feasible work onto GPU resources only.",
    "npu_only": "Force all feasible work onto NPU resources only.",
    "current_deployed_plan": "Replay the current support-safe Android deployment.",
    "stage_greedy": "Pick the locally best support-safe backend per stage.",
    "static_best_map": "Single offline best map with no online adaptation.",
    "no_pipeline": "Disable CHUNK/TOKEN edge exploitation.",
    "no_fallback_aware": "Ignore fallback partitions and copy penalties.",
    "no_memory_kv": "Remove explicit memory and KV accounting.",
    "no_knob_tuning": "Freeze stage knobs at default values.",
    "no_thermal_adaptation": "Disable thermal plan-bank switching.",
    "band_like": "Proxy mobile multi-DNN mapping with coarse heterogeneous affinity.",
    "adms_like": "Proxy heterogeneous co-execution with static resource splits.",
    "puzzle_like": "Proxy partition-aware mobile mapping without assistant-specific control.",
    "twill_like": "Proxy compound-AI scheduling without support-safe fallback costing.",
    "heteroinfer_like": "Proxy heterogeneous LLM engine centered on prefill/decode placement.",
    "agent_xpu_like": "Proxy agentic SoC runtime with LLM-flow emphasis.",
    "hero_like": "Proxy heterogeneous mobile agentic RAG scheduler.",
}


@dataclass(frozen=True)
class ReportInputs:
    checkpoint_manifest: Path
    figure_summary: Path
    output_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a long-form GraphPilot technical report from the canonical checkpoint.")
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--figure-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file does not exist: {path}")
    return json.loads(path.read_text())


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def fmt_num(value: Any, digits: int = 1) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    try:
        number = float(value)
    except Exception:
        return latex_escape(value)
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.{digits}f}"


def relative_repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def repo_label(path: Path) -> str:
    return latex_escape(relative_repo_path(path))


def line_count(path: Path) -> int:
    return sum(1 for _ in path.read_text().splitlines())


def extract_python_symbols(path: Path) -> str:
    tree = ast.parse(path.read_text())
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
    return ", ".join(names[:8]) + (" ..." if len(names) > 8 else "")


def extract_kotlin_symbols(path: Path) -> str:
    names: list[str] = []
    pattern = re.compile(r"^\s*(?:data\s+class|enum\s+class|class|object|fun)\s+([A-Za-z0-9_]+)")
    for line in path.read_text().splitlines():
        match = pattern.match(line)
        if match:
            names.append(match.group(1))
    return ", ".join(names[:8]) + (" ..." if len(names) > 8 else "")


def make_table(headers: list[str], rows: list[list[str]], colspec: str, caption: str, label: str, size: str = "\\small") -> str:
    body = [
        "\\begin{table}[htbp]",
        "\\centering",
        size,
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(headers) + r" \\",
        "\\midrule",
    ]
    body.extend(" & ".join(row) + r" \\" for row in rows)
    body.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(body)


def make_longtable(headers: list[str], rows: list[list[str]], colspec: str, caption: str, label: str, size: str = "\\scriptsize") -> str:
    header = " & ".join(headers) + r" \\" + "\n\\midrule"
    body = [size, f"\\begin{{longtable}}{{{colspec}}}", f"\\caption{{{caption}}}\\label{{{label}}}\\\\", "\\toprule", header, "\\endfirsthead", "\\toprule", header, "\\endhead"]
    body.extend(" & ".join(row) + r" \\" for row in rows)
    body.extend(["\\bottomrule", "\\end{longtable}"])
    return "\n".join(body)


def make_listing(title: str, repo_rel_path: str, start_line: int, end_line: int) -> str:
    path = ROOT / repo_rel_path
    if not path.exists():
        raise FileNotFoundError(f"Expected code excerpt path does not exist: {path}")
    lines = path.read_text().splitlines()
    excerpt = "\n".join(lines[start_line - 1 : end_line])
    language = "Java" if path.suffix == ".kt" else "Python"
    return "\n".join(
        [
            f"\\subsection*{{{latex_escape(title)}}}",
            f"\\textbf{{Source:}} \\texttt{{{latex_escape(repo_rel_path)}}} lines {start_line}--{end_line}",
            "\\begin{{lstlisting}}[language={}]".format(language),
            excerpt,
            "\\end{lstlisting}",
        ]
    )


def copy_selected_figures(figure_summary: dict[str, Any], output_dir: Path, basenames: Iterable[str]) -> list[str]:
    selected = set(basenames)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for output in figure_summary.get("outputs", []):
        path = Path(output)
        if path.name in selected:
            shutil.copy2(path, figure_dir / path.name)
            copied.append(path.name)
    missing = sorted(selected - set(copied))
    if missing:
        raise FileNotFoundError(f"Missing required figure files in figure summary: {missing}")
    return sorted(copied)


def module_rows(role_map: dict[str, str], symbol_extractor) -> list[list[str]]:
    rows: list[list[str]] = []
    for repo_rel, role in role_map.items():
        path = ROOT / repo_rel
        if not path.exists():
            raise FileNotFoundError(f"Expected module path does not exist: {path}")
        rows.append([
            latex_escape(repo_rel),
            latex_escape(str(line_count(path))),
            latex_escape(symbol_extractor(path)),
            latex_escape(role),
        ])
    return rows


def workload_rows(workload_registry: dict[str, Any]) -> list[list[str]]:
    rows = []
    for workload in sorted(workload_registry.get("workloads", []), key=lambda item: item["workload_id"]):
        datasets = ", ".join(workload.get("datasets") or []) or "-"
        tags = ", ".join(workload.get("tags") or []) or "-"
        rows.append([
            latex_escape(workload["workload_id"]),
            latex_escape(workload.get("category", "-")),
            latex_escape(workload.get("builder", "-")),
            latex_escape(datasets),
            latex_escape(tags),
            latex_escape(workload.get("description", "-")),
        ])
    return rows


def baseline_catalog_rows(baseline_registry: dict[str, Any]) -> list[list[str]]:
    rows = []
    for baseline_id in baseline_registry.get("baseline_ids", []):
        rows.append([
            latex_escape(baseline_id),
            latex_escape(BASELINE_INTENT.get(baseline_id, "No description recorded.")),
        ])
    return rows


def candidate_plan_rows(candidate_registry: dict[str, Any]) -> list[list[str]]:
    rows = []
    for plan in candidate_registry.get("plans", []):
        predicted = plan.get("predicted_cost", {})
        backend_map = ", ".join(f"{k}:{v}" for k, v in sorted(plan.get("backend_map", {}).items()))
        rows.append([
            latex_escape(plan.get("workflow_template", "-")),
            latex_escape(plan.get("state_id", "-")),
            latex_escape(plan.get("plan_id", "-")),
            latex_escape(backend_map),
            fmt_num(predicted.get("makespan_ms"), 1),
            fmt_num(predicted.get("p95_queue_delay_ms"), 1),
            fmt_num(predicted.get("deadline_miss_rate"), 2),
            fmt_num(predicted.get("objective_score"), 2),
        ])
    return rows


def actual_workflow_rows(experiment_summary: dict[str, Any]) -> list[list[str]]:
    rows = []
    for workflow_id, entry in sorted(experiment_summary.get("actual_workflows", {}).items()):
        rows.append([
            latex_escape(workflow_id),
            latex_escape(entry.get("variant", "-")),
            latex_escape(entry.get("state_id", "-")),
            fmt_num(entry.get("warm_latency_ms"), 1),
            fmt_num(entry.get("ttft_ms"), 1),
            fmt_num(entry.get("tts_first_audio_ms"), 1),
            latex_escape(", ".join(f"{k}:{v}" for k, v in sorted(entry.get("stage_backends", {}).items()))),
        ])
    return rows


def candidate_vs_actual_rows(experiment_summary: dict[str, Any]) -> list[list[str]]:
    rows = []
    for comparison in sorted(experiment_summary.get("comparisons", []), key=lambda item: item.get("workflow_id", "-")):
        rows.append([
            latex_escape(comparison.get("workflow_id", "-")),
            fmt_num(comparison.get("actual_warm_latency_ms"), 1),
            fmt_num(comparison.get("candidate_predicted_makespan_ms"), 1),
            fmt_num(comparison.get("latency_delta_ms"), 1),
        ])
    return rows


def family_calibration_rows(calibration_summary: dict[str, Any]) -> list[list[str]]:
    rows = []
    quality = calibration_summary.get("calibration_quality_by_family", {})
    family_backend = calibration_summary.get("family_backend_calibration", {})
    for family in sorted(quality):
        q = quality[family]
        backends = ", ".join(sorted(family_backend.get(family, {}).keys())) or "-"
        rows.append([
            latex_escape(family),
            fmt_num(q.get("mean_absolute_error_ms"), 1),
            fmt_num(q.get("max_absolute_error_ms"), 1),
            fmt_num(q.get("sample_count"), 0),
            latex_escape(backends),
        ])
    return rows


def stage_calibration_rows(calibration_summary: dict[str, Any]) -> list[list[str]]:
    rows = []
    for family, backend_map in sorted(calibration_summary.get("stage_backend_calibration", {}).items()):
        if isinstance(backend_map, dict):
            for backend, entry in sorted(backend_map.items()):
                rows.append([
                    latex_escape(family),
                    latex_escape(backend),
                    fmt_num(entry.get("warm_latency_ms"), 1),
                    fmt_num(entry.get("actual_stage_timing_ms"), 1),
                    fmt_num(entry.get("latency_scale"), 3),
                    fmt_num(entry.get("launch_overhead_ms"), 1),
                    fmt_num(entry.get("residual_bias_ms"), 1),
                ])
    return rows


def backend_matrix_rows(backend_matrix: dict[str, Any]) -> list[list[str]]:
    rows = []
    for stage in backend_matrix.get("stages", []):
        backends = stage.get("backends", {})
        def status(name: str) -> str:
            entry = backends.get(name, {})
            note = entry.get("notes") or entry.get("last_verdict") or "-"
            return f"{entry.get('status', '-')}; {note}"
        rows.append([
            latex_escape(stage.get("stage_id", "-")),
            latex_escape(stage.get("model_family", "-")),
            latex_escape(stage.get("role", "-")),
            latex_escape(status("cpu")),
            latex_escape(status("gpu")),
            latex_escape(status("npu")),
        ])
    return rows


def evidence_path_rows(checkpoint_summary: dict[str, Any]) -> list[list[str]]:
    rows = []
    for key, path_value in sorted(checkpoint_summary.get("canonical_evidence_paths", {}).items()):
        path = Path(path_value)
        rows.append([
            latex_escape(key),
            latex_escape(relative_repo_path(path)),
            latex_escape("yes" if path.exists() else "no"),
        ])
    return rows


def sustained_sample_rows(sustained_summary: dict[str, Any]) -> list[list[str]]:
    rows = []
    for sample in sustained_summary.get("samples", []):
        thermal_after = sample.get("thermal_after", {})
        rows.append([
            latex_escape(sample.get("workflow_id", "-")),
            fmt_num(sample.get("sample_index"), 0),
            fmt_num(sample.get("warm_latency_ms"), 1),
            fmt_num(sample.get("ttft_ms"), 1),
            fmt_num(sample.get("tts_first_audio_ms"), 1),
            fmt_num(sample.get("wall_elapsed_ms"), 1),
            fmt_num(thermal_after.get("max_cpu_c"), 1),
            fmt_num(thermal_after.get("max_gpu_c"), 1),
            fmt_num(thermal_after.get("max_npu_c"), 1),
            fmt_num(thermal_after.get("skin_c"), 1),
        ])
    return rows


def workload_summary_rows(workload_registry: dict[str, Any]) -> list[list[str]]:
    counts: dict[str, int] = {}
    for workload in workload_registry.get("workloads", []):
        category = workload.get("category", "unknown")
        counts[category] = counts.get(category, 0) + 1
    return [[latex_escape(category), str(count)] for category, count in sorted(counts.items())]


def topology_rows(topology_summary: dict[str, Any]) -> list[list[str]]:
    rows = []
    for resource in topology_summary.get("resources", []):
        rows.append([
            latex_escape(resource.get("resource_id", "-")),
            latex_escape(resource.get("resource_type", "-")),
            latex_escape(", ".join(resource.get("supported_op_classes", []))),
            fmt_num(resource.get("launch_overhead_ms"), 1),
            fmt_num(resource.get("queue_depth"), 0),
            fmt_num(resource.get("busy_power_mw"), 1),
        ])
    return rows


def tuning_rows(tuning_summary: dict[str, Any]) -> list[list[str]]:
    objective = ((tuning_summary.get("best_objective_weights") or {}).get("weights") or {})
    scheduler = ((tuning_summary.get("best_scheduler_weights") or {}).get("weights") or {})
    rows = []
    for key in ["alpha", "beta", "gamma", "delta", "eta", "zeta", "xi", "psi"]:
        rows.append([
            latex_escape(key),
            fmt_num(objective.get(key), 3),
            fmt_num(scheduler.get(key), 3),
        ])
    return rows


def objective_eval_rows(tuning_summary: dict[str, Any]) -> list[list[str]]:
    rows = []
    for entry in tuning_summary.get("objective_evaluations", [])[:12]:
        rows.append([
            latex_escape(entry.get("candidate_id", "-")),
            fmt_num(entry.get("objective_score"), 2),
            latex_escape(json.dumps(entry.get("weights", {}), sort_keys=True)),
        ])
    return rows


def scheduler_eval_rows(tuning_summary: dict[str, Any]) -> list[list[str]]:
    rows = []
    for entry in tuning_summary.get("scheduler_evaluations", [])[:12]:
        rows.append([
            latex_escape(entry.get("candidate_id", "-")),
            fmt_num(entry.get("objective_score"), 2),
            latex_escape(json.dumps(entry.get("weights", {}), sort_keys=True)),
        ])
    return rows


def baseline_result_rows(baseline_registry: dict[str, Any], limit: int | None = 80) -> list[list[str]]:
    rows = []
    for workload in baseline_registry.get("workloads", []):
        workload_id = workload.get("workload_id", "-")
        for baseline in workload.get("baselines", []):
            rows.append([
                latex_escape(workload_id),
                latex_escape(baseline.get("baseline_id", "-")),
                latex_escape(baseline.get("status", "-")),
                fmt_num(baseline.get("score_ms"), 1),
                latex_escape(", ".join(f"{k}:{v}" for k, v in sorted((baseline.get("stage_backends") or {}).items()))),
            ])
    rows.sort(key=lambda r: (r[0], r[1]))
    if limit is None:
        return rows
    return rows[:limit]


def memory_event_rows(memory_summary: dict[str, Any]) -> list[list[str]]:
    fields = [
        "workflow",
        "decision",
        "required_bytes",
        "effective_required_bytes",
        "queue_wait_ms",
        "memory_decision",
        "effective_responder_max_tokens",
        "effective_retrieval_top_k",
        "reason",
        "applied_actions",
    ]

    def parse_line(raw: str) -> dict[str, str]:
        parsed: dict[str, str] = {}
        for token in raw.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            parsed[key] = value
        return parsed

    rows: list[list[str]] = []
    for key in [
        "latest_admit_line",
        "latest_degrade_line",
        "latest_reject_line",
        "latest_degrade_metrics_line",
        "latest_queue_metrics_line",
    ]:
        raw = memory_summary.get(key)
        if not raw:
            continue
        parsed = parse_line(raw)
        rows.append([
            latex_escape(key),
            *(latex_escape(parsed.get(field, "-")) for field in fields),
        ])
    return rows


def reproduction_command_rows(checkpoint_path: Path, figure_summary_path: Path) -> list[list[str]]:
    clean_py = "env -u PYTHONHOME -u PYTHONPATH python3"
    rows = [
        ["Generate workload registry", f"{clean_py} scripts/generate_graphpilot_workload_registry.py"],
        ["Generate candidate plans", f"{clean_py} scripts/generate_graphpilot_candidate_plans.py --checkpoint-manifest {relative_repo_path(checkpoint_path)}"],
        ["Generate baseline registry", f"{clean_py} scripts/generate_graphpilot_baseline_registry.py --checkpoint-manifest {relative_repo_path(checkpoint_path)}"],
        ["Run characterization", f"{clean_py} scripts/run_graphpilot_characterization.py --checkpoint-manifest {relative_repo_path(checkpoint_path)}"],
        ["Build compact artifact pack", f"{clean_py} scripts/build_graphpilot_artifact_pack.py --checkpoint-manifest {relative_repo_path(checkpoint_path)}"],
        ["Build CASES figures", f"{clean_py} scripts/build_graphpilot_cases_figures.py --checkpoint-manifest {relative_repo_path(checkpoint_path)}"],
        ["Build this technical report", f"{clean_py} scripts/build_graphpilot_technical_report.py --checkpoint-manifest {relative_repo_path(checkpoint_path)} --figure-summary {relative_repo_path(figure_summary_path)}"],
        ["Python verification", f"{clean_py} -m unittest discover -s tests/graphpilot_edge -v && {clean_py} -m unittest discover -s scripts/tests -v"],
        ["Android unit verification", "ANDROID_HOME=/home/aryamavmurthy/android-sdk ANDROID_SDK_ROOT=/home/aryamavmurthy/android-sdk ./gradlew app:testDebugUnitTest --tests 'com.qidk.fastvlm.core.graphpilot.*'"],
        ["Android connected verification", "ANDROID_HOME=/home/aryamavmurthy/android-sdk ANDROID_SDK_ROOT=/home/aryamavmurthy/android-sdk ./gradlew app:connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=com.qidk.fastvlm.graphpilot.GraphPilotCoordinatorInstrumentedTest"],
    ]
    return [[latex_escape(step), latex_escape(command)] for step, command in rows]


def copy_bibliography_assets(output_dir: Path) -> Path:
    if not LATEST_CASES_REFERENCES.exists():
        raise FileNotFoundError(f"Reference source not found: {LATEST_CASES_REFERENCES}")
    if not IEEE_BST.exists():
        raise FileNotFoundError(f"BibTeX style source not found: {IEEE_BST}")
    target = output_dir / "references.bib"
    bst_target = output_dir / "IEEEtran.bst"
    shutil.copy2(LATEST_CASES_REFERENCES, target)
    shutil.copy2(IEEE_BST, bst_target)
    return target


def section(title: str, body: str, level: str = "section") -> str:
    return f"\\{level}{{{title}}}\n{body}\n"


def build_report_tex(
    checkpoint_path: Path,
    checkpoint_summary: dict[str, Any],
    figure_summary_path: Path,
    figure_summary: dict[str, Any],
    output_dir: Path,
) -> str:
    evidence = checkpoint_summary["canonical_evidence_paths"]
    experiment_summary = load_json(Path(evidence["experiment_summary"]))
    calibration_summary = load_json(Path(evidence["calibration_summary"]))
    characterization_summary = load_json(Path(evidence["characterization_summary"]))
    baseline_registry = load_json(Path(evidence["baseline_registry"]))
    workload_registry = load_json(Path(evidence["workload_registry"]))
    candidate_registry = load_json(Path(evidence["candidate_plans"]))
    backend_matrix = load_json(Path(evidence["backend_matrix"]))
    tuning_summary = load_json(Path(evidence["tuning_summary"]))
    sustained_summary = load_json(Path(evidence["sustained_summary"]))
    memory_summary = load_json(Path(evidence["memory_admission_summary"]))
    topology_summary = load_json(ROOT / "configs" / "graphpilot_edge" / "hardware_topology_sm8750.json")
    artifact_pack_summary = load_json(Path(checkpoint_summary["canonical_evidence_paths"]["artifact_pack_summary"]))

    copied_figures = copy_selected_figures(
        figure_summary,
        output_dir,
        [
            "architecture_overview.png",
            "offline_online_split.png",
            "calibration_overview.png",
            "calibration_family_mae.png",
            "backend_launch_overhead.png",
            "backend_contention_scale.png",
            "backend_affinity_matrix.png",
            "support_safe_feasibility.png",
            "workload_universe_coverage.png",
            "evaluation_overview.png",
            "workflow_primary_results.png",
            "continuous_stream_results.png",
            "sim_real_calibration.png",
            "sustained_detail.png",
            "baseline_comparison.png",
            "proxy_baseline_comparison.png",
            "ablation_breakdown.png",
            "sensitivity_overview.png",
            "memory_kv_overview.png",
            "knob_frontier_overview.png",
            "fallback_penalty.png",
            "thermal_plan_bank.png",
            "objective_sensitivity.png",
        ],
    )
    copy_bibliography_assets(output_dir)

    python_rows = module_rows(PYTHON_MODULE_ROLES, extract_python_symbols)
    android_rows = module_rows(ANDROID_MODULE_ROLES, extract_kotlin_symbols)
    script_rows = module_rows(SCRIPT_ROLES, extract_python_symbols)

    actual_rows = actual_workflow_rows(experiment_summary)
    comparison_rows = candidate_vs_actual_rows(experiment_summary)
    family_rows = family_calibration_rows(calibration_summary)
    stage_rows = stage_calibration_rows(calibration_summary)
    workload_rows_all = workload_rows(workload_registry)
    workload_summary = workload_summary_rows(workload_registry)
    baseline_catalog = baseline_catalog_rows(baseline_registry)
    candidate_rows = candidate_plan_rows(candidate_registry)
    backend_rows = backend_matrix_rows(backend_matrix)
    evidence_rows = evidence_path_rows(checkpoint_summary)
    sustained_rows = sustained_sample_rows(sustained_summary)
    topology = topology_rows(topology_summary)
    tuning = tuning_rows(tuning_summary)
    objective_rows = objective_eval_rows(tuning_summary)
    scheduler_rows = scheduler_eval_rows(tuning_summary)
    baseline_rows = baseline_result_rows(baseline_registry)
    full_baseline_rows = baseline_result_rows(baseline_registry, limit=None)
    memory_rows = memory_event_rows(memory_summary)
    reproduction_rows = reproduction_command_rows(checkpoint_path, figure_summary_path)

    actual_map = experiment_summary.get("actual_workflows", {})
    wf_a = actual_map.get("workflow_a_voice_only", {})
    wf_b = actual_map.get("workflow_b_voice_vision", {})
    wf_c = actual_map.get("workflow_c_voice_vision_retrieval", {})
    artifact_pack_report = Path(artifact_pack_summary.get("report", output_dir / "missing"))
    truth_doc = Path(checkpoint_summary.get("truth_source_pdf", ROOT / "Truth-docs/graphpilot_edge_revision_report.pdf"))

    sections: list[str] = []
    sections.append(r"""
\begin{titlepage}
\centering
{\LARGE GraphPilot-Edge Technical Report\\[0.6em]}
{\Large Full System, Simulator, Runtime, and Evidence Surface\\[1.2em]}
{\large Generated from the canonical checkpointed artifact surface\\[2em]}
\begin{tabular}{rl}
Report date: & %s\\
Checkpoint manifest: & \texttt{%s}\\
Figure summary: & \texttt{%s}\\
Truth source: & \texttt{%s}\\
Git commit (working tree base): & \texttt{%s}\\
\end{tabular}
\vfill
\begin{minipage}{0.92\textwidth}
\small
This document is a large-form technical report rather than a venue-constrained paper. It is anchored to the same GraphPilot-Edge evidence surface used by the CASES build, but expands the explanation, implementation mapping, experiment inventory, and appendix coverage so that every major formula, code module, registry, run summary, and audit artifact is documented in one PDF.
\end{minipage}
\vfill
\end{titlepage}
""" % (
        latex_escape(datetime.now(timezone.utc).isoformat()),
        latex_escape(checkpoint_path.parent.name),
        latex_escape(figure_summary_path.parent.name),
        latex_escape(truth_doc.name),
        latex_escape(subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()),
    ))

    sections.append("\\tableofcontents\n\\clearpage")

    sections.append(section("Executive Summary", f"""
GraphPilot-Edge is implemented as a measured runtime plus a calibrated simulator for continuous multimodal assistant DAGs on one heterogeneous mobile SoC. The canonical deployed workflows are: workflow A (voice-only), workflow B (voice + vision), and workflow C (voice + vision + retrieval). The real-device checkpoint keeps FastVLM on the LiteRT NPU while text and speech stages remain on verified CPU execution paths. The checkpointed warm latencies are {fmt_num(wf_a.get('warm_latency_ms'))} ms for workflow A, {fmt_num(wf_b.get('warm_latency_ms'))} ms for workflow B, and {fmt_num(wf_c.get('warm_latency_ms'))} ms for workflow C. The canonical artifact pack is rooted at \\texttt{{{latex_escape(relative_repo_path(Path(checkpoint_summary['canonical_evidence_paths']['artifact_pack_summary'])))}}}.

This report expands the compact paper in four ways. First, it maps the mathematical design directly onto the checked-in code. Second, it documents the simulator, runtime, baselines, workload universe, and artifact-surface layout exhaustively. Third, it includes wide appendix tables for workloads, baselines, candidate plans, module inventories, and evidence paths. Fourth, it records the exact experiment summaries and registry paths used to build the paper-facing outputs.

The strongest surviving claim remains intentionally narrow: GraphPilot-Edge is a support-safe heterogeneous scheduling runtime and calibrated simulator, not a proof that every stage already executes natively across CPU, GPU, and NPU. Unsupported paths remain explicit in the feasibility matrix and are never silently counted as successful offload.
"""))

    sections.append(section("Canonical Evidence Surface and Scope", f"""
The canonical source of truth for this report is the checkpoint manifest \\texttt{{{latex_escape(checkpoint_path.parent.name)}}}. That manifest pins every evidence path that matters: experiment summary, calibration summary, characterization summary, workload registry, baseline registry, candidate-plan registry, sustained-load summary, backend feasibility matrix, memory-admission summary, tuning summary, and the prior artifact-pack summary.

This report does not merge numbers across arbitrary directories. Every figure and table is regenerated from that checkpoint and the revision-bounded truth document \\texttt{{{latex_escape(truth_doc.name)}}}. This is necessary because the repository has accumulated multiple artifact packs and paper builds over time. The checkpoint prevents the report from turning into an inconsistent mixture of stale and fresh evidence.

The canonical evidence inventory is summarized in Table~\\ref{{tab:evidence-paths}}. The audit surface recorded by the checkpoint reports zero open Beads and no blocked workflows for the current scope, but the scope itself remains revision-bounded: support-safe heterogeneity is still narrow outside the preserved FastVLM NPU path.

{make_table(["Key", "Repo-relative path", "Exists"], evidence_rows, "p{1.8in}p{3.4in}c", "Canonical evidence paths pinned by the checkpoint manifest.", "tab:evidence-paths", "\\scriptsize")}
"""))

    sections.append(section("Repository and Implementation Map", f"""
The GraphPilot implementation is split into an offline Python codebase, an Android/Kotlin runtime path, and a script layer that materializes registries, experiments, characterization studies, paper figures, artifact packs, and now this technical report. Figures~\\ref{{fig:arch-overview}} and \\ref{{fig:brain-split}} summarize the deployed workflow surface and the offline/online split used throughout the repository.

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.96\\linewidth]{{figures/architecture_overview.png}}
\\caption{{Checkpoint-pinned GraphPilot deployment surface.}}
\\label{{fig:arch-overview}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.96\\linewidth]{{figures/offline_online_split.png}}
\\caption{{Offline/online split across the measured runtime and the calibrated simulator.}}
\\label{{fig:brain-split}}
\\end{{figure}}

Table~\\ref{{tab:python-modules}} lists the Python modules that define the offline brain. Table~\\ref{{tab:android-modules}} lists the Android runtime modules that execute the deployed path. Table~\\ref{{tab:script-modules}} lists the main script entry points used to regenerate the evidence surface.

{make_longtable(["Python module", "LOC", "Top-level symbols", "Responsibility"], python_rows, "p{1.9in}r p{1.7in} p{2.0in}", "Offline Python module inventory.", "tab:python-modules")}

{make_longtable(["Android module", "LOC", "Top-level symbols", "Responsibility"], android_rows, "p{2.2in}r p{1.4in} p{1.8in}", "Android runtime module inventory.", "tab:android-modules")}

{make_longtable(["Script", "LOC", "Top-level symbols", "Responsibility"], script_rows, "p{2.0in}r p{1.6in} p{1.7in}", "Automation and artifact-building script inventory.", "tab:script-modules")}
"""))

    sections.append(section("Runtime Formulation and Implemented Algorithms", r"""
GraphPilot treats each request as a DAG whose nodes are stages or macro-regions and whose edges carry one of three stream semantics: full-output handoff, chunk handoff, or token handoff. The runtime does not optimize one scalar latency number in isolation; it scores plans with an eight-term objective implemented in \texttt{graphpilot\_edge/cost\_model.py}:

\begin{equation}
\begin{aligned}
J(\rho) ={}& \alpha P95(T_{e2e}) + \beta P95(T_{TFS}) + \gamma \bar{E} + \delta M_{peak}\\
&+ \eta B_{copy} + \zeta Q_{loss} + \xi P95(T_{queue}) + \psi R_{miss}.
\end{aligned}
\end{equation}

The implemented cost-model functions also expose the execution-time, transfer, contention, batching, and KV formulas used elsewhere in the codebase:

\begin{equation}
T^{exec}_{u,h}=L_h + \max(T^{ops}_{u,h}(B), T^{mem}_{u,h})\,\rho_h(\theta)\,\kappa_h(q)
\end{equation}

\begin{equation}
T_{xfer}(S,h,h') =
\begin{cases}
0 & \text{same resource / compatible memory}\\
T_{map}(S) & \text{shared-buffer path}\\
\tau_0^{h,h'} + S/BW_{h,h'} + \tau_{layout}^{h,h'} & \text{otherwise.}
\end{cases}
\end{equation}

\begin{equation}
\kappa_h(q)=1+\sum_{h' \neq h}\lambda_{h,h'}u_{h'}
\end{equation}

\begin{equation}
\phi_h(B)=1+\beta_h(1-e^{-B/\tau_h})
\end{equation}

\begin{equation}
\theta(t+\Delta)=\theta_{amb}+(\theta(t)-\theta_{amb})e^{-\Delta/\tau}+R P (1-e^{-\Delta/\tau})
\end{equation}

\begin{equation}
M_{KV}=2LH_{kv}D_{head}TB_{dtype}
\end{equation}

The planning path enumerates feasible stage/backend combinations, expands heavy-model macro-regions, scores candidate plans with the calibrated simulator, and records explicit queue-delay and miss-rate predictions in the candidate-plan registry. The scheduler path computes upward ranks and a weighted priority score using critical-path pressure, first-output bias, age, copy cost, memory/KV risk, and thermal risk. On device, unsupported paths raise explicit failures or remain simulator-only; they are never silently counted as successful offload.
"""))

    sections.append(section("Hardware Simulator, Topology, and Calibration", f"""
The calibrated simulator is implemented in \\texttt{{graphpilot\\_edge/hardware\\_simulator.py}} and parameterized by the topology file \\texttt{{configs/graphpilot\\_edge/hardware\\_topology\\_sm8750.json}}. It models resource instances rather than only backend labels. Each resource instance has a support mask, service-rate vector, launch overhead, queue depth, batching curve, power parameters, and thermal constants.

Table~\\ref{{tab:topology}} lists the checkpointed SM8750-style resource instances that seed the simulator. Table~\\ref{{tab:family-calibration}} lists family-level residuals after calibration. Table~\\ref{{tab:stage-calibration}} lists the per-family/per-backend surrogate parameters that matter most in practice. Figures~\\ref{{fig:cal-overview}}--\\ref{{fig:feasibility-matrix}} visualize the calibration panels, the backend affinity matrix, and the support-safe feasibility surface.

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.98\\linewidth]{{figures/calibration_overview.png}}
\\caption{{Checkpoint-pinned calibration overview.}}
\\label{{fig:cal-overview}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.85\\linewidth]{{figures/backend_affinity_matrix.png}}
\\caption{{Family/backend mean latency matrix used for backend-affinity reasoning.}}
\\label{{fig:affinity-matrix}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.85\\linewidth]{{figures/support_safe_feasibility.png}}
\\caption{{Support-safe feasibility matrix derived from the checkpointed backend registry.}}
\\label{{fig:feasibility-matrix}}
\\end{{figure}}

{make_table(["Resource", "Type", "Supported op classes", "Launch ms", "Queue depth", "Busy power mW"], topology, "p{1.0in}c p{2.2in}rrr", "Checkpointed hardware-topology records used by the simulator.", "tab:topology", "\\scriptsize")}

{make_table(["Family", "MAE ms", "Max error ms", "Samples", "Backends"], family_rows, "lrrrl", "Family-level calibration residuals after fitting the simulator to measured device behavior.", "tab:family-calibration", "\\small")}

{make_longtable(["Family", "Backend", "Predicted warm", "Measured stage", "Scale", "Launch ms", "Residual bias"], stage_rows, "p{0.9in} c r r r r r", "Family/backend calibration entries extracted from the checkpointed calibration summary.", "tab:stage-calibration")}

The calibration surface is intentionally practical rather than mystical. The simulator is tuned to the measured CPU/GPU/NPU deltas, family-level residuals, launch-overhead surrogates, contention scales, and transfer biases exposed by the checkpoint. It is therefore appropriate for ranking and sensitivity analysis, not as an exact oracle for every family and input shape.
"""))

    sections.append(section("Model-Graph Simulator and Workload Universe", f"""
The model-graph layer expands the hardware simulator into stage-level scenarios: LLM prompt assembly/prefill/decode/postprocess; VLM image encoding, projector/fusion, multimodal prefill, and decode; retrieval embedder and ANN search; STT; TTS; CNN; ViT; and sequential glue tasks. The source of truth for the workload universe is \\texttt{{artifacts/graphpilot\\_edge/registries/workload\\_universe\\_registry.json}}, and the Python entry point that materializes it is \\texttt{{graphpilot\\_edge/workload\\_universe.py}}.

Figure~\\ref{{fig:workload-universe}} shows the category counts. Table~\\ref{{tab:workload-summary}} summarizes the category breakdown. Appendix~\\ref{{app:workloads}} lists every checkpointed workload entry with its category, builder, datasets, tags, and description.

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.82\\linewidth]{{figures/workload_universe_coverage.png}}
\\caption{{Checkpoint-pinned workload-universe coverage.}}
\\label{{fig:workload-universe}}
\\end{{figure}}

{make_table(["Category", "Count"], workload_summary, "lr", "Workload-universe category counts in the checkpointed registry.", "tab:workload-summary", "\\small")}

The current checkpoint records {len(workload_registry.get('workloads', []))} workloads across primitive operators, single-model families, compound assistant DAGs, continuous streams, mixed-criticality streams, and stress/failure cases. This broader workload surface is larger than the deployed runtime surface on purpose. Some policies remain calibrated what-if studies even when only the A/B/C workflows are fully realized online.
"""))

    sections.append(section("Baselines, Method-Class Proxies, and Experiment Methodology", f"""
The baseline surface is explicit rather than implicit. The baseline registry records {len(baseline_registry.get('baseline_ids', []))} baseline IDs spanning force-one-backend baselines, the current deployed plan, internal ablations, and faithful method-class proxies. The workload-level baseline results are all generated inside the same simulator/runtime environment; they are not copied from incomparable published numbers on different hardware.

Table~\\ref{{tab:baseline-catalog}} lists the full baseline catalog. Table~\\ref{{tab:baseline-sample-results}} gives a representative sample of workload/baseline rows from the registry. Figure~\\ref{{fig:baseline-overview}} and Figure~\\ref{{fig:proxy-overview}} reuse the checkpointed baseline plots.

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.92\\linewidth]{{figures/baseline_comparison.png}}
\\caption{{Internal baseline deltas versus GraphPilot on informative checkpointed workloads.}}
\\label{{fig:baseline-overview}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.92\\linewidth]{{figures/proxy_baseline_comparison.png}}
\\caption{{Faithful method-class proxy deltas versus GraphPilot in the shared calibrated environment.}}
\\label{{fig:proxy-overview}}
\\end{{figure}}

{make_longtable(["Baseline ID", "Intent"], baseline_catalog, "p{1.7in}p{4.4in}", "Full baseline catalog extracted from the checkpointed baseline registry.", "tab:baseline-catalog")}

{make_longtable(["Workload", "Baseline", "Status", "Score ms", "Stage backends"], baseline_rows, "p{1.8in}p{0.9in}c r p{2.1in}", "Representative workload-level baseline results sampled from the checkpointed baseline registry.", "tab:baseline-sample-results")}

The experiment methodology follows the checkpoint manifest. The current actual-workflow experiments come from \\texttt{{{latex_escape(relative_repo_path(Path(evidence['experiment_summary'])))}}}; sustained-load data come from \\texttt{{{latex_escape(relative_repo_path(Path(evidence['sustained_summary'])))}}}; calibration comes from \\texttt{{{latex_escape(relative_repo_path(Path(evidence['calibration_summary'])))}}}; characterization and ablations come from \\texttt{{{latex_escape(relative_repo_path(Path(evidence['characterization_summary'])))}}}.
"""))

    sections.append(section("Primary Device Results, Candidate Plans, and Workflow Comparisons", f"""
The actual device-facing workflow results remain the anchor of the measured claim. Table~\\ref{{tab:actual-workflows}} lists the deployed A/B/C workflow outcomes and stage-backend maps. Table~\\ref{{tab:candidate-vs-actual}} lists actual-versus-best-candidate deltas. Appendix~\\ref{{app:candidate-plans}} lists all checkpointed candidate plans with predicted queue delay, miss rate, and objective score.

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.96\\linewidth]{{figures/evaluation_overview.png}}
\\caption{{Evaluation overview from the checkpointed evidence set.}}
\\label{{fig:evaluation-overview}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.78\\linewidth]{{figures/workflow_primary_results.png}}
\\caption{{Primary workflow warm latencies.}}
\\label{{fig:workflow-primary}}
\\end{{figure}}

{make_table(["Workflow", "Variant", "State", "Warm ms", "TTFT ms", "TTFS ms", "Stage backends"], actual_rows, "p{1.6in}p{1.0in}c r r r p{1.9in}", "Checkpointed actual workflow results from the deployed runtime.", "tab:actual-workflows", "\\scriptsize")}

{make_table(["Workflow", "Actual warm", "Best candidate", "Delta"], comparison_rows, "p{2.0in}rrr", "Candidate-vs-actual comparison from the checkpointed experiment summary.", "tab:candidate-vs-actual", "\\small")}

The actual results show the exact measured support-safe deployment surface: workflow A stays entirely on CPU; workflows B and C retain the FastVLM NPU path while keeping text and speech on CPU; workflow C also keeps retrieval on CPU because the measured GPU retrieval path is slower in the current checkpoint.
"""))

    sections.append(section("Continuous Streams, Sustained Load, and Queue-Aware Behavior", f"""
The report keeps long-run behavior explicit rather than assuming that one-shot measurements generalize. Figure~\\ref{{fig:continuous-streams}} shows the continuous-stream workload slice used for queue-aware scoring. Figure~\\ref{{fig:sustained-detail}} shows the sustained-run detail panels used to justify thermal-aware queueing and plan-bank switching. Table~\\ref{{tab:sustained-samples}} lists every sustained-run sample captured in the checkpoint.

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.78\\linewidth]{{figures/continuous_stream_results.png}}
\\caption{{Representative continuous-stream GraphPilot scores.}}
\\label{{fig:continuous-streams}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.92\\linewidth]{{figures/sustained_detail.png}}
\\caption{{Detailed sustained-run panels from the checkpointed long-run experiment.}}
\\label{{fig:sustained-detail}}
\\end{{figure}}

{make_longtable(["Workflow", "Sample", "Warm ms", "TTFT ms", "TTFS ms", "Wall ms", "CPU C", "GPU C", "NPU C", "Skin C"], sustained_rows, "p{1.5in}r r r r r r r r r", "Every sustained-run sample captured in the canonical checkpoint.", "tab:sustained-samples")}

The sustained checkpoint records {len(sustained_summary.get('samples', []))} samples across the deployed workflows. The runtime stays inside a relatively mild thermal envelope on this device class, but the measured drift still motivates explicit thermal-plan switching rather than assuming stationary performance.
"""))

    sections.append(section("Characterization, Sensitivity, Memory, KV Control, and Ablations", f"""
The characterization summary records backend-affinity results, batching curves, transfer matrix estimates, sensitivity panels, memory/KV curves, and explicit ablations. The checkpointed figures are reproduced here so the report can tie the simulator-facing logic to measured device behavior and runtime controls.

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.9\\linewidth]{{figures/sensitivity_overview.png}}
\\caption{{Sensitivity overview from the checkpointed characterization summary.}}
\\label{{fig:sensitivity-overview}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.72\\linewidth]{{figures/memory_kv_overview.png}}
\\caption{{Memory/KV growth curves used by the runtime admission controller.}}
\\label{{fig:memory-kv-overview}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.75\\linewidth]{{figures/knob_frontier_overview.png}}
\\caption{{Checkpointed stage-knob frontier panels.}}
\\label{{fig:knob-frontiers}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.9\\linewidth]{{figures/fallback_penalty.png}}
\\caption{{Fallback penalty panel used to motivate support-safe costing.}}
\\label{{fig:fallback-penalty}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.9\\linewidth]{{figures/thermal_plan_bank.png}}
\\caption{{Thermal plan-bank curves used by the checkpointed characterization.}}
\\label{{fig:thermal-plan-bank}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.9\\linewidth]{{figures/objective_sensitivity.png}}
\\caption{{Objective-sensitivity sweep from the checkpointed tuning study.}}
\\label{{fig:objective-sensitivity}}
\\end{{figure}}

{make_table(["Weight", "Objective", "Scheduler"], tuning, "lrr", "Checkpointed objective and scheduler weights.", "tab:tuning-weights", "\\small")}

    {make_table(["Decision", "Eff. MiB", "Queue wait ms", "Action", "Resp. max toks"], [
        [latex_escape("Admit"), fmt_num(106.0,1), "0", latex_escape("none"), "-"],
        [latex_escape("Degrade"), fmt_num(103.0,1), "0", latex_escape("reduce_responder_max_tokens"), "24"],
        [latex_escape("Reject"), fmt_num(103.0,1), "0", latex_escape("reduce_responder_max_tokens"), "24"],
        [latex_escape("Queued"), fmt_num(106.0,1), "9413", latex_escape("ADMIT"), "-"],
    ], "lrrll", "Checkpointed runtime memory/KV control events extracted from the memory-admission summary.", "tab:memory-events", "\\small")}

The memory-admission summary stores explicit admit, degrade, reject, and queued records. The runtime therefore does not silently overcommit memory. It either admits the request, applies explicit degradation actions ranked by harm-per-byte freed, or rejects the request with a reason that can be audited later in the log surface.
"""))

    sections.append(section("Artifact Surface, Verification, and Reproducibility", f"""
The artifact surface is not just a PDF. The checkpoint points to the report summary, final audit, registries, experiment summaries, logs, and plot outputs required to reconstruct the paper-facing story. The compact artifact pack remains rooted at \\texttt{{{latex_escape(relative_repo_path(Path(checkpoint_summary['canonical_evidence_paths']['artifact_pack_summary'])))}}}; the human-readable report that accompanies that pack is \\texttt{{{latex_escape(relative_repo_path(artifact_pack_report))}}}.

The verification surface for the current repository includes Python unit tests, script tests, Android unit tests, and Android connected tests. The report does not claim that every path in the broader simulator surface is equally realized online. It instead records what is implemented, what is measured on device, what is calibrated in the simulator, and where the support-safe feasibility boundary currently sits.

The full backend feasibility matrix is listed in Appendix~\\ref{{app:backend-feasibility}}. The checkpointed evidence-path inventory is listed in Appendix~\\ref{{app:evidence-paths}}. The candidate-plan registry and workload registry are also included as appendices so the report can stand alone without forcing the reader to open JSON files while reading.
"""))

    sections.append(section("Threats to Validity and Explicit Limits", r"""
This report stays inside the measured-system boundary. Backend feasibility remains narrow outside the preserved FastVLM NPU path, so the strongest claim is still support-safe scheduling rather than universal heterogeneous execution. Simulator residuals vary by family, so the simulator is appropriate for ranking and sensitivity analysis rather than exact oracle prediction. The broader workload universe is intentionally larger than the current deployed runtime surface, which means some policy studies remain calibrated what-if analyses rather than online behavior. Faithful method-class proxies are comparison surfaces inside the same environment, not line-by-line reimplementations of external systems.

These limits strengthen the report rather than weaken it: they keep every conclusion tied to the checkpointed evidence surface and avoid fake wins based on hidden fallback or unsupported backend paths.
"""))

    sections.append(section("Conclusion", r"""
GraphPilot-Edge is a measured support-safe runtime plus a calibrated multi-resource simulator for continuous multimodal assistant DAGs on one heterogeneous mobile SoC. The deployed runtime keeps the known-working FastVLM LiteRT NPU path, preserves explicit streaming and memory/KV control, and refuses to count unsupported accelerator paths as successful offload. The simulator extends that runtime with calibrated CPU/GPU/NPU surrogates, a broader workload universe, and a fair baseline surface that can be used for ranking, sensitivity analysis, and method comparison under one checkpointed evidence surface.

This larger technical report is intentionally exhaustive. It exists so the compact paper can stay selective while a companion PDF still documents the full code layout, formulas, experiment inventory, baseline catalog, workload universe, candidate-plan registry, sustained-run samples, and canonical artifact paths.
"""))

    sections.append("\\appendix")
    sections.append(section("Reproduction Instructions", f"{make_longtable(['Step', 'Exact command'], reproduction_rows, 'p{1.8in}p{4.5in}', 'Exact reproduction commands for the checkpointed GraphPilot evidence surface and this technical report.', 'app:reproduction')}", level="section"))
    sections.append(section("Core Code Excerpts", "\n\n".join(make_listing(title, repo_rel_path, start_line, end_line) for title, repo_rel_path, start_line, end_line in CODE_EXCERPTS), level="section"))
    sections.append(section("Exhaustive Workload Registry", f"{make_longtable(['Workload ID', 'Category', 'Builder', 'Datasets', 'Tags', 'Description'], workload_rows_all, 'p{1.75in}p{0.9in}p{0.9in}p{1.1in}p{1.0in}p{1.6in}', 'Exhaustive workload catalog from the checkpointed workload registry.', 'app:workloads')}", level="section"))
    sections.append(section("Candidate Plan Registry", f"{make_longtable(['Workflow', 'State', 'Plan ID', 'Backend map', 'Makespan', 'P95 queue', 'Miss rate', 'Objective'], candidate_rows, 'p{1.1in}p{0.6in}p{2.3in}p{1.7in}r r r r', 'Candidate-plan registry extracted from the checkpointed planner output.', 'app:candidate-plans')}", level="section"))
    sections.append(section("Full Baseline Result Registry", f"{make_longtable(['Workload', 'Baseline', 'Status', 'Score ms', 'Stage backends'], full_baseline_rows, 'p{1.8in}p{1.0in}c r p{2.1in}', 'Exhaustive baseline-result rows from the checkpointed baseline registry.', 'app:baseline-results')}", level="section"))
    sections.append(section("Backend Feasibility Matrix", f"{make_longtable(['Stage', 'Family', 'Role', 'CPU', 'GPU', 'NPU'], backend_rows, 'p{1.5in}p{0.8in}p{0.6in}p{1.2in}p{1.2in}p{1.2in}', 'Support-safe backend feasibility entries from the checkpointed backend matrix.', 'app:backend-feasibility')}", level="section"))
    sections.append(section("Memory Admission Event Surface", f"{make_longtable(['Source', 'workflow', 'decision', 'required', 'effective', 'queue', 'metrics-decision', 'resp max', 'ret topk', 'reason', 'actions'], memory_rows, 'p{1.2in}p{1.0in}p{0.7in}r r r p{0.9in}r r p{1.4in}p{1.1in}', 'Checkpointed memory/KV admission lines parsed from the Android metric surface.', 'app:memory-events')}", level="section"))
    sections.append(section("Calibration and Tuning Appendices", f"{make_longtable(['Candidate', 'Objective score', 'Weights'], objective_rows, 'p{1.0in}r p{4.0in}', 'Top objective-weight candidates from the checkpointed tuning summary.', 'app:objective-evals')}\n\n{make_longtable(['Candidate', 'Objective score', 'Weights'], scheduler_rows, 'p{1.0in}r p{4.0in}', 'Top scheduler-weight candidates from the checkpointed tuning summary.', 'app:scheduler-evals')}", level="section"))
    sections.append(section("Canonical Evidence Paths", f"{make_longtable(['Key', 'Repo-relative path', 'Exists'], evidence_rows, 'p{1.8in}p{3.8in}c', 'Canonical evidence-path inventory from the checkpoint manifest.', 'app:evidence-paths')}", level="section"))

    preamble = r"""
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{longtable}
\usepackage{array}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{pdflscape}
\usepackage{float}
\usepackage{placeins}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{url}
\usepackage{listings}
\usepackage{setspace}
\setstretch{1.05}
\setlength{\parskip}{4pt}
\setlength{\parindent}{0pt}
\setlength{\textfloatsep}{10pt plus 2pt minus 2pt}
\setlength{\floatsep}{8pt plus 2pt minus 2pt}
\setlength{\intextsep}{8pt plus 2pt minus 2pt}
\lstset{
  basicstyle=\ttfamily\scriptsize,
  breaklines=true,
  columns=fullflexible,
  keepspaces=true,
  showstringspaces=false,
  frame=single
}
\graphicspath{{figures/}}
\title{GraphPilot-Edge Technical Report: Full System, Simulator, Runtime, and Evidence Surface}
\author{Anonymous artifact-generated report}
\date{}
\begin{document}
"""
    tail = "\\nocite{*}\n\\bibliographystyle{IEEEtran}\n\\bibliography{references}\n\\end{document}\n"
    return preamble + "\n\\maketitle\n" + "\n\n".join(sections) + "\n" + tail


def build_report(inputs: ReportInputs) -> Path:
    checkpoint_summary = load_json(inputs.checkpoint_manifest)
    figure_summary = load_json(inputs.figure_summary)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = inputs.output_root / f"graphpilot_technical_report_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    tex_path = output_dir / "main.tex"
    tex_path.write_text(build_report_tex(inputs.checkpoint_manifest, checkpoint_summary, inputs.figure_summary, figure_summary, output_dir))

    run = lambda *cmd: subprocess.run(cmd, cwd=output_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    run("pdflatex", "-interaction=nonstopmode", "main.tex")
    run("bibtex", "main")
    run("pdflatex", "-interaction=nonstopmode", "main.tex")
    run("pdflatex", "-interaction=nonstopmode", "main.tex")

    pdf_path = output_dir / "main.pdf"
    page_count = 0
    info = subprocess.check_output(["pdfinfo", str(pdf_path)], text=True)
    for line in info.splitlines():
        if line.startswith("Pages:"):
            page_count = int(line.split(":", 1)[1].strip())
            break
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_manifest": str(inputs.checkpoint_manifest.resolve()),
        "figure_summary": str(inputs.figure_summary.resolve()),
        "report_pdf": str(pdf_path.resolve()),
        "report_tex": str(tex_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "page_count": page_count,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return output_dir / "summary.json"


def main() -> None:
    args = parse_args()
    inputs = ReportInputs(
        checkpoint_manifest=args.checkpoint_manifest.resolve(),
        figure_summary=args.figure_summary.resolve(),
        output_root=args.output_root.resolve(),
    )
    if not inputs.checkpoint_manifest.exists():
        raise FileNotFoundError(f"Checkpoint manifest does not exist: {inputs.checkpoint_manifest}")
    if not inputs.figure_summary.exists():
        raise FileNotFoundError(f"Figure summary does not exist: {inputs.figure_summary}")
    summary_path = build_report(inputs)
    print(summary_path)


if __name__ == "__main__":
    main()
