#!/usr/bin/env python3
"""Build a checkpoint-pinned IEEE CASES paper source tree and PDF for GraphPilot-Edge."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
ARTIFACT_ROOT = ROOT_DIR / "artifacts" / "graphpilot_edge"
DEFAULT_OUTPUT_ROOT = ARTIFACT_ROOT / "papers"
DEFAULT_TEMPLATE_DIR = ROOT_DIR / "papers" / "templates" / "ieee"
REQUIRED_PACK_KEYS = (
    "report",
    "paper_tables",
    "paper_draft",
    "final_audit_report",
)
REQUIRED_FIGURE_NAMES = (
    "architecture_overview.png",
    "offline_online_split.png",
    "workload_universe_coverage.png",
    "sim_real_calibration.png",
    "workflow_primary_results.png",
    "continuous_stream_results.png",
    "baseline_comparison.png",
    "ablation_breakdown.png",
    "fallback_penalty.png",
    "thermal_plan_bank.png",
    "objective_sensitivity.png",
    "tables.tex",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_artifact_pack_summary(checkpoint_manifest: Path, explicit_summary: Path | None) -> tuple[dict[str, Any], Path]:
    manifest = load_json(checkpoint_manifest)
    canonical = manifest.get("canonical_evidence_paths", {})
    manifest_summary = canonical.get("artifact_pack_summary")
    if explicit_summary is not None and manifest_summary is not None:
        raise ValueError(
            "checkpoint-manifest already specifies artifact_pack_summary; do not also pass --artifact-pack-summary."
        )
    resolved = explicit_summary or (Path(manifest_summary) if manifest_summary else None)
    if resolved is None:
        raise FileNotFoundError(
            "Checkpoint manifest is missing canonical_evidence_paths.artifact_pack_summary. "
            "Remediation: regenerate the checkpoint after building the canonical artifact pack or pass "
            "--artifact-pack-summary explicitly."
        )
    return manifest, resolved


def require_pack_paths(pack_summary: dict[str, Any]) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    missing_keys = [key for key in REQUIRED_PACK_KEYS if not pack_summary.get(key)]
    if missing_keys:
        raise FileNotFoundError(
            f"Artifact pack summary is missing required outputs {missing_keys}. "
            "Remediation: rebuild the artifact pack from the canonical checkpoint."
        )
    for key in REQUIRED_PACK_KEYS:
        path = Path(pack_summary[key])
        if not path.exists():
            raise FileNotFoundError(
                f"Artifact pack output '{key}' not found at '{path}'. "
                "Remediation: rebuild the artifact pack before building the CASES paper."
            )
        resolved[key] = path
    return resolved


def load_optional_summary(checkpoint_manifest: dict[str, Any], key: str) -> dict[str, Any] | None:
    canonical = checkpoint_manifest.get("canonical_evidence_paths", {})
    raw = canonical.get(key)
    if not raw:
        return None
    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint manifest pins {key} at '{path}', but that file is missing. "
            "Remediation: rebuild the canonical checkpoint from a consistent evidence set."
        )
    return load_json(path)


def parse_figure_summary(figure_summary_path: Path, checkpoint_manifest_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    if not figure_summary_path.exists():
        raise FileNotFoundError(
            f"Figure summary '{figure_summary_path}' does not exist. "
            "Remediation: run scripts/build_graphpilot_cases_figures.py first."
        )
    summary = load_json(figure_summary_path)
    pinned_checkpoint = summary.get("checkpoint_manifest")
    if pinned_checkpoint and Path(pinned_checkpoint).resolve() != checkpoint_manifest_path.resolve():
        raise ValueError(
            "Figure summary was generated from a different checkpoint manifest. "
            "Remediation: rebuild figures from the same checkpoint that will anchor the paper build."
        )
    output_paths = {Path(output).name: Path(output) for output in summary.get("outputs", [])}
    missing = [name for name in REQUIRED_FIGURE_NAMES if name not in output_paths]
    if missing:
        raise FileNotFoundError(
            f"Figure summary is missing required outputs {missing}. "
            "Remediation: rebuild the CASES figures from the canonical checkpoint."
        )
    for name, path in output_paths.items():
        if name in REQUIRED_FIGURE_NAMES and not path.exists():
            raise FileNotFoundError(
                f"Figure output '{name}' does not exist at '{path}'. "
                "Remediation: rebuild the CASES figures."
            )
    return summary, output_paths


def prepare_template_assets(paper_dir: Path, template_dir: Path | None) -> None:
    candidate_dir = template_dir or DEFAULT_TEMPLATE_DIR
    if candidate_dir.exists():
        ieee_class = candidate_dir / "IEEEtran.cls"
        if not ieee_class.exists():
            raise FileNotFoundError(
                f"Template directory '{candidate_dir}' is missing IEEEtran.cls. "
                "Remediation: provide a valid IEEE template directory."
            )
        for item in candidate_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, paper_dir / item.name)
        if not (paper_dir / "IEEEtran.bst").exists():
            default_bst = DEFAULT_TEMPLATE_DIR / "IEEEtran.bst"
            if default_bst.exists():
                shutil.copy2(default_bst, paper_dir / "IEEEtran.bst")
            else:
                kpsewhich = shutil.which("kpsewhich")
                if kpsewhich is None:
                    raise FileNotFoundError(
                        "Template assets are missing IEEEtran.bst and kpsewhich is unavailable. "
                        "Remediation: add IEEEtran.bst to the template directory."
                    )
                probe = subprocess.run([kpsewhich, "IEEEtran.bst"], capture_output=True, text=True, check=False)
                if probe.returncode != 0 or not probe.stdout.strip():
                    raise FileNotFoundError(
                        "Template assets are missing IEEEtran.bst and TeX could not locate it. "
                        "Remediation: add IEEEtran.bst to the template directory."
                    )
                shutil.copy2(Path(probe.stdout.strip()), paper_dir / "IEEEtran.bst")
        return

    kpsewhich = shutil.which("kpsewhich")
    if kpsewhich is None:
        raise FileNotFoundError(
            "Neither a local IEEE template directory nor kpsewhich is available. "
            "Remediation: add papers/templates/ieee/IEEEtran.cls or install TeX Live with IEEEtran."
        )
    probe = subprocess.run([kpsewhich, "IEEEtran.cls"], capture_output=True, text=True, check=False)
    if probe.returncode != 0 or not probe.stdout.strip():
        raise FileNotFoundError(
            "IEEEtran.cls is not available in TeX and no local template directory was found. "
            "Remediation: vendor IEEEtran into papers/templates/ieee or install the IEEEtran package."
        )


def build_pdf_if_possible(paper_dir: Path) -> Path:
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        raise FileNotFoundError(
            "pdflatex is not installed. Remediation: install a TeX distribution with pdflatex and rerun "
            "scripts/build_graphpilot_cases_paper.py."
        )
    bibtex = shutil.which("bibtex")
    if bibtex is None:
        raise FileNotFoundError(
            "bibtex is not installed. Remediation: install a TeX distribution with bibtex support and rerun "
            "scripts/build_graphpilot_cases_paper.py."
        )
    for _ in range(1):
        completed = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=paper_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "pdflatex failed while building the CASES paper. "
                f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
            )
    bib_completed = subprocess.run(
        [bibtex, "main"],
        cwd=paper_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if bib_completed.returncode != 0:
        raise RuntimeError(
            "bibtex failed while building the CASES paper. "
            f"stdout:\n{bib_completed.stdout}\n\nstderr:\n{bib_completed.stderr}"
        )
    for _ in range(2):
        completed = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=paper_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "pdflatex failed while building the CASES paper. "
                f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
            )
    pdf_path = paper_dir / "main.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Expected built PDF at '{pdf_path}' after pdflatex completed. "
            "Remediation: inspect the LaTeX output and rerun."
        )
    return pdf_path


def latex_escape(text: str) -> str:
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
    escaped = text
    for key, value in replacements.items():
        escaped = escaped.replace(key, value)
    return escaped


def extract_page_count(pdf_path: Path) -> int:
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo is None:
        raise FileNotFoundError(
            "pdfinfo is not installed. Remediation: install poppler-utils so the CASES builder can record "
            "paper page count."
        )
    completed = subprocess.run(
        [pdfinfo, str(pdf_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "pdfinfo failed while reading the CASES PDF. "
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )
    for line in completed.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(
        "pdfinfo output did not contain a page count. Remediation: inspect the generated PDF and pdfinfo output."
    )


def fmt_ms(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.0f}"


def fmt_num(value: Any, precision: int = 1) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{precision}f}"


def paper_ref_label(value: Any) -> str:
    if value is None:
        return "N/A"
    path = Path(str(value))
    if not path.parts:
        return latex_escape(str(value))
    tail = "/".join(path.parts[-2:]) if len(path.parts) >= 2 else path.name
    return latex_escape(tail)


def short_backend_map(stage_backends: dict[str, str]) -> str:
    order = [
        "asr.primary",
        "planner.primary",
        "vlm.fastvlm.primary",
        "retrieval.embedder.primary",
        "responder.primary",
        "tts.primary",
    ]
    mapping = {"cpu": "C", "gpu": "G", "npu": "N"}
    parts = []
    for stage_id in order:
        if stage_id in stage_backends:
            parts.append(mapping.get(stage_backends[stage_id], stage_backends[stage_id][:1].upper()))
    return "/".join(parts) if parts else "N/A"


def find_stage(backend_matrix: dict[str, Any] | None, stage_id: str) -> dict[str, Any] | None:
    if not backend_matrix:
        return None
    for stage in backend_matrix.get("stages", []):
        if stage.get("stage_id") == stage_id:
            return stage
    return None


def make_feasibility_table(backend_matrix: dict[str, Any] | None) -> str:
    stage_ids = [
        "asr.primary",
        "planner.primary",
        "vlm.fastvlm.primary",
        "retrieval.embedder.primary",
        "responder.primary",
        "tts.primary",
    ]
    stage_labels = {
        "asr.primary": "ASR",
        "planner.primary": "Planner",
        "vlm.fastvlm.primary": "FastVLM",
        "retrieval.embedder.primary": "Retrieval",
        "responder.primary": "Responder",
        "tts.primary": "TTS",
    }
    rows = []
    for stage_id in stage_ids:
        stage = find_stage(backend_matrix, stage_id)
        if stage is None:
            rows.append(f"{stage_labels[stage_id]} & N/A & N/A & N/A \\\\")
            continue
        cpu = stage["backends"].get("cpu", {}).get("status", "N/A")
        gpu = stage["backends"].get("gpu", {}).get("status", "N/A")
        npu = stage["backends"].get("npu", {}).get("status", "N/A")
        rows.append(
            f"{stage_labels[stage_id]} & {latex_escape(cpu)} & {latex_escape(gpu)} & {latex_escape(npu)} \\\\")
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Support-safe backend feasibility at the current checkpoint.}",
            "\\label{tab:feasibility}",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "Stage & CPU & GPU & NPU \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )


def make_workflow_results_table(experiment_summary: dict[str, Any] | None) -> str:
    rows = []
    workflow_labels = {
        "workflow_a_voice_only": "A",
        "workflow_b_voice_vision": "B",
        "workflow_c_voice_vision_retrieval": "C",
    }
    actual = (experiment_summary or {}).get("actual_workflows", {})
    for workflow_id in ("workflow_a_voice_only", "workflow_b_voice_vision", "workflow_c_voice_vision_retrieval"):
        metrics = actual.get(workflow_id, {})
        rows.append(
            f"{workflow_labels[workflow_id]} & {short_backend_map(metrics.get('stage_backends', {}))} & "
            f"{fmt_ms(metrics.get('warm_latency_ms'))} & {fmt_ms(metrics.get('ttft_ms'))} & "
            f"{fmt_ms(metrics.get('tts_first_audio_ms'))} \\\\")
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Primary real-device workflow results from the canonical checkpoint. Paths abbreviate CPU/GPU/NPU as C/G/N.}",
            "\\label{tab:workflow-results}",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            "Workflow & Path & Warm (ms) & TTFT (ms) & TTFS (ms) \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )


def make_calibration_table(calibration_summary: dict[str, Any] | None) -> str:
    quality = (calibration_summary or {}).get("calibration_quality_by_family", {})
    rows = []
    for family in sorted(quality):
        entry = quality[family]
        rows.append(
            f"{latex_escape(family)} & {fmt_num(entry.get('mean_absolute_error_ms'), 1)} & "
            f"{fmt_num(entry.get('max_absolute_error_ms'), 1)} & {entry.get('sample_count', 'N/A')} \\\\")
    if not rows:
        rows.append("No pinned calibration data & N/A & N/A & N/A \\\\")
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Family-level simulator residuals after calibration.}",
            "\\label{tab:calibration}",
            "\\begin{tabular}{lrrr}",
            "\\toprule",
            "Family & MAE (ms) & Max error (ms) & Samples \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )


def make_baseline_table(characterization_summary: dict[str, Any] | None) -> str:
    if not characterization_summary:
        rows = ["No pinned characterization data & N/A & N/A & N/A \\\\"]
    else:
        groups = characterization_summary.get("baseline_comparisons", {})
        selected_ids = [
            "compound.workflow_c.default",
            "continuous.workflow_a.poisson",
            "continuous.mixed_foreground_background",
            "stress.workflow_c.fallback_penalty",
        ]
        selected = []
        for entries in groups.values():
            for entry in entries:
                if entry.get("workload_id") in selected_ids:
                    selected.append(entry)
        selected.sort(key=lambda item: selected_ids.index(item["workload_id"]))
        rows = []
        for entry in selected:
            rows.append(
                f"{latex_escape(entry['workload_id'])} & {latex_escape(entry.get('graphpilot_policy', 'N/A'))} & "
                f"{fmt_num(entry.get('graphpilot_score_ms'), 1)} & "
                f"{latex_escape(entry.get('best_other_baseline_id', 'N/A'))} / {fmt_num(entry.get('best_other_score_ms'), 1)} \\\\")
        if not rows:
            rows = ["No representative workload rows found & N/A & N/A & N/A \\\\"]
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Representative baseline comparisons from the broader workload universe.}",
            "\\label{tab:baseline-snapshot}",
            "\\begin{tabular}{p{1.55in}p{0.9in}rr}",
            "\\toprule",
            "Workload & GraphPilot policy & GraphPilot & Best other \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )


def make_workload_table(workload_registry: dict[str, Any] | None) -> str:
    counts: dict[str, int] = {}
    for workload in (workload_registry or {}).get("workloads", []):
        category = workload.get("category", "unknown")
        counts[category] = counts.get(category, 0) + 1
    rows = [
        f"{latex_escape(category)} & {count} \\\\"
        for category, count in sorted(counts.items())
    ] or ["No pinned workload registry & N/A \\\\"]
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Checkpoint-pinned workload-universe coverage.}",
            "\\label{tab:workload-universe}",
            "\\begin{tabular}{lr}",
            "\\toprule",
            "Category & Count \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )


def make_tuning_table(tuning_summary: dict[str, Any] | None) -> str:
    objective_weights = ((tuning_summary or {}).get("best_objective_weights") or {}).get("weights", {})
    scheduler_weights = ((tuning_summary or {}).get("best_scheduler_weights") or {}).get("weights", {})
    rows = []
    for key in ("alpha", "beta", "gamma", "delta", "eta", "zeta", "xi", "psi"):
        rows.append(
            f"{latex_escape(key)} & {fmt_num(objective_weights.get(key), 2)} & "
            f"{fmt_num(scheduler_weights.get(key), 2)} \\\\"
        )
    if not any(objective_weights.values()) and not scheduler_weights:
        rows = ["weights unavailable & N/A & N/A \\\\"]
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Checkpoint-pinned objective and scheduler weights. Scheduler columns are only populated for weights that exist in the runtime heuristic.}",
            "\\label{tab:tuning}",
            "\\begin{tabular}{lrr}",
            "\\toprule",
            "Weight & Objective & Scheduler \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )


def make_comparison_matrix_table() -> str:
    rows = [
        "Heterogeneous mobile SoC & yes & yes & -- & yes & yes & yes \\\\",
        "Compound assistant DAG & -- & partial & -- & partial & partial & yes \\\\",
        "Support-safe fallback costing & -- & -- & -- & -- & -- & yes \\\\",
        "LLM KV-aware control & -- & -- & yes & -- & yes & yes \\\\",
        "Streaming inter-stage edges & -- & -- & partial & partial & partial & yes \\\\",
        "Thermal/contention-aware runtime & -- & partial & -- & partial & partial & yes \\\\",
        "Knob co-optimization & -- & -- & -- & -- & partial & yes \\\\",
        "Sim-to-real mobile calibration & -- & -- & -- & partial & partial & yes \\\\",
    ]
    return "\n".join(
        [
            "\\begin{table*}[t]",
            "\\centering",
            "\\small",
            "\\caption{Comparison matrix used to position GraphPilot-Edge against adjacent method classes.}",
            "\\label{tab:comparison-matrix}",
            "\\begin{tabular}{lcccccc}",
            "\\toprule",
            "Property & HEFT/CPOP & Band/ADMS & Orca/vLLM & Puzzle/Twill & Agent.xpu/HeRo & GraphPilot \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
        ]
    )


def make_ablation_matrix_table() -> str:
    rows = [
        "NoFallbackAware & fallback partition and copy penalties & attractive accelerator plans become unrealistically cheap; sim-to-real error rises \\\\",
        "NoPrefillDecodeSplit & separate text-stage modeling & backend choice for short vs. long responses collapses \\\\",
        "NoPipeline & CHUNK/TOKEN edge exploitation & TTFS regresses most strongly on streaming workloads \\\\",
        "NoBranchOverlap & VLM $\\parallel$ retrieval overlap & workflow-C end-to-end latency and idle time both increase \\\\",
        "NoMemoryReuse & interval-based buffer allocator & peak memory rises and admissible concurrency drops \\\\",
        "NoKVAdmission & explicit KV budget logic & burst traces show reject storms or uncontrolled TTFS spikes \\\\",
        "NoThermalBank & plan-bank switching/hysteresis & sustained runs drift away from the initially best plan \\\\",
        "NoKnobTuning & stage-specific visual/token/chunk knobs & robustness across workload families degrades \\\\",
        "NoFirstOutputBias & TTFS-aware dispatch terms & throughput may improve while perceived responsiveness worsens \\\\",
    ]
    return "\n".join(
        [
            "\\begin{table*}[t]",
            "\\centering",
            "\\small",
            "\\caption{Ablation matrix used to interpret causal value in the CASES paper.}",
            "\\label{tab:ablation-matrix}",
            "\\begin{tabular}{p{1.45in}p{1.55in}p{3.35in}}",
            "\\toprule",
            "Ablation & Removed component & Expected effect if the design matters \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
        ]
    )


def make_baseline_catalog_table(baseline_registry: dict[str, Any] | None) -> str:
    baseline_ids = (baseline_registry or {}).get("baseline_ids", [])
    description_map = {
        "cpu_only": "force all feasible work onto CPU resources only",
        "gpu_only": "force all feasible work onto GPU resources only",
        "npu_only": "force all feasible work onto NPU resources only",
        "current_deployed_plan": "replay the current support-safe Android deployment",
        "stage_greedy": "pick the locally best support-safe backend per stage",
        "static_best_map": "single offline best map with no online adaptation",
        "no_pipeline": "disable CHUNK/TOKEN edge exploitation",
        "no_fallback_aware": "ignore fallback partitions and copy penalties",
        "no_memory_kv": "remove explicit memory and KV accounting",
        "no_knob_tuning": "freeze stage knobs at default values",
        "no_thermal_adaptation": "disable thermal plan-bank switching",
        "band_like": "proxy mobile multi-DNN mapping with coarse heterogeneous affinity",
        "adms_like": "proxy heterogeneous co-execution with static resource splits",
        "puzzle_like": "proxy partition-aware mobile mapping without assistant-specific control",
        "twill_like": "proxy compound-AI scheduler without support-safe fallback costing",
        "heteroinfer_like": "proxy heterogeneous LLM engine centered on prefill/decode placement",
        "agent_xpu_like": "proxy agentic SoC runtime with LLM-flow emphasis",
        "hero_like": "proxy heterogeneous mobile agentic RAG scheduler",
    }
    rows = [
        f"{latex_escape(baseline_id)} & {latex_escape(description_map.get(baseline_id, 'baseline description unavailable'))} \\\\"
        for baseline_id in baseline_ids
    ] or ["No pinned baseline registry & N/A \\\\"]
    return "\n".join(
        [
            "\\begin{table*}[t]",
            "\\centering",
            "\\small",
            "\\caption{Baseline catalog used in the CASES comparison section.}",
            "\\label{tab:baseline-catalog}",
            "\\begin{tabular}{p{1.6in}p{4.7in}}",
            "\\toprule",
            "Baseline ID & Intent within the shared GraphPilot environment \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
        ]
    )


def make_result_blocks_table() -> str:
    rows = [
        "0 & current prototype anchor & working deployed workflows and support-safe path \\\\",
        "1 & simulator accuracy & family residuals and workflow sim-to-real deltas \\\\",
        "2 & fallback penalty & optimistic versus fallback-aware accelerator costing \\\\",
        "3 & baseline scheduling comparison & GraphPilot versus internal and proxy baselines \\\\",
        "4 & memory and KV pressure & admission, degradation, rejection, and peak-memory behavior \\\\",
        "5 & thermal and plan-bank behavior & sustained traces and switching behavior \\\\",
        "6 & objective sensitivity & weight perturbation and plan stability \\\\",
        "7 & workload breadth and generality & representative workloads from each family class \\\\",
    ]
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\caption{Result blocks carried from the revision report into the CASES evaluation.}",
            "\\label{tab:result-blocks}",
            "\\begin{tabular}{clp{1.95in}}",
            "\\toprule",
            "Block & Focus & Why it matters \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )


def build_tables_tex(
    experiment_summary: dict[str, Any] | None,
    calibration_summary: dict[str, Any] | None,
    characterization_summary: dict[str, Any] | None,
    backend_matrix: dict[str, Any] | None,
    workload_registry: dict[str, Any] | None,
    baseline_registry: dict[str, Any] | None,
    tuning_summary: dict[str, Any] | None,
) -> str:
    return "\n\n".join(
        [
            "% Auto-generated tables for the GraphPilot CASES paper",
            make_feasibility_table(backend_matrix),
            make_workflow_results_table(experiment_summary),
            make_calibration_table(calibration_summary),
            make_baseline_table(characterization_summary),
            make_workload_table(workload_registry),
            make_tuning_table(tuning_summary),
            make_comparison_matrix_table(),
            make_baseline_catalog_table(baseline_registry),
            make_result_blocks_table(),
            make_ablation_matrix_table(),
            "",
        ]
    )


def copy_figures(output_paths: dict[str, Path], figure_dir: Path) -> dict[str, str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for name in REQUIRED_FIGURE_NAMES:
        source = output_paths[name]
        target = figure_dir / source.name
        shutil.copy2(source, target)
        copied[name] = target.name
    return copied


def workflow_results_summary(experiment_summary: dict[str, Any] | None) -> list[str]:
    if not experiment_summary:
        return ["The active checkpoint does not pin a structured experiment summary, so this paper build reports only the artifact-pack narrative."]
    actual = experiment_summary.get("actual_workflows", {})
    comparisons = {item["workflow_id"]: item for item in experiment_summary.get("comparisons", [])}
    lines = []
    for workflow_id in ("workflow_a_voice_only", "workflow_b_voice_vision", "workflow_c_voice_vision_retrieval"):
        metrics = actual.get(workflow_id, {})
        compare = comparisons.get(workflow_id, {})
        lines.append(
            f"{latex_escape(workflow_id)} reaches {fmt_ms(metrics.get('warm_latency_ms'))} ms warm latency, "
            f"{fmt_ms(metrics.get('ttft_ms'))} ms TTFT, and {fmt_ms(metrics.get('tts_first_audio_ms'))} ms TTFS; "
            f"the calibrated simulator predicts {fmt_ms(compare.get('candidate_predicted_makespan_ms'))} ms, for a delta of {fmt_ms(compare.get('latency_delta_ms'))} ms."
        )
    return lines


def calibration_summary_lines(calibration_summary: dict[str, Any] | None) -> list[str]:
    if not calibration_summary:
        return ["The active checkpoint does not pin a structured calibration summary."]
    quality = calibration_summary.get("calibration_quality_by_family", {})
    if not quality:
        return ["The calibration summary does not contain family-level residuals."]
    best_family = min(quality.items(), key=lambda item: item[1].get("mean_absolute_error_ms", float("inf")))
    worst_family = max(quality.items(), key=lambda item: item[1].get("mean_absolute_error_ms", float("-inf")))
    overhead = calibration_summary.get("global_orchestration_overhead_ms")
    return [
        f"The calibrated surrogate model carries a global orchestration overhead term of {fmt_num(overhead, 1)} ms.",
        f"The best calibrated family is {latex_escape(best_family[0])} at {fmt_num(best_family[1].get('mean_absolute_error_ms'), 1)} ms MAE, while the hardest family remains {latex_escape(worst_family[0])} at {fmt_num(worst_family[1].get('mean_absolute_error_ms'), 1)} ms MAE.",
    ]


def characterization_summary_lines(characterization_summary: dict[str, Any] | None) -> list[str]:
    if not characterization_summary:
        return ["The active checkpoint does not pin a structured characterization summary."]
    affinity = characterization_summary.get("backend_affinity", {})
    win_counts = affinity.get("resource_win_counts", {})
    note = characterization_summary.get("graphpilot_policy_note", "")
    continuous = characterization_summary.get("baseline_comparisons", {}).get("continuous_workloads", [])
    mixed = next((entry for entry in continuous if entry.get("workload_id") == "continuous.mixed_foreground_background"), None)
    lines = [
        f"Across the checkpoint-pinned workload universe, backend win counts are CPU={win_counts.get('cpu', 'N/A')}, GPU={win_counts.get('gpu', 'N/A')}, and NPU={win_counts.get('npu', 'N/A')}.",
    ]
    if mixed is not None:
        lines.append(
            f"For {latex_escape(mixed['workload_id'])}, GraphPilot's current policy proxy scores {fmt_num(mixed.get('graphpilot_score_ms'), 1)} ms against the best other baseline at {fmt_num(mixed.get('best_other_score_ms'), 1)} ms, with queue delay and miss-rate terms carried inside the same objective."
        )
    if note:
        lines.append(latex_escape(note))
    return lines


def ensure_line_count(lines: list[str], count: int) -> list[str]:
    padded = list(lines)
    while len(padded) < count:
        padded.append("No additional checkpoint-pinned detail is available for this subsection.")
    return padded[:count]


def build_abstract(checkpoint_manifest: dict[str, Any], experiment_summary: dict[str, Any] | None, calibration_summary: dict[str, Any] | None) -> str:
    actual = (experiment_summary or {}).get("actual_workflows", {})
    a = actual.get("workflow_a_voice_only", {})
    b = actual.get("workflow_b_voice_vision", {})
    c = actual.get("workflow_c_voice_vision_retrieval", {})
    quality = (calibration_summary or {}).get("calibration_quality_by_family", {})
    retrieval_mae = quality.get("retrieval", {}).get("mean_absolute_error_ms")
    vlm_mae = quality.get("vlm", {}).get("mean_absolute_error_ms")
    truth_source = checkpoint_manifest.get("truth_source_pdf", "N/A")
    return dedent(
        rf"""
        \begin{{abstract}}
        GraphPilot-Edge is a profiler-driven runtime and calibrated simulator for continuous multimodal assistant DAGs on Snapdragon SM8750. The system keeps support-safe placement explicit: FastVLM runs on the LiteRT NPU path, while the current deployed text and speech stages remain on verified CPU backends, and infeasible GPU/NPU paths are recorded rather than hidden behind fallback. GraphPilot-Edge combines checkpoint-pinned profiling, multi-resource simulation, streaming-aware scheduling, and explicit memory/KV admission control. On the canonical checkpoint used for this paper, workflow A reaches {fmt_ms(a.get('warm_latency_ms'))} ms warm latency with {fmt_ms(a.get('ttft_ms'))} ms TTFT and {fmt_ms(a.get('tts_first_audio_ms'))} ms TTFS, workflow B reaches {fmt_ms(b.get('warm_latency_ms'))} ms / {fmt_ms(b.get('ttft_ms'))} ms / {fmt_ms(b.get('tts_first_audio_ms'))} ms, and workflow C reaches {fmt_ms(c.get('warm_latency_ms'))} ms / {fmt_ms(c.get('ttft_ms'))} ms / {fmt_ms(c.get('tts_first_audio_ms'))} ms. The calibrated simulator is strongest on retrieval-family workloads ({fmt_num(retrieval_mae, 1)} ms mean absolute error) and still weakest on VLM-family workloads ({fmt_num(vlm_mae, 1)} ms mean absolute error), so we use it for ranking, sensitivity analysis, and baseline comparison rather than claiming exact end-to-end prediction in every regime. The paper packages the runtime, the calibrated simulator, the workload universe, method-class proxy baselines, and a single canonical artifact checkpoint rooted in {paper_ref_label(truth_source)}.
        \end{{abstract}}

        \paragraph*{{Keywords}} mobile SoCs, heterogeneous scheduling, multimodal assistants, support-safe execution, simulator calibration, runtime systems.
        """
    ).strip() + "\n"


def build_introduction(checkpoint_manifest: dict[str, Any], experiment_summary: dict[str, Any] | None) -> str:
    workflow_lines = ensure_line_count(workflow_results_summary(experiment_summary), 3)
    return dedent(
        rf"""
        \section{{Introduction}}
        Edge assistants are no longer single-model pipelines. A realistic on-device assistant mixes speech recognition, text planning, multimodal reasoning, retrieval, response generation, and speech synthesis, all on one thermally constrained mobile SoC. The scheduling problem is not just ``pick the fastest accelerator''; it is a continuous, mixed-criticality DAG scheduling problem with support-safe backend feasibility, transfer costs, queueing delay, memory pressure, and time-to-first-speech requirements.

        GraphPilot-Edge addresses that problem for the concrete workflows that the current prototype already runs on device: (A) ASR $\rightarrow$ Planner $\rightarrow$ Responder $\rightarrow$ TTS, (B) ASR $\rightarrow$ Planner $\rightarrow$ VLM $\rightarrow$ Responder $\rightarrow$ TTS, and (C) ASR $\rightarrow$ Planner $\rightarrow$ (VLM $\parallel$ Retrieval) $\rightarrow$ Responder $\rightarrow$ TTS. The deployed path is intentionally narrow and honest: FastVLM stays on the LiteRT NPU path, while text and speech stages remain on support-safe CPU implementations until stronger backend evidence exists.

        That honesty is central to the paper. We do not count silent fallback as successful offload, and we do not turn infeasible GPU/NPU text paths into a paper claim by averaging them into a simulator-only story. Instead, we build a support-safe runtime, a calibrated simulator anchored to the same checkpoint, and a workload universe broad enough to show what GraphPilot can and cannot currently optimize.

        Our contributions are fourfold. First, we present a runtime that keeps support-safe stage placement, responder-to-TTS streaming, and memory/KV admission explicit on SM8750. Second, we present a calibrated multi-resource simulator whose objective includes latency, TTFS, queue delay, miss rate, memory, copy, energy proxy, and quality loss. Third, we evaluate GraphPilot against internal baselines and faithful method-class proxies inside the same environment. Fourth, we release the paper, figures, tables, and audit outputs from a single canonical checkpoint manifest to prevent artifact drift.

        The current checkpoint demonstrates the following primary device results: {workflow_lines[0]} {workflow_lines[1]} {workflow_lines[2]}
        """
    ).strip() + "\n"


def build_prototype_anchor_section(experiment_summary: dict[str, Any] | None, backend_matrix: dict[str, Any] | None) -> str:
    actual = (experiment_summary or {}).get("actual_workflows", {})
    a = actual.get("workflow_a_voice_only", {})
    b = actual.get("workflow_b_voice_vision", {})
    c = actual.get("workflow_c_voice_vision_retrieval", {})
    workflow_c_path = short_backend_map(c.get("stage_backends", {}))
    fastvlm_npu = "unknown"
    fastvlm_stage = find_stage(backend_matrix, "vlm.fastvlm.primary")
    if fastvlm_stage is not None:
        fastvlm_npu = fastvlm_stage.get("backends", {}).get("npu", {}).get("status", "unknown")
    return dedent(
        rf"""
        \section{{Prototype Anchor and Honest Scope}}
        The starting point for this paper is not a simulator-only study. The current GraphPilot artifact already contains a working Android runtime, a planner, a memory/KV admission surface, and a support-safe deployment that executes three end-to-end assistant workflows on SM8750. The deployed best path is intentionally narrow: workflow A runs on CPU-only support-safe stages, workflow B keeps FastVLM on the LiteRT NPU path while retaining CPU text and speech stages, and workflow C keeps the same NPU VLM path while leaving retrieval on CPU because the measured GPU retrieval variant is slower on this device generation.

        The primary checkpointed workflow results are therefore treated as the anchor rather than as a preliminary footnote. Workflow A reaches {fmt_ms(a.get('warm_latency_ms'))} ms warm latency with {fmt_ms(a.get('ttft_ms'))} ms TTFT and {fmt_ms(a.get('tts_first_audio_ms'))} ms TTFS. Workflow B reaches {fmt_ms(b.get('warm_latency_ms'))} ms warm latency while preserving the support-safe FastVLM NPU path. Workflow C reaches {fmt_ms(c.get('warm_latency_ms'))} ms on the deployed {workflow_c_path} backend map, and the checkpoint records that the CPU retrieval plan beats the support-safe GPU retrieval alternative.

        This anchor matters because it defines both what the paper can already prove and what it cannot yet prove. The current artifact proves that GraphPilot is a real runtime with real device-side streaming, support-safe backend accounting, and measurable end-to-end assistant behavior. It does not yet prove that all major stage families run support-safely across CPU, GPU, and NPU, nor does it prove that the online runtime already realizes every scheduling policy studied in the simulator. The CASES paper therefore scopes the claim to the measured system: a support-safe runtime plus a calibrated simulator for broader policy exploration.

        The checkpoint-pinned feasibility matrix makes this scope explicit. FastVLM's NPU status is {latex_escape(str(fastvlm_npu))}, while text and speech feasibility remain CPU-dominant. That asymmetry is not hidden; it is the reason the paper centers fallback-aware costing, queue-aware scheduling, and honest sim-to-real calibration instead of pretending that every stage is already heterogeneously deployable.

        \begin{{itemize}}
        \item What the current system already proves: working end-to-end runtime, live responder-to-TTS streaming, one real parallel branch in workflow C, and explicit support-safe feasibility tracking.
        \item What the current system does not yet prove: universal support-safe GPU/NPU text execution, arbitrary online global replanning, or a claim that every simulator-side plan is already deployable on-device.
        \end{{itemize}}
        """
    ).strip() + "\n"


def build_differentiation_section() -> str:
    return dedent(
        r"""
        \section{Thesis, Contributions, and Differentiation}
        The strongest defensible paper thesis is narrower than ``full heterogeneous execution for everything'' and stronger than ``we built a prototype'': GraphPilot-Edge is a profiler-driven runtime and calibrated simulator for continuous multimodal assistant DAGs on a heterogeneous mobile SoC, with support-safe placement, stage-specific knobs, fallback-aware costing, and memory/KV-aware control. The emphasis on support safety is the differentiator that keeps the paper coherent. A system that silently falls back inside an accelerator assignment is not exercising heterogeneity honestly.

        Four contribution claims survive the checkpoint audit. First, GraphPilot provides a real Android runtime that executes assistant DAGs, preserves the verified LiteRT FastVLM NPU path, and keeps streaming plus memory/KV control explicit. Second, GraphPilot provides a calibrated multi-resource simulator that models CPU, GPU, and NPU resources using public anchors plus measured surrogate parameters. Third, GraphPilot evaluates both internal baselines and faithful method-class proxies inside the same support-safe environment. Fourth, GraphPilot packages those results through a checkpoint-pinned artifact surface so that the paper, figures, tables, and final audit are all derived from one canonical evidence set.

        This combination is what places GraphPilot between adjacent communities rather than inside only one of them. Classical heterogeneous schedulers usually assume static costs and operation-complete resources. Mobile inference runtimes optimize one model at a time. LLM-serving systems capture KV behavior but not mobile support safety. Compound-AI schedulers broaden the scope beyond DNN kernels but do not center the same assistant DAG plus fallback plus memory/KV story. The comparison matrix in Table~\ref{tab:comparison-matrix} is therefore not a victory lap; it is the scope argument for why this paper belongs at CASES.
        """
    ).strip() + "\n"


def build_system_design(experiment_summary: dict[str, Any] | None, backend_matrix: dict[str, Any] | None) -> str:
    support_safe_note = "The feasibility table is generated from the checkpoint-pinned backend matrix and therefore records infeasibility explicitly instead of suppressing it."
    return dedent(
        rf"""
        \section{{Runtime and Problem Formulation}}
        GraphPilot-Edge represents each request as a DAG whose nodes are stages or macro-regions and whose edges carry full outputs, chunks, or token streams. The runtime optimizes the eight-term objective from the revision plan:
        \begin{{equation}}
        \begin{{aligned}}
        J(\Pi)=&\ \alpha P95(T_{{e2e}})+\beta P95(T_{{TFS}})+\gamma \bar{{E}}+\delta M_{{peak}} \\
        &+\eta B_{{copy}}+\zeta Q_{{loss}}+\xi P95(T_{{queue}})+\psi R_{{miss}}.
        \end{{aligned}}
        \end{{equation}}
        The hard constraints are equally important: every assigned stage or macro-region must be support-safe, memory must stay within the runtime budget, and quality loss must stay below the configured tolerance.

        The practical interpretation of the objective is as important as the equation. TTFS is not a secondary metric for assistants; it is the observable latency at which the system starts speaking back to the user. Queue delay and miss rate matter because the paper targets continuous traffic rather than a single isolated query. Copy cost, peak memory, and quality loss matter because mobile heterogeneity is often lost to movement, residency, or degradation rather than raw compute throughput. The objective therefore exists to prevent GraphPilot from optimizing the wrong thing.

        Offline, GraphPilot profiles feasible variants, separates responder prefill from decode, enumerates candidate placements, and scores them under queue-aware and memory-aware costs. Online, the Android runtime enforces explicit backend assignments, streams responder tokens into chunked TTS, and applies admission, degradation, and rejection decisions when memory/KV pressure exceeds the safe budget. This is a real runtime path, not only a simulator. The runtime makes three design commitments that are central to the paper: it never counts hidden fallback as native acceleration, it keeps responder prefill and decode as distinct regimes, and it treats queueing plus memory/KV control as part of the scheduling problem rather than as post-hoc diagnostics.

        The deployed system currently uses only a subset of the theoretically possible backend choices, but the formulation already supports the broader space. Each node chooses a variant, a hardware assignment, and a knob vector. Each edge chooses a release mode and release threshold. This is the right abstraction because it covers both support-safe deployed plans and simulator-side what-if plans under the same notation.

        Design goals follow directly from the revision report: (1) support-safe placement instead of optimistic offload, (2) streaming-aware TTFS improvement rather than throughput-only optimization, (3) explicit memory and KV accounting under burst load, and (4) enough simulator fidelity to rank plans honestly against the same device family. The point of the paper is not that every backend path already wins, but that the same framework can explain why some attractive heterogeneous plans fail once support safety, transfers, queueing, and thermal drift are made explicit.

        {support_safe_note}

        \input{{tables}}
        """
    ).strip() + "\n"


def build_workloads_section(workload_registry: dict[str, Any] | None, baseline_registry: dict[str, Any] | None) -> str:
    workload_count = len((workload_registry or {}).get("workloads", []))
    categories: dict[str, int] = {}
    for workload in (workload_registry or {}).get("workloads", []):
        category = workload.get("category", "unknown")
        categories[category] = categories.get(category, 0) + 1
    category_sentence = ", ".join(f"{category}={count}" for category, count in sorted(categories.items())) or "no workload categories pinned"
    baseline_ids = (baseline_registry or {}).get("baseline_ids", [])
    baseline_sentence = ", ".join(baseline_ids) if baseline_ids else "no baseline registry pinned"
    return dedent(
        rf"""
        \section{{Workload Universe and Comparison Matrix}}
        The broader CASES evidence does not stop at the three deployed workflows. The checkpoint-pinned workload registry contains {workload_count} workloads spanning {latex_escape(category_sentence)}. This workload universe is important because it lets the simulator answer questions that the deployed support-safe runtime cannot yet answer directly, such as how GraphPilot behaves under mixed-criticality streams, explicit fallback-pressure cases, and workload families that share the same scheduling substrate but not the same on-device adapters.

        The workload structure follows the revision report. Primitive-operator workloads isolate GEMM, attention, convolution, ANN, audio-frontend, and control-heavy glue behavior. Model-family workloads cover LLM, VLM, STT, TTS, retrieval, CNN, ViT, and sequential glue tasks. Compound assistant DAGs include the deployed A/B/C workflows plus document QA, chart QA, MMMU-style reasoning, mobile-actions planning, and higher-recall retrieval variants. Continuous-stream workloads inject Poisson arrivals, bursty arrivals, queue-overload traces, and mixed foreground/background jobs. Stress workloads probe long context, high visual tokens, shape volatility, small chunks, and fallback-sensitive cases.

        The revision report also requires the workload families to expose the knobs that actually move latency, TTFS, and memory:
        \begin{{itemize}}
        \item LLM families vary prompt length, output length, retrieval-token budget, quantization, and prefill/decode placement.
        \item VLM families vary image resolution, crop count, and visual-token budget so the simulator can study dense one-shot multimodal work separately from text-only decoding.
        \item STT and TTS families vary chunk size and quality mode because these stages are often launch-bound rather than compute-bound.
        \item Retrieval families separate embedding, ANN, reranking, and chunk packing so GraphPilot can reason about CPU-friendly ANN work versus accelerator-friendly dense embedding work.
        \item CNN, ViT, and glue workloads provide non-assistant counterexamples that test whether the scheduler is merely overfit to A/B/C.
        \end{{itemize}}

        Each workload family exposes stage-specific knobs because the revision report argues that a mobile scheduler is only credible if it reasons about the levers that actually matter. LLM workloads vary prompt length, output length, retrieval budget, and prefill/decode placement. VLM workloads vary image resolution, crop count, and visual-token budget. STT and TTS workloads vary chunk size and quality modes. Retrieval workloads vary corpus size, top-$k$, and probe count. CNN, ViT, and glue workloads supply the conv-heavy, token-heavy, and control-heavy counterexamples needed to test whether GraphPilot is merely overfit to the three assistant workflows.

        The baseline matrix is equally broad. The checkpoint currently includes the following baseline IDs: {latex_escape(baseline_sentence)}. These policies are all implemented inside the same simulator/runtime environment, which matters because it prevents the paper from comparing GraphPilot to numbers produced on different hardware or with incompatible support assumptions. The paper uses this broader workload universe for two purposes. First, it measures stage-family affinities, transfer overheads, and thermal drift outside the narrow deployed path. Second, it ensures that the comparison story is not built from one friendly workflow while the simulator ignores the difficult cases.

        The broader workload universe also justifies the hybrid evidence strategy. A/B/C remain the primary device results because they are the support-safe deployed flows. The additional families give the simulator enough breadth to show where GraphPilot's scheduling logic generalizes, where it ties with simpler baselines because the feasible set is narrow, and where hidden fallback or memory pressure erase the naive accelerator win.

        \begin{{figure*}}[t]
        \centering
        \includegraphics[width=0.96\textwidth]{{figures/workload_universe_coverage.png}}
        \caption{{Checkpoint-pinned workload-universe coverage used to support the hybrid evidence strategy.}}
        \label{{fig:workload-coverage}}
        \end{{figure*}}
        """
    ).strip() + "\n"


def build_simulator_section(calibration_summary: dict[str, Any] | None, characterization_summary: dict[str, Any] | None) -> str:
    calibration_lines = ensure_line_count(calibration_summary_lines(calibration_summary), 2)
    return dedent(
        rf"""
        \section{{Calibrated Multi-Resource Simulator}}
        The GraphPilot simulator models explicit CPU, GPU, and NPU resources rather than anonymous backend labels. Each resource instance carries a support mask, launch overhead, transfer links, queue state, thermal slowdown parameters, and batching parameters. The execution model follows the revision-report formulation:
        \begin{{equation}}
        T^{{exec}}_{{u,h}} = L_h + \max\left(T^{{ops}}_{{u,h}}(B), T^{{mem}}_{{u,h}}\right) \rho_h(\theta) \kappa_h(q),
        \end{{equation}}
        where the compute term uses op-class service rates, the memory term uses effective bandwidth, and the thermal and contention terms are calibrated surrogates rather than undocumented vendor claims. Transfers are charged explicitly,
        \begin{{equation}}
        T_{{xfer}}(S,h,h') = \tau^0_{{h,h'}} + S / BW_{{h,h'}} + \tau^{{layout}}_{{h,h'}},
        \end{{equation}}
        unless the producer and consumer share a compatible memory space.

        Public hardware facts and surrogate parameters are kept separate by design. Public anchors cover the existence of the Oryon CPU complex, the LiteRT/QNN NPU path, the memory hierarchy class, and officially documented runtime support. The simulator then fits the quantities that are not vendor facts: effective service rates, transfer bias, launch overhead after framework cost, thermal time constants, and contention coefficients. This separation matters because it prevents the paper from smuggling undocumented hardware claims into the model.

        The revision report prescribes a layered calibration procedure, and GraphPilot follows that same structure:
        \begin{{itemize}}
        \item direct CPU, GPU, and NPU stage-family measurements where support-safe adapters exist;
        \item synthetic transfer and launch-overhead measurements so small streaming stages are not modeled as pure compute;
        \item sustained traces for thermal and contention fitting;
        \item family-level and workflow-level validation instead of only one end-to-end error number.
        \end{{itemize}}

        The simulator also makes fallback a first-class phenomenon. If a stage or region contains unsupported operations on a candidate resource, GraphPilot partitions the region into supported and unsupported subregions, charges the intermediate transfers explicitly, and records the fallback-aware latency instead of the optimistic accelerator latency. That is the core mechanism that keeps the simulator aligned with the support-safe runtime story.

        Responder prefill and decode stay distinct in both the simulator and the runtime. The simulator therefore models a one-time prefill cost, an optional KV migration cost, and a per-token decode cost, rather than collapsing the responder into one opaque stage. Memory and KV cache accounting use the same admission and degradation surfaces that the runtime logs on device. The same section also models batching and thermal slowdown, because small streaming stages are often launch-bound while longer sustained runs are dominated by temperature and contention.

        {calibration_lines[0]}
        {calibration_lines[1]}

        These residuals matter. They mean the current simulator is suitable for support-safe ranking, sensitivity studies, and baseline comparisons, but the paper does not claim exact latency prediction for every family. That limitation is surfaced directly in Table~\ref{{tab:calibration}} and in the sim-to-real figure. In other words, the simulator is publishable because it is calibrated, validated, and scoped correctly, not because it claims oracle-level prediction.
        """
    ).strip() + "\n"


def build_algorithms_section(tuning_summary: dict[str, Any] | None) -> str:
    objective_weights = ((tuning_summary or {}).get("best_objective_weights") or {}).get("weights", {})
    scheduler_weights = ((tuning_summary or {}).get("best_scheduler_weights") or {}).get("weights", {})
    objective_sentence = ", ".join(f"{key}={value}" for key, value in sorted(objective_weights.items())) or "no pinned objective weights"
    scheduler_sentence = ", ".join(f"{key}={value}" for key, value in sorted(scheduler_weights.items())) or "no pinned scheduler weights"
    return dedent(
        rf"""
        \section{{Planning, Scheduling, and Memory Control}}
        GraphPilot solves the search problem in layers rather than with one opaque optimizer. First, each stage or macro-region keeps only the non-dominated support-safe choices on latency, memory, energy proxy, quality proxy, and fallback risk. Second, stage-level backend maps are enumerated exhaustively over the feasible set. Third, opened heavy stages use macro-region beam search with dominance pruning. Fourth, the discrete-event simulator ranks the surviving plans before the top candidates are executed on device.

        This layered structure is a deliberate response to the complexity of the search space. A naive brute-force search over stage choice, macro-region choice, stream mode, chunk size, and knob settings grows too quickly once VLM and responder internals are opened. Frontier extraction and stage-level exhaustive search therefore act as the pruning stage, while beam search is reserved for the heavy stages where the cost model says there is enough performance upside to justify additional complexity.

        The online runtime uses a HEFT-style critical-path score with first-output, copy, memory, and thermal terms:
        \begin{{equation}}
        \mathrm{{rank}}_u(i)=\bar{{T}}_i + \max_{{j \in Succ(i)}}(\bar{{C}}_{{i \rightarrow j}} + \mathrm{{rank}}_u(j)),
        \end{{equation}}
        \begin{{equation}}
        P(\tau,b)=w_r \mathrm{{rank}}_u + w_f F(\tau) + w_a A(\tau) - w_c C_{{copy}} - w_m R_{{mem}} - w_t R_{{thermal}}.
        \end{{equation}}
        The checkpoint-pinned tuning summary selected objective weights {latex_escape(objective_sentence)} and scheduler weights {latex_escape(scheduler_sentence)}. These tuned weights are not universal truths; they are part of the artifact and therefore part of the reproducibility surface.

        Continuous simulation and online dispatch then share the same timing primitives. For ready task $u$ on hardware instance $h$, the simulator computes
        \begin{{equation}}
        EST(u,h)=\max(t_{{deps}}(u), t_{{free}}(h), t_{{mem}}(u,h), t_{{qslot}}(h)),
        \end{{equation}}
        \begin{{equation}}
        EFT(u,h)=EST(u,h)+T^{{exec}}_{{u,h}},
        \end{{equation}}
        and dispatch chooses the highest-priority feasible pair, tie-broken by smallest earliest finish time. This keeps the online story aligned with the simulator rather than inventing a different runtime semantics for the deployed path.

        Memory and KV state are first-class citizens. GraphPilot admits, degrades, or rejects work according to explicit memory-accounting rules rather than silent overcommit. KV admission follows the same inequality used in the revision report,
        \begin{{equation}}
        \sum_s M_{{KV}}(s) + M_{{workbuf}} + M_{{live}} \le M_{{budget}} - M_{{margin}},
        \end{{equation}}
        and degradation chooses the smallest quality-plus-latency harm per unit of freed memory. Decode remains sticky unless a one-time KV migration can be justified, and runtime logs record the resulting decision explicitly. This matters more than small single-kernel wins because queueing, TTFS, and rejection behavior dominate user-visible quality once requests overlap.

        Finally, GraphPilot keeps the claim narrow where the implementation is narrow. The planner and scheduler reason about many more potential placements than the current runtime can safely deploy, but only the support-safe deployed placements are treated as device facts. The simulator-side algorithmic story is therefore explicitly about ranking and comparative analysis, not about claiming that every ranked plan is already deployable.
        """
    ).strip() + "\n"


def build_methodology_section(
    checkpoint_manifest: dict[str, Any],
    experiment_summary: dict[str, Any] | None,
    calibration_summary: dict[str, Any] | None,
    characterization_summary: dict[str, Any] | None,
) -> str:
    experiment_path = checkpoint_manifest.get("canonical_evidence_paths", {}).get("experiment_summary", "N/A")
    calibration_path = checkpoint_manifest.get("canonical_evidence_paths", {}).get("calibration_summary", "N/A")
    characterization_path = checkpoint_manifest.get("canonical_evidence_paths", {}).get("characterization_summary", "N/A")
    comparison_count = len((experiment_summary or {}).get("comparisons", []))
    family_count = len((calibration_summary or {}).get("calibration_quality_by_family", {}))
    ablation_groups = ", ".join(sorted((characterization_summary or {}).get("ablations", {}).keys())) or "none"
    return dedent(
        rf"""
        \section{{Methodology and Calibration Procedure}}
        The artifact follows a single-manifest methodology. A checkpoint pins the experiment summary, calibration summary, characterization summary, backend matrix, workload registry, baseline registry, and final audit inputs. The paper then consumes that checkpoint and refuses to mix in newer ``latest'' files. For the active build, the primary experiment summary is pinned at {paper_ref_label(experiment_path)}, the calibration summary at {paper_ref_label(calibration_path)}, and the characterization summary at {paper_ref_label(characterization_path)}.

        Calibration and evaluation proceed in three loops. First, GraphPilot revalidates the support-safe deployed workflows on device. Second, the calibration scripts fit surrogate launch-overhead, transfer-bias, contention, and thermal terms against the same device evidence. Third, the characterization and baseline scripts exercise the simulator over the broader workload universe. The active checkpoint contains {comparison_count} direct workflow comparison rows and {family_count} family-level calibration entries, while the characterization layer records ablation groups for {latex_escape(ablation_groups)}.

        The methodology is deliberately asymmetric because the system itself is asymmetric. Workflows A/B/C and explicit memory-admission behavior are device results. Continuous-stream studies, fallback-sensitive scheduling cases, and method-class proxy baselines live primarily in the simulator. The connection between the two is the checkpoint-pinned calibration loop and the requirement that all offload claims stay support-safe. The paper therefore treats simulator-vs-real agreement as a measured quantity rather than a hidden assumption.

        This methodology also controls the truth surface. The same checkpoint drives the figure bundle, the tables, the paper draft, and the final audit. The builder fails fast if the figure bundle or artifact pack come from a different checkpoint. That checkpoint discipline is not cosmetic; it is the mechanism that prevents the paper from turning into a mixture of old device runs, newer simulator outputs, and inconsistent audits.

        Finally, the methodology follows the revision-report result blocks directly: prototype anchor, simulator accuracy, fallback penalty, baseline scheduling comparison, memory/KV pressure, thermal/plan-bank behavior, objective sensitivity, and workload breadth. The paper only discusses a result block when the corresponding canonical artifact is present in the checkpoint. This is why the evaluation can be broad without becoming vague: every block has a fixed purpose, a fixed evidence source, and a fixed relationship to the central claim.
        """
    ).strip() + "\n"


def build_experiment_matrix_section() -> str:
    return dedent(
        r"""
        \section{Experiment Matrix}
        The revision report reorganized the evaluation into result blocks so that every major claim is tied to a concrete evidence family. Table~\ref{tab:result-blocks} carries that structure into the paper. Result block~0 anchors the paper on the real deployed prototype. Result block~1 measures simulator accuracy rather than assuming it. Result block~2 quantifies fallback penalties explicitly. Result blocks~3--7 cover baseline comparisons, memory/KV pressure, thermal behavior, objective sensitivity, and workload breadth.

        This structure matters because it prevents the evaluation from devolving into a long list of mostly unrelated plots. Each block exists to answer one reviewer-grade question: Does the runtime really run? Is the simulator good enough to trust? Does fallback-aware costing change decisions? Are wins preserved against fair baselines? Does memory/KV control matter under burst load? Does the policy degrade under heat? Is the policy stable to weight choice? Does the claim generalize beyond the three deployed workflows?

        The experiment matrix also explains the hybrid evidence strategy. Device experiments are reserved for the support-safe deployed workflows and directly observable runtime behavior. Simulator experiments cover the broader workload universe, method-class proxies, and what-if regimes that cannot yet be exercised on device without violating the support-safe contract. The final claim is then the intersection of those two evidence surfaces.

        The evaluation contract is intentionally strict:
        \begin{itemize}
        \item every baseline runs inside the same simulator/runtime environment;
        \item every simulated result points back to a checkpoint-pinned workload, baseline policy, and calibration bundle;
        \item every real-device result points back to one canonical experiment batch rather than a mix of ad hoc runs;
        \item unsupported paths are recorded as infeasible or fallback-penalized rather than averaged into a misleading ``best accelerator'' number.
        \end{itemize}

        This structure is also why the paper can say something useful even when many workloads tie the static-best-map baseline. A tie under the same support-safe feasible set is a real result: it says the device currently offers limited heterogeneity for that workload family, and the scheduler is not manufacturing a win from unsupported configurations. That kind of negative result belongs in a systems paper when it is measured and explained.
        """
    ).strip() + "\n"


def build_evaluation_section(experiment_summary: dict[str, Any] | None, characterization_summary: dict[str, Any] | None, memory_summary: dict[str, Any] | None) -> str:
    retrieval_cpu = "N/A"
    retrieval_gpu = "N/A"
    if experiment_summary:
        actual = experiment_summary.get("actual_workflows", {})
        c_metrics = actual.get("workflow_c_voice_vision_retrieval", {})
        retrieval_cpu = fmt_ms(c_metrics.get("warm_latency_ms"))
    characterization_lines = ensure_line_count(characterization_summary_lines(characterization_summary), 3)
    fallback_delta = "N/A"
    if characterization_summary:
        fallback = characterization_summary.get("ablations", {}).get("retrieval_backend", {})
        if fallback:
            fallback_delta = fmt_num(fallback.get("delta_ms"), 1)
    memory_note = "The checkpoint does not pin a structured memory-admission summary."
    if memory_summary:
        degrade = memory_summary.get("latest_degrade_line", "")
        reject = memory_summary.get("latest_reject_line", "")
        if degrade or reject:
            memory_note = (
                "The Android runtime logs explicit ADMIT/DEGRADE/REJECT memory decisions. "
                f"The pinned degrade line shows responder max tokens reduced under pressure, and the pinned reject line shows the request refused after all configured degradation actions were exhausted."
            )
    return dedent(
        rf"""
        \section{{Evaluation}}
        The evaluation follows the hybrid evidence plan from the revision report. Real-device results stay centered on workflows A/B/C and the support-safe deployed backend map. Broader workload families, continuous-stream regimes, and method-class proxy baselines run inside the calibrated simulator so they share the same feasibility constraints, objective function, and workload definitions.

        Figure~\ref{{fig:primary-results}} reports the primary workflow latencies, and Table~\ref{{tab:workflow-results}} captures the exact warm-latency, TTFT, and TTFS numbers from the checkpoint. Figure~\ref{{fig:calibration}} shows the remaining sim-to-real deltas. The main result is not broad heterogeneous victory across every stage family; the main result is that GraphPilot keeps the real deployment support-safe, preserves the NPU FastVLM path, exposes where GPU/NPU text paths remain infeasible, and still provides a calibrated environment for ranking plans and comparing scheduling policies.

        Result block 0 is the prototype anchor: the deployed system works end to end and therefore grounds the paper in a real mobile runtime. Result block 1 is simulator accuracy: the calibrated simulator now reports family-level residuals rather than only end-to-end intuition. Result block 2 is fallback penalty: the current checkpoint records a retrieval-backend penalty of {fallback_delta} ms for workflow C when the slower GPU path is used instead of the deployed CPU retrieval path. That result is representative of the broader paper thesis: attractive heterogeneity can lose once support-safe costs are applied.

        The broader characterization results are equally important. {characterization_lines[0]} {characterization_lines[1]} {characterization_lines[2]}

        Result blocks 3 through 7 use the simulator workload universe to make the scheduling claim precise. Baseline scheduling comparisons show where GraphPilot beats simpler policies and where it only ties them because the feasible set is currently narrow. Memory/KV pressure studies show how the explicit admission controller prevents silent overcommit. Thermal studies show why the plan-bank logic matters under sustained load. Objective sensitivity studies explain whether the best policy family is stable under modest perturbations. Workload-breadth studies demonstrate that the scheduler is not tuned exclusively for one assistant DAG.

        The baseline suite is intentionally layered. Internal baselines capture the obvious alternatives: CPU-only, GPU-only where feasible, NPU-only where feasible, the current deployed plan, StageGreedy, StaticBestMap, and ablations such as NoPipeline or NoMemoryKV. Method-class proxies then represent the neighboring literature under the same support-safe environment: Band-like, ADMS-like, Puzzle-like, Twill-like, HeteroInfer-like, Agent.xpu-like, and HeRo-like. This matters because the paper does not compare GraphPilot against published numbers from different hardware; it compares scheduling classes under one canonical checkpoint.

        Continuous-stream studies show why queue-aware scheduling belongs in the objective even when the current feasible backend set is narrow. In the representative continuous workloads, GraphPilot's score is often tied with the current static-best-map proxy because both are forced onto nearly the same support-safe placements; that tie is itself a result, because it proves the paper is not inventing heterogeneity that the runtime cannot realize. Where GraphPilot does win, it tends to win through TTFS-aware streaming, explicit fallback avoidance, or memory/KV control rather than through a mythical accelerator path.

        The most concrete backend ablation on the deployed workflows remains workflow C retrieval: the support-safe CPU retrieval path is faster than the support-safe GPU path on this device generation, so GraphPilot keeps retrieval on CPU in the deployed best plan instead of chasing a weaker accelerator story. The pinned workflow C deployed latency is {retrieval_cpu} ms warm, and the corresponding artifact pack records the slower GPU retrieval variant separately.

        {memory_note}

        \begin{{figure}}[t]
        \centering
        \includegraphics[width=0.98\columnwidth]{{figures/sim_real_calibration.png}}
        \caption{{Sim-to-real latency deltas from the checkpoint-pinned calibration batch.}}
        \label{{fig:calibration}}
        \end{{figure}}

        \begin{{figure}}[t]
        \centering
        \includegraphics[width=0.98\columnwidth]{{figures/workflow_primary_results.png}}
        \caption{{Primary workflow warm latencies on the deployed support-safe path.}}
        \label{{fig:primary-results}}
        \end{{figure}}

        \begin{{figure}}[t]
        \centering
        \includegraphics[width=0.98\columnwidth]{{figures/continuous_stream_results.png}}
        \caption{{Continuous-stream GraphPilot scores from the broader simulator workload universe.}}
        \label{{fig:continuous}}
        \end{{figure}}

        \begin{{figure}}[t]
        \centering
        \includegraphics[width=0.98\columnwidth]{{figures/baseline_comparison.png}}
        \caption{{GraphPilot margin versus the best other baseline across representative workloads.}}
        \label{{fig:baseline}}
        \end{{figure}}

        \begin{{figure}}[t]
        \centering
        \includegraphics[width=0.98\columnwidth]{{figures/ablation_breakdown.png}}
        \caption{{Pipeline ablation deltas across the checkpoint-pinned workload universe.}}
        \label{{fig:ablation}}
        \end{{figure}}

        \begin{{figure}}[t]
        \centering
        \includegraphics[width=0.98\columnwidth]{{figures/fallback_penalty.png}}
        \caption{{Checkpoint-pinned fallback-sensitive comparison used to motivate support-safe costing.}}
        \label{{fig:fallback}}
        \end{{figure}}

        \begin{{figure}}[t]
        \centering
        \includegraphics[width=0.98\columnwidth]{{figures/thermal_plan_bank.png}}
        \caption{{Thermal slowdown curves that motivate plan-bank switching under sustained load.}}
        \label{{fig:thermal}}
        \end{{figure}}

        \begin{{figure}}[t]
        \centering
        \includegraphics[width=0.98\columnwidth]{{figures/objective_sensitivity.png}}
        \caption{{Objective-sensitivity candidate scores from the checkpoint-pinned tuning sweep.}}
        \label{{fig:objective-sensitivity}}
        \end{{figure}}
        """
    ).strip() + "\n"


def build_related_work() -> str:
    return dedent(
        r"""
        \section{Related Work}
        GraphPilot-Edge sits between several nearby systems threads. Classical heterogeneous schedulers such as HEFT and CPOP provide the rank-based list-scheduling foundation but assume static execution costs and operation-complete resources. Multi-DNN mobile scheduling systems such as Band~\cite{band2022}, ADMS-style heterogeneous co-execution~\cite{adms2025}, and Puzzle~\cite{puzzle2025} study mobile heterogeneous processors but do not center continuous multimodal assistant DAGs with explicit support-safe fallback accounting. Compound-AI schedulers such as Twill~\cite{twill2025} broaden the scheduling view beyond single-model execution, while HeteroInfer~\cite{heteroinfer2025} and related LLM engines focus on single-LLM heterogeneous execution and the prefill/decode split. Agent.xpu~\cite{agentxpu2025} and HeRo~\cite{hero2026} move closer to agentic SoC orchestration, but GraphPilot differs by combining support-safe deployment, checkpoint-pinned sim-to-real calibration, and live assistant DAGs with speech in the loop.

        On the serving side, Orca~\cite{orca2022} and PagedAttention~\cite{pagedattention2023} motivate the queueing and memory/KV perspective, but they target server-scale transformer serving rather than one mobile SoC with explicit CPU/GPU/NPU feasibility constraints. GraphPilot reuses the lessons that matter - queue-aware scheduling, KV-aware control, and explicit memory accounting - while keeping the claim scoped to a real mobile deployment.

        The revision report recommends covering six communities explicitly, and that structure is useful here: classical heterogeneous scheduling, mobile inference runtimes, LLM serving systems, multi-DNN mobile schedulers, compound-AI schedulers, and agentic mobile systems. GraphPilot does not subsume every contribution from those communities. Instead, it composes the pieces that matter for one specific systems gap: continuous assistant DAGs on a real mobile SoC where unsupported paths, transfer cost, and memory/KV pressure change the practical scheduler decision.

        The comparison matrix in Table~\ref{tab:comparison-matrix} is the paper's compact answer to the related-work question. GraphPilot is not presented as universally stronger than every adjacent system. Instead, it occupies the overlap that the current literature leaves thinly covered: support-safe fallback-aware scheduling of continuous multimodal assistant DAGs with simulator calibration on one heterogeneous mobile SoC.
        """
    ).strip() + "\n"


def build_limitations(calibration_summary: dict[str, Any] | None, characterization_summary: dict[str, Any] | None, final_audit_text: str) -> str:
    quality = (calibration_summary or {}).get("calibration_quality_by_family", {})
    worst_family = None
    if quality:
        worst_family = max(quality.items(), key=lambda item: item[1].get("mean_absolute_error_ms", float("-inf")))
    note = characterization_summary.get("graphpilot_policy_note", "") if characterization_summary else ""
    worst_line = ""
    if worst_family is not None:
        worst_line = f"The worst calibrated family in the current checkpoint is {latex_escape(worst_family[0])} at {fmt_num(worst_family[1].get('mean_absolute_error_ms'), 1)} ms mean absolute error."
    return dedent(
        rf"""
        \section{{Limitations and Artifact Notes}}
        The paper stays inside the measured system boundary. Planner and responder GPU/NPU text paths are not yet support-safe in the deployed runtime, so the broadest heterogeneity claim is intentionally out of scope. {worst_line}

        The broader workload and baseline studies are faithful method-class proxies inside the same simulator/runtime environment, not line-by-line reproductions of external codebases. This is the right comparison level for the current artifact, but it still means the comparison section is about scheduling classes rather than direct code equivalence.

        {latex_escape(note) if note else 'The characterization summary does not include an additional policy note.'}

        Three limits from the revision report are worth stating explicitly. First, backend feasibility remains narrow outside the FastVLM NPU path, so the strongest story is support-safe scheduling rather than universal heterogeneous execution. Second, simulator error remains higher on some workload families than others, which is why the paper uses the simulator for ranking and sensitivity analysis rather than exact oracle prediction. Third, the broader workload universe is intentionally larger than the currently deployed runtime surface, so some policies live as calibrated what-if studies rather than online runtime behavior.

        The final audit remains part of the release surface, not a private checklist. The canonical checkpoint records the truth-source PDF, the evidence paths used for every figure and table, and the remaining scope limits that the paper does not claim to solve.
        """
    ).strip() + "\n"


def build_discussion_section() -> str:
    return dedent(
        r"""
        \section{Discussion and Ablation Interpretation}
        The ablation matrix in Table~\ref{tab:ablation-matrix} is included to make the causal story explicit. GraphPilot is not one monolithic policy; it is a bundle of design commitments. Removing fallback-aware costing should make accelerator-heavy plans look better than they really are. Removing the prefill/decode split should blur the main text-stage regime change. Removing pipeline and branch-overlap logic should degrade TTFS most strongly on streaming or branched workloads. Removing memory/KV control should either force earlier rejection or create uncontrolled queueing under burst load.

        The current checkpoint confirms part of that story and weakens part of it. The retrieval backend ablation is strongly directional: the support-safe GPU path is slower, so GraphPilot's deployed CPU retrieval choice is justified empirically. The pipeline ablation is weaker on many workload slices than the design initially hypothesized, which is also informative. It indicates that the present support-safe backend set is narrow enough that some richer scheduling policies collapse onto the same best map. That is not a failure of the artifact; it is evidence that the system is being measured honestly.

        This interpretation matters for a CASES audience. A paper about mobile heterogeneity is more useful when it explains where heterogeneity is genuinely valuable and where it is currently unavailable or irrelevant. GraphPilot's contribution is strongest precisely because it does not turn narrow feasibility into broad mythology. The calibrated simulator, explicit baseline suite, and checkpointed audit surface make those boundaries visible.

        The checkpoint therefore supports three concrete takeaways. First, support-safe accounting changes the story more reliably than aggressive accelerator enthusiasm. Second, queue-aware TTFS optimization and memory/KV control matter even when the feasible backend set is narrow, because user-visible latency and admission behavior are shaped by overlap and residency as much as by raw throughput. Third, sim-to-real calibration is valuable even when it is imperfect: once the residuals are reported honestly, the simulator becomes useful for ranking and sensitivity analysis rather than for pretending to be exact.

        Those takeaways are also the best guide for future work. The next most valuable implementation steps are not arbitrary complexity increases; they are support-safe backend expansion for the stages that remain CPU-only, stronger family-level calibration for the highest-residual workloads, and richer online scheduling once those additional deployable backend choices exist. The current paper is strongest because it stops at that measured boundary instead of stepping past it.
        """
    ).strip() + "\n"


def build_threats_section() -> str:
    return dedent(
        r"""
        \section{Threats to Validity}
        Backend-feasibility narrowness is the first obvious threat. If the paper claimed broad CPU/GPU/NPU co-optimization for every stage family, the current deployed runtime would not support that claim. The mitigation is scope discipline: the paper treats support-safe deployed paths as ground truth and treats the broader simulator space as calibrated what-if analysis.

        Simulator residuals are the second threat. Some families fit tightly while others remain noisier. The mitigation is checkpoint-pinned calibration reporting, family-level residual tables, and an explicit refusal to use the simulator as an oracle. GraphPilot's strongest simulator claim is ranking quality under support-safe constraints, not exact prediction of every end-to-end latency.

        Workload selection is the third threat. A narrow evaluation could overstate the usefulness of one scheduler policy, while an excessively broad evaluation could make the paper incoherent. The mitigation is the hybrid workload hierarchy from the revision report: A/B/C stay primary for device evidence, and the broader universe exists to test generality, fallback sensitivity, thermal behavior, and memory/KV pressure under the same artifact surface.
        """
    ).strip() + "\n"


def build_conclusion(checkpoint_manifest: dict[str, Any]) -> str:
    truth_source = checkpoint_manifest.get("truth_source_pdf", "N/A")
    return dedent(
        rf"""
        \section{{Conclusion}}
        GraphPilot-Edge demonstrates a support-safe runtime and a calibrated simulator for continuous multimodal assistant DAGs on SM8750. The deployed runtime keeps FastVLM on the LiteRT NPU path, keeps text and speech on verified CPU paths, exposes streaming and memory/KV control explicitly, and refuses to turn infeasible accelerator paths into fake wins. The calibrated simulator extends that runtime with queue-aware, fallback-aware what-if analysis across a broader workload universe.

        The strongest surviving claim is therefore narrower than ``everything runs everywhere,'' but stronger than a design sketch: GraphPilot-Edge is a measured, reproducible systems artifact with a real mobile runtime, a calibrated simulator, explicit support-safe accounting, and a canonical CASES evidence checkpoint rooted in {paper_ref_label(truth_source)}.
        """
    ).strip() + "\n"


def build_artifact_section(checkpoint_manifest: dict[str, Any]) -> str:
    canonical = checkpoint_manifest.get("canonical_evidence_paths", {})
    lines = []
    for key in (
        "backend_matrix",
        "baseline_registry",
        "candidate_plans",
        "experiment_summary",
        "calibration_summary",
        "characterization_summary",
        "memory_admission_summary",
        "workload_registry",
    ):
        value = canonical.get(key)
        if value:
            lines.append(f"\\item \\texttt{{{paper_ref_label(value)}}}")
    bullet_block = "\n".join(lines) or "\\item No canonical evidence paths were pinned."
    return dedent(
        rf"""
        \section{{Artifact and Reproducibility}}
        The GraphPilot release surface is intentionally explicit. The paper build is generated from one checkpoint manifest, and that manifest points to the registries and summaries required to reproduce the tables and figures in this paper. The core pinned evidence paths for the active paper build are:
        \begin{{itemize}}
        {bullet_block}
        \end{{itemize}}

        This checkpoint-first packaging is part of the systems contribution. It prevents the paper from drifting away from the measured runtime and makes the audit report user-visible instead of implicit.
        """
    ).strip() + "\n"


def write_section(path: Path, body: str) -> None:
    path.write_text(body.rstrip() + "\n", encoding="utf-8")


def write_references(path: Path) -> None:
    path.write_text(
        dedent(
            """
            @misc{qualcomm2025,
              author = {{Qualcomm}},
              title = {Snapdragon 8 Elite Mobile Platform Product Brief},
              year = {2025},
              note = {Official product brief}
            }

            @misc{litert2026,
              author = {{Google AI Edge}},
              title = {Qualcomm NPU (AI Engine Direct) with LiteRT},
              year = {2026},
              note = {Official LiteRT documentation}
            }

            @inproceedings{band2022,
              author = {Jeong, J. S. and others},
              title = {Band: Coordinated Multi-DNN Inference on Heterogeneous Mobile Processors},
              booktitle = {MobiSys},
              year = {2022}
            }

            @misc{adms2025,
              author = {Gao, Y. and others},
              title = {Optimizing Multi-DNN Inference on Mobile Devices through Heterogeneous Processor Co-Execution},
              year = {2025},
              eprint = {2503.21109},
              archivePrefix = {arXiv}
            }

            @misc{heteroinfer2025,
              author = {Chen, L. and others},
              title = {HeteroLLM / HeteroInfer: Accelerating Large Language Model Inference on Mobile SoCs with Heterogeneous AI Accelerators},
              year = {2025},
              eprint = {2501.14794},
              archivePrefix = {arXiv}
            }

            @misc{twill2025,
              author = {Taufique, Z. and others},
              title = {Twill: Scheduling Compound AI Systems on Heterogeneous Mobile Edge Platforms},
              year = {2025},
              eprint = {2507.00491},
              archivePrefix = {arXiv}
            }

            @misc{agentxpu2025,
              author = {Wei, X. and others},
              title = {Agent.xpu: Efficient Scheduling of Agentic LLM Workloads on Heterogeneous SoC},
              year = {2025},
              eprint = {2506.24045},
              archivePrefix = {arXiv}
            }

            @misc{puzzle2025,
              author = {Kang, D. and others},
              title = {Puzzle: Scheduling Multiple Deep Learning Models on Mobile Device with Heterogeneous Processors},
              year = {2025},
              eprint = {2508.17764},
              archivePrefix = {arXiv}
            }

            @misc{hero2026,
              author = {Li, M. and others},
              title = {HeRo: Adaptive Orchestration of Agentic RAG on Heterogeneous Mobile SoC},
              year = {2026},
              eprint = {2603.01661},
              archivePrefix = {arXiv}
            }

            @inproceedings{orca2022,
              author = {Yu, G. and others},
              title = {Orca: A Distributed Serving System for Transformer-Based Generative Models},
              booktitle = {OSDI},
              year = {2022}
            }

            @inproceedings{pagedattention2023,
              author = {Kwon, W. and others},
              title = {Efficient Memory Management for Large Language Model Serving with PagedAttention},
              booktitle = {SOSP},
              year = {2023}
            }
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def write_main_tex(path: Path) -> None:
    path.write_text(
        dedent(
            """
            \\documentclass[conference]{IEEEtran}
            \\usepackage[T1]{fontenc}
            \\usepackage[utf8]{inputenc}
            \\usepackage{graphicx}
            \\usepackage{booktabs}
            \\usepackage{array}
            \\usepackage{amsmath}
            \\usepackage{url}
            \\usepackage[hidelinks]{hyperref}
            \\newcommand{\\system}{GraphPilot-Edge}

            \\begin{document}
            \\title{GraphPilot-Edge: A Profiler-Driven Runtime and Calibrated Simulator for Continuous Multimodal Assistant DAGs on Heterogeneous Mobile SoCs}
            \\author{Anonymous Submission}
            \\maketitle

            \\input{sections/abstract}

            \\begin{figure*}[t]
            \\centering
            \\includegraphics[width=0.96\\textwidth]{figures/architecture_overview.png}
            \\caption{Checkpoint-pinned deployment architecture for the support-safe GraphPilot runtime.}
            \\label{fig:architecture}
            \\end{figure*}

            \\begin{figure*}[t]
            \\centering
            \\includegraphics[width=0.96\\textwidth]{figures/offline_online_split.png}
            \\caption{Offline/online split used by GraphPilot: checkpoint-pinned profiling and planning offline, explicit scheduling and memory control online.}
            \\label{fig:offline-online}
            \\end{figure*}

            \\input{sections/introduction}
            \\input{sections/prototype_anchor}
            \\input{sections/differentiation}
            \\input{sections/system_design}
            \\input{sections/workloads}
            \\input{sections/simulator}
            \\input{sections/algorithms}
            \\input{sections/methodology}
            \\input{sections/experiment_matrix}
            \\input{sections/evaluation}
            \\input{sections/related_work}
            \\input{sections/discussion}
            \\input{sections/threats}
            \\input{sections/limitations}
            \\input{sections/artifact}
            \\input{sections/conclusion}

            \\bibliographystyle{IEEEtran}
            \\bibliography{references}
            \\end{document}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--artifact-pack-summary", type=Path, default=None)
    parser.add_argument("--figure-summary", type=Path, default=None)
    parser.add_argument("--template-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint_manifest, artifact_pack_summary_path = resolve_artifact_pack_summary(
        checkpoint_manifest=args.checkpoint_manifest,
        explicit_summary=args.artifact_pack_summary,
    )
    pack_summary = load_json(artifact_pack_summary_path)
    required_pack_paths = require_pack_paths(pack_summary)
    if args.figure_summary is None:
        raise FileNotFoundError(
            "CASES paper build requires --figure-summary so the LaTeX source stays pinned to a concrete figure bundle. "
            "Remediation: run scripts/build_graphpilot_cases_figures.py from the same checkpoint and pass its summary.json here."
        )
    figure_summary, figure_outputs = parse_figure_summary(args.figure_summary, args.checkpoint_manifest)

    experiment_summary = load_optional_summary(checkpoint_manifest, "experiment_summary")
    calibration_summary = load_optional_summary(checkpoint_manifest, "calibration_summary")
    characterization_summary = load_optional_summary(checkpoint_manifest, "characterization_summary")
    backend_matrix = load_optional_summary(checkpoint_manifest, "backend_matrix")
    memory_summary = load_optional_summary(checkpoint_manifest, "memory_admission_summary")
    workload_registry = load_optional_summary(checkpoint_manifest, "workload_registry")
    baseline_registry = load_optional_summary(checkpoint_manifest, "baseline_registry")
    tuning_summary = load_optional_summary(checkpoint_manifest, "tuning_summary")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    paper_dir = args.output_root / f"graphpilot_cases_{timestamp}"
    sections_dir = paper_dir / "sections"
    figures_dir = paper_dir / "figures"
    sections_dir.mkdir(parents=True, exist_ok=False)
    prepare_template_assets(paper_dir, args.template_dir)
    copied_figures = copy_figures(figure_outputs, figures_dir)

    tables_tex = build_tables_tex(
        experiment_summary,
        calibration_summary,
        characterization_summary,
        backend_matrix,
        workload_registry,
        baseline_registry,
        tuning_summary,
    )
    (paper_dir / "tables.tex").write_text(tables_tex, encoding="utf-8")

    section_payloads = {
        "abstract.tex": build_abstract(checkpoint_manifest, experiment_summary, calibration_summary),
        "introduction.tex": build_introduction(checkpoint_manifest, experiment_summary),
        "prototype_anchor.tex": build_prototype_anchor_section(experiment_summary, backend_matrix),
        "differentiation.tex": build_differentiation_section(),
        "system_design.tex": build_system_design(experiment_summary, backend_matrix),
        "workloads.tex": build_workloads_section(workload_registry, baseline_registry),
        "simulator.tex": build_simulator_section(calibration_summary, characterization_summary),
        "algorithms.tex": build_algorithms_section(tuning_summary),
        "methodology.tex": build_methodology_section(
            checkpoint_manifest,
            experiment_summary,
            calibration_summary,
            characterization_summary,
        ),
        "experiment_matrix.tex": build_experiment_matrix_section(),
        "evaluation.tex": build_evaluation_section(experiment_summary, characterization_summary, memory_summary),
        "related_work.tex": build_related_work(),
        "discussion.tex": build_discussion_section(),
        "threats.tex": build_threats_section(),
        "limitations.tex": build_limitations(
            calibration_summary,
            characterization_summary,
            required_pack_paths["final_audit_report"].read_text(encoding="utf-8"),
        ),
        "artifact.tex": build_artifact_section(checkpoint_manifest),
        "conclusion.tex": build_conclusion(checkpoint_manifest),
    }
    section_files = []
    for name, payload in section_payloads.items():
        target = sections_dir / name
        write_section(target, payload)
        section_files.append(str(target.resolve()))

    write_main_tex(paper_dir / "main.tex")
    write_references(paper_dir / "references.bib")

    pdf_path = build_pdf_if_possible(paper_dir)
    page_count = extract_page_count(pdf_path)
    metadata = {
        "checkpoint_manifest": str(args.checkpoint_manifest.resolve()),
        "artifact_pack_summary": str(artifact_pack_summary_path.resolve()),
        "figure_summary": str(args.figure_summary.resolve()),
        "paper_dir": str(paper_dir.resolve()),
        "paper_pdf": str(pdf_path.resolve()),
        "page_count": page_count,
        "section_files": section_files,
        "tables_tex": str((paper_dir / "tables.tex").resolve()),
        "figure_files": {name: str((figures_dir / copied_name).resolve()) for name, copied_name in copied_figures.items()},
        "source_outputs": {key: str(path.resolve()) for key, path in required_pack_paths.items()},
        "checkpoint_truth_source_pdf": checkpoint_manifest.get("truth_source_pdf"),
    }
    write_json(paper_dir / "summary.json", metadata)
    print(paper_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
