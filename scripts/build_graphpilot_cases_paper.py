#!/usr/bin/env python3
"""Build a checkpoint-pinned GraphPilot CASES paper source tree."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
ARTIFACT_ROOT = ROOT_DIR / "artifacts" / "graphpilot_edge"
DEFAULT_OUTPUT_ROOT = ARTIFACT_ROOT / "papers"
REQUIRED_PACK_KEYS = (
    "report",
    "paper_tables",
    "paper_draft",
    "final_audit_report",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_artifact_pack_summary(
    checkpoint_manifest: Path,
    explicit_summary: Path | None,
) -> tuple[dict[str, Any], Path]:
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


def copy_figures(pack_summary: dict[str, Any], figure_dir: Path) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for path_str in pack_summary.get("plot_outputs", []):
        source = Path(path_str)
        if not source.exists():
            raise FileNotFoundError(
                f"Artifact plot output not found at '{source}'. "
                "Remediation: regenerate the canonical artifact pack so the figure exists."
            )
        target = figure_dir / source.name
        shutil.copy2(source, target)
        copied.append(target.name)
    return copied


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


def write_text_section(path: Path, title: str, source: Path) -> None:
    body = source.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(
            [
                f"% Source: {source}",
                f"\\section{{{latex_escape(title)}}}",
                "\\begin{verbatim}",
                body,
                "\\end{verbatim}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_main_tex(path: Path, figure_names: list[str]) -> None:
    figure_block = ""
    if figure_names:
        first_figure = figure_names[0]
        figure_block = "\n".join(
            [
                "\\begin{figure}[t]",
                "\\centering",
                f"\\includegraphics[width=0.95\\columnwidth]{{figures/{latex_escape(first_figure)}}}",
                "\\caption{GraphPilot artifact-pack figure copied from the canonical checkpoint.}",
                "\\label{fig:artifact-pack-overview}",
                "\\end{figure}",
                "",
            ]
        )
    path.write_text(
        "\n".join(
            [
                "\\documentclass[conference]{IEEEtran}",
                "\\usepackage[T1]{fontenc}",
                "\\usepackage[utf8]{inputenc}",
                "\\usepackage{graphicx}",
                "\\usepackage{verbatim}",
                "\\begin{document}",
                "\\title{GraphPilot-Edge: CASES Paper Build from a Canonical Checkpoint}",
                "\\author{GraphPilot-Edge Artifact Builder}",
                "\\maketitle",
                "\\input{sections/abstract}",
                figure_block,
                "\\input{sections/introduction}",
                "\\input{sections/results}",
                "\\input{sections/final_audit}",
                "\\bibliographystyle{IEEEtran}",
                "\\bibliography{references}",
                "\\end{document}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_pdf_if_possible(paper_dir: Path) -> Path:
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        raise FileNotFoundError(
            "pdflatex is not installed. Remediation: install a TeX distribution with pdflatex and rerun "
            "scripts/build_graphpilot_cases_paper.py."
        )
    for _ in range(2):
        completed = subprocess.run(
            [
                pdflatex,
                "-interaction=nonstopmode",
                "-halt-on-error",
                "main.tex",
            ],
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--artifact-pack-summary", type=Path, default=None)
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

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    paper_dir = args.output_root / f"graphpilot_cases_{timestamp}"
    sections_dir = paper_dir / "sections"
    figures_dir = paper_dir / "figures"
    sections_dir.mkdir(parents=True, exist_ok=False)
    figure_names = copy_figures(pack_summary, figures_dir)

    write_text_section(sections_dir / "abstract.tex", "Abstract", required_pack_paths["paper_draft"])
    write_text_section(sections_dir / "introduction.tex", "Introduction and System Scope", required_pack_paths["report"])
    write_text_section(sections_dir / "results.tex", "Results and Tables", required_pack_paths["paper_tables"])
    write_text_section(sections_dir / "final_audit.tex", "Final Audit", required_pack_paths["final_audit_report"])
    write_main_tex(paper_dir / "main.tex", figure_names)
    (paper_dir / "references.bib").write_text("", encoding="utf-8")
    shutil.copy2(required_pack_paths["paper_tables"], paper_dir / "tables.tex")

    pdf_path = build_pdf_if_possible(paper_dir)
    metadata = {
        "checkpoint_manifest": str(args.checkpoint_manifest.resolve()),
        "artifact_pack_summary": str(artifact_pack_summary_path.resolve()),
        "paper_dir": str(paper_dir.resolve()),
        "paper_pdf": str(pdf_path.resolve()),
        "figure_names": figure_names,
        "source_outputs": {key: str(path.resolve()) for key, path in required_pack_paths.items()},
        "checkpoint_truth_source_pdf": checkpoint_manifest.get("truth_source_pdf"),
    }
    write_json(paper_dir / "summary.json", metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
