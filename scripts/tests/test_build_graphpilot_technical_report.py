from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import build_graphpilot_technical_report as report_builder


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "artifacts" / "graphpilot_edge" / "checkpoints" / "graphpilot_checkpoint_20260312_195112" / "summary.json"
FIGURE_SUMMARY = ROOT / "artifacts" / "graphpilot_edge" / "papers" / "graphpilot_cases_figures_20260313_063751" / "summary.json"


class BuildGraphPilotTechnicalReportTests(unittest.TestCase):
    def test_build_report_tex_contains_expected_sections_and_appendices(self) -> None:
        checkpoint_summary = json.loads(CHECKPOINT.read_text())
        figure_summary = json.loads(FIGURE_SUMMARY.read_text())

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            tex = report_builder.build_report_tex(
                CHECKPOINT,
                checkpoint_summary,
                FIGURE_SUMMARY,
                figure_summary,
                output_dir,
            )

        self.assertIn(r"\section{Runtime Formulation and Implemented Algorithms}", tex)
        self.assertIn(r"\section{Hardware Simulator, Topology, and Calibration}", tex)
        self.assertIn(r"\section{Model-Graph Simulator and Workload Universe}", tex)
        self.assertIn(r"\section{Exhaustive Workload Registry}", tex)
        self.assertIn(r"\section{Full Baseline Result Registry}", tex)
        self.assertIn(r"\section{Memory Admission Event Surface}", tex)
        self.assertIn(r"\section{Reproduction Instructions}", tex)
        self.assertIn(r"\section{Core Code Excerpts}", tex)
        self.assertIn(r"\section{Canonical Evidence Paths}", tex)
        self.assertIn(report_builder.latex_escape("workflow_a_voice_only"), tex)
        self.assertIn(report_builder.latex_escape("band_like"), tex)
        self.assertIn(report_builder.latex_escape("graphpilot_edge/hardware_simulator.py"), tex)
        self.assertIn(r"\nocite{*}", tex)
        self.assertIn(
            report_builder.latex_escape("scripts/build_graphpilot_technical_report.py"),
            tex,
        )
        self.assertIn(r"\begin{lstlisting}[language=Python]", tex)
        self.assertNotIn("\x08", tex)
        self.assertNotIn("\r", tex)

    def test_memory_event_rows_parse_expected_decisions(self) -> None:
        summary_path = ROOT / "artifacts" / "graphpilot_edge" / "analysis" / "graphpilot_memory_admission_20260312_034310" / "summary.json"
        summary = json.loads(summary_path.read_text())
        rows = report_builder.memory_event_rows(summary)
        self.assertEqual(len(rows), 5)
        sources = {row[0] for row in rows}
        self.assertIn(report_builder.latex_escape("latest_admit_line"), sources)
        self.assertIn(report_builder.latex_escape("latest_queue_metrics_line"), sources)
        flat = "\n".join(" | ".join(row) for row in rows)
        self.assertIn("ADMIT", flat)
        self.assertIn("DEGRADE", flat)
        self.assertIn("REJECT", flat)


if __name__ == "__main__":
    unittest.main()
