import importlib.util
import json
import pathlib
import tempfile
import unittest


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "build_graphpilot_cases_paper.py"
    assert module_path.exists(), f"Missing CASES paper builder: {module_path}"
    spec = importlib.util.spec_from_file_location("build_graphpilot_cases_paper", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BuildGraphPilotCasesPaperTest(unittest.TestCase):
    def test_cases_paper_builder_requires_checkpoint_manifest(self) -> None:
        module = load_module()

        with self.assertRaises(SystemExit) as exc:
            module.main([])

        self.assertNotEqual(exc.exception.code, 0)

    def test_cases_paper_builder_fails_fast_without_required_pack_outputs(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            checkpoint_manifest = root / "checkpoint_summary.json"
            artifact_pack_summary = root / "artifact_pack_summary.json"
            artifact_pack_summary.write_text(
                json.dumps(
                    {
                        "report": str(root / "report.md"),
                        "paper_tables": str(root / "paper_tables.md"),
                        "paper_draft": str(root / "paper_draft.md"),
                        "final_audit_report": str(root / "final_audit_report.md"),
                    }
                ),
                encoding="utf-8",
            )
            checkpoint_manifest.write_text(
                json.dumps(
                    {
                        "canonical_evidence_paths": {
                            "artifact_pack_summary": str(artifact_pack_summary),
                        }
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(FileNotFoundError):
                module.main(
                    [
                        "--checkpoint-manifest",
                        str(checkpoint_manifest),
                        "--output-root",
                        str(root / "paper_out"),
                    ]
                )

    def test_cases_paper_builder_builds_pdf_from_template_dir(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            artifact_pack_summary = root / "artifact_pack_summary.json"
            checkpoint_manifest = root / "checkpoint_summary.json"
            figure_summary = root / "figure_summary.json"
            template_dir = root / "template"
            markdown_dir = root / "markdown"
            figures_dir = root / "figure_bundle"
            figures_dir.mkdir()
            markdown_sections = markdown_dir / "sections"
            markdown_sections.mkdir(parents=True)
            png_blob = (
                b"\x89PNG\r\n\x1a\n"
                b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
                b"\x00\x00\x00\x0cIDAT\x08\x99c```\x00\x00\x00\x04\x00\x01\xf6\x178U"
                b"\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            for name in (
                "workflow_primary_results.png",
                "sim_real_calibration.png",
                "sustained_detail.png",
                "sustained_overview.png",
                "architecture_overview.png",
                "offline_online_split.png",
                "workload_universe_coverage.png",
                "backend_affinity_matrix.png",
                "support_safe_feasibility.png",
                "calibration_family_mae.png",
                "backend_launch_overhead.png",
                "backend_contention_scale.png",
                "proxy_baseline_comparison.png",
                "memory_kv_overview.png",
                "knob_frontier_overview.png",
                "calibration_overview.png",
                "evaluation_overview.png",
                "sensitivity_overview.png",
                "continuous_stream_results.png",
                "baseline_comparison.png",
                "ablation_breakdown.png",
                "fallback_penalty.png",
                "thermal_plan_bank.png",
                "objective_sensitivity.png",
            ):
                (figures_dir / name).write_bytes(png_blob)
            (figures_dir / "tables.tex").write_text(
                "\\begin{tabular}{ll}\nA & B \\\\\n\\end{tabular}\n",
                encoding="utf-8",
            )
            (root / "report.md").write_text("# Report\n", encoding="utf-8")
            (root / "paper_tables.md").write_text("| A |\n| - |\n| 1 |\n", encoding="utf-8")
            (root / "paper_draft.md").write_text("# Draft\n", encoding="utf-8")
            (root / "final_audit_report.md").write_text("# Audit\n", encoding="utf-8")
            artifact_pack_summary.write_text(
                json.dumps(
                    {
                        "report": str(root / "report.md"),
                        "paper_tables": str(root / "paper_tables.md"),
                        "paper_draft": str(root / "paper_draft.md"),
                        "final_audit_report": str(root / "final_audit_report.md"),
                    }
                ),
                encoding="utf-8",
            )
            figure_summary.write_text(
                json.dumps(
                    {
                        "checkpoint_manifest": str(checkpoint_manifest),
                        "output_dir": str(figures_dir),
                        "outputs": [
                            str(figures_dir / "architecture_overview.png"),
                            str(figures_dir / "offline_online_split.png"),
                            str(figures_dir / "workload_universe_coverage.png"),
                            str(figures_dir / "evaluation_overview.png"),
                            str(figures_dir / "sensitivity_overview.png"),
                            str(figures_dir / "sim_real_calibration.png"),
                            str(figures_dir / "workflow_primary_results.png"),
                            str(figures_dir / "continuous_stream_results.png"),
                            str(figures_dir / "sustained_detail.png"),
                            str(figures_dir / "sustained_overview.png"),
                            str(figures_dir / "backend_affinity_matrix.png"),
                            str(figures_dir / "support_safe_feasibility.png"),
                            str(figures_dir / "calibration_family_mae.png"),
                            str(figures_dir / "backend_launch_overhead.png"),
                            str(figures_dir / "backend_contention_scale.png"),
                            str(figures_dir / "proxy_baseline_comparison.png"),
                            str(figures_dir / "memory_kv_overview.png"),
                            str(figures_dir / "knob_frontier_overview.png"),
                            str(figures_dir / "calibration_overview.png"),
                            str(figures_dir / "baseline_comparison.png"),
                            str(figures_dir / "ablation_breakdown.png"),
                            str(figures_dir / "fallback_penalty.png"),
                            str(figures_dir / "thermal_plan_bank.png"),
                            str(figures_dir / "objective_sensitivity.png"),
                            str(figures_dir / "tables.tex"),
                        ],
                    }
                ),
                encoding="utf-8",
            )
            checkpoint_manifest.write_text(
                json.dumps(
                    {
                        "truth_source_pdf": str(root / "truth.pdf"),
                        "canonical_evidence_paths": {
                            "artifact_pack_summary": str(artifact_pack_summary),
                        },
                    }
                ),
                encoding="utf-8",
            )
            template_dir.mkdir()
            (template_dir / "IEEEtran.cls").write_text(
                "\\NeedsTeXFormat{LaTeX2e}\n"
                "\\ProvidesClass{IEEEtran}[2026/03/12 test class]\n"
                "\\LoadClass{article}\n",
                encoding="utf-8",
            )
            (markdown_sections / "01_abstract.md").write_text(
                "GraphPilot-Edge keeps the paper source in markdown first.\n",
                encoding="utf-8",
            )
            (markdown_sections / "02_introduction.md").write_text(
                "# Introduction\n\n"
                "A concise markdown-first paper section.\n\n"
                "```latex\n"
                "\\begin{figure}[t]\n"
                "\\centering\n"
                "\\includegraphics[width=0.45\\textwidth]{figures/workflow_primary_results.png}\n"
                "\\caption{Primary workflow figure.}\n"
                "\\label{fig:test_primary}\n"
                "\\end{figure}\n"
                "```\n",
                encoding="utf-8",
            )
            (markdown_sections / "03_conclusion.md").write_text(
                "# Conclusion\n\n"
                "- calibrated simulator\n"
                "- support-safe runtime\n",
                encoding="utf-8",
            )

            rc = module.main(
                [
                    "--checkpoint-manifest",
                    str(checkpoint_manifest),
                    "--figure-summary",
                    str(figure_summary),
                    "--markdown-dir",
                    str(markdown_dir),
                    "--template-dir",
                    str(template_dir),
                    "--output-root",
                    str(root / "paper_out"),
                ]
            )

            self.assertEqual(rc, 0)
            summary_path = next((root / "paper_out").glob("graphpilot_cases_*/summary.json"))
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertTrue(pathlib.Path(payload["paper_pdf"]).exists())
            self.assertTrue(pathlib.Path(payload["paper_markdown"]).exists())
            self.assertEqual(payload["markdown_dir"], str(markdown_dir.resolve()))
            self.assertTrue(payload["markdown_section_files"])
            self.assertTrue(
                {
                    "01_abstract.tex",
                    "02_introduction.tex",
                    "03_conclusion.tex",
                }.issubset({pathlib.Path(path).name for path in payload["section_files"]})
            )
            main_tex = (summary_path.parent / "main.tex").read_text(encoding="utf-8")
            self.assertIn("\\input{sections/02_introduction}", main_tex)
            self.assertIn("\\usepackage{float}", main_tex)
            self.assertNotIn("\\usepackage{balance}", main_tex)
            self.assertNotIn("\\balance", main_tex)
            self.assertIn("\\FloatBarrier", main_tex)
            self.assertNotIn("\\clearpage", main_tex)
            intro_tex = (summary_path.parent / "sections" / "02_introduction.tex").read_text(encoding="utf-8")
            self.assertIn("\\includegraphics", intro_tex)
            self.assertEqual(payload["figure_summary"], str(figure_summary.resolve()))
            self.assertIsInstance(payload["page_count"], int)
            self.assertGreaterEqual(payload["page_count"], 1)

    def test_build_markdown_context_shortens_path_references(self) -> None:
        module = load_module()

        context = module.build_markdown_context(
            pathlib.Path("/tmp/checkpoints/graphpilot_checkpoint_test/summary.json"),
            {
                "truth_source_pdf": "/tmp/truth/Truth-docs/graphpilot_edge_revision_report.pdf",
            },
            {},
            {},
            {},
            {},
        )

        self.assertEqual(context["TRUTH_SOURCE_REF"], "Truth-docs/graphpilot_edge_revision_report.pdf")
        self.assertEqual(context["CHECKPOINT_MANIFEST_REF"], "graphpilot_checkpoint_test/summary.json")


if __name__ == "__main__":
    unittest.main()
