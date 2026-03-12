import importlib.util
import json
import pathlib
import tempfile
import unittest


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "build_graphpilot_cases_figures.py"
    assert module_path.exists(), f"Missing CASES figures builder: {module_path}"
    spec = importlib.util.spec_from_file_location("build_graphpilot_cases_figures", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BuildGraphPilotCasesFiguresTest(unittest.TestCase):
    def test_cases_figure_builder_requires_checkpoint_manifest(self) -> None:
        module = load_module()

        with self.assertRaises(SystemExit) as exc:
            module.main([])

        self.assertNotEqual(exc.exception.code, 0)

    def test_cases_figure_builder_emits_required_outputs(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pack_summary = root / "artifact_pack_summary.json"
            calibration_summary = root / "calibration_summary.json"
            characterization_summary = root / "characterization_summary.json"
            experiment_summary = root / "experiment_summary.json"
            workload_registry = root / "workload_registry.json"
            checkpoint_manifest = root / "checkpoint_summary.json"
            paper_tables = root / "paper_tables.md"
            dummy_plot = root / "workflow_latency.svg"
            dummy_plot.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
            paper_tables.write_text("| table |\n| --- |\n| row |\n", encoding="utf-8")

            pack_summary.write_text(
                json.dumps(
                    {
                        "paper_tables": str(paper_tables),
                        "plot_outputs": [str(dummy_plot)],
                    }
                ),
                encoding="utf-8",
            )
            calibration_summary.write_text(
                json.dumps(
                    {
                        "comparisons": [
                            {
                                "workflow_id": "workflow_a_voice_only",
                                "actual_warm_latency_ms": 10091,
                                "candidate_predicted_makespan_ms": 9548,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            characterization_summary.write_text(
                json.dumps(
                    {
                        "baseline_comparisons": {
                            "compound_workloads": [
                                {
                                    "workload_id": "compound.workflow_a.default",
                                    "graphpilot_margin_vs_best_other_ms": -120.0,
                                }
                            ],
                            "continuous_workloads": [
                                {
                                    "workload_id": "continuous.workflow_a.poisson",
                                    "graphpilot_score_ms": 93864.0,
                                    "best_other_score_ms": 100160.0,
                                }
                            ]
                        },
                        "ablations": {
                            "pipeline": [
                                {
                                    "workload_id": "primitive.gemm_heavy.medium",
                                    "graphpilot_score_ms": 3053.0,
                                    "ablation_score_ms": 3200.0,
                                    "delta_ms": 147.0,
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            experiment_summary.write_text(
                json.dumps(
                    {
                        "actual_workflows": {
                            "workflow_a_voice_only": {
                                "warm_latency_ms": 10091,
                                "ttft_ms": 241,
                                "tts_first_audio_ms": 3276,
                            }
                        },
                        "comparisons": [
                            {
                                "workflow_id": "workflow_a_voice_only",
                                "actual_warm_latency_ms": 10091,
                                "candidate_predicted_makespan_ms": 9548,
                                "latency_delta_ms": 543,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            workload_registry.write_text(
                json.dumps(
                    {
                        "workloads": [
                            {"workload_id": "compound.workflow_a.default", "category": "compound_workflow"},
                            {"workload_id": "continuous.workflow_a.poisson", "category": "continuous_stream"},
                            {"workload_id": "stress.workflow_c.fallback_penalty", "category": "stress_failure"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            checkpoint_manifest.write_text(
                json.dumps(
                    {
                        "canonical_evidence_paths": {
                            "artifact_pack_summary": str(pack_summary),
                            "calibration_summary": str(calibration_summary),
                            "characterization_summary": str(characterization_summary),
                            "experiment_summary": str(experiment_summary),
                            "workload_registry": str(workload_registry),
                        }
                    }
                ),
                encoding="utf-8",
            )

            rc = module.main(
                [
                    "--checkpoint-manifest",
                    str(checkpoint_manifest),
                    "--output-root",
                    str(root / "figures_out"),
                ]
            )

            self.assertEqual(rc, 0)
            summary_path = next((root / "figures_out").glob("graphpilot_cases_figures_*/summary.json"))
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            for name in (
                "architecture_overview.svg",
                "architecture_overview.png",
                "offline_online_split.svg",
                "offline_online_split.png",
                "workload_universe_coverage.svg",
                "workload_universe_coverage.png",
                "evaluation_overview.png",
                "sensitivity_overview.png",
                "sim_real_calibration.svg",
                "sim_real_calibration.png",
                "workflow_primary_results.svg",
                "workflow_primary_results.png",
                "continuous_stream_results.svg",
                "continuous_stream_results.png",
                "baseline_comparison.svg",
                "baseline_comparison.png",
                "ablation_breakdown.svg",
                "ablation_breakdown.png",
                "fallback_penalty.svg",
                "fallback_penalty.png",
                "thermal_plan_bank.svg",
                "thermal_plan_bank.png",
                "objective_sensitivity.svg",
                "objective_sensitivity.png",
                "tables.tex",
            ):
                self.assertTrue((summary_path.parent / name).exists(), msg=name)
            self.assertEqual(payload["checkpoint_manifest"], str(checkpoint_manifest.resolve()))

    def test_signed_bar_rendering_supports_negative_and_positive_values(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            svg_path = root / "signed_bars.svg"
            png_path = root / "signed_bars.png"

            rows = [
                ("negative_case", -120.0),
                ("zero_case", 0.0),
                ("positive_case", 240.0),
            ]

            module.write_bar_svg(svg_path, "Signed bar test", rows, "delta ms")
            module.write_bar_png(png_path, "Signed bar test", rows, "delta ms")

            self.assertTrue(svg_path.exists())
            self.assertTrue(png_path.exists())
            svg_text = svg_path.read_text(encoding="utf-8")
            self.assertIn("Signed bar test", svg_text)
            self.assertIn("-120.0", svg_text)
            self.assertIn("240.0", svg_text)


if __name__ == "__main__":
    unittest.main()
