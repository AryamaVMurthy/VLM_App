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
                        "calibration_quality_by_family": {
                            "asr": {"mean_absolute_error_ms": 320.0, "max_absolute_error_ms": 320.0, "sample_count": 1},
                            "vlm": {"mean_absolute_error_ms": 1800.0, "max_absolute_error_ms": 2400.0, "sample_count": 2},
                        },
                        "launch_overhead_ms_by_backend": {"cpu": 119.0, "gpu": 0.0, "npu": 0.0},
                        "contention_scale_by_backend": {"cpu": 1.8, "gpu": 1.1, "npu": 29.7},
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
                                    "baselines": [
                                        {"baseline_id": "current_deployed_plan", "status": "ok", "score_ms": 6400.0},
                                        {"baseline_id": "stage_greedy", "status": "ok", "score_ms": 6200.0},
                                    ],
                                    "graphpilot_score_ms": 6000.0,
                                    "workload_id": "compound.workflow_a.default",
                                    "graphpilot_margin_vs_best_other_ms": -120.0,
                                }
                            ],
                            "model_family": [
                                {
                                    "baselines": [
                                        {"baseline_id": "twill_like", "status": "ok", "score_ms": 14400.0},
                                        {"baseline_id": "agent_xpu_like", "status": "ok", "score_ms": 13500.0},
                                    ],
                                    "graphpilot_score_ms": 12000.0,
                                    "workload_id": "model.llm.long_long",
                                }
                            ],
                            "continuous_workloads": [
                                {
                                    "baselines": [
                                        {"baseline_id": "no_pipeline", "status": "ok", "score_ms": 101500.0},
                                        {"baseline_id": "no_memory_kv", "status": "ok", "score_ms": 104200.0},
                                    ],
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
                        "knob_frontiers": {
                            "llm": [
                                {"context_cap": 1024, "max_output_tokens": 64, "quality_proxy_loss": 0.3, "score_ms": 15000.0},
                                {"context_cap": 4096, "max_output_tokens": 192, "quality_proxy_loss": 0.0, "score_ms": 36000.0},
                            ]
                        },
                        "memory_kv_curves": {
                            "kv_curve": [
                                {"cached_tokens": 256.0, "kv_bytes": 37748736.0},
                                {"cached_tokens": 1024.0, "kv_bytes": 150994944.0},
                            ],
                            "llm_context_curve": [
                                {"context_tokens": 256.0, "aggregate_memory_bytes": 32636928.0},
                                {"context_tokens": 1024.0, "aggregate_memory_bytes": 89260032.0},
                            ],
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
                            "backend_matrix": str(root / "backend_matrix.json"),
                            "sustained_summary": str(root / "sustained_summary.json"),
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "backend_matrix.json").write_text(
                json.dumps(
                    {
                        "stages": [
                            {
                                "stage_id": "vlm.fastvlm.primary",
                                "backends": {
                                    "cpu": {"status": "feasible_smoke_pass"},
                                    "gpu": {"status": "infeasible_no_backend_adapter"},
                                    "npu": {"status": "known_working"},
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (root / "sustained_summary.json").write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "sample_index": 0,
                                "workflow_id": "workflow_a_voice_only",
                                "warm_latency_ms": 10091,
                                "thermal_after": {
                                    "max_cpu_c": 48.1,
                                    "max_npu_c": 45.2,
                                    "skin_c": 37.8,
                                },
                            },
                            {
                                "sample_index": 1,
                                "workflow_id": "workflow_b_voice_vision",
                                "warm_latency_ms": 15742,
                                "thermal_after": {
                                    "max_cpu_c": 49.5,
                                    "max_npu_c": 46.7,
                                    "skin_c": 38.2,
                                },
                            },
                            {
                                "sample_index": 2,
                                "workflow_id": "workflow_a_voice_only",
                                "warm_latency_ms": 10144,
                                "thermal_after": {
                                    "max_cpu_c": 49.3,
                                    "max_npu_c": 46.0,
                                    "skin_c": 38.0,
                                },
                            },
                        ]
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
                "backend_affinity_matrix.svg",
                "backend_affinity_matrix.png",
                "support_safe_feasibility.svg",
                "support_safe_feasibility.png",
                "calibration_family_mae.svg",
                "calibration_family_mae.png",
                "backend_launch_overhead.svg",
                "backend_launch_overhead.png",
                "backend_contention_scale.svg",
                "backend_contention_scale.png",
                "proxy_baseline_comparison.svg",
                "proxy_baseline_comparison.png",
                "memory_kv_overview.svg",
                "memory_kv_overview.png",
                "knob_frontier_overview.svg",
                "knob_frontier_overview.png",
                "calibration_overview.svg",
                "calibration_overview.png",
                "evaluation_overview.svg",
                "evaluation_overview.png",
                "sensitivity_overview.svg",
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
                "sustained_latency_drift.svg",
                "sustained_latency_drift.png",
                "sustained_thermal_drift.svg",
                "sustained_thermal_drift.png",
                "sustained_detail.svg",
                "sustained_detail.png",
                "sustained_overview.svg",
                "sustained_overview.png",
                "tables.tex",
            ):
                self.assertTrue((summary_path.parent / name).exists(), msg=name)
            self.assertEqual(payload["checkpoint_manifest"], str(checkpoint_manifest.resolve()))

    def test_baseline_delta_rows_use_explicit_baseline_scores(self) -> None:
        module = load_module()
        characterization_summary = {
            "baseline_comparisons": {
                "compound_workloads": [
                    {
                        "workload_id": "compound.workflow_a.default",
                        "graphpilot_score_ms": 6000.0,
                        "baselines": [
                            {"baseline_id": "current_deployed_plan", "status": "ok", "score_ms": 6400.0},
                            {"baseline_id": "stage_greedy", "status": "ok", "score_ms": 6200.0},
                        ],
                    }
                ]
            }
        }

        rows = module.build_baseline_delta_rows(characterization_summary)

        self.assertTrue(rows)
        self.assertIn(("compound.workflow_a.default :: current_deployed_plan", 400.0), rows)

    def test_proxy_baseline_rows_capture_method_class_deltas(self) -> None:
        module = load_module()
        characterization_summary = {
            "baseline_comparisons": {
                "model_family": [
                    {
                        "workload_id": "model.llm.long_long",
                        "graphpilot_score_ms": 12000.0,
                        "baselines": [
                            {"baseline_id": "twill_like", "status": "ok", "score_ms": 14400.0},
                            {"baseline_id": "agent_xpu_like", "status": "ok", "score_ms": 13500.0},
                        ],
                    }
                ]
            }
        }

        rows = module.build_proxy_baseline_rows(characterization_summary)

        self.assertTrue(rows)
        self.assertIn(("model.llm.long_long :: twill_like", 2400.0), rows)

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
