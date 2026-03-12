import importlib.util
import json
import pathlib
import tempfile
import unittest


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "build_graphpilot_artifact_pack.py"
    spec = importlib.util.spec_from_file_location("build_graphpilot_artifact_pack", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BuildGraphPilotArtifactPackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_build_repeat_stats_accepts_gpu_retrieval_workflow_c_variant(self) -> None:
        profiler_registry = {
            "entries": [
                {
                    "stage_id": "workflow_c_voice_vision_retrieval",
                    "backend": "mixed",
                    "variant": "graphpilot_vlm_npu_retrieval_gpu",
                    "recorded_at": "2026-03-11T00:00:00+00:00",
                    "metrics": {
                        "warm_latency_ms": 3000,
                        "ttft_ms": 700,
                        "tts_first_audio_ms": 900,
                    },
                },
                {
                    "stage_id": "workflow_c_voice_vision_retrieval",
                    "backend": "mixed",
                    "variant": "graphpilot_vlm_npu_retrieval_gpu",
                    "recorded_at": "2026-03-11T00:01:00+00:00",
                    "metrics": {
                        "warm_latency_ms": 3100,
                        "ttft_ms": 710,
                        "tts_first_audio_ms": 920,
                    },
                },
            ]
        }

        stats = self.module.build_repeat_stats(profiler_registry)

        self.assertIn("workflow_c_voice_vision_retrieval", stats)
        self.assertEqual(
            stats["workflow_c_voice_vision_retrieval"]["variant"],
            "graphpilot_vlm_npu_retrieval_gpu",
        )
        self.assertEqual(
            stats["workflow_c_voice_vision_retrieval"]["warm_latency_ms"]["count"],
            2.0,
        )

    def test_build_workflow_c_retrieval_ablation_detects_cpu_and_gpu_entries(self) -> None:
        profiler_registry = {
            "entries": [
                {
                    "stage_id": "workflow_c_voice_vision_retrieval",
                    "backend": "mixed",
                    "variant": "graphpilot_vlm_npu_retrieval_cpu",
                    "recorded_at": "2026-03-11T00:00:00+00:00",
                    "metrics": {"warm_latency_ms": 14744, "ttft_ms": 4749, "tts_first_audio_ms": 10022},
                },
                {
                    "stage_id": "workflow_c_voice_vision_retrieval",
                    "backend": "mixed",
                    "variant": "graphpilot_vlm_npu_retrieval_gpu",
                    "recorded_at": "2026-03-11T00:01:00+00:00",
                    "metrics": {"warm_latency_ms": 21627, "ttft_ms": 6400, "tts_first_audio_ms": 12000},
                },
            ]
        }

        ablation = self.module.build_workflow_c_retrieval_ablation(profiler_registry)

        self.assertIsNotNone(ablation)
        self.assertEqual(ablation["cpu"]["warm_latency_ms"], 14744)
        self.assertEqual(ablation["gpu"]["warm_latency_ms"], 21627)

    def test_write_simple_bar_svg_creates_svg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "plot.svg"
            self.module.write_simple_bar_svg(
                out,
                title="Latency",
                rows=[("A", 10.0), ("B", 20.0)],
                x_label="ms",
            )
            text = out.read_text(encoding="utf-8")
            self.assertIn("<svg", text)
            self.assertIn("Latency", text)

    def test_main_generates_pack_and_updates_plot_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            backend = root / "backend.json"
            profiler = root / "profiler.json"
            plans = root / "plans.json"
            experiments = root / "experiments.json"
            plot_registry = root / "plots.json"
            state = root / "state.json"
            experiment_summary = root / "batch_summary.json"
            sustained_summary = root / "sustained_summary.json"
            calibration_summary = root / "calibration_summary.json"
            tuning_summary = root / "tuning_summary.json"
            output_root = root / "out"

            backend.write_text(
                json.dumps(
                    {
                        "stages": [
                            {
                                "stage_id": "asr.primary",
                                "backends": {
                                    "cpu": {"status": "feasible_smoke_pass"},
                                    "gpu": {"status": "infeasible_no_backend_adapter"},
                                    "npu": {"status": "infeasible_no_backend_adapter"},
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            profiler.write_text(json.dumps({"entries": []}), encoding="utf-8")
            plans.write_text(json.dumps({"plans": []}), encoding="utf-8")
            plans.write_text(
                json.dumps(
                    {
                        "plans": [
                            {
                                "workflow_template": "workflow_a_voice_only",
                                "state_id": "cool",
                                "plan_id": "cool:workflow_a_voice_only:test",
                                "predicted_cost": {
                                    "makespan_ms": 900.0,
                                    "p95_queue_delay_ms": 40.0,
                                    "deadline_miss_rate": 0.25,
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            experiments.write_text(
                json.dumps(
                    {
                        "experiments": [
                            {
                                "workflow_template": "graphpilot_phase7_sustained_load",
                                "metrics_path": str(sustained_summary),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            plot_registry.write_text(json.dumps({"plots": []}), encoding="utf-8")
            state.write_text(json.dumps({"current_phase": "Phase 7"}), encoding="utf-8")
            experiment_summary.write_text(
                json.dumps(
                    {
                        "experiment_id": "batch1",
                        "actual_workflows": {
                            "workflow_a_voice_only": {
                                "variant": "graphpilot_cpu_stack",
                                "warm_latency_ms": 1000,
                                "ttft_ms": None,
                                "tts_first_chunk_queued_ms": 250,
                                "tts_first_audio_ms": 300,
                            }
                        },
                        "candidate_workflows": {},
                        "comparisons": [],
                        "blocked_workflows": [
                            {
                                "workflow_id": "workflow_c_voice_vision_retrieval",
                                "reason": "retrieval_infeasible",
                                "detail": "missing artifact",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            sustained_summary.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "workflow_id": "workflow_a_voice_only",
                                "warm_latency_ms": 1000,
                                "thermal_after": {"skin_c": 33.0},
                            },
                            {
                                "workflow_id": "workflow_a_voice_only",
                                "warm_latency_ms": 1100,
                                "thermal_after": {"skin_c": 34.0},
                            },
                        ],
                        "workflow_summary": {
                            "workflow_a_voice_only": {
                                "count": 2,
                                "warm_latency_ms": {"drift": 100.0},
                                "ttft_ms": None,
                                "tts_first_audio_ms": {"drift": 20.0},
                                "thermal": {"skin_c": {"drift": 1.0}},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            calibration_summary.write_text(
                json.dumps(
                    {
                        "global_orchestration_overhead_ms": 321.0,
                        "thermal_scale_by_workflow": {
                            "workflow_a_voice_only": {"warm_latency_scale": 1.1}
                        },
                    }
                ),
                encoding="utf-8",
            )
            tuning_summary.write_text(
                json.dumps(
                    {
                        "best_objective_weights": {
                            "weights": {
                                "alpha": 2.0,
                                "beta": 1.0,
                                "gamma": 0.0,
                                "delta": 0.0,
                                "eta": 0.0,
                                "zeta": 0.0,
                            }
                        },
                        "best_scheduler_weights": {
                            "weights": {
                                "rank": 2.0,
                                "first_output": 6.0,
                                "age": 1.0,
                                "copy": 1.0,
                                "memory": 1.0,
                                "thermal": 1.0,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            rc = self.module.main(
                [
                    "--backend-matrix",
                    str(backend),
                    "--profiler-registry",
                    str(profiler),
                    "--candidate-plans",
                    str(plans),
                    "--experiment-registry",
                    str(experiments),
                    "--plot-registry",
                    str(plot_registry),
                    "--state-ledger",
                    str(state),
                    "--experiment-summary",
                    str(experiment_summary),
                    "--calibration-summary",
                    str(calibration_summary),
                    "--tuning-summary",
                    str(tuning_summary),
                    "--output-root",
                    str(output_root),
                ]
            )

            self.assertEqual(rc, 0)
            packs = list(output_root.glob("artifact_pack_*/summary.json"))
            self.assertEqual(len(packs), 1)
            payload = json.loads(packs[0].read_text(encoding="utf-8"))
            self.assertIsNotNone(payload["sustained_summary"])
            self.assertEqual(payload["calibration_summary"], str(calibration_summary.resolve()))
            self.assertEqual(payload["tuning_summary"], str(tuning_summary.resolve()))
            self.assertEqual(payload["calibration_summary_inline"]["global_orchestration_overhead_ms"], 321.0)
            self.assertEqual(payload["tuning_summary_inline"]["best_objective_weights"]["weights"]["alpha"], 2.0)
            self.assertTrue(pathlib.Path(payload["paper_tables"]).exists())
            self.assertTrue(pathlib.Path(payload["paper_draft"]).exists())
            report_text = pathlib.Path(payload["report"]).read_text(encoding="utf-8")
            self.assertIn("Calibration", report_text)
            self.assertIn("Hyperparameter tuning", report_text)
            self.assertIn("Stream scheduling", report_text)
            self.assertIn("p95_queue_delay_ms=40.0", report_text)
            self.assertIn("tts_first_chunk_queued_ms=250", report_text)
            draft_text = pathlib.Path(payload["paper_draft"]).read_text(encoding="utf-8")
            self.assertIn("GraphPilot-Edge", draft_text)
            self.assertIn("J(Pi)", draft_text)
            self.assertIn("Offline Brain", draft_text)
            self.assertIn("Online Brain", draft_text)
            self.assertIn("P95(T_queue)", draft_text)
            updated_plot_registry = json.loads(plot_registry.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(updated_plot_registry["plots"]), 5)


if __name__ == "__main__":
    unittest.main()
