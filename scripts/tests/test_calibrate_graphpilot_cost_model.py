import importlib.util
import json
import pathlib
import tempfile
import unittest


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "calibrate_graphpilot_cost_model.py"
    spec = importlib.util.spec_from_file_location("calibrate_graphpilot_cost_model", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CalibrateGraphPilotCostModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_main_writes_calibration_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            experiment_summary = root / "experiment_summary.json"
            sustained_summary = root / "sustained_summary.json"
            output_root = root / "out"
            experiment_summary.write_text(
                json.dumps(
                    {
                        "actual_workflows": {
                            "workflow_a_voice_only": {
                                "warm_latency_ms": 1000,
                                "stage_backends": {
                                    "asr.primary": "cpu",
                                    "planner.primary": "cpu",
                                },
                            },
                            "workflow_b_voice_vision": {
                                "warm_latency_ms": 2000,
                                "stage_backends": {
                                    "asr.primary": "cpu",
                                    "vlm.fastvlm.primary": "npu",
                                },
                            },
                        },
                        "comparisons": [
                            {
                                "workflow_id": "workflow_a_voice_only",
                                "actual_warm_latency_ms": 1000,
                                "candidate_predicted_makespan_ms": 700,
                                "latency_delta_ms": 300,
                            },
                            {
                                "workflow_id": "workflow_b_voice_vision",
                                "actual_warm_latency_ms": 2000,
                                "candidate_predicted_makespan_ms": 1500,
                                "latency_delta_ms": 500,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            sustained_summary.write_text(
                json.dumps(
                    {
                        "workflow_summary": {
                            "workflow_a_voice_only": {
                                "warm_latency_ms": {"first": 1000.0, "last": 1100.0},
                                "thermal": {"cpu_c": {"first": 35.0, "last": 45.0}, "skin_c": {"first": 33.0, "last": 38.0}},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            rc = self.module.main(
                [
                    "--experiment-summary",
                    str(experiment_summary),
                    "--sustained-summary",
                    str(sustained_summary),
                    "--output-root",
                    str(output_root),
                ]
            )

            self.assertEqual(rc, 0)
            summaries = list(output_root.glob("graphpilot_cost_calibration_*/summary.json"))
            self.assertEqual(len(summaries), 1)
            payload = json.loads(summaries[0].read_text(encoding="utf-8"))
            self.assertIn("orchestration_overhead_ms", payload)
            self.assertIn("residual_bias_ms", payload)
            self.assertIn("thermal_scale_by_workflow", payload)
            self.assertIn("backend_thermal_factors_by_workflow", payload)
            self.assertIn("backend_utilizations_by_workflow", payload)
            self.assertEqual(payload["backend_thermal_factors_by_workflow"]["workflow_a_voice_only"]["cpu"], 1.1)
            self.assertEqual(payload["backend_utilizations_by_workflow"]["workflow_a_voice_only"]["cpu"], 1.0)

    def test_negative_delta_becomes_residual_bias_not_negative_overhead(self) -> None:
        payload = self.module.calibrate(
            experiment_summary={
                "actual_workflows": {
                    "workflow_a_voice_only": {
                        "stage_backends": {"asr.primary": "cpu"}
                    }
                },
                "comparisons": [
                    {
                        "workflow_id": "workflow_a_voice_only",
                        "actual_warm_latency_ms": 900,
                        "candidate_predicted_makespan_ms": 1000,
                        "latency_delta_ms": -100,
                    }
                ],
            },
            sustained_summary={
                "workflow_summary": {
                    "workflow_a_voice_only": {
                        "warm_latency_ms": {"first": 1000.0, "last": 1000.0},
                        "thermal": {"cpu_c": {"first": 35.0, "last": 35.0}, "skin_c": {"first": 33.0, "last": 33.0}},
                    }
                }
            },
        )

        self.assertEqual(payload["orchestration_overhead_ms"]["workflow_a_voice_only"], 0.0)
        self.assertEqual(payload["global_orchestration_overhead_ms"], 0.0)
        self.assertEqual(payload["residual_bias_ms"]["workflow_a_voice_only"], -100.0)

    def test_main_writes_family_and_stage_backend_surrogates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            experiment_summary = root / "experiment_summary.json"
            sustained_summary = root / "sustained_summary.json"
            profiler_registry = root / "profiler_registry.json"
            output_root = root / "out"
            experiment_summary.write_text(
                json.dumps(
                    {
                        "actual_workflows": {
                            "workflow_a_voice_only": {
                                "warm_latency_ms": 9805,
                                "stage_backends": {
                                    "asr.primary": "cpu",
                                    "planner.primary": "cpu",
                                    "responder.primary": "cpu",
                                    "tts.primary": "cpu",
                                },
                            }
                        },
                        "comparisons": [
                            {
                                "workflow_id": "workflow_a_voice_only",
                                "actual_warm_latency_ms": 9805,
                                "candidate_predicted_makespan_ms": 13630,
                                "latency_delta_ms": -3825,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            sustained_summary.write_text(
                json.dumps(
                    {
                        "workflow_summary": {
                            "workflow_a_voice_only": {
                                "warm_latency_ms": {"first": 9805.0, "last": 10120.0},
                                "thermal": {
                                    "cpu_c": {"first": 35.0, "last": 44.0, "drift": 9.0},
                                    "skin_c": {"first": 33.0, "last": 37.0, "drift": 4.0},
                                },
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            profiler_registry.write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "stage_id": "asr.primary",
                                "backend": "cpu",
                                "variant": "whisper_stt",
                                "metrics": {
                                    "warm_latency_ms": 320.0,
                                    "cold_latency_ms": 360.0,
                                    "prefill_latency_ms": 0.0,
                                    "decode_latency_ms": 320.0,
                                    "transfer_time_ms": 3.0,
                                    "peak_memory_bytes": 1048576,
                                },
                            },
                            {
                                "stage_id": "planner.primary",
                                "backend": "cpu",
                                "variant": "gemma3_1b_it",
                                "metrics": {
                                    "warm_latency_ms": 810.0,
                                    "cold_latency_ms": 920.0,
                                    "prefill_latency_ms": 530.0,
                                    "decode_latency_ms": 280.0,
                                    "transfer_time_ms": 2.0,
                                    "peak_memory_bytes": 2097152,
                                },
                            },
                            {
                                "stage_id": "workflow_a_voice_only",
                                "backend": "mixed",
                                "variant": "graphpilot_cpu_stack",
                                "metrics": {
                                    "stage_backends": {
                                        "asr.primary": "cpu",
                                        "planner.primary": "cpu",
                                    },
                                    "stage_timings_ms": {
                                        "asr.primary": 314.0,
                                        "planner.primary": 810.0,
                                    },
                                },
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            rc = self.module.main(
                [
                    "--experiment-summary",
                    str(experiment_summary),
                    "--sustained-summary",
                    str(sustained_summary),
                    "--profiler-registry",
                    str(profiler_registry),
                    "--output-root",
                    str(output_root),
                ]
            )

            self.assertEqual(rc, 0)
            summaries = list(output_root.glob("graphpilot_cost_calibration_*/summary.json"))
            self.assertEqual(len(summaries), 1)
            payload = json.loads(summaries[0].read_text(encoding="utf-8"))
            self.assertIn("family_backend_calibration", payload)
            self.assertIn("stage_backend_calibration", payload)
            self.assertIn("calibration_quality_by_family", payload)
            self.assertIn("launch_overhead_ms_by_backend", payload)
            self.assertIn("transfer_bias_ms_by_backend", payload)
            self.assertIn("contention_scale_by_backend", payload)
            self.assertIn("cpu", payload["family_backend_calibration"]["asr"])
            self.assertIn("cpu", payload["stage_backend_calibration"]["asr.primary"])

    def test_stage_backend_calibration_uses_latest_workflow_stage_timing(self) -> None:
        payload = self.module.calibrate(
            experiment_summary={
                "actual_workflows": {
                    "workflow_a_voice_only": {
                        "warm_latency_ms": 900,
                        "stage_backends": {"asr.primary": "cpu"},
                    }
                },
                "comparisons": [
                    {
                        "workflow_id": "workflow_a_voice_only",
                        "actual_warm_latency_ms": 900,
                        "candidate_predicted_makespan_ms": 850,
                        "latency_delta_ms": 50,
                    }
                ],
            },
            sustained_summary={
                "workflow_summary": {
                    "workflow_a_voice_only": {
                        "warm_latency_ms": {"first": 900.0, "last": 920.0},
                        "thermal": {
                            "cpu_c": {"first": 35.0, "last": 36.0},
                            "skin_c": {"first": 33.0, "last": 33.5},
                        },
                    }
                }
            },
            profiler_registry={
                "entries": [
                    {
                        "stage_id": "asr.primary",
                        "backend": "cpu",
                        "variant": "whisper_stt",
                        "metrics": {
                            "warm_latency_ms": 100.0,
                            "cold_latency_ms": 120.0,
                            "prefill_latency_ms": 0.0,
                            "decode_latency_ms": 100.0,
                            "transfer_time_ms": 1.0,
                            "peak_memory_bytes": 1024.0,
                        },
                    },
                    {
                        "stage_id": "workflow_a_voice_only",
                        "backend": "mixed",
                        "variant": "graphpilot_cpu_stack",
                        "recorded_at": "2026-03-12T10:00:00+00:00",
                        "metrics": {
                            "stage_backends": {"asr.primary": "cpu"},
                            "stage_timings_ms": {"asr.primary": 220.0},
                        },
                    },
                    {
                        "stage_id": "workflow_a_voice_only",
                        "backend": "mixed",
                        "variant": "graphpilot_cpu_stack",
                        "recorded_at": "2026-03-12T11:00:00+00:00",
                        "metrics": {
                            "stage_backends": {"asr.primary": "cpu"},
                            "stage_timings_ms": {"asr.primary": 150.0},
                        },
                    },
                ]
            },
        )

        self.assertEqual(
            payload["stage_backend_calibration"]["asr.primary"]["cpu"]["actual_stage_timing_ms"],
            150.0,
        )

    def test_thermal_surrogate_never_speeds_up_backends_below_one(self) -> None:
        payload = self.module.calibrate(
            experiment_summary={
                "actual_workflows": {
                    "workflow_b_voice_vision": {
                        "warm_latency_ms": 16000,
                        "stage_backends": {
                            "asr.primary": "cpu",
                            "vlm.fastvlm.primary": "npu",
                        },
                    }
                },
                "comparisons": [
                    {
                        "workflow_id": "workflow_b_voice_vision",
                        "actual_warm_latency_ms": 16000,
                        "candidate_predicted_makespan_ms": 15000,
                        "latency_delta_ms": 1000,
                    }
                ],
            },
            sustained_summary={
                "workflow_summary": {
                    "workflow_b_voice_vision": {
                        "warm_latency_ms": {"first": 20000.0, "last": 15000.0},
                        "thermal": {
                            "cpu_c": {"first": 42.0, "last": 39.0},
                            "skin_c": {"first": 37.0, "last": 35.0},
                        },
                    }
                }
            },
            profiler_registry={"entries": []},
        )

        self.assertEqual(payload["thermal_scale_by_workflow"]["workflow_b_voice_vision"]["warm_latency_scale"], 1.0)
        self.assertEqual(payload["backend_thermal_factors_by_workflow"]["workflow_b_voice_vision"]["cpu"], 1.0)
        self.assertEqual(payload["backend_thermal_factors_by_workflow"]["workflow_b_voice_vision"]["npu"], 1.0)


if __name__ == "__main__":
    unittest.main()
