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


if __name__ == "__main__":
    unittest.main()
