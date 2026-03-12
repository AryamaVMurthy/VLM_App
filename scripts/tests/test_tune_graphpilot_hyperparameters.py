import importlib.util
import json
import pathlib
import subprocess
import tempfile
import unittest


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "tune_graphpilot_hyperparameters.py"
    spec = importlib.util.spec_from_file_location("tune_graphpilot_hyperparameters", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TuneGraphPilotHyperparametersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_main_writes_tuning_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            config_path = root / "optimization_defaults.json"
            candidate_plans = root / "candidate_plans.json"
            experiment_summary = root / "experiment_summary.json"
            output_root = root / "out"
            config_path.write_text(
                json.dumps(
                    {
                        "objective_weight_grid": {
                            "alpha": [1.0, 2.0],
                            "beta": [0.5],
                            "gamma": [0.0],
                            "delta": [0.0],
                            "eta": [0.0],
                            "zeta": [0.0],
                        },
                        "scheduler_weight_grid": {
                            "rank": [1.0],
                            "first_output": [2.0, 4.0],
                            "age": [1.0],
                            "copy": [1.0],
                            "memory": [1.0],
                            "thermal": [1.0],
                        },
                    }
                ),
                encoding="utf-8",
            )
            candidate_plans.write_text(
                json.dumps(
                    {
                        "plans": [
                            {
                                "plan_id": "cool:workflow_b:plan0",
                                "workflow_template": "workflow_b_voice_vision",
                                "state_id": "cool",
                                "predicted_cost": {
                                    "makespan_ms": 4000,
                                    "ttfs_ms": 2200,
                                    "energy_mj": 300.0,
                                    "memory_mb": 512,
                                    "copy_bytes": 2048,
                                    "quality_loss": 0.0,
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            experiment_summary.write_text(
                json.dumps(
                    {
                        "actual_workflows": {
                            "workflow_b_voice_vision": {
                                "warm_latency_ms": 4500,
                                "tts_first_audio_ms": 2500,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            rc = self.module.main(
                [
                    "--config",
                    str(config_path),
                    "--candidate-plans",
                    str(candidate_plans),
                    "--experiment-summary",
                    str(experiment_summary),
                    "--output-root",
                    str(output_root),
                ]
            )

            self.assertEqual(rc, 0)
            summaries = list(output_root.glob("graphpilot_hparam_tuning_*/summary.json"))
            self.assertEqual(len(summaries), 1)
            payload = json.loads(summaries[0].read_text(encoding="utf-8"))
            self.assertIn("best_objective_weights", payload)
            self.assertIn("best_scheduler_weights", payload)

    def test_script_runs_as_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            config_path = root / "optimization_defaults.json"
            candidate_plans = root / "candidate_plans.json"
            experiment_summary = root / "experiment_summary.json"
            output_root = root / "out"
            config_path.write_text(
                json.dumps(
                    {
                        "objective_weight_grid": {
                            "alpha": [1.0],
                            "beta": [0.5],
                            "gamma": [0.0],
                            "delta": [0.0],
                            "eta": [0.0],
                            "zeta": [0.0],
                        },
                        "scheduler_weight_grid": {
                            "rank": [1.0],
                            "first_output": [2.0],
                            "age": [1.0],
                            "copy": [1.0],
                            "memory": [1.0],
                            "thermal": [1.0],
                        },
                    }
                ),
                encoding="utf-8",
            )
            candidate_plans.write_text(
                json.dumps(
                    {
                        "plans": [
                            {
                                "plan_id": "cool:workflow_a:test",
                                "workflow_template": "workflow_a_voice_only",
                                "state_id": "cool",
                                "predicted_cost": {
                                    "makespan_ms": 4000,
                                    "ttfs_ms": 2000,
                                    "energy_mj": 0.0,
                                    "memory_mb": 0,
                                    "copy_bytes": 0,
                                    "quality_loss": 0.0,
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            experiment_summary.write_text(
                json.dumps(
                    {
                        "actual_workflows": {
                            "workflow_a_voice_only": {
                                "warm_latency_ms": 4500,
                                "tts_first_audio_ms": 2500,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            script_path = pathlib.Path(__file__).resolve().parents[1] / "tune_graphpilot_hyperparameters.py"
            result = subprocess.run(
                [
                    "python3",
                    str(script_path),
                    "--config",
                    str(config_path),
                    "--candidate-plans",
                    str(candidate_plans),
                    "--experiment-summary",
                    str(experiment_summary),
                    "--output-root",
                    str(output_root),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={"PATH": "/usr/bin:/bin"},
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            summaries = list(output_root.glob("graphpilot_hparam_tuning_*/summary.json"))
            self.assertEqual(len(summaries), 1)


if __name__ == "__main__":
    unittest.main()
