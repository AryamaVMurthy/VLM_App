import importlib.util
import json
import pathlib
import tempfile
import sys
import unittest


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "generate_graphpilot_candidate_plans.py"
    spec = importlib.util.spec_from_file_location(
        "generate_graphpilot_candidate_plans", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GenerateGraphPilotCandidatePlansTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_resolve_latest_analysis_summary_picks_newest_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            older = root / "graphpilot_cost_calibration_20260311_010000"
            newer = root / "graphpilot_cost_calibration_20260311_020000"
            older.mkdir()
            newer.mkdir()
            (older / "summary.json").write_text("{}", encoding="utf-8")
            (newer / "summary.json").write_text("{}", encoding="utf-8")

            resolved = self.module.resolve_latest_analysis_summary(root, "graphpilot_cost_calibration")

            self.assertEqual(resolved, newer / "summary.json")

    def test_select_workflows_skips_infeasible_retrieval_workflow(self):
        workflows = {
            "workflow_a_voice_only": object(),
            "workflow_c_voice_vision_retrieval": object(),
        }
        matrix = {
            "stages": [
                {
                    "stage_id": "retrieval.embedder.primary",
                    "backends": {
                        "cpu": {"status": "infeasible_missing_artifact"},
                        "gpu": {"status": "infeasible_missing_artifact"},
                        "npu": {"status": "infeasible_missing_artifact"},
                    },
                }
            ]
        }
        selected = self.module.select_workflow_ids(workflows, matrix)
        self.assertEqual(selected, ["workflow_a_voice_only"])

    def test_resolve_objective_weights_prefers_tuned_weights(self):
        weights = self.module.resolve_objective_weights(
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
                }
            }
        )
        self.assertEqual(weights.alpha, 2.0)
        self.assertEqual(weights.beta, 1.0)

    def test_resolve_simulation_calibration_includes_backend_surrogates(self):
        calibration = self.module.resolve_simulation_calibration(
            "workflow_a_voice_only",
            {
                "global_orchestration_overhead_ms": 25.0,
                "orchestration_overhead_ms": {
                    "workflow_a_voice_only": 7.0,
                },
                "thermal_scale_by_workflow": {
                    "workflow_a_voice_only": {"warm_latency_scale": 1.2}
                },
                "backend_thermal_factors_by_workflow": {
                    "workflow_a_voice_only": {"cpu": 1.2}
                },
                "backend_utilizations_by_workflow": {
                    "workflow_a_voice_only": {"cpu": 1.0}
                },
                "stage_backend_calibration": {
                    "asr.primary": {
                        "cpu": {"latency_scale": 0.9, "residual_bias_ms": -6.0}
                    }
                },
                "family_backend_calibration": {
                    "asr": {
                        "cpu": {"latency_scale": 0.92, "residual_bias_ms": -4.0}
                    }
                },
                "launch_overhead_ms_by_backend": {"cpu": 3.0},
                "transfer_bias_ms_by_backend": {"cpu": 1.5},
                "contention_scale_by_backend": {"cpu": 1.1},
            },
        )

        self.assertIsNotNone(calibration)
        self.assertEqual(calibration.orchestration_overhead_ms, 7.0)
        self.assertEqual(calibration.workflow_thermal_scale, 1.2)
        self.assertEqual(calibration.backend_thermal_factors["cpu"], 1.2)
        self.assertEqual(calibration.backend_utilizations["cpu"], 1.0)
        self.assertEqual(calibration.stage_latency_scales[("asr.primary", "cpu")], 0.9)
        self.assertEqual(calibration.family_latency_scales[("asr", "cpu")], 0.92)
        self.assertEqual(calibration.stage_residual_bias_ms[("asr.primary", "cpu")], -6.0)
        self.assertEqual(calibration.family_residual_bias_ms[("asr", "cpu")], -4.0)
        self.assertEqual(calibration.backend_launch_overheads_ms["cpu"], 3.0)
        self.assertEqual(calibration.backend_transfer_bias_ms["cpu"], 1.5)
        self.assertEqual(calibration.backend_contention_scales["cpu"], 1.1)

    def test_main_prefers_checkpoint_manifest_for_calibration_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            workflow_path = root / "workflow_templates.json"
            stage_path = root / "stage_catalog.json"
            plan_bank_path = root / "plan_bank_templates.json"
            backend_matrix_path = root / "backend_feasibility_matrix.json"
            profiler_registry_path = root / "profiler_registry.json"
            candidate_registry_path = root / "candidate_plan_registry.json"
            tuning_summary_path = root / "tuning_summary.json"
            optimization_config_path = root / "optimization_defaults.json"
            calibration_summary_path = root / "calibration_summary.json"
            checkpoint_manifest_path = root / "checkpoint_summary.json"

            workflow_path.write_text(
                json.dumps(
                    {
                        "workflows": [
                            {
                                "workflow_id": "workflow_a_voice_only",
                                "nodes": ["asr.primary", "planner.primary"],
                                "chunk_sizes": {"asr.primary": 800},
                                "edges": [
                                    {
                                        "from": "asr.primary",
                                        "to": "planner.primary",
                                        "stream_mode": "chunk",
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            stage_path.write_text(
                json.dumps(
                    {"stages": [{"stage_id": "asr.primary", "role": "asr"}, {"stage_id": "planner.primary", "role": "planner"}]}
                ),
                encoding="utf-8",
            )
            plan_bank_path.write_text(json.dumps({"plan_bank_states": [{"state_id": "cool"}]}), encoding="utf-8")
            backend_matrix_path.write_text(
                json.dumps(
                    {
                        "stages": [
                            {"stage_id": "asr.primary", "backends": {"cpu": {"status": "feasible_smoke_pass"}}},
                            {"stage_id": "planner.primary", "backends": {"cpu": {"status": "feasible_smoke_pass"}}},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            profiler_registry_path.write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "stage_id": "asr.primary",
                                "variant": "whisper_stt",
                                "backend": "cpu",
                                "metrics": {
                                    "warm_latency_ms": 100,
                                    "peak_memory_bytes": 1048576,
                                    "output_bytes": 2048,
                                    "average_power_mw": 1200.0,
                                    "quality_loss": 0.0,
                                    "compile_cost_ms": 5,
                                },
                            },
                            {
                                "stage_id": "planner.primary",
                                "variant": "gemma3_1b_it",
                                "backend": "cpu",
                                "metrics": {
                                    "warm_latency_ms": 50,
                                    "peak_memory_bytes": 524288,
                                    "output_bytes": 1024,
                                    "average_power_mw": 900.0,
                                    "quality_loss": 0.0,
                                    "compile_cost_ms": 3,
                                    "ttft_ms": 45,
                                },
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            candidate_registry_path.write_text(json.dumps({"plans": []}), encoding="utf-8")
            tuning_summary_path.write_text(
                json.dumps({"best_objective_weights": {"weights": {"alpha": 2.0, "beta": 1.0, "gamma": 0.0, "delta": 0.0, "eta": 0.0, "zeta": 0.0, "xi": 3.0, "psi": 100.0}}}),
                encoding="utf-8",
            )
            optimization_config_path.write_text(
                json.dumps(
                    {
                        "objective_weight_grid": {
                            "alpha": [1.0], "beta": [0.5], "gamma": [0.0], "delta": [0.0],
                            "eta": [0.0], "zeta": [0.0], "xi": [1.0], "psi": [10.0],
                        },
                        "stream_workloads": {"workflow_a_voice_only": {"arrivals_ms": [0], "deadline_ms": 140}},
                    }
                ),
                encoding="utf-8",
            )
            calibration_summary_path.write_text(
                json.dumps(
                    {
                        "global_orchestration_overhead_ms": 25.0,
                        "thermal_scale_by_workflow": {"workflow_a_voice_only": {"warm_latency_scale": 1.5}},
                    }
                ),
                encoding="utf-8",
            )
            checkpoint_manifest_path.write_text(
                json.dumps(
                    {
                        "canonical_evidence_paths": {
                            "calibration_summary": str(calibration_summary_path.resolve()),
                            "tuning_summary": str(tuning_summary_path.resolve()),
                        }
                    }
                ),
                encoding="utf-8",
            )

            rc = self.module.main(
                [
                    "--workflow-path", str(workflow_path),
                    "--stage-path", str(stage_path),
                    "--plan-bank-path", str(plan_bank_path),
                    "--backend-matrix", str(backend_matrix_path),
                    "--profiler-registry", str(profiler_registry_path),
                    "--candidate-registry", str(candidate_registry_path),
                    "--optimization-config", str(optimization_config_path),
                    "--checkpoint-manifest", str(checkpoint_manifest_path),
                ]
            )

            self.assertEqual(rc, 0)
            payload = json.loads(candidate_registry_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["calibration_summary"], str(calibration_summary_path.resolve()))
            self.assertEqual(payload["objective_weights_source"], str(tuning_summary_path.resolve()))
            self.assertEqual(payload["checkpoint_manifest"], str(checkpoint_manifest_path.resolve()))

    def test_main_emits_objective_aware_predicted_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            workflow_path = root / "workflow_templates.json"
            stage_path = root / "stage_catalog.json"
            plan_bank_path = root / "plan_bank_templates.json"
            backend_matrix_path = root / "backend_feasibility_matrix.json"
            profiler_registry_path = root / "profiler_registry.json"
            candidate_registry_path = root / "candidate_plan_registry.json"
            tuning_summary_path = root / "tuning_summary.json"
            optimization_config_path = root / "optimization_defaults.json"
            calibration_summary_path = root / "calibration_summary.json"

            workflow_path.write_text(
                json.dumps(
                    {
                        "workflows": [
                            {
                                "workflow_id": "workflow_a_voice_only",
                                "nodes": ["asr.primary", "planner.primary"],
                                "chunk_sizes": {"asr.primary": 800},
                                "edges": [
                                    {
                                        "from": "asr.primary",
                                        "to": "planner.primary",
                                        "stream_mode": "chunk",
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            stage_path.write_text(
                json.dumps(
                    {
                        "stages": [
                            {"stage_id": "asr.primary", "role": "asr"},
                            {"stage_id": "planner.primary", "role": "planner"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            plan_bank_path.write_text(
                json.dumps(
                    {
                        "plan_bank_states": [
                            {"state_id": "cool"},
                            {"state_id": "warm"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            backend_matrix_path.write_text(
                json.dumps(
                    {
                        "stages": [
                            {
                                "stage_id": "asr.primary",
                                "backends": {"cpu": {"status": "feasible_smoke_pass"}},
                            },
                            {
                                "stage_id": "planner.primary",
                                "backends": {"cpu": {"status": "feasible_smoke_pass"}},
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            profiler_registry_path.write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "stage_id": "asr.primary",
                                "variant": "whisper_stt",
                                "backend": "cpu",
                                "metrics": {
                                    "warm_latency_ms": 100,
                                    "peak_memory_bytes": 1048576,
                                    "output_bytes": 2048,
                                    "average_power_mw": 1200.0,
                                    "quality_loss": 0.0,
                                    "compile_cost_ms": 5,
                                },
                            },
                            {
                                "stage_id": "planner.primary",
                                "variant": "gemma3_1b_it",
                                "backend": "cpu",
                                "metrics": {
                                    "warm_latency_ms": 50,
                                    "peak_memory_bytes": 524288,
                                    "output_bytes": 1024,
                                    "average_power_mw": 900.0,
                                    "quality_loss": 0.0,
                                    "compile_cost_ms": 3,
                                    "ttft_ms": 45,
                                },
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            candidate_registry_path.write_text(json.dumps({"plans": []}), encoding="utf-8")
            tuning_summary_path.write_text(
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
                                "xi": 3.0,
                                "psi": 100.0,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            optimization_config_path.write_text(
                json.dumps(
                    {
                        "objective_weight_grid": {
                            "alpha": [1.0],
                            "beta": [0.5],
                            "gamma": [0.0],
                            "delta": [0.0],
                            "eta": [0.0],
                            "zeta": [0.0],
                            "xi": [1.0],
                            "psi": [10.0],
                        },
                        "stream_workloads": {
                            "workflow_a_voice_only": {
                                "arrivals_ms": [0, 1],
                                "deadline_ms": 140,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            calibration_summary_path.write_text(
                json.dumps(
                    {
                        "global_orchestration_overhead_ms": 25.0,
                        "thermal_scale_by_workflow": {
                            "workflow_a_voice_only": {"warm_latency_scale": 1.5}
                        },
                    }
                ),
                encoding="utf-8",
            )

            rc = self.module.main(
                [
                    "--workflow-path",
                    str(workflow_path),
                    "--stage-path",
                    str(stage_path),
                    "--plan-bank-path",
                    str(plan_bank_path),
                    "--backend-matrix",
                    str(backend_matrix_path),
                    "--profiler-registry",
                    str(profiler_registry_path),
                    "--candidate-registry",
                    str(candidate_registry_path),
                    "--optimization-config",
                    str(optimization_config_path),
                    "--tuning-summary",
                    str(tuning_summary_path),
                    "--calibration-summary",
                    str(calibration_summary_path),
                ]
            )

            self.assertEqual(rc, 0)
            payload = json.loads(candidate_registry_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["objective_weights"]["alpha"], 2.0)
            self.assertEqual(payload["objective_weights"]["xi"], 3.0)
            self.assertEqual(payload["objective_weights"]["psi"], 100.0)
            self.assertEqual(payload["objective_weights_source"], str(tuning_summary_path.resolve()))
            self.assertEqual(payload["calibration_summary"], str(calibration_summary_path.resolve()))
            self.assertEqual(len(payload["plans"]), 2)
            self.assertEqual(payload["plans"][0]["chunk_sizes"], {"asr.primary": 800})
            predicted_cost = payload["plans"][0]["predicted_cost"]
            self.assertIn("objective_score", predicted_cost)
            self.assertIn("copy_bytes", predicted_cost)
            self.assertIn("energy_mj", predicted_cost)
            self.assertIn("peak_memory_bytes", predicted_cost)
            self.assertIn("p95_e2e_ms", predicted_cost)
            self.assertIn("p95_ttfs_ms", predicted_cost)
            self.assertIn("avg_energy_mj", predicted_cost)
            self.assertIn("p95_queue_delay_ms", predicted_cost)
            self.assertIn("deadline_miss_rate", predicted_cost)
            self.assertEqual(predicted_cost["makespan_ms"], 255)


if __name__ == "__main__":
    unittest.main()
