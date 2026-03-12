import importlib.util
import json
import pathlib
import tempfile
import unittest


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "run_graphpilot_experiments.py"
    spec = importlib.util.spec_from_file_location("run_graphpilot_experiments", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RunGraphPilotExperimentsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_default_rerun_profiles_use_graphpilot_support_safe_workflows_only(self) -> None:
        self.assertEqual(
            self.module.DEFAULT_RERUN_PROFILES,
            (
                "graphpilot_workflow_a",
                "graphpilot_workflow_b",
                "graphpilot_workflow_c",
            ),
        )

    def test_summarize_results_marks_blocked_workflow_c(self) -> None:
        profiler_registry = {
            "entries": [
                {
                    "stage_id": "workflow_a_voice_only",
                    "backend": "mixed",
                    "variant": "graphpilot_cpu_stack",
                    "recorded_at": "2026-03-10T00:00:00+00:00",
                    "metrics": {
                        "warm_latency_ms": 1000,
                        "tts_first_audio_ms": 250,
                        "tts_first_chunk_queued_ms": 200,
                        "ttft_ms": None,
                    },
                },
                {
                    "stage_id": "workflow_b_voice_vision",
                    "backend": "mixed",
                    "variant": "graphpilot_vlm_npu",
                    "recorded_at": "2026-03-10T00:00:01+00:00",
                    "metrics": {
                        "warm_latency_ms": 2000,
                        "tts_first_audio_ms": 750,
                        "tts_first_chunk_queued_ms": 700,
                        "ttft_ms": 500,
                    },
                },
            ]
        }
        candidate_plans = {
            "plans": [
                {
                    "plan_id": "cool:workflow_b_voice_vision:test",
                    "workflow_template": "workflow_b_voice_vision",
                    "state_id": "cool",
                    "backend_map": {"vlm.fastvlm.primary": "npu"},
                    "predicted_cost": {"makespan_ms": 1500, "latency_sum_ms": 1500, "memory_mb": 0},
                }
            ]
        }
        backend_matrix = {
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

        summary = self.module.summarize_results(profiler_registry, candidate_plans, backend_matrix)

        blocked = {item["workflow_id"]: item for item in summary["blocked_workflows"]}
        self.assertIn("workflow_c_voice_vision_retrieval", blocked)
        self.assertEqual(blocked["workflow_c_voice_vision_retrieval"]["reason"], "retrieval_infeasible")
        self.assertEqual(summary["actual_workflows"]["workflow_b_voice_vision"]["warm_latency_ms"], 2000)
        self.assertEqual(
            summary["actual_workflows"]["workflow_b_voice_vision"]["tts_first_chunk_queued_ms"],
            700,
        )

    def test_summarize_results_accepts_profiled_workflow_c_when_retrieval_is_feasible(self) -> None:
        profiler_registry = {
            "entries": [
                {
                    "stage_id": "workflow_c_voice_vision_retrieval",
                    "backend": "mixed",
                    "variant": "graphpilot_vlm_npu_retrieval_cpu",
                    "recorded_at": "2026-03-10T00:00:02+00:00",
                    "metrics": {"warm_latency_ms": 3200, "tts_first_audio_ms": 900, "ttft_ms": 700},
                }
            ]
        }
        candidate_plans = {"plans": []}
        backend_matrix = {
            "stages": [
                {
                    "stage_id": "retrieval.embedder.primary",
                    "backends": {
                        "cpu": {"status": "feasible_smoke_pass"},
                        "gpu": {"status": "infeasible_no_backend_adapter"},
                        "npu": {"status": "infeasible_no_backend_adapter"},
                    },
                }
            ]
        }

        summary = self.module.summarize_results(profiler_registry, candidate_plans, backend_matrix)

        self.assertEqual(summary["blocked_workflows"], [])
        self.assertEqual(
            summary["actual_workflows"]["workflow_c_voice_vision_retrieval"]["warm_latency_ms"],
            3200,
        )

    def test_summarize_results_accepts_gpu_retrieval_workflow_c_variant(self) -> None:
        profiler_registry = {
            "entries": [
                {
                    "stage_id": "workflow_c_voice_vision_retrieval",
                    "backend": "mixed",
                    "variant": "graphpilot_vlm_npu_retrieval_gpu",
                    "recorded_at": "2026-03-11T00:00:02+00:00",
                    "metrics": {
                        "workflow_id": "workflow_c_voice_vision_retrieval",
                        "warm_latency_ms": 2800,
                        "tts_first_audio_ms": 850,
                        "ttft_ms": 650,
                        "stage_backends": {
                            "vlm.fastvlm.primary": "npu",
                            "retrieval.embedder.primary": "gpu",
                        },
                    },
                }
            ]
        }
        candidate_plans = {"plans": []}
        backend_matrix = {
            "stages": [
                {
                    "stage_id": "retrieval.embedder.primary",
                    "backends": {
                        "cpu": {"status": "feasible_smoke_pass"},
                        "gpu": {"status": "feasible_smoke_pass"},
                        "npu": {"status": "infeasible_no_backend_adapter"},
                    },
                }
            ]
        }

        summary = self.module.summarize_results(profiler_registry, candidate_plans, backend_matrix)

        self.assertEqual(summary["blocked_workflows"], [])
        self.assertEqual(
            summary["actual_workflows"]["workflow_c_voice_vision_retrieval"]["variant"],
            "graphpilot_vlm_npu_retrieval_gpu",
        )
        self.assertEqual(
            summary["actual_workflows"]["workflow_c_voice_vision_retrieval"]["stage_backends"]["retrieval.embedder.primary"],
            "gpu",
        )

    def test_summarize_results_prefers_lowest_objective_score_when_present(self) -> None:
        profiler_registry = {
            "entries": [
                {
                    "stage_id": "workflow_b_voice_vision",
                    "backend": "mixed",
                    "variant": "graphpilot_vlm_npu",
                    "recorded_at": "2026-03-10T00:00:01+00:00",
                    "metrics": {"warm_latency_ms": 2000, "tts_first_audio_ms": 750, "ttft_ms": 500},
                },
            ]
        }
        candidate_plans = {
            "plans": [
                {
                    "plan_id": "cool:workflow_b_voice_vision:fast_but_worse_objective",
                    "workflow_template": "workflow_b_voice_vision",
                    "state_id": "cool",
                    "backend_map": {"vlm.fastvlm.primary": "npu"},
                    "predicted_cost": {
                        "makespan_ms": 1400,
                        "latency_sum_ms": 1400,
                        "memory_mb": 0,
                        "objective_score": 3000.0,
                    },
                },
                {
                    "plan_id": "cool:workflow_b_voice_vision:better_objective",
                    "workflow_template": "workflow_b_voice_vision",
                    "state_id": "cool",
                    "backend_map": {"vlm.fastvlm.primary": "npu"},
                    "predicted_cost": {
                        "makespan_ms": 1500,
                        "latency_sum_ms": 1500,
                        "memory_mb": 0,
                        "objective_score": 1000.0,
                    },
                },
            ]
        }
        backend_matrix = {
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

        summary = self.module.summarize_results(profiler_registry, candidate_plans, backend_matrix)

        self.assertEqual(
            summary["comparisons"][0]["candidate_plan_id"],
            "cool:workflow_b_voice_vision:better_objective",
        )

    def test_upsert_experiment_registry_adds_batch(self) -> None:
        registry = {"experiments": []}
        summary_path = pathlib.Path("/tmp/fake_summary.json")
        payload = {"experiment_id": "graphpilot_batch", "output_dir": "/tmp/batch"}

        self.module.upsert_experiment_registry(
            registry=registry,
            summary_path=summary_path,
            payload=payload,
            commit="deadbeef",
            adb_serial="serial123",
        )

        self.assertEqual(len(registry["experiments"]), 1)
        self.assertEqual(registry["experiments"][0]["experiment_id"], "graphpilot_batch")
        self.assertEqual(registry["experiments"][0]["device_state"]["adb_serial"], "serial123")

    def test_main_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            profiler = root / "profiler.json"
            plans = root / "plans.json"
            matrix = root / "matrix.json"
            experiments = root / "experiments.json"
            output_root = root / "out"
            profiler.write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "stage_id": "workflow_a_voice_only",
                                "backend": "mixed",
                                "variant": "graphpilot_cpu_stack",
                                "recorded_at": "2026-03-10T00:00:00+00:00",
                                "metrics": {"warm_latency_ms": 1000, "tts_first_audio_ms": 250, "ttft_ms": None},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            plans.write_text(json.dumps({"plans": []}), encoding="utf-8")
            matrix.write_text(
                json.dumps(
                    {
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
                ),
                encoding="utf-8",
            )
            experiments.write_text(json.dumps({"experiments": []}), encoding="utf-8")

            rc = self.module.main(
                [
                    "--profiler-registry",
                    str(profiler),
                    "--candidate-plans",
                    str(plans),
                    "--backend-matrix",
                    str(matrix),
                    "--experiment-registry",
                    str(experiments),
                    "--output-root",
                    str(output_root),
                ]
            )

            self.assertEqual(rc, 0)
            summaries = list(output_root.glob("graphpilot_experiment_batch_*/summary.json"))
            self.assertEqual(len(summaries), 1)


if __name__ == "__main__":
    unittest.main()
