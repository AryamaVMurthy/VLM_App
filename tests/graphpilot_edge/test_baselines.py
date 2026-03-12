import unittest

from graphpilot_edge.baselines import build_baseline_candidate
from graphpilot_edge.hardware_simulator import HardwareInstance, HardwareSimulator, PartitionAssignment, TaskProfile
from graphpilot_edge.model_graph_simulator import ModelGraphNodeProfile, ModelGraphScenario
from graphpilot_edge.workload_universe import DEFAULT_WORKLOAD_UNIVERSE_PATH, load_workload_universe
from graphpilot_edge.workflow import WorkflowDag


class GraphPilotBaselinesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.simulator = HardwareSimulator(
            resources=(
                HardwareInstance(
                    resource_id="cpu0",
                    resource_type="cpu",
                    supported_op_classes=("control", "gather", "audio", "gemm", "attn", "ann", "conv", "elem"),
                    service_rates={
                        "control": 100.0,
                        "gather": 300.0,
                        "audio": 200.0,
                        "gemm": 200.0,
                        "attn": 120.0,
                        "ann": 150.0,
                        "conv": 200.0,
                        "elem": 250.0,
                    },
                    memory_bandwidth_bytes_per_ms=2_000.0,
                    launch_overhead_ms=1.0,
                    busy_power_mw=2_500.0,
                    idle_power_mw=200.0,
                    queue_depth=2,
                    batching_beta=0.2,
                    batching_tau=2.0,
                    thermal_time_constant_ms=10_000.0,
                    thermal_resistance_c_per_mw=0.005,
                    thermal_threshold_c=40.0,
                    thermal_gamma=0.05,
                    contention_domain="dram",
                ),
                HardwareInstance(
                    resource_id="gpu0",
                    resource_type="gpu",
                    supported_op_classes=("gemm", "attn", "conv", "elem"),
                    service_rates={
                        "gemm": 600.0,
                        "attn": 250.0,
                        "conv": 500.0,
                        "elem": 350.0,
                    },
                    memory_bandwidth_bytes_per_ms=4_500.0,
                    launch_overhead_ms=1.5,
                    busy_power_mw=2_000.0,
                    idle_power_mw=180.0,
                    queue_depth=2,
                    batching_beta=0.4,
                    batching_tau=3.0,
                    thermal_time_constant_ms=9_000.0,
                    thermal_resistance_c_per_mw=0.004,
                    thermal_threshold_c=41.0,
                    thermal_gamma=0.08,
                    contention_domain="dram",
                ),
                HardwareInstance(
                    resource_id="npu0",
                    resource_type="npu",
                    supported_op_classes=("gemm", "attn", "conv", "elem"),
                    service_rates={
                        "gemm": 1200.0,
                        "attn": 700.0,
                        "conv": 900.0,
                        "elem": 400.0,
                    },
                    memory_bandwidth_bytes_per_ms=5_500.0,
                    launch_overhead_ms=2.0,
                    busy_power_mw=1_700.0,
                    idle_power_mw=160.0,
                    queue_depth=1,
                    batching_beta=0.6,
                    batching_tau=4.0,
                    thermal_time_constant_ms=8_000.0,
                    thermal_resistance_c_per_mw=0.004,
                    thermal_threshold_c=42.0,
                    thermal_gamma=0.1,
                    contention_domain="dram",
                ),
            ),
            transfer_bandwidth_bytes_per_ms={
                ("cpu0", "gpu0"): 1_500.0,
                ("gpu0", "cpu0"): 1_500.0,
                ("cpu0", "npu0"): 1_200.0,
                ("npu0", "cpu0"): 1_200.0,
                ("gpu0", "npu0"): 1_100.0,
                ("npu0", "gpu0"): 1_100.0,
            },
            transfer_fixed_overhead_ms={
                ("cpu0", "gpu0"): 0.5,
                ("gpu0", "cpu0"): 0.5,
                ("cpu0", "npu0"): 0.5,
                ("npu0", "cpu0"): 0.5,
                ("gpu0", "npu0"): 0.75,
                ("npu0", "gpu0"): 0.75,
            },
            transfer_layout_overhead_ms={
                ("cpu0", "gpu0"): 0.25,
                ("gpu0", "cpu0"): 0.25,
                ("cpu0", "npu0"): 0.25,
                ("npu0", "cpu0"): 0.25,
                ("gpu0", "npu0"): 0.3,
                ("npu0", "gpu0"): 0.3,
            },
            contention_sensitivities={
                ("cpu0", "gpu0"): 0.1,
                ("cpu0", "npu0"): 0.15,
                ("gpu0", "cpu0"): 0.12,
                ("gpu0", "npu0"): 0.1,
                ("npu0", "cpu0"): 0.2,
                ("npu0", "gpu0"): 0.15,
            },
            ambient_temperature_c=30.0,
        )
        universe = load_workload_universe(DEFAULT_WORKLOAD_UNIVERSE_PATH)
        self.workflow_a = universe.build_scenario("compound.workflow_a.default")
        self.workflow_b = universe.build_scenario("compound.workflow_b.default")
        self.workflow_c = universe.build_scenario("compound.workflow_c.default")

    def test_cpu_only_assigns_cpu_to_every_stage(self):
        candidate = build_baseline_candidate(
            self.workflow_a,
            simulator=self.simulator,
            baseline_id="cpu_only",
        )

        self.assertTrue(all(resource_id == "cpu0" for resource_id in candidate.resource_assignment.values()))
        self.assertTrue(all(option.backend == "cpu" for option in candidate.plan.stage_options))

    def test_stage_greedy_picks_npu_for_vlm_and_cpu_for_retrieval(self):
        candidate = build_baseline_candidate(
            self.workflow_c,
            simulator=self.simulator,
            baseline_id="stage_greedy",
        )

        self.assertEqual(candidate.resource_assignment["vlm.fastvlm.primary"], "npu0")
        self.assertEqual(candidate.resource_assignment["retrieval.embedder.primary"], "cpu0")

    def test_current_deployed_plan_matches_support_safe_prototype_map(self):
        candidate = build_baseline_candidate(
            self.workflow_c,
            simulator=self.simulator,
            baseline_id="current_deployed_plan",
        )

        self.assertEqual(candidate.resource_assignment["asr.primary"], "cpu0")
        self.assertEqual(candidate.resource_assignment["planner.primary"], "cpu0")
        self.assertEqual(candidate.resource_assignment["vlm.fastvlm.primary"], "npu0")
        self.assertEqual(candidate.resource_assignment["retrieval.embedder.primary"], "cpu0")
        self.assertEqual(candidate.resource_assignment["responder.primary"], "cpu0")
        self.assertEqual(candidate.resource_assignment["tts.primary"], "cpu0")

    def test_no_pipeline_converts_stream_edges_to_full(self):
        candidate = build_baseline_candidate(
            self.workflow_a,
            simulator=self.simulator,
            baseline_id="no_pipeline",
        )

        self.assertTrue(candidate.workflow.has_edge("asr.primary", "planner.primary", "full"))
        self.assertTrue(candidate.workflow.has_edge("responder.primary", "tts.primary", "full"))

    def test_no_pipeline_scores_slower_than_stage_greedy_for_workflow_a(self):
        pipelined = build_baseline_candidate(
            self.workflow_a,
            simulator=self.simulator,
            baseline_id="stage_greedy",
        )
        no_pipeline = build_baseline_candidate(
            self.workflow_a,
            simulator=self.simulator,
            baseline_id="no_pipeline",
        )

        self.assertGreater(no_pipeline.score_ms, pipelined.score_ms)

    def test_static_best_map_beats_cpu_only_for_workflow_b(self):
        cpu_only = build_baseline_candidate(
            self.workflow_b,
            simulator=self.simulator,
            baseline_id="cpu_only",
        )
        static_best = build_baseline_candidate(
            self.workflow_b,
            simulator=self.simulator,
            baseline_id="static_best_map",
        )

        self.assertLess(static_best.plan.total_latency_ms, cpu_only.plan.total_latency_ms)
        self.assertEqual(static_best.resource_assignment["vlm.fastvlm.primary"], "npu0")

    def test_npu_only_fails_fast_when_support_safe_path_does_not_exist(self):
        with self.assertRaisesRegex(ValueError, "No support-safe resource"):
            build_baseline_candidate(
                self.workflow_a,
                simulator=self.simulator,
                baseline_id="npu_only",
            )

    def test_no_fallback_aware_can_pick_partitioned_fallback_path(self):
        scenario = ModelGraphScenario(
            scenario_id="toy.fallback",
            workflow=WorkflowDag(workflow_id="toy.fallback", stage_ids=("toy.stage",), edges=()),
            node_profiles={
                "toy.stage": ModelGraphNodeProfile(
                    stage_id="toy.stage",
                    family="toy",
                    variant_id="toy",
                    task_profile=TaskProfile(
                        task_id="toy.stage",
                        op_volume={"gemm": 500.0, "control": 40.0},
                        memory_bytes=256 * 1024,
                        input_bytes=128 * 1024,
                        output_bytes=64 * 1024,
                        fallback_partitions=(
                            PartitionAssignment(
                                partition_id="toy.npu",
                                resource_id="npu0",
                                op_volume={"gemm": 500.0},
                                memory_bytes=128 * 1024,
                                output_bytes=64 * 1024,
                            ),
                            PartitionAssignment(
                                partition_id="toy.cpu",
                                resource_id="cpu0",
                                op_volume={"control": 40.0},
                                memory_bytes=64 * 1024,
                                output_bytes=16 * 1024,
                            ),
                        ),
                    ),
                )
            },
        )

        support_safe = build_baseline_candidate(
            scenario,
            simulator=self.simulator,
            baseline_id="stage_greedy",
        )
        fallback_unaware = build_baseline_candidate(
            scenario,
            simulator=self.simulator,
            baseline_id="no_fallback_aware",
        )

        self.assertEqual(support_safe.resource_assignment["toy.stage"], "cpu0")
        self.assertNotEqual(fallback_unaware.resource_assignment["toy.stage"], "cpu0")

    def test_no_thermal_adaptation_is_slower_than_static_best_map_under_hot_state(self):
        static_best = build_baseline_candidate(
            self.workflow_b,
            simulator=self.simulator,
            baseline_id="static_best_map",
        )
        no_thermal = build_baseline_candidate(
            self.workflow_b,
            simulator=self.simulator,
            baseline_id="no_thermal_adaptation",
        )

        self.assertGreater(no_thermal.score_ms, static_best.score_ms)

    def test_no_memory_kv_and_no_knob_tuning_are_runnable(self):
        no_memory_kv = build_baseline_candidate(
            self.workflow_c,
            simulator=self.simulator,
            baseline_id="no_memory_kv",
        )
        no_knob_tuning = build_baseline_candidate(
            self.workflow_c,
            simulator=self.simulator,
            baseline_id="no_knob_tuning",
        )

        self.assertGreater(no_memory_kv.score_ms, 0.0)
        self.assertGreater(no_knob_tuning.score_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
