import unittest

from graphpilot_edge.hardware_simulator import HardwareInstance, HardwareSimulator
from graphpilot_edge.workload_universe import DEFAULT_WORKLOAD_UNIVERSE_PATH, load_workload_universe


class WorkloadUniverseTest(unittest.TestCase):
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
        self.universe = load_workload_universe(DEFAULT_WORKLOAD_UNIVERSE_PATH)

    def test_default_universe_covers_required_categories(self):
        categories = {spec.category for spec in self.universe.specs.values()}
        self.assertIn("primitive_operator", categories)
        self.assertIn("model_family", categories)
        self.assertIn("compound_assistant", categories)
        self.assertIn("continuous_stream", categories)
        self.assertIn("stress_failure", categories)

    def test_default_universe_covers_revision_report_workload_families(self):
        self.assertIn("model.vlm.document_qa", self.universe.specs)
        self.assertIn("model.vlm.chart_qa", self.universe.specs)
        self.assertIn("model.vlm.mmmu_reasoning", self.universe.specs)
        self.assertIn("model.llm.mobile_actions_planner", self.universe.specs)
        self.assertIn("compound.workflow_a.mobile_actions", self.universe.specs)
        self.assertIn("compound.workflow_b.document_qa", self.universe.specs)
        self.assertIn("compound.workflow_b.chart_qa", self.universe.specs)
        self.assertIn("compound.workflow_b.mmmu_reasoning", self.universe.specs)
        self.assertIn("compound.workflow_c.high_recall_rag", self.universe.specs)
        self.assertIn("model.glue.tool_call_marshal", self.universe.specs)
        self.assertIn("model.glue.retrieval_chunk_pack", self.universe.specs)
        self.assertIn("continuous.workflow_b.poisson", self.universe.specs)
        self.assertIn("continuous.workflow_c.poisson_heavy", self.universe.specs)
        self.assertIn("continuous.workflow_c.queue_overload", self.universe.specs)
        self.assertIn("stress.workflow_b.high_visual_tokens", self.universe.specs)
        self.assertIn("stress.workflow_b.shape_volatility", self.universe.specs)
        self.assertIn("stress.workflow_c.fallback_penalty", self.universe.specs)

        datasets = {
            dataset
            for spec in self.universe.specs.values()
            for dataset in spec.datasets
        }
        self.assertIn("MMMU", datasets)
        self.assertIn("DocVQA", datasets)
        self.assertIn("ChartQA", datasets)
        self.assertIn("MobileActions", datasets)

    def test_compound_workflow_c_scenario_preserves_parallel_branches_and_stream_edges(self):
        scenario = self.universe.build_scenario("compound.workflow_c.default")

        self.assertTrue(scenario.workflow.has_edge("asr.primary", "planner.primary", "chunk"))
        self.assertTrue(scenario.workflow.has_edge("planner.primary", "vlm.fastvlm.primary", "full"))
        self.assertTrue(scenario.workflow.has_edge("planner.primary", "retrieval.embedder.primary", "full"))
        self.assertTrue(scenario.workflow.has_edge("responder.primary", "tts.primary", "token"))
        self.assertEqual(scenario.workflow.chunk_size("asr.primary"), 800)

        plan = scenario.instantiate_candidate_plan(
            simulator=self.simulator,
            resource_assignment={
                "asr.primary": "cpu0",
                "planner.primary": "cpu0",
                "vlm.fastvlm.primary": "npu0",
                "retrieval.embedder.primary": "cpu0",
                "responder.primary": "cpu0",
                "tts.primary": "cpu0",
            },
        )
        self.assertEqual(plan.option_for_stage("vlm.fastvlm.primary").backend, "npu")

    def test_continuous_stream_workload_emits_request_specs(self):
        requests = self.universe.build_request_specs(
            "continuous.workflow_a.poisson",
            simulator=self.simulator,
            resource_assignment={
                "asr.primary": "cpu0",
                "planner.primary": "cpu0",
                "responder.primary": "cpu0",
                "tts.primary": "cpu0",
            },
        )

        self.assertEqual(len(requests), 4)
        self.assertEqual([request.arrival_ms for request in requests], [0, 1500, 3000, 4500])
        self.assertTrue(all(request.deadline_ms == 12000 for request in requests))
        self.assertTrue(all(request.plan.workflow_id == "compound.workflow_a.default" for request in requests))

    def test_new_multimodal_and_rag_workloads_build(self):
        scenario = self.universe.build_scenario("compound.workflow_b.document_qa")
        self.assertIn("vlm.fastvlm.primary", scenario.workflow.stage_ids)
        self.assertEqual(
            scenario.node_profiles["vlm.fastvlm.primary"].knob_values["image_resolution"],
            1024,
        )

        rag_scenario = self.universe.build_scenario("compound.workflow_c.high_recall_rag")
        self.assertTrue(
            rag_scenario.workflow.has_edge(
                "planner.primary",
                "retrieval.embedder.primary",
                "full",
            )
        )
        self.assertEqual(
            rag_scenario.node_profiles["retrieval.embedder.primary"].knob_values["top_k"],
            8,
        )

    def test_mixed_criticality_stream_builds_requests_from_multiple_base_workloads(self):
        requests = self.universe.build_request_specs(
            "continuous.mixed_foreground_background",
            simulator=self.simulator,
            resource_assignment={
                "asr.primary": "cpu0",
                "planner.primary": "cpu0",
                "responder.primary": "cpu0",
                "tts.primary": "cpu0",
                "vlm.fastvlm.primary": "npu0",
                "retrieval.embedder.primary": "cpu0",
                "glue.tokenize": "cpu0",
                "glue.validate": "cpu0",
                "glue.pack": "cpu0",
            },
        )

        self.assertEqual(len(requests), 6)
        self.assertEqual(
            [request.source_workload_id for request in requests[:3]],
            ["compound.workflow_c.high_recall_rag"] * 3,
        )
        self.assertEqual(
            [request.source_workload_id for request in requests[3:]],
            ["model.glue.retrieval_chunk_pack"] * 3,
        )
        self.assertEqual(requests[0].criticality_class, "foreground")
        self.assertEqual(requests[-1].criticality_class, "background")

    def test_stress_workload_carries_quality_speed_knobs_into_scenario(self):
        scenario = self.universe.build_scenario("stress.workflow_c.long_context")
        responder = scenario.node_profiles["responder.primary"]
        self.assertEqual(responder.knob_values["max_output_tokens"], 192)
        self.assertGreater(responder.task_profile.memory_bytes, 0)

    def test_fallback_and_queue_stress_workloads_are_configured(self):
        queue_stress = self.universe.specs["continuous.workflow_c.queue_overload"]
        self.assertEqual(queue_stress.base_workload_id, "compound.workflow_c.high_recall_rag")
        self.assertGreater(len(queue_stress.arrivals_ms), 4)

        fallback = self.universe.build_scenario("stress.workflow_c.fallback_penalty")
        self.assertEqual(
            fallback.node_profiles["retrieval.embedder.primary"].knob_values["top_k"],
            10,
        )

        shape_volatility = self.universe.build_scenario("stress.workflow_b.shape_volatility")
        self.assertEqual(
            shape_volatility.node_profiles["vlm.fastvlm.primary"].knob_values["visual_token_budget"],
            384,
        )


if __name__ == "__main__":
    unittest.main()
