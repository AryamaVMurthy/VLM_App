import unittest

from graphpilot_edge.hardware_simulator import HardwareInstance, HardwareSimulator
from graphpilot_edge.model_graph_simulator import (
    build_llm_scenario,
    build_retrieval_scenario,
    build_tts_scenario,
    build_vlm_scenario,
)


class ModelGraphSimulatorTest(unittest.TestCase):
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

    def test_llm_scenario_separates_prefill_and_decode_and_instantiates_plan(self):
        scenario = build_llm_scenario(
            scenario_id="llm.short_long",
            prompt_tokens=512,
            output_tokens=64,
            context_tokens=256,
            max_output_tokens=64,
        )

        self.assertEqual(
            scenario.workflow.stage_ids,
            (
                "llm.prompt_assembly",
                "llm.prefill",
                "llm.decode",
                "llm.postprocess",
            ),
        )
        self.assertEqual(scenario.workflow.predecessors("llm.decode"), ("llm.prefill",))

        plan = scenario.instantiate_candidate_plan(
            simulator=self.simulator,
            resource_assignment={
                "llm.prompt_assembly": "cpu0",
                "llm.prefill": "npu0",
                "llm.decode": "gpu0",
                "llm.postprocess": "cpu0",
            },
        )

        self.assertEqual(plan.option_for_stage("llm.prefill").backend, "npu")
        self.assertEqual(plan.option_for_stage("llm.decode").backend, "gpu")
        self.assertGreater(plan.option_for_stage("llm.decode").memory_mb, 0)
        self.assertGreater(plan.option_for_stage("llm.prefill").latency_ms, 0)

    def test_vlm_visual_token_budget_changes_prefill_profile(self):
        high_budget = build_vlm_scenario(
            scenario_id="vlm.high",
            prompt_tokens=64,
            output_tokens=32,
            image_resolution=896,
            visual_token_budget=256,
        )
        low_budget = build_vlm_scenario(
            scenario_id="vlm.low",
            prompt_tokens=64,
            output_tokens=32,
            image_resolution=896,
            visual_token_budget=64,
        )

        high_prefill = high_budget.node_profiles["vlm.multimodal_prefill"].task_profile
        low_prefill = low_budget.node_profiles["vlm.multimodal_prefill"].task_profile
        self.assertGreater(high_prefill.op_volume["attn"], low_prefill.op_volume["attn"])
        self.assertGreater(high_prefill.memory_bytes, low_prefill.memory_bytes)

    def test_retrieval_scenario_includes_optional_reranker(self):
        without_rerank = build_retrieval_scenario(
            scenario_id="retrieval.base",
            query_tokens=32,
            corpus_size=1_000,
            top_k=4,
            rerank=False,
        )
        with_rerank = build_retrieval_scenario(
            scenario_id="retrieval.rerank",
            query_tokens=32,
            corpus_size=1_000,
            top_k=4,
            rerank=True,
        )

        self.assertNotIn("retrieval.rerank", without_rerank.workflow.stage_ids)
        self.assertIn("retrieval.rerank", with_rerank.workflow.stage_ids)
        self.assertEqual(with_rerank.workflow.predecessors("retrieval.rerank"), ("retrieval.ann",))

    def test_tts_scenario_uses_chunked_vocoder_edge(self):
        scenario = build_tts_scenario(
            scenario_id="tts.streaming",
            text_tokens=96,
            chunk_size_chars=80,
            quality_mode="balanced",
        )

        self.assertTrue(scenario.workflow.has_edge("tts.acoustic", "tts.vocoder", "chunk"))
        plan = scenario.instantiate_candidate_plan(
            simulator=self.simulator,
            resource_assignment={
                "tts.text_normalize": "cpu0",
                "tts.acoustic": "cpu0",
                "tts.vocoder": "gpu0",
            },
        )
        self.assertEqual(plan.option_for_stage("tts.vocoder").backend, "gpu")
        self.assertGreater(plan.option_for_stage("tts.vocoder").tts_first_audio_ms or 0, 0)


if __name__ == "__main__":
    unittest.main()
