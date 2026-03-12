import importlib.util
import pathlib
import sys
import unittest


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "run_graphpilot_stage_profiler.py"
    spec = importlib.util.spec_from_file_location(
        "run_graphpilot_stage_profiler", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RunGraphPilotStageProfilerTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_parse_asr_metrics(self):
        metrics = self.module.parse_asr_metrics(
            "WhisperSttInstrumentedTest: first_elapsed_ms=731 second_elapsed_ms=412"
        )
        self.assertEqual(metrics["cold_latency_ms"], 731)
        self.assertEqual(metrics["warm_latency_ms"], 412)

    def test_parse_text_metrics(self):
        metrics = self.module.parse_text_metrics(
            "TEXT_STAGE role=PLANNER backend=CPU model=/tmp/model init_elapsed_ms=1200 generate_elapsed_ms=345 output='ok'"
        )
        self.assertEqual(metrics["cold_latency_ms"], 1545)
        self.assertEqual(metrics["backend_reported"], "CPU")

    def test_parse_tts_metrics(self):
        metrics = self.module.parse_tts_metrics(
            "TTS validated init_elapsed_ms=32 speak_elapsed_ms=12769"
        )
        self.assertEqual(metrics["steady_state_latency_ms"], 12769)

    def test_parse_retrieval_metrics(self):
        metrics = self.module.parse_retrieval_metrics(
            "GRAPH_PILOT_RETRIEVAL backend=cpu elapsed_ms=87 top_doc_id=jfk_inaugural top_score=0.991"
        )
        self.assertEqual(metrics["warm_latency_ms"], 87)
        self.assertEqual(metrics["top_doc_id"], "jfk_inaugural")

    def test_parse_vlm_metrics(self):
        metrics = self.module.parse_vlm_metrics(
            'FastVlmBridgeInstrumentedTest: metrics={"backend_status":{"backend_config_actual":"CPU"},"stage_timings":{"prefill_ms":468,"decode_ms":1535,"total_ms":5198},"token_stats":{"ttft_ms":3683},"error":null}'
        )
        self.assertEqual(metrics["ttft_ms"], 3683)
        self.assertEqual(metrics["prefill_latency_ms"], 468)

    def test_parse_graphpilot_metrics(self):
        metrics = self.module.parse_graphpilot_metrics(
            "GRAPHPILOT_METRICS workflow=workflow_b_voice_vision "
            "plan_id=cool:workflow_b_voice_vision:test state_id=cool "
            "request_id=1 queue_depth_at_admission=0 queue_wait_ms=0 "
            "memory_decision=ADMIT memory_effective_required_bytes=111149056 "
            "stage_backends=asr.primary:cpu,planner.primary:cpu,vlm.fastvlm.primary:cpu,responder.primary:cpu,tts.primary:cpu "
            "stage_timings_ms=asr.primary:310,planner.primary:512,vlm.fastvlm.primary:4021,responder.primary:750,tts.primary:5200 "
            "total_ms=11200 planner_first_partial_ms=444 asr_chunk_count=2 ttft_ms=3585 tts_first_chunk_queued_ms=8800 "
            "tts_first_audio_ms=8931 vlm_prefill_ms=480 vlm_decode_ms=1500"
        )
        self.assertEqual(metrics["workflow_id"], "workflow_b_voice_vision")
        self.assertEqual(metrics["plan_id"], "cool:workflow_b_voice_vision:test")
        self.assertEqual(metrics["state_id"], "cool")
        self.assertEqual(metrics["planner_first_partial_ms"], 444)
        self.assertEqual(metrics["asr_chunk_count"], 2)
        self.assertEqual(metrics["ttft_ms"], 3585)
        self.assertEqual(metrics["tts_first_chunk_queued_ms"], 8800)
        self.assertEqual(metrics["stage_timings_ms"]["tts.primary"], 5200)
        self.assertEqual(metrics["stage_backends"]["vlm.fastvlm.primary"], "cpu")

    def test_resolve_profile_variant_uses_retrieval_backend_for_workflow_c(self):
        variant = self.module.resolve_profile_variant(
            "graphpilot_workflow_c",
            {
                "stage_id": "workflow_c_voice_vision_retrieval",
                "backend": "mixed",
                "variant": "graphpilot_vlm_npu_retrieval_cpu",
            },
            {
                "stage_backends": {
                    "vlm.fastvlm.primary": "npu",
                    "retrieval.embedder.primary": "gpu",
                }
            },
        )
        self.assertEqual(variant, "graphpilot_vlm_npu_retrieval_gpu")

    def test_stage_candidate_plans_requires_registry(self):
        with self.assertRaises(FileNotFoundError):
            self.module.stage_candidate_plans(pathlib.Path("/tmp/missing_candidate_plan_registry.json"))

    def test_parse_voice_pipeline_metrics(self):
        metrics = self.module.parse_voice_pipeline_metrics(
            "PIPELINE_TIMINGS stt_init_ms=127 stt_warmup_ms=470 stt_transcribe_ms=375 "
            "vlm_init_ms=990 vlm_ttft_ms=3683 vlm_prefill_ms=468 vlm_decode_ms=1535 "
            "vlm_total_ms=5198 vlm_wall_ms=5211 tts_init_ms=32 tts_speak_ms=12769"
        )
        self.assertEqual(metrics["prefill_latency_ms"], 468)
        self.assertEqual(metrics["tts_first_audio_ms"], 12801)


if __name__ == "__main__":
    unittest.main()
