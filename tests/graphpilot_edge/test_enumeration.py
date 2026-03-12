import unittest

from graphpilot_edge.enumeration import enumerate_candidate_plans
from graphpilot_edge.errors import EnumerationError
from graphpilot_edge.models import StageOption
from graphpilot_edge.workflow import WorkflowDag, WorkflowEdge


class CandidatePlanEnumerationTest(unittest.TestCase):
    def setUp(self):
        self.workflow = WorkflowDag(
            workflow_id="voice_pipeline",
            stage_ids=("asr.primary", "planner.primary", "tts.primary"),
            edges=(
                WorkflowEdge("asr.primary", "planner.primary", "chunk"),
                WorkflowEdge("planner.primary", "tts.primary", "token"),
            ),
        )

    def test_exhaustive_stage_enumeration_builds_every_plan(self):
        plans = enumerate_candidate_plans(
            self.workflow,
            {
                "asr.primary": (
                    StageOption("asr.primary", "whisper_tiny", "cpu", 5, 128),
                    StageOption("asr.primary", "whisper_small", "gpu", 3, 256),
                ),
                "planner.primary": (
                    StageOption("planner.primary", "gemma_fast", "npu", 7, 512),
                ),
                "tts.primary": (
                    StageOption("tts.primary", "android_tts", "cpu", 2, 64),
                    StageOption("tts.primary", "streaming_tts", "dsp", 4, 32),
                ),
            },
        )

        self.assertEqual(len(plans), 4)
        self.assertEqual(plans[0].plan_id, "voice_pipeline:asr.primary=whisper_tiny|planner.primary=gemma_fast|tts.primary=android_tts")
        self.assertEqual(plans[-1].total_latency_ms, 14)
        self.assertEqual(plans[-1].total_memory_mb, 800)

    def test_enumeration_requires_explicit_stage_options_for_every_stage(self):
        with self.assertRaisesRegex(EnumerationError, "Missing stage options"):
            enumerate_candidate_plans(
                self.workflow,
                {
                    "asr.primary": (
                        StageOption("asr.primary", "whisper_tiny", "cpu", 5, 128),
                    ),
                    "planner.primary": (
                        StageOption("planner.primary", "gemma_fast", "npu", 7, 512),
                    ),
                },
            )


if __name__ == "__main__":
    unittest.main()
