from __future__ import annotations

import unittest

from graphpilot_edge.catalog import StageDefinition
from graphpilot_edge.macro_regions import (
    beam_search_macro_region_plans,
    expand_workflow_with_macro_regions,
    macro_region_stage_id,
)
from graphpilot_edge.models import StageOption
from graphpilot_edge.workflow import WorkflowDag, WorkflowEdge


class MacroRegionPlanningTest(unittest.TestCase):
    def test_expand_workflow_replaces_opened_stages_with_macro_regions(self) -> None:
        workflow = WorkflowDag(
            workflow_id="workflow_b_voice_vision",
            stage_ids=(
                "asr.primary",
                "planner.primary",
                "vlm.fastvlm.primary",
                "responder.primary",
                "tts.primary",
            ),
            edges=(
                WorkflowEdge("asr.primary", "planner.primary", "chunk"),
                WorkflowEdge("planner.primary", "vlm.fastvlm.primary", "full"),
                WorkflowEdge("vlm.fastvlm.primary", "responder.primary", "full"),
                WorkflowEdge("responder.primary", "tts.primary", "token"),
            ),
        )
        stage_catalog = {
            "vlm.fastvlm.primary": StageDefinition(
                stage_id="vlm.fastvlm.primary",
                role="vlm",
                macro_regions=("vision_encoder", "prefill", "decode"),
            ),
            "responder.primary": StageDefinition(
                stage_id="responder.primary",
                role="responder",
                macro_regions=("prefill", "decode"),
            ),
        }

        expanded = expand_workflow_with_macro_regions(
            workflow,
            stage_catalog,
            opened_stage_ids={"vlm.fastvlm.primary", "responder.primary"},
        )

        self.assertEqual(
            expanded.stage_ids,
            (
                "asr.primary",
                "planner.primary",
                "vlm.fastvlm.primary::vision_encoder",
                "vlm.fastvlm.primary::prefill",
                "vlm.fastvlm.primary::decode",
                "responder.primary::prefill",
                "responder.primary::decode",
                "tts.primary",
            ),
        )
        edge_pairs = {(edge.source_stage_id, edge.target_stage_id, edge.stream_mode) for edge in expanded.edges}
        self.assertIn(
            (
                "planner.primary",
                "vlm.fastvlm.primary::vision_encoder",
                "full",
            ),
            edge_pairs,
        )
        self.assertIn(
            (
                "vlm.fastvlm.primary::decode",
                "responder.primary::prefill",
                "full",
            ),
            edge_pairs,
        )
        self.assertIn(
            (
                "responder.primary::decode",
                "tts.primary",
                "token",
            ),
            edge_pairs,
        )

    def test_beam_search_returns_best_completed_macro_plan(self) -> None:
        workflow = WorkflowDag(
            workflow_id="workflow_b_voice_vision::macro",
            stage_ids=(
                macro_region_stage_id("vlm.fastvlm.primary", "vision_encoder"),
                macro_region_stage_id("vlm.fastvlm.primary", "prefill"),
                macro_region_stage_id("responder.primary", "prefill"),
            ),
            edges=(
                WorkflowEdge(
                    macro_region_stage_id("vlm.fastvlm.primary", "vision_encoder"),
                    macro_region_stage_id("vlm.fastvlm.primary", "prefill"),
                    "full",
                ),
                WorkflowEdge(
                    macro_region_stage_id("vlm.fastvlm.primary", "prefill"),
                    macro_region_stage_id("responder.primary", "prefill"),
                    "full",
                ),
            ),
        )
        stage_options = {
            macro_region_stage_id("vlm.fastvlm.primary", "vision_encoder"): (
                StageOption(
                    stage_id=macro_region_stage_id("vlm.fastvlm.primary", "vision_encoder"),
                    variant_id="fastvlm_encoder_npu",
                    backend="npu",
                    latency_ms=100,
                    memory_mb=64,
                    output_bytes=2048,
                    average_power_mw=400.0,
                ),
                StageOption(
                    stage_id=macro_region_stage_id("vlm.fastvlm.primary", "vision_encoder"),
                    variant_id="fastvlm_encoder_cpu",
                    backend="cpu",
                    latency_ms=180,
                    memory_mb=64,
                    output_bytes=2048,
                    average_power_mw=200.0,
                ),
            ),
            macro_region_stage_id("vlm.fastvlm.primary", "prefill"): (
                StageOption(
                    stage_id=macro_region_stage_id("vlm.fastvlm.primary", "prefill"),
                    variant_id="fastvlm_prefill_npu",
                    backend="npu",
                    latency_ms=140,
                    memory_mb=80,
                    output_bytes=4096,
                    average_power_mw=450.0,
                ),
                StageOption(
                    stage_id=macro_region_stage_id("vlm.fastvlm.primary", "prefill"),
                    variant_id="fastvlm_prefill_cpu",
                    backend="cpu",
                    latency_ms=220,
                    memory_mb=80,
                    output_bytes=4096,
                    average_power_mw=220.0,
                ),
            ),
            macro_region_stage_id("responder.primary", "prefill"): (
                StageOption(
                    stage_id=macro_region_stage_id("responder.primary", "prefill"),
                    variant_id="responder_prefill_cpu",
                    backend="cpu",
                    latency_ms=160,
                    memory_mb=96,
                    output_bytes=1024,
                    average_power_mw=180.0,
                ),
                StageOption(
                    stage_id=macro_region_stage_id("responder.primary", "prefill"),
                    variant_id="responder_prefill_npu",
                    backend="npu",
                    latency_ms=110,
                    memory_mb=96,
                    output_bytes=1024,
                    average_power_mw=320.0,
                ),
            ),
        }

        ranked, stats = beam_search_macro_region_plans(
            workflow,
            stage_options,
            beam_width=2,
        )

        self.assertGreaterEqual(stats.expanded_partials, 4)
        self.assertGreaterEqual(stats.completed_plans, 1)
        self.assertTrue(ranked)
        self.assertEqual(
            [option.variant_id for option in ranked[0].plan.stage_options],
            ["fastvlm_encoder_npu", "fastvlm_prefill_npu", "responder_prefill_npu"],
        )


if __name__ == "__main__":
    unittest.main()
