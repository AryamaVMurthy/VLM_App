import unittest

from graphpilot_edge.errors import ValidationError
from graphpilot_edge.workflow import WorkflowDag, WorkflowEdge


class WorkflowDagTest(unittest.TestCase):
    def test_topological_order_is_stable_for_parallel_graphs(self):
        workflow = WorkflowDag(
            workflow_id="voice_vision",
            stage_ids=(
                "asr.primary",
                "planner.primary",
                "vlm.fastvlm.primary",
                "retrieval.embedder.primary",
                "responder.primary",
            ),
            edges=(
                WorkflowEdge("asr.primary", "planner.primary", "chunk"),
                WorkflowEdge("planner.primary", "vlm.fastvlm.primary", "full"),
                WorkflowEdge("planner.primary", "retrieval.embedder.primary", "full"),
                WorkflowEdge("vlm.fastvlm.primary", "responder.primary", "full"),
                WorkflowEdge("retrieval.embedder.primary", "responder.primary", "full"),
            ),
            chunk_sizes=(("asr.primary", 800),),
        )

        self.assertEqual(
            workflow.topological_order(),
            (
                "asr.primary",
                "planner.primary",
                "vlm.fastvlm.primary",
                "retrieval.embedder.primary",
                "responder.primary",
            ),
        )
        self.assertEqual(workflow.root_stage_ids(), ("asr.primary",))
        self.assertEqual(
            workflow.predecessors("responder.primary"),
            ("retrieval.embedder.primary", "vlm.fastvlm.primary"),
        )
        self.assertEqual(workflow.chunk_size("asr.primary"), 800)
        self.assertIsNone(workflow.chunk_size("planner.primary"))

    def test_cycle_detection_fails_fast(self):
        with self.assertRaisesRegex(ValidationError, "cycle"):
            WorkflowDag(
                workflow_id="invalid_cycle",
                stage_ids=("a", "b", "c"),
                edges=(
                    WorkflowEdge("a", "b", "full"),
                    WorkflowEdge("b", "c", "full"),
                    WorkflowEdge("c", "a", "full"),
                ),
            )

    def test_chunk_sizes_validate_stage_membership_and_positive_values(self):
        with self.assertRaisesRegex(ValidationError, "unknown stage"):
            WorkflowDag(
                workflow_id="invalid_chunks",
                stage_ids=("asr.primary", "planner.primary"),
                edges=(WorkflowEdge("asr.primary", "planner.primary", "chunk"),),
                chunk_sizes=(("missing.primary", 800),),
            )

        with self.assertRaisesRegex(ValidationError, "must be > 0"):
            WorkflowDag(
                workflow_id="invalid_chunk_value",
                stage_ids=("asr.primary", "planner.primary"),
                edges=(WorkflowEdge("asr.primary", "planner.primary", "chunk"),),
                chunk_sizes=(("asr.primary", 0),),
            )


if __name__ == "__main__":
    unittest.main()
