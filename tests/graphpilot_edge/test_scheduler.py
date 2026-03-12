import unittest

from graphpilot_edge.scheduler import compute_dynamic_priority, compute_upward_ranks
from graphpilot_edge.workflow import WorkflowDag, WorkflowEdge


class GraphPilotSchedulerTest(unittest.TestCase):
    def test_compute_upward_ranks_counts_downstream_cost(self):
        workflow = WorkflowDag(
            workflow_id="workflow_b",
            stage_ids=("asr.primary", "planner.primary", "responder.primary"),
            edges=(
                WorkflowEdge("asr.primary", "planner.primary", "chunk"),
                WorkflowEdge("planner.primary", "responder.primary", "full"),
            ),
        )
        stage_costs = {
            "asr.primary": 100,
            "planner.primary": 200,
            "responder.primary": 300,
        }
        communication_costs = {
            ("asr.primary", "planner.primary"): 10,
            ("planner.primary", "responder.primary"): 20,
        }

        ranks = compute_upward_ranks(workflow, stage_costs, communication_costs)
        self.assertEqual(ranks["responder.primary"], 300)
        self.assertEqual(ranks["planner.primary"], 520)
        self.assertEqual(ranks["asr.primary"], 630)

    def test_compute_dynamic_priority_combines_rank_bonuses_and_penalties(self):
        value = compute_dynamic_priority(
            upward_rank=520.0,
            first_output_bonus=40.0,
            age_bonus=10.0,
            copy_penalty=5.0,
            memory_penalty=7.0,
            thermal_penalty=8.0,
            weights={
                "rank": 1.0,
                "first_output": 2.0,
                "age": 1.5,
                "copy": 3.0,
                "memory": 4.0,
                "thermal": 5.0,
            },
        )
        self.assertAlmostEqual(value, 520.0 + 80.0 + 15.0 - 15.0 - 28.0 - 40.0)


if __name__ == "__main__":
    unittest.main()
