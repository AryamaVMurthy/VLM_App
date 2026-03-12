import json
import tempfile
import unittest
from pathlib import Path

from graphpilot_edge.models import CandidatePlan, StageOption
from graphpilot_edge.plan_bank import PlanBank, load_plan_bank, select_plan_for_state


class GraphPilotPlanBankTest(unittest.TestCase):
    def setUp(self):
        self.fast = CandidatePlan(
            workflow_id="workflow_a",
            stage_options=(
                StageOption("planner.primary", "gemma_fast", "cpu", 10, 128),
            ),
        )
        self.slow = CandidatePlan(
            workflow_id="workflow_a",
            stage_options=(
                StageOption("planner.primary", "gemma_slow", "cpu", 20, 64),
            ),
        )

    def test_load_plan_bank_reads_state_ids_from_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan_bank.json"
            path.write_text(
                json.dumps(
                    {
                        "plan_bank_states": [
                            {"state_id": "cool"},
                            {"state_id": "lowmem"},
                        ]
                    }
                )
            )
            bank = load_plan_bank(path)
            self.assertEqual(bank.state_ids, ("cool", "lowmem"))

    def test_select_plan_for_state_prefers_lowest_latency_by_default(self):
        bank = PlanBank(state_ids=("cool", "hot"))
        selected = select_plan_for_state(bank, "cool", [self.slow, self.fast])
        self.assertEqual(selected.plan_id, self.fast.plan_id)

    def test_select_plan_for_low_memory_state_prefers_lowest_memory(self):
        bank = PlanBank(state_ids=("lowmem",))
        selected = select_plan_for_state(bank, "lowmem", [self.fast, self.slow])
        self.assertEqual(selected.plan_id, self.slow.plan_id)


if __name__ == "__main__":
    unittest.main()
