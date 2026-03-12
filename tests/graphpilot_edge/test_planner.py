import unittest

from graphpilot_edge.cost_model import ObjectiveWeights, SimulationCalibration
from graphpilot_edge.models import StageOption
from graphpilot_edge.planner import build_stage_options, enumerate_and_rank_plans
from graphpilot_edge.profiles import StageProfile
from graphpilot_edge.workflow import WorkflowDag, WorkflowEdge


class GraphPilotPlannerTest(unittest.TestCase):
    def setUp(self):
        self.workflow = WorkflowDag(
            workflow_id="workflow_a",
            stage_ids=("asr.primary", "planner.primary"),
            edges=(WorkflowEdge("asr.primary", "planner.primary", "chunk"),),
        )
        self.matrix = {
            "stages": [
                {
                    "stage_id": "asr.primary",
                    "backends": {
                        "cpu": {"status": "feasible_smoke_pass"},
                        "gpu": {"status": "unverified"},
                    },
                },
                {
                    "stage_id": "planner.primary",
                    "backends": {
                        "cpu": {"status": "feasible_smoke_pass"},
                        "npu": {"status": "known_working"},
                    },
                },
            ]
        }
        self.profiles = {
            ("asr.primary", "cpu"): StageProfile("asr.primary", "cpu", "whisper", 200, 300, 200, 10 * 1024 * 1024),
            ("planner.primary", "cpu"): StageProfile("planner.primary", "cpu", "gemma", 400, 600, 400, 20 * 1024 * 1024),
            ("planner.primary", "npu"): StageProfile("planner.primary", "npu", "gemma", 250, 500, 250, 24 * 1024 * 1024),
        }

    def test_build_stage_options_uses_only_feasible_backends_with_profiles(self):
        options = build_stage_options(self.workflow, self.matrix, self.profiles)
        self.assertEqual([option.backend for option in options["asr.primary"]], ["cpu"])
        self.assertEqual([option.backend for option in options["planner.primary"]], ["cpu", "npu"])

    def test_enumerate_and_rank_plans_prefers_lower_makespan(self):
        options = {
            "asr.primary": (StageOption("asr.primary", "whisper", "cpu", 200, 10),),
            "planner.primary": (
                StageOption("planner.primary", "gemma", "cpu", 400, 20),
                StageOption("planner.primary", "gemma", "npu", 250, 24),
            ),
        }
        ranked = enumerate_and_rank_plans(self.workflow, options, resource_capacities={"cpu": 1, "npu": 1})
        self.assertEqual(ranked[0].plan.option_for_stage("planner.primary").backend, "npu")
        self.assertLess(ranked[0].score_makespan_ms, ranked[1].score_makespan_ms)

    def test_enumerate_and_rank_plans_exposes_objective_score(self):
        options = {
            "asr.primary": (StageOption("asr.primary", "whisper", "cpu", 200, 10, average_power_mw=200.0),),
            "planner.primary": (
                StageOption("planner.primary", "gemma_cpu", "cpu", 300, 20, average_power_mw=150.0),
                StageOption("planner.primary", "gemma_npu", "npu", 250, 40, average_power_mw=300.0),
            ),
        }
        ranked = enumerate_and_rank_plans(
            self.workflow,
            options,
            resource_capacities={"cpu": 1, "npu": 1},
            objective_weights=ObjectiveWeights(alpha=1.0, beta=0.0, gamma=0.0, delta=1e-3, eta=0.0, zeta=0.0),
        )
        self.assertIsNotNone(ranked[0].objective_score)
        self.assertIsNotNone(ranked[0].simulation_result)

    def test_enumerate_and_rank_plans_applies_simulation_calibration(self):
        options = {
            "asr.primary": (StageOption("asr.primary", "whisper", "cpu", 200, 10),),
            "planner.primary": (StageOption("planner.primary", "gemma", "npu", 250, 24),),
        }
        ranked = enumerate_and_rank_plans(
            self.workflow,
            options,
            resource_capacities={"cpu": 1, "npu": 1},
            simulation_calibration=SimulationCalibration(
                orchestration_overhead_ms=50.0,
                workflow_thermal_scale=1.5,
            ),
        )
        self.assertEqual(ranked[0].score_makespan_ms, 575)


if __name__ == "__main__":
    unittest.main()
