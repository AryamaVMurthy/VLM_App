import unittest

from graphpilot_edge.cost_model import ObjectiveWeights, SimulationCalibration
from graphpilot_edge.models import CandidatePlan, RequestSpec, StageOption
from graphpilot_edge.simulation import simulate_candidate_plan, simulate_request_stream
from graphpilot_edge.workflow import WorkflowDag, WorkflowEdge


class DiscreteEventSimulationTest(unittest.TestCase):
    def setUp(self):
        self.workflow = WorkflowDag(
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
        )
        self.plan = CandidatePlan(
            workflow_id="voice_vision",
            stage_options=(
                StageOption("asr.primary", "whisper_tiny", "cpu", 2, 64),
                StageOption("planner.primary", "gemma_fast", "npu", 3, 256),
                StageOption("vlm.fastvlm.primary", "fastvlm", "npu", 5, 512),
                StageOption("retrieval.embedder.primary", "embedder", "cpu", 4, 128),
                StageOption("responder.primary", "gemma_response", "cpu", 1, 256),
            ),
        )

    def test_simulation_emits_stage_events_and_makespan(self):
        result = simulate_candidate_plan(
            self.workflow,
            self.plan,
            resource_capacities={"cpu": 1, "npu": 1},
        )

        self.assertEqual(result.makespan_ms, 11)
        self.assertEqual(
            [event.stage_id for event in result.events if event.event_type == "start"],
            [
                "asr.primary",
                "planner.primary",
                "vlm.fastvlm.primary",
                "retrieval.embedder.primary",
                "responder.primary",
            ],
        )
        self.assertEqual(result.stage_timings["vlm.fastvlm.primary"].start_ms, 5)
        self.assertEqual(result.stage_timings["retrieval.embedder.primary"].finish_ms, 9)

    def test_simulation_requires_explicit_resource_capacities(self):
        with self.assertRaisesRegex(ValueError, "Missing explicit resource capacity"):
            simulate_candidate_plan(
                self.workflow,
                self.plan,
                resource_capacities={"cpu": 1},
            )

    def test_simulation_accounts_for_copy_bytes_and_objective_score(self):
        plan = CandidatePlan(
            workflow_id="voice_vision",
            stage_options=(
                StageOption("asr.primary", "whisper_tiny", "cpu", 2, 64, output_bytes=4_096, average_power_mw=500.0),
                StageOption("planner.primary", "gemma_fast", "npu", 3, 256, output_bytes=8_192, average_power_mw=900.0),
                StageOption("vlm.fastvlm.primary", "fastvlm", "npu", 5, 512, output_bytes=12_288, average_power_mw=1_200.0),
                StageOption("retrieval.embedder.primary", "embedder", "cpu", 4, 128, output_bytes=2_048, average_power_mw=400.0),
                StageOption("responder.primary", "gemma_response", "cpu", 1, 256, output_bytes=1_024, average_power_mw=700.0),
            ),
        )
        result = simulate_candidate_plan(
            self.workflow,
            plan,
            resource_capacities={"cpu": 1, "npu": 1},
            bandwidth_bytes_per_ms=1024.0,
            transfer_fixed_overhead_ms=1.0,
            transfer_layout_ms=1.0,
            objective_weights=ObjectiveWeights(alpha=1.0, beta=0.0, gamma=1.0, delta=1e-6, eta=1e-6, zeta=0.0),
        )
        self.assertGreater(result.copy_bytes, 0)
        self.assertGreater(result.copy_time_ms, 0.0)
        self.assertGreater(result.energy_mj, 0.0)
        self.assertGreater(result.peak_memory_bytes, 0)
        self.assertIsNotNone(result.objective_score)

    def test_simulation_applies_calibration_overhead_and_thermal_scaling(self):
        baseline = simulate_candidate_plan(
            self.workflow,
            self.plan,
            resource_capacities={"cpu": 1, "npu": 1},
        )
        calibrated = simulate_candidate_plan(
            self.workflow,
            self.plan,
            resource_capacities={"cpu": 1, "npu": 1},
            calibration=SimulationCalibration(
                orchestration_overhead_ms=5.0,
                workflow_thermal_scale=2.0,
            ),
        )

        self.assertEqual(baseline.makespan_ms, 11)
        self.assertEqual(calibrated.makespan_ms, 27)
        self.assertGreater(calibrated.ttft_ms, baseline.ttft_ms)

    def test_stream_simulation_reports_queue_delay_and_deadline_miss_rate(self):
        requests = (
            RequestSpec(
                request_id="req0",
                arrival_ms=0,
                workflow=self.workflow,
                plan=self.plan,
                deadline_ms=10,
            ),
            RequestSpec(
                request_id="req1",
                arrival_ms=1,
                workflow=self.workflow,
                plan=self.plan,
                deadline_ms=8,
            ),
        )

        result = simulate_request_stream(
            requests,
            resource_capacities={"cpu": 1, "npu": 1},
        )

        self.assertEqual(result.request_results["req0"].queue_delay_ms, 0)
        self.assertGreater(result.request_results["req1"].queue_delay_ms, 0)
        self.assertGreater(result.p95_queue_delay_ms, 0)
        self.assertGreater(result.deadline_miss_rate, 0.0)
        self.assertGreater(result.makespan_ms, result.request_results["req0"].makespan_ms)

    def test_stream_simulation_aggregates_copy_bytes_and_peak_memory(self):
        copy_plan = CandidatePlan(
            workflow_id="voice_vision",
            stage_options=(
                StageOption("asr.primary", "whisper_tiny", "cpu", 2, 64, output_bytes=4_096, average_power_mw=500.0),
                StageOption("planner.primary", "gemma_fast", "npu", 3, 256, output_bytes=8_192, average_power_mw=900.0),
                StageOption("vlm.fastvlm.primary", "fastvlm", "npu", 5, 512, output_bytes=12_288, average_power_mw=1_200.0),
                StageOption("retrieval.embedder.primary", "embedder", "cpu", 4, 128, output_bytes=2_048, average_power_mw=400.0),
                StageOption("responder.primary", "gemma_response", "cpu", 1, 256, output_bytes=1_024, average_power_mw=700.0),
            ),
        )
        requests = (
            RequestSpec(request_id="req0", arrival_ms=0, workflow=self.workflow, plan=copy_plan, deadline_ms=10),
            RequestSpec(request_id="req1", arrival_ms=0, workflow=self.workflow, plan=copy_plan, deadline_ms=5),
        )

        result = simulate_request_stream(
            requests,
            resource_capacities={"cpu": 1, "npu": 1},
            bandwidth_bytes_per_ms=1024.0,
            transfer_fixed_overhead_ms=1.0,
            transfer_layout_ms=1.0,
            objective_weights=ObjectiveWeights(alpha=1.0, beta=0.25, gamma=1.0, delta=1e-6, eta=1e-6, zeta=0.0, xi=2.0, psi=50.0),
        )

        self.assertGreater(result.copy_bytes, 0)
        self.assertGreater(result.copy_time_ms, 0.0)
        self.assertGreater(result.peak_memory_bytes, 0)
        self.assertGreater(result.energy_mj, 0.0)
        self.assertIsNotNone(result.objective_score)
        self.assertGreater(result.p95_queue_delay_ms, 0.0)
        self.assertGreater(result.deadline_miss_rate, 0.0)

    def test_stream_simulation_objective_includes_queue_delay_and_deadline_miss(self):
        requests = (
            RequestSpec(
                request_id="req0",
                arrival_ms=0,
                workflow=self.workflow,
                plan=self.plan,
                deadline_ms=10,
            ),
            RequestSpec(
                request_id="req1",
                arrival_ms=0,
                workflow=self.workflow,
                plan=self.plan,
                deadline_ms=5,
            ),
        )

        baseline = simulate_request_stream(
            requests,
            resource_capacities={"cpu": 1, "npu": 1},
            objective_weights=ObjectiveWeights(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0, eta=0.0, zeta=0.0, xi=0.0, psi=0.0),
        )
        penalized = simulate_request_stream(
            requests,
            resource_capacities={"cpu": 1, "npu": 1},
            objective_weights=ObjectiveWeights(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0, eta=0.0, zeta=0.0, xi=10.0, psi=500.0),
        )

        self.assertIsNotNone(baseline.objective_score)
        self.assertIsNotNone(penalized.objective_score)
        self.assertGreater(penalized.objective_score, baseline.objective_score)


if __name__ == "__main__":
    unittest.main()
