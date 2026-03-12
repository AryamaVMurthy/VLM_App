import math
import unittest

from graphpilot_edge.hardware_simulator import (
    HardwareInstance,
    HardwarePrediction,
    HardwareSimulator,
    PartitionAssignment,
    ResourceQueueState,
    TaskProfile,
    compute_batching_gain,
    compute_thermal_slowdown,
    evolve_thermal_state,
)


class HardwareSimulatorTest(unittest.TestCase):
    def setUp(self):
        self.cpu = HardwareInstance(
            resource_id="cpu0",
            resource_type="cpu",
            supported_op_classes=("gemm", "attn", "control"),
            service_rates={"gemm": 200.0, "attn": 100.0, "control": 50.0},
            memory_bandwidth_bytes_per_ms=2_000.0,
            launch_overhead_ms=1.0,
            busy_power_mw=2_000.0,
            idle_power_mw=200.0,
            queue_depth=2,
            batching_beta=0.2,
            batching_tau=2.0,
            thermal_time_constant_ms=10_000.0,
            thermal_resistance_c_per_mw=0.005,
            thermal_threshold_c=40.0,
            thermal_gamma=0.05,
            contention_domain="shared_mem",
        )
        self.npu = HardwareInstance(
            resource_id="npu0",
            resource_type="npu",
            supported_op_classes=("gemm", "attn"),
            service_rates={"gemm": 1000.0, "attn": 500.0},
            memory_bandwidth_bytes_per_ms=5_000.0,
            launch_overhead_ms=2.0,
            busy_power_mw=1_500.0,
            idle_power_mw=150.0,
            queue_depth=1,
            batching_beta=0.6,
            batching_tau=4.0,
            thermal_time_constant_ms=8_000.0,
            thermal_resistance_c_per_mw=0.004,
            thermal_threshold_c=42.0,
            thermal_gamma=0.1,
            contention_domain="shared_mem",
        )
        self.simulator = HardwareSimulator(
            resources=(self.cpu, self.npu),
            transfer_bandwidth_bytes_per_ms={
                ("cpu0", "npu0"): 1_000.0,
                ("npu0", "cpu0"): 1_000.0,
            },
            transfer_fixed_overhead_ms={
                ("cpu0", "npu0"): 0.5,
                ("npu0", "cpu0"): 0.5,
            },
            transfer_layout_overhead_ms={
                ("cpu0", "npu0"): 0.25,
                ("npu0", "cpu0"): 0.25,
            },
            contention_sensitivities={
                ("cpu0", "npu0"): 0.2,
                ("npu0", "cpu0"): 0.3,
            },
            ambient_temperature_c=30.0,
        )

    def test_compute_batching_gain_matches_formula(self):
        observed = compute_batching_gain(batch_size=4, beta=0.5, tau=2.0)
        expected = 1.0 + 0.5 * (1.0 - math.exp(-2.0))
        self.assertAlmostEqual(observed, expected)

    def test_evolve_thermal_state_and_slowdown_follow_formula(self):
        updated = evolve_thermal_state(
            current_temp_c=35.0,
            ambient_temp_c=30.0,
            delta_ms=2_000.0,
            time_constant_ms=4_000.0,
            thermal_resistance_c_per_mw=0.01,
            power_mw=1_000.0,
        )
        expected = 30.0 + (35.0 - 30.0) * math.exp(-0.5) + 0.01 * 1_000.0 * (1.0 - math.exp(-0.5))
        self.assertAlmostEqual(updated, expected)
        self.assertAlmostEqual(compute_thermal_slowdown(temp_c=45.0, threshold_c=40.0, gamma=0.1), 1.5)

    def test_predict_task_execution_uses_compute_memory_thermal_and_contention_terms(self):
        task = TaskProfile(
            task_id="vlm.prefill",
            op_volume={"gemm": 4_000.0, "attn": 1_000.0},
            memory_bytes=12_000,
            input_bytes=4_096,
            output_bytes=8_192,
        )

        prediction = self.simulator.predict_task(
            task=task,
            resource_id="npu0",
            batch_size=4,
            current_temp_c={"npu0": 47.0},
            utilizations={"cpu0": 0.5},
        )

        base_compute = ((4_000.0 / 1000.0) + (1_000.0 / 500.0)) / compute_batching_gain(4, 0.6, 4.0)
        base_memory = 12_000 / 5_000.0
        expected_exec = 2.0 + max(base_compute, base_memory) * 1.5 * 1.15
        self.assertIsInstance(prediction, HardwarePrediction)
        self.assertAlmostEqual(prediction.compute_time_ms, base_compute)
        self.assertAlmostEqual(prediction.memory_time_ms, base_memory)
        self.assertAlmostEqual(prediction.execution_time_ms, expected_exec)
        self.assertGreater(prediction.energy_mj, 0.0)

    def test_predict_partitioned_task_charges_fallback_and_transfer(self):
        task = TaskProfile(
            task_id="responder.decode_region",
            op_volume={"gemm": 4_000.0, "control": 100.0},
            memory_bytes=8_000,
            input_bytes=2_048,
            output_bytes=1_024,
            fallback_partitions=(
                PartitionAssignment(
                    partition_id="supported_head",
                    resource_id="npu0",
                    op_volume={"gemm": 2_000.0},
                    memory_bytes=4_000,
                    output_bytes=1_500,
                ),
                PartitionAssignment(
                    partition_id="unsupported_middle",
                    resource_id="cpu0",
                    op_volume={"control": 100.0},
                    memory_bytes=2_000,
                    output_bytes=1_000,
                ),
                PartitionAssignment(
                    partition_id="supported_tail",
                    resource_id="npu0",
                    op_volume={"gemm": 1_000.0},
                    memory_bytes=2_000,
                    output_bytes=512,
                ),
            ),
        )

        prediction = self.simulator.predict_task(
            task=task,
            resource_id="npu0",
            batch_size=1,
            current_temp_c={"cpu0": 35.0, "npu0": 35.0},
            utilizations={},
        )

        self.assertTrue(prediction.used_fallback)
        self.assertGreater(prediction.transfer_time_ms, 0.0)
        self.assertGreater(prediction.execution_time_ms, prediction.transfer_time_ms)

    def test_predict_earliest_finish_respects_resource_free_time_and_queue_depth(self):
        task = TaskProfile(
            task_id="planner.primary",
            op_volume={"gemm": 1_000.0},
            memory_bytes=2_000,
            input_bytes=256,
            output_bytes=256,
        )

        start_ms, finish_ms = self.simulator.predict_earliest_finish(
            task=task,
            resource_id="cpu0",
            arrival_ms=5.0,
            queue_state=ResourceQueueState(free_at_ms=12.0, queued_tasks=1),
            batch_size=1,
            current_temp_c={"cpu0": 35.0},
            utilizations={},
        )

        self.assertEqual(start_ms, 12.0)
        self.assertGreater(finish_ms, start_ms)

        with self.assertRaisesRegex(ValueError, "queue depth"):
            self.simulator.predict_earliest_finish(
                task=task,
                resource_id="npu0",
                arrival_ms=0.0,
                queue_state=ResourceQueueState(free_at_ms=0.0, queued_tasks=1),
                batch_size=1,
                current_temp_c={"npu0": 35.0},
                utilizations={},
            )


if __name__ == "__main__":
    unittest.main()
