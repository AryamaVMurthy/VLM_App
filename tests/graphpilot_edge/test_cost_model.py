import unittest

from graphpilot_edge.cost_model import (
    ObjectiveWeights,
    compute_contention_factor,
    compute_energy_mj,
    compute_execution_time_ms,
    compute_objective_score,
    compute_transfer_cost_ms,
    evaluate_hybrid_prefill_decode,
    kv_cache_size_bytes,
)


class GraphPilotCostModelTest(unittest.TestCase):
    def test_compute_execution_time_uses_thermal_contention_and_compile(self):
        value = compute_execution_time_ms(
            base_latency_ms=100.0,
            thermal_factor=1.1,
            contention_factor=1.2,
            compile_cost_ms=20.0,
            cold=True,
        )
        self.assertAlmostEqual(value, 152.0)

    def test_compute_transfer_cost_covers_same_backend_zero_copy_and_copy(self):
        self.assertEqual(
            compute_transfer_cost_ms(
                size_bytes=4096,
                src_backend="cpu",
                dst_backend="cpu",
                same_memory=True,
                bandwidth_bytes_per_ms=1024.0,
                fixed_overhead_ms=1.0,
                layout_ms=2.0,
            ),
            0.0,
        )
        self.assertEqual(
            compute_transfer_cost_ms(
                size_bytes=4096,
                src_backend="cpu",
                dst_backend="gpu",
                zero_copy=True,
                map_cost_ms=0.75,
                bandwidth_bytes_per_ms=1024.0,
                fixed_overhead_ms=1.0,
                layout_ms=2.0,
            ),
            0.75,
        )
        self.assertAlmostEqual(
            compute_transfer_cost_ms(
                size_bytes=4096,
                src_backend="cpu",
                dst_backend="npu",
                bandwidth_bytes_per_ms=1024.0,
                fixed_overhead_ms=1.0,
                layout_ms=2.0,
            ),
            7.0,
        )

    def test_compute_contention_factor_matches_lambda_dot_utilization(self):
        value = compute_contention_factor(
            backend="npu",
            sensitivities={
                ("npu", "cpu"): 0.2,
                ("npu", "gpu"): 0.4,
            },
            utilizations={"cpu": 0.5, "gpu": 0.25},
        )
        self.assertAlmostEqual(value, 1.2)

    def test_energy_kv_and_hybrid_helpers_match_design_examples(self):
        self.assertAlmostEqual(compute_energy_mj(avg_power_mw=2000.0, time_ms=500.0), 1000.0)
        self.assertEqual(
            kv_cache_size_bytes(layers=2, kv_heads=4, head_dim=8, cached_tokens=16, bytes_per_element=2),
            4096,
        )
        long_answer = evaluate_hybrid_prefill_decode(
            prefill_npu_ms=180.0,
            prefill_alt_ms=320.0,
            decode_npu_per_token_ms=9.0,
            decode_alt_per_token_ms=7.0,
            output_tokens=50,
            kv_migration_ms=80.0,
        )
        self.assertTrue(long_answer["beats_all_alt"])
        self.assertTrue(long_answer["beats_all_npu"])
        short_answer = evaluate_hybrid_prefill_decode(
            prefill_npu_ms=180.0,
            prefill_alt_ms=320.0,
            decode_npu_per_token_ms=9.0,
            decode_alt_per_token_ms=7.0,
            output_tokens=10,
            kv_migration_ms=80.0,
        )
        self.assertTrue(short_answer["beats_all_alt"])
        self.assertFalse(short_answer["beats_all_npu"])

    def test_compute_objective_score_applies_weights(self):
        score = compute_objective_score(
            p95_e2e_ms=1000.0,
            p95_ttfs_ms=800.0,
            avg_energy_mj=200.0,
            peak_memory_bytes=100 * 1024 * 1024,
            copy_bytes=10 * 1024 * 1024,
            quality_loss=0.05,
            weights=ObjectiveWeights(
                alpha=1.0,
                beta=2.0,
                gamma=0.5,
                delta=1e-6,
                eta=1e-6,
                zeta=1000.0,
            ),
        )
        self.assertGreater(score, 2865.0)
        self.assertLess(score, 2866.0)


if __name__ == "__main__":
    unittest.main()
