import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


def load_module():
    module_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "fastvlm_overlap_stream_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location(
        "fastvlm_overlap_stream_benchmark", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FastVlmOverlapStreamBenchmarkTest(unittest.TestCase):
    def test_parse_battery_dumpsys_marks_ac_power_unavailable(self):
        module = load_module()

        parsed = module.parse_battery_dumpsys(
            "\n".join(
                [
                    "Current Battery Service state:",
                    "  AC powered: true",
                    "  Charge counter: 2371000",
                    "  voltage: 12755",
                ]
            )
        )

        self.assertFalse(parsed["power_estimate_available"])
        self.assertEqual(
            parsed["power_estimate_unavailable_reason"], "device_ac_powered"
        )

    def test_parse_overlap_metrics_extracts_ttft_and_handoff_stats(self):
        module = load_module()
        payloads = [
            {
                "type": "PREFILL_START",
                "request_id": "q1",
                "timestamp_unix_nanos": 1000,
            },
            {
                "type": "PREFILL_DONE",
                "request_id": "q1",
                "duration_ms": 200.0,
                "timestamp_unix_nanos": 201000000,
            },
            {
                "type": "FIRST_TOKEN",
                "request_id": "q1",
                "ttft_ms": 260.0,
                "decode_first_token_latency_ms": 60.0,
                "timestamp_unix_nanos": 261000000,
                "text": "A ",
            },
            {
                "type": "DECODE_DONE",
                "request_id": "q1",
                "duration_ms": 400.0,
                "timestamp_unix_nanos": 601000000,
            },
            {
                "type": "HANDOFF_EXPORT",
                "request_id": "q1",
                "kv_cache_buffer_count": 8,
                "kv_cache_total_bytes": 4096,
                "processed_token_count": 12,
            },
            {
                "type": "HANDOFF_IMPORT",
                "request_id": "q1",
                "kv_cache_buffer_count": 8,
                "kv_cache_total_bytes": 4096,
            },
            {
                "type": "PROCESS_CPU_SUMMARY",
                "cpu_summary_available": True,
                "avg_cpu_util_percent_total_capacity": 12.5,
            },
            {
                "type": "OVERLAP_RESPONSE",
                "request_id": "q1",
                "text": "A forest scene.",
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = pathlib.Path(tmpdir) / "run.log"
            log_path.write_text(
                "\n".join(f"VLM_EVENT {json.dumps(item)}" for item in payloads),
                encoding="utf-8",
            )

            parsed = module.parse_overlap_metrics(log_path)

        self.assertAlmostEqual(parsed["event_window_s"], 0.601, places=3)
        self.assertAlmostEqual(parsed["first_token_ttft_ms_mean"], 260.0)
        self.assertAlmostEqual(parsed["first_token_decode_latency_ms_mean"], 60.0)
        self.assertAlmostEqual(parsed["serial_stage_sum_s"], 0.6)
        self.assertAlmostEqual(parsed["overlap_pipeline_speedup"], 0.6 / 0.601, places=5)
        self.assertEqual(parsed["handoff_export_buffer_count_total"], 8)
        self.assertEqual(parsed["handoff_export_total_bytes"], 4096)
        self.assertEqual(parsed["handoff_import_event_count"], 1)
        self.assertEqual(parsed["processed_token_count_total"], 12)
        self.assertEqual(parsed["response_count"], 1)
        self.assertEqual(parsed["example_outputs"], ["A forest scene."])

    def test_build_variants_includes_adaptive_overlap_variant(self):
        module = load_module()

        variants = module.build_variants(64, 48)
        adaptive = [item for item in variants if "adaptive" in item.name]

        self.assertEqual(len(variants), 4)
        self.assertEqual(len(adaptive), 1)
        self.assertEqual(adaptive[0].adaptive_budgets, "32,64,128")

    def test_summarize_proc_stat_delta(self):
        module = load_module()

        summary = module.summarize_proc_stat_delta(
            {"busy": 100, "idle": 100, "iowait": 10, "total": 230},
            {"busy": 160, "idle": 140, "iowait": 12, "total": 332},
        )

        self.assertTrue(summary["available"])
        self.assertAlmostEqual(summary["busy_percent"], 58.8235294118)
        self.assertAlmostEqual(summary["idle_percent"], 39.2156862745)
        self.assertAlmostEqual(summary["iowait_percent"], 1.9607843137)


if __name__ == "__main__":
    unittest.main()
