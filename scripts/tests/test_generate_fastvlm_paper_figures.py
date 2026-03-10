import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


def load_module():
    module_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "generate_fastvlm_paper_figures.py"
    )
    spec = importlib.util.spec_from_file_location(
        "generate_fastvlm_paper_figures", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GenerateFastVlmPaperFiguresTest(unittest.TestCase):
    def test_build_variant_comparison_svg_renders_svg(self):
        module = load_module()
        summary = {
            "variants": [
                {
                    "name": "seq",
                    "mode": "sequential",
                    "metrics": {
                        "host_wall_clock_s": 10.0,
                        "throughput_img_per_s_host": 0.9,
                        "first_token_ttft_ms_mean": 100.0,
                    },
                },
                {
                    "name": "ovl",
                    "mode": "overlap",
                    "metrics": {
                        "host_wall_clock_s": 5.0,
                        "throughput_img_per_s_host": 1.8,
                        "first_token_ttft_ms_mean": 140.0,
                    },
                },
            ]
        }

        svg = module.build_variant_comparison_svg(summary)

        self.assertIn("<svg", svg)
        self.assertIn("Stream benchmark comparison", svg)
        self.assertIn("seq", svg)
        self.assertIn("ovl", svg)

    def test_build_partition_map_svg_renders_backends(self):
        module = load_module()
        partition = {
            "TF_LITE_VISION_ADAPTER": {
                "subgraphs": [
                    {"runtime_backend": "CPU_XNNPACK"},
                ]
            },
            "TF_LITE_PREFILL_DECODE": {
                "subgraphs": [
                    {
                        "runtime_backend": "NPU_DISPATCHDELEGATE",
                        "mapping_limitation": "opaque",
                    }
                ]
            },
        }

        svg = module.build_partition_map_svg(partition)

        self.assertIn("<svg", svg)
        self.assertIn("TF_LITE_VISION_ADAPTER", svg)
        self.assertIn("NPU_DISPATCHDELEGATE", svg)

    def test_build_timeline_svg_uses_decode_start_timestamp(self):
        module = load_module()
        summary = {"variants": []}
        payloads = [
            {
                "type": "PREFILL_START",
                "request_id": "request_0",
                "timestamp_unix_nanos": 1000,
            },
            {
                "type": "PREFILL_DONE",
                "request_id": "request_0",
                "timestamp_unix_nanos": 101000000,
            },
            {
                "type": "DECODE_START",
                "request_id": "request_0",
                "timestamp_unix_nanos": 151000000,
            },
            {
                "type": "FIRST_TOKEN",
                "request_id": "request_0",
                "timestamp_unix_nanos": 181000000,
            },
            {
                "type": "DECODE_DONE",
                "request_id": "request_0",
                "timestamp_unix_nanos": 251000000,
                "duration_ms": 100.0,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = pathlib.Path(tmpdir) / "overlap.log"
            log_path.write_text(
                "\n".join(f"VLM_EVENT {json.dumps(item)}" for item in payloads),
                encoding="utf-8",
            )
            svg = module.build_timeline_svg(summary, log_path)

        self.assertIn("<svg", svg)
        self.assertIn("CPU/NPU overlap timeline", svg)
        self.assertNotIn('x="140.0" y="60" width="980.0" height="18" fill="#d95f02"', svg)


if __name__ == "__main__":
    unittest.main()
