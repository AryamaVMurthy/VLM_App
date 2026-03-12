import importlib.util
import pathlib
import sys
import tempfile
import unittest


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "ingest_fastvlm_quick_profile.py"
    spec = importlib.util.spec_from_file_location(
        "ingest_fastvlm_quick_profile", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class IngestFastVlmQuickProfileTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_build_registry_entry_maps_summary_to_vlm_npu_profile(self):
        summary = {
            "prefill_mean_s": 0.091024,
            "decode_mean_s": 0.0675235,
            "ttft_mean_s": 0.1027,
            "output_dir": "/tmp/gqa_quick",
        }
        entry = self.module.build_registry_entry(pathlib.Path("/tmp/gqa_quick/summary.json"), summary)
        self.assertEqual(entry["stage_id"], "vlm.fastvlm.primary")
        self.assertEqual(entry["backend"], "npu")
        self.assertEqual(entry["metrics"]["prefill_latency_ms"], 91)
        self.assertEqual(entry["metrics"]["ttft_ms"], 103)


if __name__ == "__main__":
    unittest.main()
