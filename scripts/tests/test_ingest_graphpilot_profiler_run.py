import importlib.util
import pathlib
import sys
import tempfile
import unittest


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "ingest_graphpilot_profiler_run.py"
    spec = importlib.util.spec_from_file_location(
        "ingest_graphpilot_profiler_run", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class IngestGraphPilotProfilerRunTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_summarize_profile_coverage_counts_stage_backend_pairs(self):
        summary = {
            "profiles": [
                {"stage_id": "planner.primary", "backend": "cpu"},
                {"stage_id": "workflow_b_voice_vision", "backend": "mixed"},
            ]
        }
        coverage = self.module.summarize_profile_coverage(summary)
        self.assertEqual(coverage["profile_count"], 2)
        self.assertIn("planner.primary@cpu", coverage["covered_pairs"])

    def test_write_phase3_report_mentions_graphpilot_workflows(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path = pathlib.Path(tmp) / "phase3.md"
            original = self.module.PHASE3_REPORT_PATH
            self.module.PHASE3_REPORT_PATH = report_path
            try:
                summary_path = pathlib.Path("/tmp/profile_run/summary.json")
                coverage = {
                    "profile_count": 2,
                    "covered_pairs": ["workflow_a_voice_only@mixed", "workflow_b_voice_vision@mixed"],
                }
                self.module.write_phase3_report(summary_path, coverage)
                text = report_path.read_text(encoding="utf-8")
                self.assertIn("workflow_b_voice_vision@mixed", text)
                self.assertIn("Phase 3 Profiler Status", text)
            finally:
                self.module.PHASE3_REPORT_PATH = original


if __name__ == "__main__":
    unittest.main()
