import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


def load_module():
    module_path = (
        pathlib.Path(__file__).resolve().parents[1] / "ingest_graphpilot_stage_feasibility.py"
    )
    spec = importlib.util.spec_from_file_location(
        "ingest_graphpilot_stage_feasibility", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class IngestGraphPilotStageFeasibilityTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_update_backend_matrix_marks_stage_backend_with_evidence(self):
        matrix = {
            "stages": [
                {
                    "stage_id": "tts.primary",
                    "backends": {"cpu": {"status": "unverified", "evidence": []}},
                }
            ]
        }
        summary_path = pathlib.Path("/tmp/stage_feasibility_20260310_190754/summary.json")
        summary = {
            "generated_at": "2026-03-10T19:07:54+00:00",
            "stage_runs": [
                {
                    "stage_key": "tts",
                    "stage_id": "tts.primary",
                    "verdict": "pass",
                    "summary": "OK (1 test)",
                    "stdout_log": "/tmp/tts/stdout.log",
                    "stderr_log": "/tmp/tts/stderr.log",
                    "test_class": "com.qidk.fastvlm.speech.AndroidTtsSpeakerInstrumentedTest",
                }
            ],
        }

        self.module.update_backend_matrix(matrix, summary_path, summary)
        cpu_state = matrix["stages"][0]["backends"]["cpu"]
        self.assertEqual(cpu_state["status"], "feasible_smoke_pass")
        self.assertEqual(cpu_state["last_verdict"], "pass")
        self.assertIn(str(summary_path), cpu_state["evidence"])

    def test_write_phase2_report_contains_stage_verdicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path = pathlib.Path(tmp) / "phase2.md"
            original = self.module.PHASE2_REPORT_PATH
            self.module.PHASE2_REPORT_PATH = report_path
            try:
                summary_path = pathlib.Path("/tmp/stage_feasibility/summary.json")
                summary = {
                    "output_dir": "/tmp/stage_feasibility",
                    "stage_runs": [
                        {
                            "stage_id": "tts.primary",
                            "stage_key": "tts",
                            "verdict": "pass",
                            "test_class": "AndroidTtsSpeakerInstrumentedTest",
                            "stdout_log": "/tmp/stdout.log",
                            "stderr_log": "/tmp/stderr.log",
                            "logcat_log": "/tmp/logcat.log",
                            "summary": "OK (1 test)",
                        }
                    ],
                }
                backend_matrix = {
                    "stages": [
                        {
                            "stage_id": "tts.primary",
                            "backends": {
                                "cpu": {
                                    "status": "feasible_smoke_pass",
                                    "last_verdict": "pass",
                                }
                            },
                        }
                    ]
                }
                self.module.write_phase2_report(summary_path, summary, backend_matrix)
                text = report_path.read_text(encoding="utf-8")
                self.assertIn("`tts.primary` / `tts`", text)
                self.assertIn("`/tmp/logcat.log`", text)
                self.assertIn("Root causes observed", text)
            finally:
                self.module.PHASE2_REPORT_PATH = original

    def test_update_backend_matrix_sets_failure_notes_for_retrieval_gpu(self):
        matrix = {
            "stages": [
                {
                    "stage_id": "retrieval.embedder.primary",
                    "backends": {"gpu": {"status": "unverified", "evidence": []}},
                }
            ]
        }
        summary_path = pathlib.Path("/tmp/stage_feasibility_20260311_054938/summary.json")
        summary = {
            "generated_at": "2026-03-11T05:49:38+00:00",
            "stage_runs": [
                {
                    "stage_key": "retrieval_gpu",
                    "stage_id": "retrieval.embedder.primary",
                    "verdict": "fail",
                    "summary": "FAILURES!!!",
                    "stdout_log": "/tmp/retrieval_gpu/stdout.log",
                    "stderr_log": "/tmp/retrieval_gpu/stderr.log",
                    "logcat_log": "/tmp/retrieval_gpu/logcat.log",
                    "test_class": "com.qidk.fastvlm.graphpilot.GraphPilotRetrievalBridgeInstrumentedTest#gpuQueryReturnsJfkHit",
                }
            ],
        }

        self.module.update_backend_matrix(matrix, summary_path, summary)
        gpu_state = matrix["stages"][0]["backends"]["gpu"]
        self.assertEqual(gpu_state["status"], "infeasible_smoke_fail")
        self.assertIn("accelerator registration/compile failed", gpu_state["notes"])
        self.assertIn("/tmp/retrieval_gpu/logcat.log", gpu_state["evidence"])


if __name__ == "__main__":
    unittest.main()
