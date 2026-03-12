import importlib.util
import pathlib
import sys
import tempfile
import unittest


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "ingest_graphpilot_retrieval_probe.py"
    spec = importlib.util.spec_from_file_location(
        "ingest_graphpilot_retrieval_probe", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class IngestGraphPilotRetrievalProbeTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_update_backend_matrix_marks_retrieval_backends_missing_artifact(self):
        matrix = {
            "stages": [
                {
                    "stage_id": "retrieval.embedder.primary",
                    "backends": {
                        "cpu": {"status": "unverified", "evidence": []},
                        "gpu": {"status": "unverified", "evidence": []},
                        "npu": {"status": "unverified", "evidence": []},
                    },
                }
            ]
        }
        summary_path = pathlib.Path("/tmp/retrieval_artifact_probe/summary.json")
        summary = {
            "generated_at": "2026-03-10T19:38:57+00:00",
            "expected_stage": "retrieval.embedder.primary",
            "verdict": "missing_artifact",
            "remediation": "stage artifact",
        }

        self.module.update_backend_matrix(matrix, summary_path, summary)

        for backend in ("cpu", "gpu", "npu"):
            state = matrix["stages"][0]["backends"][backend]
            self.assertEqual(state["status"], "infeasible_missing_artifact")
            self.assertEqual(state["last_verdict"], "missing_artifact")
            self.assertIn(str(summary_path), state["evidence"])

    def test_write_phase2_report_mentions_retrieval_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path = pathlib.Path(tmp) / "phase2.md"
            original = self.module.PHASE2_REPORT_PATH
            self.module.PHASE2_REPORT_PATH = report_path
            try:
                summary_path = pathlib.Path("/tmp/retrieval_artifact_probe/summary.json")
                summary = {
                    "generated_at": "2026-03-10T19:38:57+00:00",
                    "expected_stage": "retrieval.embedder.primary",
                    "verdict": "missing_artifact",
                    "remediation": "stage artifact",
                }
                backend_matrix = {
                    "stages": [
                        {
                            "stage_id": "retrieval.embedder.primary",
                            "backends": {
                                "cpu": {"status": "infeasible_missing_artifact"},
                                "gpu": {"status": "infeasible_missing_artifact"},
                                "npu": {"status": "infeasible_missing_artifact"},
                            },
                        }
                    ]
                }
                self.module.write_phase2_report(summary_path, summary, backend_matrix)
                text = report_path.read_text(encoding="utf-8")
                self.assertIn("retrieval.embedder.primary", text)
                self.assertIn("missing_artifact", text)
            finally:
                self.module.PHASE2_REPORT_PATH = original


if __name__ == "__main__":
    unittest.main()
