import importlib.util
import json
import pathlib
import tempfile
import unittest


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "close_graphpilot_backend_gaps.py"
    spec = importlib.util.spec_from_file_location("close_graphpilot_backend_gaps", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CloseGraphPilotBackendGapsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_apply_gap_closures_updates_unverified_backends(self) -> None:
        matrix = {
            "stages": [
                {
                    "stage_id": "asr.primary",
                    "backends": {
                        "cpu": {"status": "feasible_smoke_pass", "evidence": []},
                        "gpu": {"status": "unverified", "evidence": []},
                        "npu": {"status": "unverified", "evidence": []},
                    },
                },
                {
                    "stage_id": "tts.primary",
                    "backends": {
                        "cpu": {"status": "feasible_smoke_pass", "evidence": []},
                        "gpu": {"status": "unverified", "evidence": []},
                        "npu": {"status": "unverified", "evidence": []},
                    },
                },
                {
                    "stage_id": "vlm.fastvlm.primary",
                    "backends": {
                        "cpu": {"status": "feasible_smoke_pass", "evidence": []},
                        "gpu": {"status": "unknown", "evidence": []},
                        "npu": {"status": "known_working", "evidence": []},
                    },
                },
            ]
        }

        updates = self.module.apply_gap_closures(matrix)

        self.assertEqual(matrix["stages"][0]["backends"]["gpu"]["status"], "infeasible_no_backend_adapter")
        self.assertEqual(matrix["stages"][0]["backends"]["npu"]["status"], "infeasible_no_backend_adapter")
        self.assertEqual(matrix["stages"][1]["backends"]["gpu"]["status"], "infeasible_no_backend_adapter")
        self.assertEqual(matrix["stages"][1]["backends"]["npu"]["status"], "infeasible_no_backend_adapter")
        self.assertEqual(matrix["stages"][2]["backends"]["gpu"]["status"], "infeasible_no_backend_adapter")
        self.assertIn("external adb runner path", matrix["stages"][2]["backends"]["npu"]["notes"])
        self.assertIn(("asr.primary", "gpu"), updates)
        self.assertIn(("tts.primary", "npu"), updates)
        self.assertIn(("vlm.fastvlm.primary", "gpu"), updates)

    def test_update_state_ledger_records_phase2_closure(self) -> None:
        ledger = {"completed_tasks": []}
        self.module.update_state_ledger(ledger, [("asr.primary", "gpu"), ("tts.primary", "npu")])
        self.assertEqual(
            ledger["next_immediate_action"],
            "Run GraphPilot experiment baselines and candidate-plan comparisons now that backend feasibility coverage is explicit.",
        )
        self.assertTrue(any(item["id"] == "phase2-closure-001" for item in ledger["completed_tasks"]))

    def test_main_writes_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            matrix_path = root / "matrix.json"
            ledger_path = root / "ledger.json"
            report_path = root / "report.md"
            matrix_path.write_text(
                json.dumps(
                    {
                        "stages": [
                            {
                                "stage_id": "asr.primary",
                                "backends": {
                                    "cpu": {"status": "feasible_smoke_pass", "evidence": []},
                                    "gpu": {"status": "unverified", "evidence": []},
                                    "npu": {"status": "unverified", "evidence": []},
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            ledger_path.write_text(json.dumps({"completed_tasks": []}), encoding="utf-8")
            report_path.write_text("# Phase 2\n", encoding="utf-8")

            rc = self.module.main(
                [
                    "--backend-matrix",
                    str(matrix_path),
                    "--state-ledger",
                    str(ledger_path),
                    "--phase2-report",
                    str(report_path),
                ]
            )

            self.assertEqual(rc, 0)
            updated_matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
            self.assertEqual(
                updated_matrix["stages"][0]["backends"]["gpu"]["status"],
                "infeasible_no_backend_adapter",
            )
            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn("Backend gap closure", report_text)


if __name__ == "__main__":
    unittest.main()
