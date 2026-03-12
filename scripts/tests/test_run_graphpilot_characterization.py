import importlib.util
import json
import pathlib
import tempfile
import unittest


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "run_graphpilot_characterization.py"
    spec = importlib.util.spec_from_file_location("run_graphpilot_characterization", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RunGraphPilotCharacterizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_build_characterization_summary_includes_baselines_and_ablations(self) -> None:
        summary = self.module.build_characterization_summary()

        self.assertIn("backend_affinity", summary)
        self.assertIn("baseline_comparisons", summary)
        self.assertIn("ablations", summary)
        self.assertIn("transfer_matrix_ms_per_mib", summary)
        self.assertIn("batching_curves", summary)
        self.assertIn("thermal_curves", summary)
        self.assertIn("memory_kv_curves", summary)
        self.assertIn("knob_frontiers", summary)

        compound = summary["baseline_comparisons"]["compound_workloads"]
        workflow_a = next(
            row for row in compound if row["workload_id"] == "compound.workflow_a.default"
        )
        baseline_ids = {row["baseline_id"] for row in workflow_a["baselines"]}
        self.assertIn("cpu_only", baseline_ids)
        self.assertIn("stage_greedy", baseline_ids)
        self.assertIn("static_best_map", baseline_ids)
        self.assertIn("no_pipeline", baseline_ids)

        pipeline_rows = summary["ablations"]["pipeline"]
        self.assertTrue(
            any(row["workload_id"] == "compound.workflow_a.default" for row in pipeline_rows)
        )

    def test_main_writes_summary_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = pathlib.Path(tmp)

            rc = self.module.main(["--output-root", str(output_root)])

            self.assertEqual(rc, 0)
            summaries = list(output_root.glob("graphpilot_characterization_*/summary.json"))
            self.assertEqual(len(summaries), 1)
            payload = json.loads(summaries[0].read_text(encoding="utf-8"))
            self.assertIn("baseline_comparisons", payload)
            self.assertIn("ablations", payload)
            report_path = pathlib.Path(payload["report"])
            self.assertTrue(report_path.exists())
            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn("Baseline comparisons", report_text)
            self.assertIn("Ablations", report_text)


if __name__ == "__main__":
    unittest.main()
