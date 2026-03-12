import importlib.util
import json
import pathlib
import tempfile
import unittest


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "build_graphpilot_checkpoint.py"
    spec = importlib.util.spec_from_file_location("build_graphpilot_checkpoint", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BuildGraphPilotCheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_main_writes_checkpoint_summary_with_pinned_paths_and_beads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            paths = {}
            for name in (
                "backend_matrix",
                "profiler_registry",
                "candidate_plans",
                "baseline_registry",
                "workload_registry",
                "experiment_registry",
                "plot_registry",
                "state_ledger",
                "experiment_summary",
                "sustained_summary",
                "calibration_summary",
                "tuning_summary",
                "characterization_summary",
                "memory_admission_summary",
            ):
                path = root / f"{name}.json"
                path.write_text(json.dumps({"name": name}), encoding="utf-8")
                paths[name] = path
            truth_pdf = root / "graphpilot_edge_revision_report.pdf"
            truth_pdf.write_bytes(b"%PDF-1.7\n%test\n")
            open_beads = root / "open_beads.json"
            open_beads.write_text(
                json.dumps(
                    [
                        {"id": "fvlm-i6n.11", "title": "Implement fair baseline suite"},
                        {"id": "fvlm-i6n.12", "title": "Generate final artifact pack report paper and audit"},
                    ]
                ),
                encoding="utf-8",
            )
            closed_beads = root / "closed_beads.json"
            closed_beads.write_text(
                json.dumps([{"id": "fvlm-i6n.1", "title": "Revalidate audited state"}]),
                encoding="utf-8",
            )
            output_root = root / "out"

            rc = self.module.main(
                [
                    "--backend-matrix",
                    str(paths["backend_matrix"]),
                    "--profiler-registry",
                    str(paths["profiler_registry"]),
                    "--candidate-plans",
                    str(paths["candidate_plans"]),
                    "--baseline-registry",
                    str(paths["baseline_registry"]),
                    "--workload-registry",
                    str(paths["workload_registry"]),
                    "--experiment-registry",
                    str(paths["experiment_registry"]),
                    "--plot-registry",
                    str(paths["plot_registry"]),
                    "--state-ledger",
                    str(paths["state_ledger"]),
                    "--experiment-summary",
                    str(paths["experiment_summary"]),
                    "--sustained-summary",
                    str(paths["sustained_summary"]),
                    "--calibration-summary",
                    str(paths["calibration_summary"]),
                    "--tuning-summary",
                    str(paths["tuning_summary"]),
                    "--characterization-summary",
                    str(paths["characterization_summary"]),
                    "--memory-admission-summary",
                    str(paths["memory_admission_summary"]),
                    "--truth-source-pdf",
                    str(truth_pdf),
                    "--open-beads-json",
                    str(open_beads),
                    "--closed-beads-json",
                    str(closed_beads),
                    "--output-root",
                    str(output_root),
                ]
            )

            self.assertEqual(rc, 0)
            summaries = list(output_root.glob("graphpilot_checkpoint_*/summary.json"))
            self.assertEqual(len(summaries), 1)
            payload = json.loads(summaries[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["truth_source_pdf"], str(truth_pdf.resolve()))
            self.assertEqual(payload["canonical_evidence_paths"]["experiment_summary"], str(paths["experiment_summary"].resolve()))
            self.assertEqual(payload["canonical_evidence_paths"]["baseline_registry"], str(paths["baseline_registry"].resolve()))
            self.assertEqual(payload["canonical_evidence_paths"]["workload_registry"], str(paths["workload_registry"].resolve()))
            self.assertEqual([item["id"] for item in payload["open_beads"]], ["fvlm-i6n.11", "fvlm-i6n.12"])
            self.assertEqual([item["id"] for item in payload["closed_beads"]], ["fvlm-i6n.1"])


if __name__ == "__main__":
    unittest.main()
