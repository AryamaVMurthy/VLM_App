import importlib.util
import json
import pathlib
import tempfile
import unittest


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "build_graphpilot_cases_paper.py"
    assert module_path.exists(), f"Missing CASES paper builder: {module_path}"
    spec = importlib.util.spec_from_file_location("build_graphpilot_cases_paper", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BuildGraphPilotCasesPaperTest(unittest.TestCase):
    def test_cases_paper_builder_requires_checkpoint_manifest(self) -> None:
        module = load_module()

        with self.assertRaises(SystemExit) as exc:
            module.main([])

        self.assertNotEqual(exc.exception.code, 0)

    def test_cases_paper_builder_fails_fast_without_required_pack_outputs(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            checkpoint_manifest = root / "checkpoint_summary.json"
            artifact_pack_summary = root / "artifact_pack_summary.json"
            artifact_pack_summary.write_text(
                json.dumps(
                    {
                        "report": str(root / "report.md"),
                        "paper_tables": str(root / "paper_tables.md"),
                        "paper_draft": str(root / "paper_draft.md"),
                        "final_audit_report": str(root / "final_audit_report.md"),
                    }
                ),
                encoding="utf-8",
            )
            checkpoint_manifest.write_text(
                json.dumps(
                    {
                        "canonical_evidence_paths": {
                            "artifact_pack_summary": str(artifact_pack_summary),
                        }
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(FileNotFoundError):
                module.main(
                    [
                        "--checkpoint-manifest",
                        str(checkpoint_manifest),
                        "--output-root",
                        str(root / "paper_out"),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
