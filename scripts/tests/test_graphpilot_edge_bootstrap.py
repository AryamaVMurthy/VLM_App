import importlib.util
import json
import pathlib
import tempfile
import unittest


def load_module():
    module_path = (
        pathlib.Path(__file__).resolve().parents[1] / "graphpilot_edge_bootstrap.py"
    )
    spec = importlib.util.spec_from_file_location("graphpilot_edge_bootstrap", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GraphPilotEdgeBootstrapTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def write_json(self, path: pathlib.Path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def create_minimal_workspace(self, root: pathlib.Path):
        output_root = root / "artifacts" / "graphpilot_edge"
        config_dir = root / "configs" / "graphpilot_edge"
        for relative in self.module.REQUIRED_SUBDIRS:
            (output_root / relative).mkdir(parents=True, exist_ok=True)
        for relative in self.module.REQUIRED_JSON_PATHS:
            self.write_json(output_root / relative, {"ok": True})
        self.write_json(
            config_dir / "workflow_templates.json",
            {"workflows": [{"workflow_id": "a"}, {"workflow_id": "b"}]},
        )
        self.write_json(
            config_dir / "stage_catalog.json",
            {"stages": [{"stage_id": "x"}, {"stage_id": "y"}, {"stage_id": "z"}]},
        )
        self.write_json(
            config_dir / "plan_bank_templates.json",
            {"plan_bank_states": [{"state_id": "cool"}, {"state_id": "warm"}]},
        )
        return output_root, config_dir

    def test_ensure_dirs_creates_required_subdirectories(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = pathlib.Path(tmp) / "artifacts" / "graphpilot_edge"
            created = self.module.ensure_dirs(output_root)
            self.assertEqual(len(created), len(self.module.REQUIRED_SUBDIRS))
            for relative in self.module.REQUIRED_SUBDIRS:
                self.assertTrue((output_root / relative).is_dir())

    def test_build_summary_counts_workflows_stages_and_plan_bank(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root, config_dir = self.create_minimal_workspace(pathlib.Path(tmp))
            summary = self.module.build_summary(output_root, config_dir, probe_adb=False)
            self.assertEqual(summary["workflow_count"], 2)
            self.assertEqual(summary["stage_count"], 3)
            self.assertEqual(summary["plan_bank_states"], ["cool", "warm"])

    def test_build_summary_fails_fast_when_required_json_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root, config_dir = self.create_minimal_workspace(pathlib.Path(tmp))
            missing = output_root / self.module.REQUIRED_JSON_PATHS[0]
            missing.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Missing required JSON file"):
                self.module.build_summary(output_root, config_dir, probe_adb=False)


if __name__ == "__main__":
    unittest.main()
