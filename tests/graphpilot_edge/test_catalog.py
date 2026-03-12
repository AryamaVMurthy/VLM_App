import json
import tempfile
import unittest
from pathlib import Path

from graphpilot_edge.catalog import load_stage_catalog, load_workflow_catalog, resolve_feasible_backends


class GraphPilotCatalogTest(unittest.TestCase):
    def test_load_catalogs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage_path = root / "stages.json"
            workflow_path = root / "workflows.json"
            stage_path.write_text(
                json.dumps({"stages": [{"stage_id": "planner.primary", "role": "planner"}]})
            )
            workflow_path.write_text(
                json.dumps({"workflows": [{"workflow_id": "workflow_a", "nodes": ["planner.primary"], "edges": []}]})
            )

            stages = load_stage_catalog(stage_path)
            workflows = load_workflow_catalog(workflow_path)

            self.assertEqual(stages["planner.primary"].role, "planner")
            self.assertEqual(workflows["workflow_a"].stage_ids, ("planner.primary",))

    def test_resolve_feasible_backends_filters_only_verified_backend_states(self):
        matrix = {
            "stages": [
                {
                    "stage_id": "planner.primary",
                    "backends": {
                        "cpu": {"status": "feasible_smoke_pass"},
                        "gpu": {"status": "infeasible_smoke_fail"},
                        "npu": {"status": "known_working"},
                    },
                }
            ]
        }

        backends = resolve_feasible_backends(matrix, "planner.primary")
        self.assertEqual(backends, ("cpu", "npu"))


if __name__ == "__main__":
    unittest.main()
