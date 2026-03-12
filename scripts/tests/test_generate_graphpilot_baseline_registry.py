import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "generate_graphpilot_baseline_registry.py"
    spec = importlib.util.spec_from_file_location(
        "generate_graphpilot_baseline_registry", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GenerateGraphPilotBaselineRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_main_writes_baseline_registry_with_fail_fast_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            output_path = root / "baseline_registry.json"

            rc = self.module.main([
                "--output",
                str(output_path),
                "--workloads",
                "compound.workflow_a.default",
                "compound.workflow_b.default",
            ])

            self.assertEqual(rc, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                sorted(payload["baseline_ids"]),
                sorted(
                    [
                        "cpu_only",
                        "gpu_only",
                        "npu_only",
                        "current_deployed_plan",
                        "stage_greedy",
                        "static_best_map",
                        "no_pipeline",
                    ]
                ),
            )
            workflow_a = next(item for item in payload["workloads"] if item["workload_id"] == "compound.workflow_a.default")
            cpu_only = next(item for item in workflow_a["baselines"] if item["baseline_id"] == "cpu_only")
            self.assertEqual(cpu_only["resource_assignment"]["asr.primary"], "cpu0")
            current_deployed = next(item for item in workflow_a["baselines"] if item["baseline_id"] == "current_deployed_plan")
            self.assertEqual(current_deployed["resource_assignment"]["asr.primary"], "cpu0")
            npu_only = next(item for item in workflow_a["baselines"] if item["baseline_id"] == "npu_only")
            self.assertEqual(npu_only["status"], "error")
            self.assertIn("No support-safe resource", npu_only["error"])

            workflow_b = next(item for item in payload["workloads"] if item["workload_id"] == "compound.workflow_b.default")
            static_best = next(item for item in workflow_b["baselines"] if item["baseline_id"] == "static_best_map")
            self.assertEqual(static_best["resource_assignment"]["vlm.fastvlm.primary"], "npu0")
            current_deployed_b = next(item for item in workflow_b["baselines"] if item["baseline_id"] == "current_deployed_plan")
            self.assertEqual(current_deployed_b["resource_assignment"]["vlm.fastvlm.primary"], "npu0")


if __name__ == "__main__":
    unittest.main()
