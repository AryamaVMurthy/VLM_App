import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "generate_graphpilot_workload_registry.py"
    spec = importlib.util.spec_from_file_location(
        "generate_graphpilot_workload_registry", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GenerateGraphPilotWorkloadRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_main_emits_registry_with_scenario_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            workload_path = root / "workload_universe.json"
            output_path = root / "workload_registry.json"
            workload_path.write_text(
                json.dumps(
                    {
                        "workloads": [
                            {
                                "workload_id": "compound.workflow_a.default",
                                "category": "compound_assistant",
                                "builder": "assistant_workflow",
                                "description": "workflow a",
                                "params": {
                                    "workflow_id": "workflow_a_voice_only",
                                    "audio_duration_ms": 3200,
                                    "planner_prompt_tokens": 96,
                                    "planner_output_tokens": 32,
                                    "responder_prompt_tokens": 384,
                                    "responder_output_tokens": 96,
                                    "tts_chunk_size_chars": 80
                                }
                            },
                            {
                                "workload_id": "continuous.workflow_a.poisson",
                                "category": "continuous_stream",
                                "description": "stream a",
                                "base_workload_id": "compound.workflow_a.default",
                                "arrivals_ms": [0, 1500],
                                "deadline_ms": 12000
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            rc = self.module.main(
                [
                    "--workload-universe",
                    str(workload_path),
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(rc, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["workloads"]), 2)
            compound = next(
                workload for workload in payload["workloads"] if workload["workload_id"] == "compound.workflow_a.default"
            )
            self.assertEqual(compound["scenario"]["stage_ids"], ["asr.primary", "planner.primary", "responder.primary", "tts.primary"])
            self.assertEqual(compound["scenario"]["chunk_sizes"], {"asr.primary": 800})
            continuous = next(
                workload for workload in payload["workloads"] if workload["workload_id"] == "continuous.workflow_a.poisson"
            )
            self.assertEqual(continuous["base_workload_id"], "compound.workflow_a.default")
            self.assertEqual(continuous["arrivals_ms"], [0, 1500])

    def test_cli_invocation_writes_registry_from_repo_root(self):
        script_path = pathlib.Path(__file__).resolve().parents[1] / "generate_graphpilot_workload_registry.py"
        result = subprocess.run(
            [sys.executable, str(script_path), "--output", str(pathlib.Path.cwd() / "artifacts" / "graphpilot_edge" / "registries" / "workload_universe_registry.json")],
            cwd=pathlib.Path(__file__).resolve().parents[2],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)


if __name__ == "__main__":
    unittest.main()
