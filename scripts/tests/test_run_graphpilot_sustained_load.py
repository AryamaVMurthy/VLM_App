import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest import mock


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "run_graphpilot_sustained_load.py"
    spec = importlib.util.spec_from_file_location("run_graphpilot_sustained_load", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


THERMAL_TEXT = """Thermal Status: 1
Current temperatures from HAL:
\tTemperature{mValue=36.1, mType=0, mName=CPU0, mStatus=0}
\tTemperature{mValue=35.5, mType=1, mName=GPU0, mStatus=0}
\tTemperature{mValue=34.2, mType=9, mName=nsp6, mStatus=0}
\tTemperature{mValue=33.4, mType=3, mName=skin, mStatus=0}
"""


class RunGraphPilotSustainedLoadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_parse_thermalservice_output_extracts_cpu_gpu_npu(self) -> None:
        parsed = self.module.parse_thermalservice_output(THERMAL_TEXT)
        self.assertEqual(parsed["thermal_status"], 1)
        self.assertEqual(parsed["max_cpu_c"], 36.1)
        self.assertEqual(parsed["max_gpu_c"], 35.5)
        self.assertEqual(parsed["max_npu_c"], 34.2)
        self.assertEqual(parsed["skin_c"], 33.4)

    def test_summarize_samples_reports_latency_and_skin_drift(self) -> None:
        samples = [
            {
                "workflow_id": "workflow_b_voice_vision",
                "variant": "graphpilot_vlm_npu",
                "warm_latency_ms": 15000,
                "ttft_ms": 4100,
                "tts_first_audio_ms": 9700,
                "thermal_after": {"thermal_status": 0, "max_cpu_c": 35.0, "max_gpu_c": 34.0, "max_npu_c": 33.0, "skin_c": 32.0},
            },
            {
                "workflow_id": "workflow_b_voice_vision",
                "variant": "graphpilot_vlm_npu",
                "warm_latency_ms": 15600,
                "ttft_ms": 4300,
                "tts_first_audio_ms": 9900,
                "thermal_after": {"thermal_status": 1, "max_cpu_c": 37.0, "max_gpu_c": 35.0, "max_npu_c": 34.5, "skin_c": 33.0},
            },
        ]
        summary = self.module.summarize_samples(samples)
        workflow = summary["workflow_summary"]["workflow_b_voice_vision"]
        self.assertEqual(workflow["warm_latency_ms"]["drift"], 600.0)
        self.assertEqual(workflow["ttft_ms"]["drift"], 200.0)
        self.assertEqual(workflow["thermal"]["skin_c"]["drift"], 1.0)
        self.assertEqual(summary["global_summary"]["thermal_status_max"], 1)

    def test_main_writes_summary_and_registry_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            registry_path = root / "experiment_registry.json"
            registry_path.write_text(json.dumps({"experiments": []}), encoding="utf-8")
            output_root = root / "out"

            fake_profile_summary = root / "profiler_summary.json"
            fake_profile_summary.write_text(json.dumps({"profiles": []}), encoding="utf-8")

            profile_entry = {
                "stage_id": "workflow_b_voice_vision",
                "variant": "graphpilot_vlm_npu",
                "artifacts": {"run_dir": "/tmp/run"},
                "metrics": {
                    "workflow_id": "workflow_b_voice_vision",
                    "warm_latency_ms": 16000,
                    "cold_latency_ms": 16000,
                    "ttft_ms": 4200,
                    "tts_first_audio_ms": 9800,
                    "stage_backends": {"vlm.fastvlm.primary": "npu"},
                },
            }

            with mock.patch.object(self.module, "adb_serial", return_value="serial123"), mock.patch.object(
                self.module, "git_commit", return_value="deadbeef"
            ), mock.patch.object(
                self.module,
                "capture_batterystats",
                return_value="currently on battery: false\nComputed drain: 0\n",
            ), mock.patch.object(
                self.module,
                "capture_thermalservice",
                side_effect=[THERMAL_TEXT, THERMAL_TEXT, THERMAL_TEXT, THERMAL_TEXT, THERMAL_TEXT],
            ), mock.patch.object(
                self.module,
                "run_profile_once",
                return_value=(profile_entry, fake_profile_summary),
            ):
                rc = self.module.main(
                    [
                        "--profiles",
                        "graphpilot_workflow_b",
                        "--max-iterations",
                        "1",
                        "--skip-install",
                        "--output-root",
                        str(output_root),
                        "--experiment-registry",
                        str(registry_path),
                    ]
                )

            self.assertEqual(rc, 0)
            summaries = list(output_root.glob("graphpilot_sustained_load_*/summary.json"))
            self.assertEqual(len(summaries), 1)
            payload = json.loads(summaries[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["device_serial"], "serial123")
            self.assertEqual(payload["workflow_summary"]["workflow_b_voice_vision"]["count"], 1)
            updated_registry = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(len(updated_registry["experiments"]), 1)
            self.assertEqual(updated_registry["experiments"][0]["workflow_template"], "graphpilot_phase7_sustained_load")


if __name__ == "__main__":
    unittest.main()
