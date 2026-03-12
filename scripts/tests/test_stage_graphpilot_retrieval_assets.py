import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest import mock


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "stage_graphpilot_retrieval_assets.py"
    spec = importlib.util.spec_from_file_location(
        "stage_graphpilot_retrieval_assets", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class StageGraphPilotRetrievalAssetsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_main_writes_summary_with_pushed_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            model = root / "embeddinggemma.tflite"
            tokenizer = root / "tokenizer.model"
            kb = root / "retrieval_kb.json"
            host_bin = root / "graphpilot_retrieval_main"
            android_bin = root / "graphpilot_retrieval_main_android"
            litert_runtime = root / "libLiteRt.so"
            gpu_accelerator = root / "libLiteRtOpenClAccelerator.so"
            gpu_sampler = root / "libLiteRtTopKOpenClSampler.so"
            gemma_provider = root / "libGemmaModelConstraintProvider.so"
            output_root = root / "out"
            for path in (
                model,
                tokenizer,
                kb,
                host_bin,
                android_bin,
                litert_runtime,
                gpu_accelerator,
                gpu_sampler,
                gemma_provider,
            ):
                path.write_text("x", encoding="utf-8")

            calls = []

            def fake_run(cmd, cwd=None):
                calls.append((tuple(cmd), str(cwd) if cwd else None))
                class Result:
                    returncode = 0
                    stdout = ""
                    stderr = ""
                return Result()

            with mock.patch.object(self.module, "run", side_effect=fake_run):
                rc = self.module.main(
                    [
                        "--host-binary",
                        str(host_bin),
                        "--android-binary",
                        str(android_bin),
                        "--litert-runtime-so",
                        str(litert_runtime),
                        "--gpu-accelerator-so",
                        str(gpu_accelerator),
                        "--gpu-sampler-so",
                        str(gpu_sampler),
                        "--gemma-constraint-provider-so",
                        str(gemma_provider),
                        "--model-path",
                        str(model),
                        "--tokenizer-path",
                        str(tokenizer),
                        "--kb-path",
                        str(kb),
                        "--output-root",
                        str(output_root),
                    ]
                )

            self.assertEqual(rc, 0)
            summaries = list(output_root.glob("retrieval_stage_*/summary.json"))
            self.assertEqual(len(summaries), 1)
            payload = json.loads(summaries[0].read_text(encoding="utf-8"))
            self.assertIn("graphpilot_retrieval_main", json.dumps(payload))
            self.assertGreaterEqual(len(calls), 9)


if __name__ == "__main__":
    unittest.main()
