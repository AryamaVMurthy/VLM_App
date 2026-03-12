import importlib.util
import pathlib
import sys
import unittest


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "probe_graphpilot_retrieval_artifact.py"
    spec = importlib.util.spec_from_file_location(
        "probe_graphpilot_retrieval_artifact", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ProbeGraphPilotRetrievalArtifactTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_summarize_probe_marks_missing_when_no_retrieval_artifact_exists(self):
        summary = self.module.summarize_probe(host_matches=[], device_matches=[])
        self.assertEqual(summary["verdict"], "missing_artifact")
        self.assertIn("EmbeddingGemma-300m", summary["remediation"])
        self.assertEqual(summary["usable_stage_variant"], None)

    def test_parse_device_probe_allows_nonzero_when_find_reports_missing_paths(self):
        matches = self.module.parse_device_probe_result(
            returncode=1,
            stdout=(
                "/data/local/tmp/graphpilot_edge/EmbeddingGemma-300m-int8.tflite\n"
                "/data/local/tmp/graphpilot_edge/gemma3-1b-it-int4.litertlm\n"
            ),
            stderr="",
        )
        self.assertEqual(matches, ["/data/local/tmp/graphpilot_edge/EmbeddingGemma-300m-int8.tflite"])

    def test_summarize_probe_accepts_embeddinggemma_candidates(self):
        device_match = "/data/local/tmp/graphpilot_edge/EmbeddingGemma-300m-int8.tflite"
        summary = self.module.summarize_probe(
            host_matches=[],
            device_matches=[device_match],
        )
        self.assertEqual(summary["verdict"], "artifact_present")
        self.assertEqual(summary["usable_stage_variant"], "embeddinggemma_300m")
        self.assertIn(device_match, summary["device_matches"])


if __name__ == "__main__":
    unittest.main()
