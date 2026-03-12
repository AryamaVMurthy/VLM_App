import importlib.util
import json
import pathlib
from unittest import mock
import sys
import tempfile
import unittest


def load_module():
    module_path = (
        pathlib.Path(__file__).resolve().parents[1] / "run_graphpilot_stage_feasibility.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_graphpilot_stage_feasibility", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RunGraphPilotStageFeasibilityTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_parse_instrumentation_output_pass(self):
        text = """
        INSTRUMENTATION_STATUS: class=com.qidk.fastvlm.speech.WhisperSttInstrumentedTest
        INSTRUMENTATION_STATUS_CODE: 1
        INSTRUMENTATION_CODE: -1
        OK (1 test)
        """
        parsed = self.module.parse_instrumentation_output(text)
        self.assertEqual(parsed["verdict"], "pass")
        self.assertEqual(parsed["test_count"], 1)

    def test_parse_instrumentation_output_fail(self):
        text = """
        INSTRUMENTATION_STATUS: stack=java.lang.AssertionError: boom
        INSTRUMENTATION_FAILED:
        Failure [TEST_FAILED]
        Tests run: 1,  Failures: 1
        """
        parsed = self.module.parse_instrumentation_output(text)
        self.assertEqual(parsed["verdict"], "fail")
        self.assertIn("TEST_FAILED", parsed["summary"])

    def test_write_summary_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = pathlib.Path(tmp)
            payload = {"stage_runs": [{"stage_id": "asr.primary", "verdict": "pass"}]}
            path = self.module.write_summary(output_dir, payload)
            self.assertEqual(path.name, "summary.json")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)

    def test_build_gradle_env_sets_android_vars(self):
        sdk = pathlib.Path("/tmp/fake-sdk")
        ndk = pathlib.Path("/tmp/fake-ndk")
        env = self.module.build_gradle_env(sdk, ndk)
        self.assertEqual(env["ANDROID_HOME"], str(sdk))
        self.assertEqual(env["ANDROID_SDK_ROOT"], str(sdk))
        self.assertEqual(env["ANDROID_NDK_HOME"], str(ndk))

    def test_tts_stage_is_registered_and_default_selected(self):
        self.assertIn("tts", self.module.STAGE_TESTS)
        args = self.module.parse_args([])
        self.assertIn("tts", args.stages)

    def test_retrieval_gpu_and_npu_stages_are_registered(self):
        self.assertIn("retrieval_gpu", self.module.STAGE_TESTS)
        self.assertIn("retrieval_npu", self.module.STAGE_TESTS)
        args = self.module.parse_args([])
        self.assertIn("retrieval_gpu", args.stages)
        self.assertIn("retrieval_npu", args.stages)

    def test_resolve_jni_lib_uses_fallback_candidate(self):
        expected = pathlib.Path("/tmp/fallback/liblitertlm_jni.so")
        with mock.patch.object(
            self.module, "DEFAULT_JNI_LIB_CANDIDATES", (pathlib.Path("/tmp/missing.so"),)
        ):
            with mock.patch.object(pathlib.Path, "is_file", autospec=True) as is_file:
                with mock.patch.object(pathlib.Path, "glob", autospec=True) as glob:
                    is_file.side_effect = lambda self: self == expected
                    glob.return_value = [expected]
                    resolved = self.module.resolve_jni_lib()
        self.assertEqual(resolved, expected)


if __name__ == "__main__":
    unittest.main()
