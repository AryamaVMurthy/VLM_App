import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


def load_module():
    module_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "run_fastvlm_litert_gqa_overlap_eval.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_fastvlm_litert_gqa_overlap_eval", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = load_module()


class GQAOverlapEvalHelpersTest(unittest.TestCase):
    def test_build_manifest_rows_assigns_device_paths(self):
        samples = [
            module.GQASample(
                question_id="q1",
                image_id="img1",
                question="What color?",
                answer="red",
            ),
            module.GQASample(
                question_id="q2",
                image_id="img2",
                question="Who is it?",
                answer="man",
            ),
        ]
        rows = module.build_manifest_rows(samples, "/data/local/tmp/vlm_overlap")
        self.assertEqual(rows[0]["request_id"], "q1")
        self.assertTrue(rows[0]["prompt"].startswith("Question: What color?"))
        self.assertEqual(rows[0]["image_path"], "/data/local/tmp/vlm_overlap/image_000.jpg")
        self.assertEqual(rows[1]["image_path"], "/data/local/tmp/vlm_overlap/image_001.jpg")

    def test_parse_overlap_log_reads_response_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = pathlib.Path(tmpdir) / "run.log"
            payloads = [
                {"type": "QUEUE_ADMISSION", "request_id": "q1"},
                {"type": "OVERLAP_RESPONSE", "request_index": 0, "request_id": "q1", "text": "Answer: red"},
                {"type": "OVERLAP_RESPONSE", "request_index": 1, "request_id": "q2", "text": "Answer: man"},
            ]
            log_path.write_text(
                "\n".join(f"VLM_EVENT {json.dumps(item)}" for item in payloads),
                encoding="utf-8",
            )
            parsed = module.parse_overlap_log(log_path)
            self.assertEqual(parsed.responses["q1"].text, "Answer: red")
            self.assertEqual(parsed.responses["q2"].request_index, 1)
            self.assertEqual(parsed.recovered_split_events, 0)

    def test_parse_overlap_log_recovers_split_event_line(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = pathlib.Path(tmpdir) / "run.log"
            log_path.write_text(
                "\n".join(
                    [
                        'VLM_EVENT      0.0ms [  INFO ]  QnnDsp <I> QnnGraph_execute started. graph = 0x11',
                        '{"type":"OVERLAP_RESPONSE","request_index":0,"request_id":"q1","text":"Answer: red"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            parsed = module.parse_overlap_log(log_path)
            self.assertEqual(parsed.responses["q1"].text, "Answer: red")
            self.assertEqual(parsed.recovered_split_events, 1)

    def test_parse_overlap_log_rejects_error_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = pathlib.Path(tmpdir) / "run.log"
            payloads = [
                {"type": "ERROR", "message": "decode_failed"},
                {"type": "OVERLAP_RESPONSE", "request_index": 0, "request_id": "q1", "text": "Answer: red"},
            ]
            log_path.write_text(
                "\n".join(f"VLM_EVENT {json.dumps(item)}" for item in payloads),
                encoding="utf-8",
            )
            with self.assertRaises(module.GQAOverlapRuntimeError):
                module.parse_overlap_log(log_path)

    def test_parse_overlap_log_rejects_malformed_event_json_with_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = pathlib.Path(tmpdir) / "run.log"
            log_path.write_text(
                'VLM_EVENT VLM_EVENT {"type":"PREPARE_START","request_id":"q1"}\n',
                encoding="utf-8",
            )
            with self.assertRaises(module.GQAOverlapRuntimeError) as ctx:
                module.parse_overlap_log(log_path)
            self.assertIn("Invalid VLM_EVENT JSON", str(ctx.exception))

    def test_build_runner_command_includes_optional_regex_only_when_set(self):
        command = module.build_runner_command(
            runner_script=pathlib.Path("scripts/run_fastvlm_litert_overlap_adb.sh"),
            model_path=pathlib.Path("artifacts/models/model.litertlm"),
            decode_model_path=pathlib.Path("artifacts/models/decode.litertlm"),
            manifest_path=pathlib.Path("requests.jsonl"),
            image_paths=[pathlib.Path("img1.jpg"), pathlib.Path("img2.jpg")],
            max_visual_tokens=96,
            visual_token_pruning_strategy="prompt_conditioned_v1",
            max_output_tokens=12,
            prepare_queue_size=4,
            constraint_regex="",
            skip_build=True,
            skip_push=True,
            device_dir="/data/local/tmp/vlm_overlap",
        )
        self.assertNotIn("--constraint-regex", command)
        command_with_regex = module.build_runner_command(
            runner_script=pathlib.Path("scripts/run_fastvlm_litert_overlap_adb.sh"),
            model_path=pathlib.Path("artifacts/models/model.litertlm"),
            decode_model_path=pathlib.Path("artifacts/models/decode.litertlm"),
            manifest_path=pathlib.Path("requests.jsonl"),
            image_paths=[pathlib.Path("img1.jpg"), pathlib.Path("img2.jpg")],
            max_visual_tokens=96,
            visual_token_pruning_strategy="prompt_conditioned_v1",
            max_output_tokens=12,
            prepare_queue_size=7,
            constraint_regex=r" ?Answer: [A-Za-z0-9]+",
            skip_build=True,
            skip_push=True,
            device_dir="/data/local/tmp/vlm_overlap",
        )
        self.assertIn("--constraint-regex", command_with_regex)
        self.assertIn("--prepare-queue-size", command_with_regex)
        queue_size_index = command_with_regex.index("--prepare-queue-size")
        self.assertEqual(command_with_regex[queue_size_index + 1], "7")


if __name__ == "__main__":
    unittest.main()
