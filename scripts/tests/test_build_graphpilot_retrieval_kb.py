import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest import mock


def load_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "build_graphpilot_retrieval_kb.py"
    spec = importlib.util.spec_from_file_location(
        "build_graphpilot_retrieval_kb", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BuildGraphPilotRetrievalKbTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()

    def test_main_builds_kb_from_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            corpus = root / "corpus.json"
            binary = root / "graphpilot_retrieval_main"
            tokenizer = root / "tokenizer.model"
            model = root / "embeddinggemma.tflite"
            out = root / "retrieval_kb.json"
            corpus.write_text(
                json.dumps(
                    {
                        "corpus_id": "demo",
                        "documents": [
                            {"doc_id": "doc1", "title": "One", "text": "hello"},
                            {"doc_id": "doc2", "title": "Two", "text": "world"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            binary.write_text("", encoding="utf-8")
            tokenizer.write_text("", encoding="utf-8")
            model.write_text("", encoding="utf-8")

            def fake_run_embed(binary_path, model_path, tokenizer_path, text):
                base = 1.0 if text == "hello" else 2.0
                return {
                    "embedding": [base, base + 0.5],
                    "timings_ms": {"total": 7},
                    "token_count": 3,
                    "sequence_length": 512,
                }

            with mock.patch.object(self.module, "run_embed", side_effect=fake_run_embed):
                rc = self.module.main(
                    [
                        "--binary",
                        str(binary),
                        "--model-path",
                        str(model),
                        "--tokenizer-path",
                        str(tokenizer),
                        "--corpus",
                        str(corpus),
                        "--output",
                        str(out),
                    ]
                )

            self.assertEqual(rc, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["corpus_id"], "demo")
            self.assertEqual(len(payload["documents"]), 2)
            self.assertEqual(payload["documents"][0]["embedding"], [1.0, 1.5])


if __name__ == "__main__":
    unittest.main()
