#!/usr/bin/env python3
"""Build a GraphPilot retrieval KB with real EmbeddingGemma embeddings."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_BINARY = ROOT_DIR / "third_party/litert-lm/bazel-bin/runtime/engine/graphpilot_retrieval_main"
DEFAULT_MODEL = ROOT_DIR / "artifacts/models/embeddinggemma-300M_seq512_mixed-precision.tflite"
DEFAULT_TOKENIZER = ROOT_DIR / "artifacts/models/tokenizer.model"
DEFAULT_CORPUS = ROOT_DIR / "configs/graphpilot_edge/retrieval_corpus.json"
DEFAULT_OUTPUT = ROOT_DIR / "artifacts/graphpilot_edge/retrieval/retrieval_kb.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_file(path: Path, remediation: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file '{path}'. Remediation: {remediation}")


def run_embed(binary_path: Path, model_path: Path, tokenizer_path: Path, text: str) -> dict[str, Any]:
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    cmd = [
        str(binary_path),
        "--mode=embed",
        f"--model_path={model_path}",
        f"--tokenizer_path={tokenizer_path}",
        f"--query_b64={encoded}",
        "--accelerator=cpu",
    ]
    result = subprocess.run(cmd, cwd=ROOT_DIR, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "EmbeddingGemma KB build embed run failed.\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
    stdout = result.stdout.strip()
    if not stdout:
        raise RuntimeError("EmbeddingGemma embed run returned empty stdout.")
    return json.loads(stdout.splitlines()[-1])


def build_kb(
    binary_path: Path,
    model_path: Path,
    tokenizer_path: Path,
    corpus_path: Path,
) -> dict[str, Any]:
    corpus = load_json(corpus_path)
    documents = []
    for document in corpus.get("documents", []):
        embed = run_embed(binary_path, model_path, tokenizer_path, document["text"])
        documents.append(
            {
                "doc_id": document["doc_id"],
                "title": document["title"],
                "text": document["text"],
                "embedding": embed["embedding"],
                "token_count": embed.get("token_count"),
                "sequence_length": embed.get("sequence_length"),
                "embed_timings_ms": embed.get("timings_ms", {}),
            }
        )
    if not documents:
        raise RuntimeError("Retrieval corpus is empty; cannot build a KB.")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_id": corpus["corpus_id"],
        "binary_path": str(binary_path.resolve()),
        "model_path": str(model_path.resolve()),
        "tokenizer_path": str(tokenizer_path.resolve()),
        "documents": documents,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require_file(
        args.binary,
        "build the host retrieval binary first: cd third_party/litert-lm && bazel build --jobs=18 //runtime/engine:graphpilot_retrieval_main",
    )
    require_file(args.model_path, "stage the real EmbeddingGemma retrieval model under artifacts/models/.")
    require_file(args.tokenizer_path, "stage tokenizer.model under artifacts/models/.")
    require_file(args.corpus, "ensure configs/graphpilot_edge/retrieval_corpus.json exists.")
    payload = build_kb(args.binary, args.model_path, args.tokenizer_path, args.corpus)
    write_json(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
