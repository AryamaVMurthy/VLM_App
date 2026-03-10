#!/usr/bin/env python3
"""Run COCO Karpathy caption evaluation through the real LiteRT adb/NPU FastVLM path."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fastvlm_cases_benchmark import extract_run_log_path, parse_run_log
from run_fastvlm_litert_gqa_eval import (
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_MODEL_PATH,
    DEFAULT_VISUAL_TOKEN_PRUNING_STRATEGY,
    GQARuntimeError,
    adb_push_image,
    build_runner_command,
    collect_pruning_env_overrides,
    require_successful_run,
    run_command,
    summarize_runtime,
    write_jsonl,
)

DEFAULT_RUNNER_SCRIPT = SCRIPT_DIR / "run_fastvlm_litert_npu_adb.sh"
DEFAULT_IMAGE_CACHE = DEFAULT_BENCHMARK_ROOT / "bench_data/coco_karpathy/image_cache"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR.parent / "artifacts/coco_eval"
DEFAULT_SPLIT = "test"
DEFAULT_SAMPLE_N = 100
DEFAULT_MAX_OUTPUT_TOKENS = 32
DEFAULT_MAX_VISUAL_TOKENS = 96
DEFAULT_PROMPT = "Describe this image in one concise sentence."


@dataclass(frozen=True)
class CocoSample:
    sample_id: str
    image_id: int
    filename: str
    references: list[str]


def caption_prompt() -> str:
    return DEFAULT_PROMPT


def _load_module(module_path: Path, module_name: str):
    if not module_path.is_file():
        raise FileNotFoundError(f"Missing module: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_coco_samples(
    benchmark_root: Path, split: str, sample_n: int, image_cache: Path
) -> list[CocoSample]:
    loader = _load_module(
        benchmark_root / "src/lfm2vl_bench/datasets/coco_karpathy.py",
        "lfm2vl_bench_datasets_coco_karpathy",
    )
    rows = loader.load_coco_split(split=split, max_rows=sample_n)
    samples: list[CocoSample] = []
    for row in rows:
        if "cocoid" not in row or "filename" not in row or "sentences" not in row:
            raise RuntimeError(f"COCO row missing required fields: {row}")
        filename = str(row["filename"])
        image_path = image_cache / filename
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Missing cached COCO image: {image_path}. "
                "Populate bench_data/coco_karpathy/image_cache before running."
            )
        references: list[str] = []
        for item in row["sentences"]:
            if isinstance(item, dict):
                text = str(item.get("raw", "")).strip()
            else:
                text = str(item).strip()
            if text:
                references.append(text)
        if not references:
            raise RuntimeError(f"COCO sample has no reference captions: {row}")
        samples.append(
            CocoSample(
                sample_id=str(row["cocoid"]),
                image_id=int(row["cocoid"]),
                filename=filename,
                references=references,
            )
        )
    if not samples:
        raise RuntimeError("No COCO samples loaded.")
    return samples


def build_prediction_row(
    sample: CocoSample, prediction: str, metrics: dict[str, Any]
) -> dict[str, Any]:
    ttft_ms = float(metrics.get("ttft_ms") or 0.0)
    prefill_latency_us = float(metrics.get("prefill_latency_us") or 0.0)
    decode_latency_us = float(metrics.get("decode_latency_us") or 0.0)
    return {
        "benchmark": "coco_karpathy",
        "sample_id": sample.sample_id,
        "image_ref": sample.filename,
        "prompt": caption_prompt(),
        "prediction": prediction,
        "references": sample.references,
        "cocoid": sample.image_id,
        "ttft_s": round(ttft_ms / 1000.0, 4),
        "prefill_s": round(prefill_latency_us / 1_000_000.0, 6),
        "decode_s": round(decode_latency_us / 1_000_000.0, 6),
        "decode_tokens_per_s": float(metrics.get("decode_tokens_per_sec") or 0.0),
        "prefill_tokens_per_s": float(metrics.get("prefill_tokens_per_sec") or 0.0),
        "vision_tokens": metrics.get("visual_tokens_kept"),
        "vision_tokens_original": metrics.get("visual_tokens_original"),
        "visual_token_pruning_strategy": metrics.get("visual_token_pruning_strategy"),
        "mean_prompt_similarity": metrics.get("mean_prompt_similarity"),
        "mean_salience": metrics.get("mean_salience"),
        "selected_token_indices": metrics.get("selected_token_indices"),
        "log_path": metrics.get("log_path"),
        "error": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def evaluate_predictions(predictions_jsonl: Path, benchmark_root: Path) -> dict[str, float]:
    evaluator = _load_module(
        benchmark_root / "src/lfm2vl_bench/eval_coco.py",
        "lfm2vl_bench_eval_coco",
    )
    metrics = evaluator.evaluate_coco_jsonl(
        predictions_jsonl, predictions_jsonl.parent / "coco_eval_work"
    )
    if "CIDEr" not in metrics:
        raise RuntimeError(f"Unexpected COCO evaluator output: {metrics}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--image-cache", type=Path, default=DEFAULT_IMAGE_CACHE)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--sample-n", type=int, default=DEFAULT_SAMPLE_N)
    parser.add_argument("--runner-script", type=Path, default=DEFAULT_RUNNER_SCRIPT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--visual-token-pruning-strategy",
        default=DEFAULT_VISUAL_TOKEN_PRUNING_STRATEGY,
    )
    parser.add_argument("--max-visual-tokens", type=int, default=DEFAULT_MAX_VISUAL_TOKENS)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-push", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_n <= 0:
        raise ValueError(f"--sample-n must be > 0, got {args.sample_n}")
    if not args.runner_script.is_file():
        raise FileNotFoundError(f"Missing runner script: {args.runner_script}")
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Missing model path: {args.model_path}")
    samples = load_coco_samples(
        args.benchmark_root, args.split, args.sample_n, args.image_cache
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"coco_{args.split}_{args.sample_n}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    pruning_env_overrides = collect_pruning_env_overrides(os.environ)
    rows: list[dict[str, Any]] = []
    skip_push = args.skip_push
    for index, sample in enumerate(samples):
        image_path = args.image_cache / sample.filename
        adb_push_image(image_path, "/data/local/tmp/vlm_phase1/image.jpg")
        command = build_runner_command(
            runner_script=args.runner_script,
            prompt=caption_prompt(),
            model_path=args.model_path,
            max_visual_tokens=args.max_visual_tokens,
            visual_token_pruning_strategy=args.visual_token_pruning_strategy,
            max_output_tokens=args.max_output_tokens,
            constraint_regex="",
            skip_build=args.skip_build,
            skip_push=skip_push,
            pruning_env_overrides=pruning_env_overrides,
            image_path=image_path,
        )
        result = run_command(command, env=os.environ.copy())
        run_log_path = require_successful_run(result)
        metrics = parse_run_log(run_log_path)
        rows.append(build_prediction_row(sample, str(metrics.get("caption") or ""), metrics))
        skip_push = True
        print(
            json.dumps(
                {
                    "index": index,
                    "sample_id": sample.sample_id,
                    "image": sample.filename,
                    "caption": rows[-1]["prediction"],
                },
                ensure_ascii=True,
            )
        )

    predictions_path = output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, rows)
    coco_metrics = evaluate_predictions(predictions_path, args.benchmark_root)
    runtime_summary = summarize_runtime(rows)
    summary = {
        "benchmark": "coco_karpathy",
        "split": args.split,
        "sample_count": len(rows),
        "visual_token_pruning_strategy": args.visual_token_pruning_strategy,
        "max_visual_tokens": args.max_visual_tokens,
        "max_output_tokens": args.max_output_tokens,
        "runtime": runtime_summary,
        "metrics": coco_metrics,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
