#!/usr/bin/env python3
"""Run exact GQA evaluation through the real LiteRT adb/NPU FastVLM path."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fastvlm_cases_benchmark import extract_run_log_path, parse_run_log

DEFAULT_BENCHMARK_ROOT = Path("/home/aryamavmurthy/work/Liquid_benchmarking_image tokens pruning")
DEFAULT_QUESTIONS_JSON = (
    DEFAULT_BENCHMARK_ROOT
    / "bench_data/gqa/questions/val_balanced_questions_quick500_local_seed42.json"
)
DEFAULT_IMAGES_DIR = DEFAULT_BENCHMARK_ROOT / "bench_data/gqa/images"
DEFAULT_RUNNER_SCRIPT = SCRIPT_DIR / "run_fastvlm_litert_npu_adb.sh"
DEFAULT_MODEL_PATH = (
    SCRIPT_DIR.parent / "artifacts/models/FastVLM-0.5B.qualcomm.sm8750.auxmaskrope_runtime.litertlm"
)
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR.parent / "artifacts/gqa_eval"
DEFAULT_DEVICE_DIR = "/data/local/tmp/vlm_phase1"
DEFAULT_DEVICE_IMAGE_PATH = f"{DEFAULT_DEVICE_DIR}/image.jpg"
DEFAULT_VISUAL_TOKEN_PRUNING_STRATEGY = "prompt_conditioned_v1"
DEFAULT_MAX_OUTPUT_TOKENS = 12
DEFAULT_MAX_VISUAL_TOKENS = 96
DEFAULT_GQA_SHORT_ANSWER_REGEX = r" ?Answer: [A-Za-z0-9]+(?: [A-Za-z0-9]+)?"


@dataclass(frozen=True)
class GQASample:
    question_id: str
    image_id: str
    question: str
    answer: str


class GQARuntimeError(RuntimeError):
    pass


def load_gqa_questions(path: Path) -> list[GQASample]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("GQA questions file must be a dict of question_id -> payload")
    samples: list[GQASample] = []
    for question_id, payload in raw.items():
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid payload for question_id={question_id}")
        for key in ("imageId", "question", "answer"):
            if key not in payload:
                raise ValueError(f"Missing required field '{key}' for question_id={question_id}")
        samples.append(
            GQASample(
                question_id=str(question_id),
                image_id=str(payload["imageId"]),
                question=str(payload["question"]),
                answer=str(payload["answer"]),
            )
        )
    return samples


def gqa_prompt(question: str) -> str:
    return f"Question: {question}\nRespond exactly as: Answer: <one or two words>."


def build_runner_command(
    *,
    runner_script: Path,
    prompt: str,
    model_path: Path,
    max_visual_tokens: int,
    visual_token_pruning_strategy: str,
    max_output_tokens: int,
    constraint_regex: str,
    skip_build: bool,
    skip_push: bool,
    image_path: Path | None = None,
    device_dir: str = DEFAULT_DEVICE_DIR,
) -> list[str]:
    command = [
        str(runner_script),
        "--model",
        str(model_path),
        "--prompt",
        prompt,
        "--max-output-tokens",
        str(max_output_tokens),
        "--constraint-regex",
        constraint_regex,
        "--max-visual-tokens",
        str(max_visual_tokens),
        "--visual-token-pruning-strategy",
        visual_token_pruning_strategy,
        "--benchmark",
        "1",
        "--event-mode",
        "1",
        "--skip-build",
        "1" if skip_build else "0",
        "--skip-push",
        "1" if skip_push else "0",
        "--device-dir",
        device_dir,
    ]
    if image_path is not None:
        command.extend(["--image", str(image_path)])
    return command


def run_command(command: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def adb_push_image(image_path: Path, device_image_path: str) -> None:
    result = run_command(["adb", "push", str(image_path), device_image_path])
    if result.returncode != 0:
        raise GQARuntimeError(
            "Failed to push GQA image to device. "
            f"image={image_path} device_image_path={device_image_path} "
            f"stdout={result.stdout.strip()} stderr={result.stderr.strip()}"
        )


def require_successful_run(result: subprocess.CompletedProcess[str]) -> Path:
    if result.returncode != 0:
        raise GQARuntimeError(
            "FastVLM adb runner failed. "
            f"exit_code={result.returncode} stdout={result.stdout.strip()} stderr={result.stderr.strip()}"
        )
    run_log_path = extract_run_log_path(result.stdout)
    if not run_log_path.is_file():
        raise GQARuntimeError(f"Runner reported missing log path: {run_log_path}")
    return run_log_path


def build_prediction_row(sample: GQASample, prediction: str, metrics: dict[str, Any]) -> dict[str, Any]:
    ttft_ms = float(metrics.get("ttft_ms") or 0.0)
    prefill_latency_us = float(metrics.get("prefill_latency_us") or 0.0)
    decode_latency_us = float(metrics.get("decode_latency_us") or 0.0)
    return {
        "benchmark": "gqa",
        "sample_id": sample.question_id,
        "image_ref": f"{sample.image_id}.jpg",
        "question": sample.question,
        "prompt": gqa_prompt(sample.question),
        "prediction": prediction,
        "answer": sample.answer,
        "references": [sample.answer],
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


def load_gqa_evaluator(benchmark_root: Path):
    evaluator_path = benchmark_root / "src/lfm2vl_bench/eval_gqa.py"
    if not evaluator_path.is_file():
        raise FileNotFoundError(f"Missing GQA evaluator: {evaluator_path}")
    spec = importlib.util.spec_from_file_location("lfm2vl_bench_eval_gqa", evaluator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load GQA evaluator module from {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_predictions(predictions_jsonl: Path, benchmark_root: Path) -> dict[str, float]:
    evaluator = load_gqa_evaluator(benchmark_root)
    metrics = evaluator.evaluate_gqa_jsonl(predictions_jsonl)
    if "accuracy" not in metrics:
        raise RuntimeError(f"Unexpected GQA evaluator result: {metrics}")
    return metrics


def summarize_runtime(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    ttfts = [float(row["ttft_s"]) for row in rows]
    prefills = [float(row["prefill_s"]) for row in rows]
    decodes = [float(row["decode_s"]) for row in rows]
    decode_tps = [float(row["decode_tokens_per_s"]) for row in rows]
    prefill_tps = [float(row["prefill_tokens_per_s"]) for row in rows]
    vision_tokens = [float(row["vision_tokens"]) for row in rows if row.get("vision_tokens") is not None]
    return {
        "sample_count": len(rows),
        "ttft_mean_s": statistics.fmean(ttfts),
        "ttft_median_s": statistics.median(ttfts),
        "prefill_mean_s": statistics.fmean(prefills),
        "decode_mean_s": statistics.fmean(decodes),
        "prefill_tokens_per_s_mean": statistics.fmean(prefill_tps),
        "decode_tokens_per_s_mean": statistics.fmean(decode_tps),
        "vision_tokens_mean": statistics.fmean(vision_tokens) if vision_tokens else 0.0,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--questions-json", type=Path, default=DEFAULT_QUESTIONS_JSON)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--runner-script", type=Path, default=DEFAULT_RUNNER_SCRIPT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-visual-tokens", type=int, default=DEFAULT_MAX_VISUAL_TOKENS)
    parser.add_argument(
        "--visual-token-pruning-strategy",
        default=DEFAULT_VISUAL_TOKEN_PRUNING_STRATEGY,
    )
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--sample-n", type=int, default=None)
    parser.add_argument("--skip-initial-build", action="store_true")
    parser.add_argument("--device-dir", default=DEFAULT_DEVICE_DIR)
    parser.add_argument("--device-image-path", default=DEFAULT_DEVICE_IMAGE_PATH)
    parser.add_argument("--constraint-regex", default=DEFAULT_GQA_SHORT_ANSWER_REGEX)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_n is not None and args.sample_n < 1:
        raise ValueError(f"--sample-n must be >= 1, got {args.sample_n}")
    if not args.runner_script.is_file():
        raise FileNotFoundError(f"Missing adb runner script: {args.runner_script}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Missing model artifact: {args.model}")
    if not args.questions_json.is_file():
        raise FileNotFoundError(f"Missing GQA questions file: {args.questions_json}")
    if not args.images_dir.is_dir():
        raise FileNotFoundError(f"Missing GQA images dir: {args.images_dir}")

    samples = load_gqa_questions(args.questions_json)
    if args.sample_n is not None:
        samples = samples[: args.sample_n]
    if not samples:
        raise RuntimeError("No GQA samples selected.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"gqa_quick500_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_jsonl = output_dir / "predictions.jsonl"
    summary_json = output_dir / "summary.json"

    rows: list[dict[str, Any]] = []
    base_env = os.environ.copy()
    first_sample = True

    for index, sample in enumerate(samples, start=1):
        image_path = args.images_dir / f"{sample.image_id}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Missing GQA image for sample_id={sample.question_id}: {image_path}"
            )
        prompt = gqa_prompt(sample.question)
        print(
            f"[{index}/{len(samples)}] question_id={sample.question_id} image_id={sample.image_id}",
            flush=True,
        )
        if first_sample:
            command = build_runner_command(
                runner_script=args.runner_script,
                prompt=prompt,
                model_path=args.model,
                max_visual_tokens=args.max_visual_tokens,
                visual_token_pruning_strategy=args.visual_token_pruning_strategy,
                max_output_tokens=args.max_output_tokens,
                constraint_regex=args.constraint_regex,
                skip_build=args.skip_initial_build,
                skip_push=False,
                image_path=image_path,
                device_dir=args.device_dir,
            )
        else:
            adb_push_image(image_path, args.device_image_path)
            command = build_runner_command(
                runner_script=args.runner_script,
                prompt=prompt,
                model_path=args.model,
                max_visual_tokens=args.max_visual_tokens,
                visual_token_pruning_strategy=args.visual_token_pruning_strategy,
                max_output_tokens=args.max_output_tokens,
                constraint_regex=args.constraint_regex,
                skip_build=True,
                skip_push=True,
                image_path=None,
                device_dir=args.device_dir,
            )
        result = run_command(command, env=base_env)
        run_log_path = require_successful_run(result)
        metrics = parse_run_log(run_log_path)
        runtime_error = metrics.get("runtime_error")
        if runtime_error:
            raise GQARuntimeError(
                f"Runtime emitted VLM_EVENT error for sample_id={sample.question_id}: {runtime_error}"
            )
        prediction = str(metrics.get("caption") or "").strip()
        row = build_prediction_row(sample, prediction, metrics)
        rows.append(row)
        write_jsonl(predictions_jsonl, rows)
        first_sample = False

    quality = evaluate_predictions(predictions_jsonl, args.benchmark_root)
    runtime = summarize_runtime(rows)
    summary = {
        "task": "gqa",
        "questions_json": str(args.questions_json.resolve()),
        "images_dir": str(args.images_dir.resolve()),
        "predictions_jsonl": str(predictions_jsonl.resolve()),
        "output_dir": str(output_dir.resolve()),
        "max_visual_tokens": args.max_visual_tokens,
        "visual_token_pruning_strategy": args.visual_token_pruning_strategy,
        "constraint_regex": args.constraint_regex,
        **quality,
        **runtime,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
