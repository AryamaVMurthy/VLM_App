#!/usr/bin/env python3
"""Run exact GQA evaluation through the real LiteRT overlap runtime on device."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fastvlm_cases_benchmark import extract_run_log_path
from run_fastvlm_litert_gqa_eval import (
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_GQA_SHORT_ANSWER_REGEX,
    DEFAULT_IMAGES_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_QUESTIONS_JSON,
    DEFAULT_VISUAL_TOKEN_PRUNING_STRATEGY,
    GQASample,
    evaluate_predictions,
    gqa_prompt,
    load_gqa_questions,
    write_jsonl,
)

DEFAULT_RUNNER_SCRIPT = SCRIPT_DIR / "run_fastvlm_litert_overlap_adb.sh"
DEFAULT_DECODE_MODEL_PATH = SCRIPT_DIR.parent / "artifacts/models/FastVLM-0.5B.litertlm"
DEFAULT_DEVICE_DIR = "/data/local/tmp/vlm_overlap"
DEFAULT_MAX_OUTPUT_TOKENS = 12
DEFAULT_PREPARE_QUEUE_SIZE = 4


class GQAOverlapRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True)
class OverlapResponse:
    request_id: str
    request_index: int
    text: str


@dataclass(frozen=True)
class ParsedOverlapLog:
    responses: dict[str, OverlapResponse]
    recovered_split_events: int


def build_manifest_rows(samples: list[GQASample], device_dir: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, sample in enumerate(samples):
        rows.append(
            {
                "request_id": sample.question_id,
                "prompt": gqa_prompt(sample.question),
                "image_path": f"{device_dir}/image_{index:03d}.jpg",
            }
        )
    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_overlap_log(log_path: Path) -> ParsedOverlapLog:
    responses: dict[str, OverlapResponse] = {}
    runtime_errors: list[str] = []
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    recovered_split_events = 0
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        line_number = line_index + 1
        line_index += 1
        if not line.startswith("VLM_EVENT "):
            continue
        payload_text = line[len("VLM_EVENT ") :].lstrip()
        if not payload_text.startswith("{"):
            if line_index < len(lines):
                split_payload_text = lines[line_index].lstrip()
                if split_payload_text.startswith("{"):
                    payload_text = split_payload_text
                    recovered_split_events += 1
                    line_index += 1
                else:
                    raise GQAOverlapRuntimeError(
                        f"Invalid VLM_EVENT JSON in {log_path} at line {line_number}: {line}"
                    )
            else:
                raise GQAOverlapRuntimeError(
                    f"Invalid VLM_EVENT JSON in {log_path} at line {line_number}: {line}"
                )
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            raise GQAOverlapRuntimeError(
                f"Invalid VLM_EVENT JSON in {log_path} at line {line_number}: {line}"
            ) from exc
        event_type = payload.get("type")
        if event_type == "ERROR":
            runtime_errors.append(str(payload.get("message", "")))
        if event_type != "OVERLAP_RESPONSE":
            continue
        request_id = str(payload.get("request_id", ""))
        if not request_id:
            raise GQAOverlapRuntimeError(
                f"Missing request_id in OVERLAP_RESPONSE event from {log_path}"
            )
        responses[request_id] = OverlapResponse(
            request_id=request_id,
            request_index=int(payload.get("request_index", -1)),
            text=str(payload.get("text", "")).strip(),
        )
    if runtime_errors:
        raise GQAOverlapRuntimeError(
            f"Overlap runtime emitted error events in {log_path}: {runtime_errors}"
        )
    if recovered_split_events:
        print(
            f"Recovered {recovered_split_events} split VLM_EVENT lines while parsing {log_path}",
            file=sys.stderr,
        )
    return ParsedOverlapLog(
        responses=responses,
        recovered_split_events=recovered_split_events,
    )


def run_command(command: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True, env=env)


def build_runner_command(
    *,
    runner_script: Path,
    model_path: Path,
    decode_model_path: Path,
    manifest_path: Path,
    image_paths: list[Path],
    max_visual_tokens: int,
    visual_token_pruning_strategy: str,
    max_output_tokens: int,
    prepare_queue_size: int,
    constraint_regex: str,
    skip_build: bool,
    skip_push: bool,
    device_dir: str,
) -> list[str]:
    command = [
        "bash",
        str(runner_script),
        "--model",
        str(model_path),
        "--decode-model",
        str(decode_model_path),
        "--request-manifest",
        str(manifest_path),
        "--max-output-tokens",
        str(max_output_tokens),
        "--prepare-queue-size",
        str(prepare_queue_size),
        "--max-visual-tokens",
        str(max_visual_tokens),
        "--visual-token-pruning-strategy",
        visual_token_pruning_strategy,
        "--event-mode",
        "1",
        "--skip-build",
        "1" if skip_build else "0",
        "--skip-push",
        "1" if skip_push else "0",
        "--device-dir",
        device_dir,
    ]
    if constraint_regex:
        command.extend(["--constraint-regex", constraint_regex])
    for image_path in image_paths:
        command.extend(["--image", str(image_path)])
    return command


def build_prediction_row(
    sample: GQASample,
    response: OverlapResponse,
    max_visual_tokens: int,
    run_log_path: Path,
) -> dict[str, Any]:
    return {
        "benchmark": "gqa",
        "sample_id": sample.question_id,
        "image_ref": f"{sample.image_id}.jpg",
        "question": sample.question,
        "prompt": gqa_prompt(sample.question),
        "prediction": response.text,
        "answer": sample.answer,
        "references": [sample.answer],
        "max_visual_tokens": max_visual_tokens,
        "runner_mode": "overlap_batch",
        "log_path": str(run_log_path.resolve()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--questions-json", type=Path, default=DEFAULT_QUESTIONS_JSON)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--runner-script", type=Path, default=DEFAULT_RUNNER_SCRIPT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--decode-model", type=Path, default=DEFAULT_DECODE_MODEL_PATH)
    parser.add_argument("--sample-n", type=int, default=500)
    parser.add_argument("--max-visual-tokens", type=int, required=True)
    parser.add_argument(
        "--visual-token-pruning-strategy",
        default=DEFAULT_VISUAL_TOKEN_PRUNING_STRATEGY,
    )
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--skip-initial-build", action="store_true")
    parser.add_argument("--skip-push", action="store_true")
    parser.add_argument("--device-dir", default=DEFAULT_DEVICE_DIR)
    parser.add_argument("--prepare-queue-size", type=int, default=DEFAULT_PREPARE_QUEUE_SIZE)
    parser.add_argument("--constraint-regex", default=DEFAULT_GQA_SHORT_ANSWER_REGEX)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_n < 2:
        raise ValueError(f"--sample-n must be >= 2 for overlap eval, got {args.sample_n}")
    if not args.runner_script.is_file():
        raise FileNotFoundError(f"Missing overlap adb runner script: {args.runner_script}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Missing model artifact: {args.model}")
    if not args.decode_model.is_file():
        raise FileNotFoundError(f"Missing decode model artifact: {args.decode_model}")
    if not args.questions_json.is_file():
        raise FileNotFoundError(f"Missing GQA questions file: {args.questions_json}")
    if not args.images_dir.is_dir():
        raise FileNotFoundError(f"Missing GQA images dir: {args.images_dir}")

    samples = load_gqa_questions(args.questions_json)[: args.sample_n]
    if len(samples) < 2:
        raise RuntimeError("Selected sample set must contain at least 2 samples.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"gqa_overlap500_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_jsonl = output_dir / "predictions.jsonl"
    summary_json = output_dir / "summary.json"
    manifest_path = output_dir / "requests.jsonl"

    manifest_rows = build_manifest_rows(samples, args.device_dir)
    write_manifest(manifest_path, manifest_rows)
    image_paths = [args.images_dir / f"{sample.image_id}.jpg" for sample in samples]
    for sample, image_path in zip(samples, image_paths):
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Missing GQA image for sample_id={sample.question_id}: {image_path}"
            )

    command = build_runner_command(
        runner_script=args.runner_script,
        model_path=args.model,
        decode_model_path=args.decode_model,
        manifest_path=manifest_path,
        image_paths=image_paths,
        max_visual_tokens=args.max_visual_tokens,
        visual_token_pruning_strategy=args.visual_token_pruning_strategy,
        max_output_tokens=args.max_output_tokens,
        prepare_queue_size=args.prepare_queue_size,
        constraint_regex=args.constraint_regex,
        skip_build=args.skip_initial_build,
        skip_push=args.skip_push,
        device_dir=args.device_dir,
    )
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    start = time.monotonic()
    result = run_command(command, env)
    elapsed_s = time.monotonic() - start
    if result.returncode != 0:
        raise GQAOverlapRuntimeError(
            "Overlap GQA runner failed. "
            f"exit_code={result.returncode} stdout={result.stdout.strip()} stderr={result.stderr.strip()}"
        )
    run_log_path = extract_run_log_path(result.stdout)
    if not run_log_path.is_file():
        raise GQAOverlapRuntimeError(f"Overlap runner reported missing log path: {run_log_path}")
    parsed_log = parse_overlap_log(run_log_path)
    responses = parsed_log.responses

    rows: list[dict[str, Any]] = []
    for sample in samples:
        response = responses.get(sample.question_id)
        if response is None:
            raise GQAOverlapRuntimeError(
                f"Missing overlap response for question_id={sample.question_id} in {run_log_path}"
            )
        rows.append(
            build_prediction_row(
                sample=sample,
                response=response,
                max_visual_tokens=args.max_visual_tokens,
                run_log_path=run_log_path,
            )
        )
    if len(responses) != len(samples):
        missing = sorted(set(responses) - {sample.question_id for sample in samples})
        if missing:
            raise GQAOverlapRuntimeError(
                f"Unexpected extra overlap responses in {run_log_path}: {missing[:5]}"
            )

    write_jsonl(predictions_jsonl, rows)
    quality = evaluate_predictions(predictions_jsonl, args.benchmark_root)
    summary = {
        "task": "gqa",
        "runner_mode": "overlap_batch",
        "questions_json": str(args.questions_json.resolve()),
        "images_dir": str(args.images_dir.resolve()),
        "predictions_jsonl": str(predictions_jsonl.resolve()),
        "output_dir": str(output_dir.resolve()),
        "run_log": str(run_log_path.resolve()),
        "request_manifest": str(manifest_path.resolve()),
        "sample_count": len(rows),
        "max_visual_tokens": args.max_visual_tokens,
        "visual_token_pruning_strategy": args.visual_token_pruning_strategy,
        "prepare_queue_size": args.prepare_queue_size,
        "wall_clock_s": round(elapsed_s, 3),
        "event_log_split_recoveries": parsed_log.recovered_split_events,
        **quality,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
