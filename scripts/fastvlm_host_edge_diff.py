#!/usr/bin/env python3
"""Compare host full-precision FastVLM outputs against edge LiteRT overlap runs."""

from __future__ import annotations

import argparse
import csv
import html
import importlib
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fastvlm_cases_benchmark import extract_run_log_path, parse_pruning_decision_line
from run_fastvlm_litert_gqa_eval import (
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_GQA_SHORT_ANSWER_REGEX,
    DEFAULT_IMAGES_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_QUESTIONS_JSON,
    DEFAULT_VISUAL_TOKEN_PRUNING_STRATEGY,
    GQASample,
    gqa_prompt,
    load_gqa_questions,
)
from run_fastvlm_litert_gqa_overlap_eval import (
    DEFAULT_DECODE_MODEL_PATH,
    DEFAULT_DEVICE_DIR,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_PREPARE_QUEUE_SIZE,
    DEFAULT_RUNNER_SCRIPT,
    GQAOverlapRuntimeError,
    build_manifest_rows,
    build_runner_command,
    run_command,
    write_manifest,
)

DEFAULT_OUTPUT_ROOT = SCRIPT_DIR.parent / "artifacts/host_edge_diff"
DEFAULT_HOST_BENCH_ROOT = Path(
    "/home/aryamavmurthy/work/Liquid_benchmarking_image tokens pruning"
)
DEFAULT_HOST_MODEL_ID = "apple/FastVLM-0.5B"


@dataclass(frozen=True)
class HostCaseResult:
    prediction: str
    correct: bool
    latency_s: float
    ttft_s: float
    prefill_s: float
    decode_s: float
    decode_tokens_per_s: float
    peak_mem_gb: float
    generated_tokens: int
    vision_tokens: int | None


@dataclass(frozen=True)
class EdgeCaseResult:
    prediction: str
    correct: bool
    prepare_ms: float | None
    prepare_queue_wait_ms: float | None
    prefill_ms: float | None
    decode_ms: float | None
    local_refinement_count: int | None
    local_refinement_candidate_count: int | None
    mean_prompt_similarity: float | None
    mean_reference_prompt_similarity: float | None
    mean_local_refinement_prompt_gain: float | None
    max_local_refinement_prompt_gain: float | None
    selected_token_indices: list[int] | None
    reference_token_indices: list[int] | None


@dataclass(frozen=True)
class ParsedEventStream:
    events: list[dict[str, Any]]
    recovered_split_events: int


@dataclass(frozen=True)
class HostRuntimeComponents:
    runtime_class: Any
    vision_config_class: Any
    quantization_config_class: Any
    answer_matcher: Any


@dataclass(frozen=True)
class EdgeRequestMetrics:
    prepare_ms: float | None = None
    prepare_queue_wait_ms: float | None = None
    prefill_ms: float | None = None
    decode_ms: float | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_html(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    headers = [
        "sample_id",
        "image_id",
        "question",
        "answer",
        "host_prediction",
        "host_correct",
        "edge_prediction",
        "edge_correct",
        "predictions_match",
        "edge_prepare_ms",
        "edge_prefill_ms",
        "edge_decode_ms",
        "edge_local_refinement_count",
        "edge_mean_local_refinement_prompt_gain",
    ]
    body_rows: list[str] = []
    for row in rows:
        body_rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(row.get(header, '')))}</td>" for header in headers
            )
            + "</tr>"
        )
    summary_items = "".join(
        f"<li><strong>{html.escape(str(key))}</strong>: {html.escape(str(value))}</li>"
        for key, value in summary.items()
    )
    doc = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>FastVLM Host vs Edge Differential</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; font-size: 13px; vertical-align: top; }}
    th {{ background: #f5f5f5; text-align: left; }}
  </style>
</head>
<body>
  <h1>FastVLM Host vs Edge Differential</h1>
  <ul>{summary_items}</ul>
  <table>
    <thead><tr>{''.join(f'<th>{html.escape(h)}</th>' for h in headers)}</tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def _parse_sample_ids(raw: str) -> list[str]:
    sample_ids: list[str] = []
    seen: set[str] = set()
    for item in raw.split(","):
        sample_id = item.strip()
        if not sample_id or sample_id in seen:
            continue
        seen.add(sample_id)
        sample_ids.append(sample_id)
    if not sample_ids:
        raise ValueError("At least one sample id is required.")
    return sample_ids


def _select_samples(samples: list[GQASample], sample_ids: list[str] | None, sample_n: int | None) -> list[GQASample]:
    if sample_ids:
        sample_by_id = {sample.question_id: sample for sample in samples}
        selected: list[GQASample] = []
        missing: list[str] = []
        for sample_id in sample_ids:
            sample = sample_by_id.get(sample_id)
            if sample is None:
                missing.append(sample_id)
            else:
                selected.append(sample)
        if missing:
            raise KeyError(f"Unknown sample ids: {missing}")
        return selected
    if sample_n is None or sample_n <= 0:
        raise ValueError("Provide either --sample-ids or --sample-n > 0.")
    return samples[:sample_n]


def _load_host_runtime_components(host_benchmark_root: Path) -> HostRuntimeComponents:
    src_root = host_benchmark_root / "src"
    if not src_root.is_dir():
        raise FileNotFoundError(f"Missing host benchmark src directory: {src_root}")
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    try:
        model_runtime = importlib.import_module("lfm2vl_bench.model_runtime")
        config = importlib.import_module("lfm2vl_bench.config")
        eval_gqa = importlib.import_module("lfm2vl_bench.eval_gqa")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Failed to import host FastVLM benchmark modules from {src_root}"
        ) from exc
    return HostRuntimeComponents(
        runtime_class=model_runtime.LFM2Runtime,
        vision_config_class=config.VisionConfig,
        quantization_config_class=config.QuantizationConfig,
        answer_matcher=eval_gqa.is_correct_gqa_answer,
    )


def parse_event_stream(log_path: Path) -> ParsedEventStream:
    events: list[dict[str, Any]] = []
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
            if line_index < len(lines) and lines[line_index].lstrip().startswith("{"):
                payload_text = lines[line_index].lstrip()
                recovered_split_events += 1
                line_index += 1
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
        events.append(payload)
    return ParsedEventStream(events=events, recovered_split_events=recovered_split_events)


def parse_edge_timings(events: list[dict[str, Any]]) -> dict[str, EdgeRequestMetrics]:
    timings: dict[str, EdgeRequestMetrics] = {}
    for event in events:
        request_id = event.get("request_id")
        if not request_id:
            continue
        request_id = str(request_id)
        current = timings.get(request_id, EdgeRequestMetrics())
        event_type = event.get("type")
        duration_ms = event.get("duration_ms")
        if duration_ms is not None:
            duration_ms = float(duration_ms)
        if event_type == "PREPARE_DONE":
            current = EdgeRequestMetrics(
                prepare_ms=duration_ms,
                prepare_queue_wait_ms=current.prepare_queue_wait_ms,
                prefill_ms=current.prefill_ms,
                decode_ms=current.decode_ms,
            )
        elif event_type == "PREPARE_QUEUE_WAIT":
            current = EdgeRequestMetrics(
                prepare_ms=current.prepare_ms,
                prepare_queue_wait_ms=duration_ms,
                prefill_ms=current.prefill_ms,
                decode_ms=current.decode_ms,
            )
        elif event_type == "PREFILL_DONE":
            current = EdgeRequestMetrics(
                prepare_ms=current.prepare_ms,
                prepare_queue_wait_ms=current.prepare_queue_wait_ms,
                prefill_ms=duration_ms,
                decode_ms=current.decode_ms,
            )
        elif event_type == "DECODE_DONE":
            current = EdgeRequestMetrics(
                prepare_ms=current.prepare_ms,
                prepare_queue_wait_ms=current.prepare_queue_wait_ms,
                prefill_ms=current.prefill_ms,
                decode_ms=duration_ms,
            )
        timings[request_id] = current
    return timings


def parse_pruning_decisions(log_path: Path) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parsed = parse_pruning_decision_line(line)
        if parsed is None:
            continue
        request_id = parsed.get("request_id")
        if not request_id:
            raise RuntimeError(
                "Pruning decision log line is missing request_id; rebuild the overlap binary "
                f"with the current observability changes and rerun. log_path={log_path}"
            )
        decisions[str(request_id)] = parsed
    return decisions


def _run_host_cases(
    *,
    samples: list[GQASample],
    images_dir: Path,
    host_components: HostRuntimeComponents,
    host_model_id: str,
    host_device: str,
    host_inference_engine: str,
    host_cpu_threads: int | None,
    host_cpu_interop_threads: int | None,
    max_output_tokens: int,
) -> dict[str, HostCaseResult]:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required for host-side differential runs.") from exc

    runtime = host_components.runtime_class.load(
        model_id=host_model_id,
        quantization_cfg=host_components.quantization_config_class(
            name="bf16_ref", method="none"
        ),
        device_preference=host_device,
        inference_engine=host_inference_engine,
        cpu_num_threads=host_cpu_threads,
        cpu_num_interop_threads=host_cpu_interop_threads,
    )
    runtime.apply_vision_config(
        host_components.vision_config_class(
            name="host_full_precision",
            min_image_tokens=64,
            max_image_tokens=256,
            do_image_splitting=False,
            vision_token_stride=1,
            collect_exact_vision_tokens=True,
        )
    )

    results: dict[str, HostCaseResult] = {}
    for sample in samples:
        image_path = images_dir / f"{sample.image_id}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Missing GQA image for sample_id={sample.question_id}: {image_path}"
            )
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
            generation = runtime.generate(
                image,
                gqa_prompt(sample.question),
                max_output_tokens,
            )
        results[sample.question_id] = HostCaseResult(
            prediction=generation.text,
            correct=bool(host_components.answer_matcher(generation.text, sample.answer)),
            latency_s=float(generation.latency_s),
            ttft_s=float(generation.ttft_s),
            prefill_s=float(generation.prefill_s),
            decode_s=float(generation.decode_s),
            decode_tokens_per_s=float(generation.decode_tokens_per_s),
            peak_mem_gb=float(generation.peak_mem_gb),
            generated_tokens=int(generation.generated_tokens),
            vision_tokens=(None if generation.vision_tokens is None else int(generation.vision_tokens)),
        )
    return results


def _run_edge_cases(
    *,
    samples: list[GQASample],
    images_dir: Path,
    output_dir: Path,
    runner_script: Path,
    model_path: Path,
    decode_model_path: Path,
    max_visual_tokens: int,
    visual_token_pruning_strategy: str,
    max_output_tokens: int,
    prepare_queue_size: int,
    constraint_regex: str,
    skip_build: bool,
    skip_push: bool,
    device_dir: str,
) -> tuple[dict[str, str], dict[str, EdgeRequestMetrics], dict[str, dict[str, Any]], Path, int, float]:
    manifest_path = output_dir / "requests.jsonl"
    manifest_rows = build_manifest_rows(samples, device_dir)
    write_manifest(manifest_path, manifest_rows)
    image_paths = [images_dir / f"{sample.image_id}.jpg" for sample in samples]
    for sample, image_path in zip(samples, image_paths):
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Missing GQA image for sample_id={sample.question_id}: {image_path}"
            )

    command = build_runner_command(
        runner_script=runner_script,
        model_path=model_path,
        decode_model_path=decode_model_path,
        manifest_path=manifest_path,
        image_paths=image_paths,
        max_visual_tokens=max_visual_tokens,
        visual_token_pruning_strategy=visual_token_pruning_strategy,
        max_output_tokens=max_output_tokens,
        prepare_queue_size=prepare_queue_size,
        constraint_regex=constraint_regex,
        skip_build=skip_build,
        skip_push=skip_push,
        device_dir=device_dir,
    )
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    start = time.monotonic()
    result = run_command(command, env)
    elapsed_s = time.monotonic() - start
    if result.returncode != 0:
        raise GQAOverlapRuntimeError(
            "Edge overlap differential runner failed. "
            f"exit_code={result.returncode} stdout={result.stdout.strip()} stderr={result.stderr.strip()}"
        )
    run_log_path = extract_run_log_path(result.stdout)
    if not run_log_path.is_file():
        raise FileNotFoundError(f"Overlap runner reported missing log path: {run_log_path}")
    parsed_stream = parse_event_stream(run_log_path)
    timings = parse_edge_timings(parsed_stream.events)
    responses: dict[str, str] = {}
    for event in parsed_stream.events:
        if event.get("type") != "OVERLAP_RESPONSE":
            continue
        request_id = str(event.get("request_id", ""))
        if not request_id:
            raise GQAOverlapRuntimeError(
                f"Missing request_id in OVERLAP_RESPONSE event from {run_log_path}"
            )
        responses[request_id] = str(event.get("text", "")).strip()
    pruning_decisions = parse_pruning_decisions(run_log_path)
    return (
        responses,
        timings,
        pruning_decisions,
        run_log_path,
        parsed_stream.recovered_split_events,
        elapsed_s,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--questions-json", type=Path, default=DEFAULT_QUESTIONS_JSON)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--runner-script", type=Path, default=DEFAULT_RUNNER_SCRIPT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--decode-model", type=Path, default=DEFAULT_DECODE_MODEL_PATH)
    parser.add_argument("--host-benchmark-root", type=Path, default=DEFAULT_HOST_BENCH_ROOT)
    parser.add_argument("--host-model-id", default=DEFAULT_HOST_MODEL_ID)
    parser.add_argument("--host-device", default="cpu")
    parser.add_argument("--host-inference-engine", default="hf_eager")
    parser.add_argument("--host-cpu-threads", type=int, default=None)
    parser.add_argument("--host-cpu-interop-threads", type=int, default=None)
    parser.add_argument("--sample-ids", default="")
    parser.add_argument("--sample-n", type=int, default=None)
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
    if not args.questions_json.is_file():
        raise FileNotFoundError(f"Missing GQA questions file: {args.questions_json}")
    if not args.images_dir.is_dir():
        raise FileNotFoundError(f"Missing GQA images dir: {args.images_dir}")
    if not args.runner_script.is_file():
        raise FileNotFoundError(f"Missing overlap adb runner script: {args.runner_script}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Missing model artifact: {args.model}")
    if not args.decode_model.is_file():
        raise FileNotFoundError(f"Missing decode model artifact: {args.decode_model}")

    sample_ids = _parse_sample_ids(args.sample_ids) if args.sample_ids.strip() else None
    all_samples = load_gqa_questions(args.questions_json)
    samples = _select_samples(all_samples, sample_ids, args.sample_n)
    if not samples:
        raise RuntimeError("No GQA samples selected for differential run.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"host_edge_diff_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    host_components = _load_host_runtime_components(args.host_benchmark_root)
    host_results = _run_host_cases(
        samples=samples,
        images_dir=args.images_dir,
        host_components=host_components,
        host_model_id=args.host_model_id,
        host_device=args.host_device,
        host_inference_engine=args.host_inference_engine,
        host_cpu_threads=args.host_cpu_threads,
        host_cpu_interop_threads=args.host_cpu_interop_threads,
        max_output_tokens=args.max_output_tokens,
    )
    (
        edge_responses,
        edge_timings,
        pruning_decisions,
        run_log_path,
        recovered_split_events,
        edge_wall_clock_s,
    ) = _run_edge_cases(
        samples=samples,
        images_dir=args.images_dir,
        output_dir=output_dir,
        runner_script=args.runner_script,
        model_path=args.model,
        decode_model_path=args.decode_model,
        max_visual_tokens=args.max_visual_tokens,
        visual_token_pruning_strategy=args.visual_token_pruning_strategy,
        max_output_tokens=args.max_output_tokens,
        prepare_queue_size=args.prepare_queue_size,
        constraint_regex=args.constraint_regex,
        skip_build=args.skip_initial_build,
        skip_push=args.skip_push,
        device_dir=args.device_dir,
    )

    rows: list[dict[str, Any]] = []
    for sample in samples:
        host_result = host_results[sample.question_id]
        edge_prediction = edge_responses.get(sample.question_id)
        if edge_prediction is None:
            raise RuntimeError(
                f"Missing edge response for sample_id={sample.question_id} in {run_log_path}"
            )
        timing = edge_timings.get(sample.question_id, EdgeRequestMetrics())
        pruning = pruning_decisions.get(sample.question_id, {})
        edge_correct = bool(host_components.answer_matcher(edge_prediction, sample.answer))
        rows.append(
            {
                "sample_id": sample.question_id,
                "image_id": sample.image_id,
                "question": sample.question,
                "answer": sample.answer,
                "host_prediction": host_result.prediction,
                "host_correct": host_result.correct,
                "host_latency_s": host_result.latency_s,
                "host_ttft_s": host_result.ttft_s,
                "host_prefill_s": host_result.prefill_s,
                "host_decode_s": host_result.decode_s,
                "host_decode_tokens_per_s": host_result.decode_tokens_per_s,
                "host_generated_tokens": host_result.generated_tokens,
                "host_vision_tokens": host_result.vision_tokens,
                "edge_prediction": edge_prediction,
                "edge_correct": edge_correct,
                "predictions_match": host_result.prediction.strip() == edge_prediction.strip(),
                "edge_prepare_ms": timing.prepare_ms,
                "edge_prepare_queue_wait_ms": timing.prepare_queue_wait_ms,
                "edge_prefill_ms": timing.prefill_ms,
                "edge_decode_ms": timing.decode_ms,
                "edge_local_refinement_count": pruning.get("local_refinement_count"),
                "edge_local_refinement_candidate_count": pruning.get(
                    "local_refinement_candidate_count"
                ),
                "edge_mean_prompt_similarity": pruning.get("mean_prompt_similarity"),
                "edge_mean_reference_prompt_similarity": pruning.get(
                    "mean_reference_prompt_similarity"
                ),
                "edge_mean_local_refinement_prompt_gain": pruning.get(
                    "mean_local_refinement_prompt_gain"
                ),
                "edge_max_local_refinement_prompt_gain": pruning.get(
                    "max_local_refinement_prompt_gain"
                ),
                "edge_selected_token_indices": pruning.get("selected_token_indices"),
                "edge_reference_token_indices": pruning.get("reference_token_indices"),
            }
        )

    deltas = Counter()
    for row in rows:
        key = (bool(row["host_correct"]), bool(row["edge_correct"]))
        deltas[key] += 1
    summary = {
        "timestamp": _utc_now(),
        "sample_count": len(rows),
        "host_correct": sum(1 for row in rows if row["host_correct"]),
        "edge_correct": sum(1 for row in rows if row["edge_correct"]),
        "prediction_match_count": sum(1 for row in rows if row["predictions_match"]),
        "host_only_correct": deltas[(True, False)],
        "edge_only_correct": deltas[(False, True)],
        "both_correct": deltas[(True, True)],
        "both_wrong": deltas[(False, False)],
        "max_visual_tokens": args.max_visual_tokens,
        "visual_token_pruning_strategy": args.visual_token_pruning_strategy,
        "host_device": args.host_device,
        "host_model_id": args.host_model_id,
        "edge_wall_clock_s": round(edge_wall_clock_s, 3),
        "event_log_split_recoveries": recovered_split_events,
        "run_log": str(run_log_path.resolve()),
    }

    json_path = output_dir / "comparison.json"
    csv_path = output_dir / "comparison.csv"
    html_path = output_dir / "report.html"
    summary_path = output_dir / "summary.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_csv(csv_path, rows)
    _render_html(html_path, summary, rows)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "output_dir": str(output_dir.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
