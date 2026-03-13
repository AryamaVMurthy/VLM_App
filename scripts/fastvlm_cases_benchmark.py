#!/usr/bin/env python3
"""Deterministic FastVLM benchmark harness for CASES experiments."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fastvlm_decode_profiles import build_decode_profile
from fastvlm_budget_controller import choose_visual_token_budget
from legacy_fastvlm_paths import LEGACY_CASES_BENCHMARK_ROOT


RUN_LOG_PATTERN = re.compile(r"^Run log:\s+(?P<path>.+)$", re.MULTILINE)
VISUAL_BUDGET_PATTERN = re.compile(
    r"Applied visual token budget: kept (?P<kept>\d+) of (?P<original>\d+) projected vision tokens\."
)
PRUNING_DECISION_PATTERN = re.compile(
    r"Visual token pruning decision: strategy=(?P<strategy>[a-zA-Z0-9_]+) "
    r"kept=(?P<kept>\d+) original=(?P<original>\d+) "
    r"mean_prompt_similarity=(?P<mean_prompt_similarity>[-0-9.eE]+) "
    r"mean_salience=(?P<mean_salience>[-0-9.eE]+)"
)
PRUNING_FLOAT_FIELD_PATTERNS = {
    "mean_reference_prompt_similarity": re.compile(
        r"mean_reference_prompt_similarity=(?P<value>[-0-9.eE]+)"
    ),
    "mean_reference_salience": re.compile(
        r"mean_reference_salience=(?P<value>[-0-9.eE]+)"
    ),
    "mean_local_refinement_prompt_gain": re.compile(
        r"mean_local_refinement_prompt_gain=(?P<value>[-0-9.eE]+)"
    ),
    "max_local_refinement_prompt_gain": re.compile(
        r"max_local_refinement_prompt_gain=(?P<value>[-0-9.eE]+)"
    ),
}
PRUNING_INT_FIELD_PATTERNS = {
    "local_refinement_count": re.compile(r"local_refinement_count=(?P<value>\d+)"),
    "proposed_local_refinement_count": re.compile(
        r"proposed_local_refinement_count=(?P<value>\d+)"
    ),
    "local_refinement_candidate_count": re.compile(
        r"local_refinement_candidate_count=(?P<value>\d+)"
    ),
}
PRUNING_BOOL_FIELD_PATTERNS = {
    "controller_kept_uniform": re.compile(
        r"controller_kept_uniform=(?P<value>true|false)"
    ),
}
PRUNING_STRING_FIELD_PATTERNS = {
    "controller_reason": re.compile(r"controller_reason=(?P<value>[^ ]+)")
}
PRUNING_INDEX_FIELD_PATTERNS = {
    "selected_token_indices": re.compile(
        r"selected_token_indices=(?P<value>\[[^\]]*\])"
    ),
    "reference_token_indices": re.compile(
        r"reference_token_indices=(?P<value>\[[^\]]*\])"
    ),
}
STAT_PATTERNS = {
    "prefill_latency_us": re.compile(r"Total prefill latency \[us\]: (?P<value>\d+)"),
    "prefill_tokens": re.compile(r"\(e2e\) Prefill num tokens: (?P<value>\d+)"),
    "prefill_tokens_per_sec": re.compile(r"\(e2e\) Prefill tokens per second: (?P<value>[0-9.]+)"),
    "prefill_tokens_per_sec_transformer": re.compile(
        r"\(TransformerStackOnly\) Prefill tokens per second: (?P<value>[0-9.]+)"
    ),
    "prefill_mask_latency_us": re.compile(
        r"Total prefill mask inference latency \[us\]: (?P<value>\d+)"
    ),
    "prefill_llm_latency_us": re.compile(
        r"Total prefill LLM inference latency \[us\]: (?P<value>\d+)"
    ),
    "decode_latency_us": re.compile(r"Total decode latency \[us\]: (?P<value>\d+)"),
    "decode_tokens": re.compile(r"Decode num tokens: (?P<value>\d+)"),
    "decode_tokens_per_sec": re.compile(r"Decode tokens per second: (?P<value>[0-9.]+)"),
    "decode_tokens_per_sec_transformer": re.compile(
        r"\(TransformerStackOnly\) Decode tokens per second: (?P<value>[0-9.]+)"
    ),
    "decode_mask_latency_us": re.compile(
        r"Total decode mask inference latency \[us\]: (?P<value>\d+)"
    ),
}
NUMERIC_FIELDS = (
    "ttft_ms",
    "prefill_tokens_per_sec",
    "prefill_tokens_per_sec_transformer",
    "mean_prompt_similarity",
    "mean_salience",
    "decode_tokens_per_sec",
    "decode_tokens_per_sec_transformer",
    "prefill_latency_us",
    "decode_latency_us",
)


def extract_run_log_path(stdout_text: str) -> Path:
    match = RUN_LOG_PATTERN.search(stdout_text)
    if match is None:
        raise ValueError("Runner output did not contain a 'Run log:' line.")
    return Path(match.group("path")).expanduser().resolve()


def parse_pruning_decision_line(line: str) -> dict[str, object] | None:
    match = PRUNING_DECISION_PATTERN.search(line)
    if match is None:
        return None
    metrics: dict[str, object] = {
        "visual_token_pruning_strategy": match.group("strategy"),
        "mean_prompt_similarity": float(match.group("mean_prompt_similarity")),
        "mean_salience": float(match.group("mean_salience")),
    }
    for field, pattern in PRUNING_FLOAT_FIELD_PATTERNS.items():
        field_match = pattern.search(line)
        if field_match is not None:
            metrics[field] = float(field_match.group("value"))
    for field, pattern in PRUNING_INT_FIELD_PATTERNS.items():
        field_match = pattern.search(line)
        if field_match is not None:
            metrics[field] = int(field_match.group("value"))
    for field, pattern in PRUNING_BOOL_FIELD_PATTERNS.items():
        field_match = pattern.search(line)
        if field_match is not None:
            metrics[field] = field_match.group("value") == "true"
    for field, pattern in PRUNING_STRING_FIELD_PATTERNS.items():
        field_match = pattern.search(line)
        if field_match is not None:
            metrics[field] = field_match.group("value")
    for field, pattern in PRUNING_INDEX_FIELD_PATTERNS.items():
        field_match = pattern.search(line)
        if field_match is None:
            continue
        raw_indices = field_match.group("value").strip("[]")
        metrics[field] = [] if not raw_indices else [
            int(item) for item in raw_indices.split(",") if item
        ]
    request_id_match = re.search(r"request_id=(?P<value>[^ ]+)$", line.strip())
    if request_id_match is not None:
        metrics["request_id"] = request_id_match.group("value")
    return metrics


def parse_run_log(log_path: Path) -> dict[str, object]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    metrics: dict[str, object] = {
        "log_path": str(log_path),
        "caption": "",
        "ttft_ms": None,
    }
    caption_parts: list[str] = []
    for line in text.splitlines():
        if not line.startswith("VLM_EVENT "):
            continue
        payload = json.loads(line[len("VLM_EVENT ") :])
        event_type = payload.get("type")
        if event_type == "TOKEN":
            caption_parts.append(str(payload.get("text", "")))
        elif event_type == "BENCHMARK":
            ttft_sec = payload.get("ttft_sec")
            if ttft_sec is not None:
                metrics["ttft_ms"] = round(float(ttft_sec) * 1000.0, 3)
            turns = payload.get("turns") or []
            if turns:
                first_turn = turns[0]
                for field in (
                    "prefill_tokens",
                    "prefill_duration_ms",
                    "prefill_tokens_per_sec",
                    "decode_tokens",
                    "decode_duration_ms",
                    "decode_tokens_per_sec",
                ):
                    if field in first_turn:
                        metrics[f"benchmark_{field}"] = first_turn[field]
        elif event_type == "ERROR":
            metrics["runtime_error"] = payload.get("message", "")
    metrics["caption"] = "".join(caption_parts).strip()

    budget_match = VISUAL_BUDGET_PATTERN.search(text)
    if budget_match is not None:
        metrics["visual_tokens_kept"] = int(budget_match.group("kept"))
        metrics["visual_tokens_original"] = int(budget_match.group("original"))

    pruning_decision_line = next(
        (
            line
            for line in text.splitlines()
            if "Visual token pruning decision:" in line
        ),
        None,
    )
    pruning_decision_match = (
        PRUNING_DECISION_PATTERN.search(pruning_decision_line)
        if pruning_decision_line is not None
        else None
    )
    if pruning_decision_match is not None:
        metrics.update(parse_pruning_decision_line(pruning_decision_line) or {})

    for field, pattern in STAT_PATTERNS.items():
        match = pattern.search(text)
        if match is None:
            continue
        raw_value = match.group("value")
        metrics[field] = float(raw_value) if "." in raw_value else int(raw_value)

    return metrics


def summarize_results(rows: list[dict[str, object]], key_fields: tuple[str, ...]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in key_fields)].append(row)

    summary: list[dict[str, object]] = []
    for group_key, group_rows in sorted(grouped.items()):
        item = {field: value for field, value in zip(key_fields, group_key)}
        item["count"] = len(group_rows)
        for field in NUMERIC_FIELDS:
            values = [float(row[field]) for row in group_rows if row.get(field) is not None]
            if values:
                item[f"{field}_median"] = statistics.median(values)
                item[f"{field}_mean"] = statistics.fmean(values)
        summary.append(item)
    return summary


def parse_budgets(raw: str) -> list[int]:
    budgets: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        budget = int(item)
        if budget < 0:
            raise ValueError(f"Visual token budgets must be non-negative, got {budget}.")
        budgets.append(budget)
    if not budgets:
        raise ValueError("At least one visual token budget is required.")
    return budgets


def collect_images(explicit_images: Iterable[str], images_dir: str | None) -> list[Path]:
    images: list[Path] = []
    for image in explicit_images:
        path = Path(image).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing image file: {path}")
        images.append(path)
    if images_dir:
        root = Path(images_dir).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Missing image directory: {root}")
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            images.extend(sorted(root.glob(pattern)))
    unique_images = []
    seen: set[Path] = set()
    for image in images:
        if image in seen:
            continue
        seen.add(image)
        unique_images.append(image)
    if not unique_images:
        raise ValueError("No images were provided.")
    return unique_images


def run_command(command: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_html_report(path: Path, summary_rows: list[dict[str, object]], measured_rows: list[dict[str, object]]) -> None:
    summary_headers = sorted({key for row in summary_rows for key in row.keys()})
    measured_headers = sorted({key for row in measured_rows for key in row.keys()})

    def render_table(headers: list[str], rows: list[dict[str, object]]) -> str:
        head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
        body_rows = []
        for row in rows:
            cols = "".join(f"<td>{html.escape(str(row.get(header, '')))}</td>" for header in headers)
            body_rows.append(f"<tr>{cols}</tr>")
        return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>FastVLM CASES Benchmark</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; font-size: 13px; }}
    th {{ background: #f5f5f5; text-align: left; }}
    h1, h2 {{ margin-top: 24px; }}
  </style>
</head>
<body>
  <h1>FastVLM CASES Benchmark</h1>
  <h2>Summary</h2>
  {render_table(summary_headers, summary_rows)}
  <h2>Measured Runs</h2>
  {render_table(measured_headers, measured_rows)}
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def build_command(
    args: argparse.Namespace,
    image_path: Path,
    budget: int,
    prompt: str,
    max_output_tokens: int | None,
    visual_token_pruning_strategy: str,
) -> list[str]:
    command = [
        str(Path(args.runner).resolve()),
        "--image",
        str(image_path),
        "--prompt",
        prompt,
        "--max-num-tokens",
        str(args.max_num_tokens),
        "--max-visual-tokens",
        str(budget),
        "--visual-token-pruning-strategy",
        visual_token_pruning_strategy,
        "--jobs",
        str(args.jobs),
        "--skip-build",
        str(args.skip_build),
        "--skip-push",
        str(args.skip_push),
        "--benchmark",
        "1",
        "--event-mode",
        "1",
    ]
    if max_output_tokens is not None:
        command.extend(["--max-output-tokens", str(max_output_tokens)])
    if args.model:
        command.extend(["--model", str(Path(args.model).resolve())])
    return command


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", default="scripts/run_fastvlm_litert_npu_adb.sh")
    parser.add_argument("--model")
    parser.add_argument("--images", nargs="*", default=[])
    parser.add_argument("--images-dir")
    parser.add_argument("--prompt", default="Describe this image in one sentence.")
    parser.add_argument("--answer-mode", choices=("none", "short", "long"), default="none")
    parser.add_argument("--short-max-output-tokens", type=int, default=64)
    parser.add_argument("--long-max-output-tokens", type=int, default=256)
    parser.add_argument("--max-num-tokens", type=int, default=512)
    parser.add_argument("--budgets", default="0")
    parser.add_argument(
        "--budget-policy", choices=("fixed", "adaptive_v1"), default="fixed"
    )
    parser.add_argument(
        "--visual-token-pruning-strategy",
        default="prompt_conditioned_v1",
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--skip-build", type=int, choices=(0, 1), default=1)
    parser.add_argument("--skip-push", type=int, choices=(0, 1), default=0)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)

    budgets = parse_budgets(args.budgets)
    images = collect_images(args.images, args.images_dir)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else LEGACY_CASES_BENCHMARK_ROOT
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)

    all_rows: list[dict[str, object]] = []
    measured_rows: list[dict[str, object]] = []
    for image_pass_index in range(1 if args.budget_policy != "fixed" else len(budgets)):
        last_metrics: dict[str, object] | None = None
        budget = budgets[image_pass_index] if args.budget_policy == "fixed" else None
        for image_index, image_path in enumerate(images):
            if args.answer_mode == "none":
                prompt = args.prompt
                max_output_tokens = None
            else:
                decode_profile = build_decode_profile(
                    answer_mode=args.answer_mode,
                    base_prompt=args.prompt,
                    short_max_output_tokens=args.short_max_output_tokens,
                    long_max_output_tokens=args.long_max_output_tokens,
                )
                prompt = decode_profile.prompt
                max_output_tokens = decode_profile.max_output_tokens
            total_runs = args.warmup_runs + args.measured_runs
            for run_index in range(total_runs):
                is_warmup = run_index < args.warmup_runs
                if args.budget_policy == "adaptive_v1":
                    recent_prefill_ms = (
                        None
                        if last_metrics is None
                        else float(last_metrics["prefill_latency_us"]) / 1000.0
                    )
                    recent_decode_ms = (
                        None
                        if last_metrics is None
                        else float(last_metrics["decode_latency_us"]) / 1000.0
                    )
                    budget_decision = choose_visual_token_budget(
                        budgets,
                        queue_depth=len(images) - image_index - 1,
                        queued_image_count=len(images) - image_index,
                        recent_prefill_ms=recent_prefill_ms,
                        recent_decode_ms=recent_decode_ms,
                        answer_mode=args.answer_mode,
                    )
                    budget = budget_decision.budget
                    budget_reason_codes = ",".join(budget_decision.reason_codes)
                else:
                    budget_reason_codes = "fixed_budget"
                command = build_command(
                    args,
                    image_path,
                    budget,
                    prompt=prompt,
                    max_output_tokens=max_output_tokens,
                    visual_token_pruning_strategy=args.visual_token_pruning_strategy,
                )
                result = run_command(command, env)
                stdout_path = output_dir / f"{image_path.stem}_budget{budget}_run{run_index}_stdout.txt"
                stderr_path = output_dir / f"{image_path.stem}_budget{budget}_run{run_index}_stderr.txt"
                stdout_path.write_text(result.stdout, encoding="utf-8")
                stderr_path.write_text(result.stderr, encoding="utf-8")
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Benchmark run failed for image={image_path} budget={budget} run={run_index}. "
                        f"See {stdout_path} and {stderr_path}."
                    )
                run_log_path = extract_run_log_path(result.stdout)
                metrics = parse_run_log(run_log_path)
                metrics.update(
                    {
                        "image": str(image_path),
                        "image_name": image_path.name,
                        "budget": budget,
                        "budget_policy": args.budget_policy,
                        "budget_reason_codes": budget_reason_codes,
                        "visual_token_pruning_strategy_requested": args.visual_token_pruning_strategy,
                        "queue_depth": len(images) - image_index - 1,
                        "queued_image_count": len(images) - image_index,
                        "run_index": run_index,
                        "is_warmup": is_warmup,
                        "answer_mode": args.answer_mode,
                        "prompt": prompt,
                        "max_output_tokens": max_output_tokens,
                    }
                )
                all_rows.append(metrics)
                last_metrics = metrics
                if not is_warmup:
                    measured_rows.append(metrics)

    summary_key_fields = (
        ("budget_policy", "budget")
        if args.budget_policy != "fixed"
        else ("budget",)
    )
    summary_rows = summarize_results(measured_rows, key_fields=summary_key_fields)
    write_csv(output_dir / "all_runs.csv", all_rows)
    write_csv(output_dir / "measured_runs.csv", measured_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "all_runs.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    render_html_report(output_dir / "report.html", summary_rows, measured_rows)
    print(f"Wrote FastVLM CASES benchmark outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
