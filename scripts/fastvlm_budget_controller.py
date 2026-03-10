#!/usr/bin/env python3
"""Discrete visual-token budget controller for FastVLM CASES runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class BudgetDecision:
    budget: int
    reason_codes: tuple[str, ...]


def _normalize_buckets(allowed_buckets: Iterable[int]) -> list[int]:
    buckets = sorted(set(int(bucket) for bucket in allowed_buckets))
    if not buckets:
        raise ValueError("At least one visual token budget bucket is required.")
    if buckets[0] <= 0:
        raise ValueError(
            f"Visual token budget buckets must be positive, got {buckets[0]}."
        )
    return buckets


def choose_visual_token_budget(
    allowed_buckets: Iterable[int],
    *,
    queue_depth: int,
    queued_image_count: int,
    recent_prefill_ms: float | None,
    recent_decode_ms: float | None,
    previous_budget: int | None = None,
    max_step_change: int = 1,
    answer_mode: str = "none",
) -> BudgetDecision:
    buckets = _normalize_buckets(allowed_buckets)
    if queue_depth < 0:
        raise ValueError(f"queue_depth must be non-negative, got {queue_depth}.")
    if queued_image_count < 0:
        raise ValueError(
            f"queued_image_count must be non-negative, got {queued_image_count}."
        )
    if max_step_change < 0:
        raise ValueError(f"max_step_change must be non-negative, got {max_step_change}.")
    normalized_answer_mode = answer_mode.strip().lower()
    if normalized_answer_mode not in {"none", "short", "long"}:
        raise ValueError(f"Unsupported answer_mode: {answer_mode}")

    bucket_index = len(buckets) - 1
    reason_codes: list[str] = []

    if queue_depth >= 4 or queued_image_count >= 4:
        bucket_index = max(0, bucket_index - 1)
        reason_codes.append("queue_pressure")
    if queue_depth >= 8 or queued_image_count >= 8:
        bucket_index = 0
        reason_codes.append("heavy_queue_pressure")

    if recent_prefill_ms is not None and recent_decode_ms is not None:
        if recent_prefill_ms <= 0 or recent_decode_ms <= 0:
            raise ValueError(
                "recent_prefill_ms and recent_decode_ms must be positive when "
                "provided."
            )
        if recent_prefill_ms > recent_decode_ms * 1.10:
            bucket_index = max(0, bucket_index - 1)
            reason_codes.append("prefill_above_decode_window")
        elif recent_prefill_ms < recent_decode_ms * 0.75:
            bucket_index = min(len(buckets) - 1, bucket_index + 1)
            reason_codes.append("prefill_below_decode_window")

    if normalized_answer_mode == "short" and bucket_index > 0:
        bucket_index = max(0, bucket_index - 1)
        reason_codes.append("short_answer_mode")
    elif normalized_answer_mode == "long" and bucket_index < len(buckets) - 1:
        bucket_index = min(len(buckets) - 1, bucket_index + 1)
        reason_codes.append("long_answer_mode")

    if previous_budget is not None:
        if previous_budget not in buckets:
            raise ValueError(
                f"previous_budget must be one of the allowed buckets, got {previous_budget}."
            )
        previous_index = buckets.index(previous_budget)
        delta = bucket_index - previous_index
        if abs(delta) > max_step_change:
            bucket_index = previous_index + (
                max_step_change if delta > 0 else -max_step_change
            )
            reason_codes.append("limited_step_change")

    if not reason_codes:
        reason_codes.append("baseline_bucket")
    return BudgetDecision(budget=buckets[bucket_index], reason_codes=tuple(reason_codes))
