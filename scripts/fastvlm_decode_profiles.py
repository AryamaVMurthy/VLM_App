#!/usr/bin/env python3
"""Explicit answer-mode scaffolding for FastVLM experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodeProfile:
    answer_mode: str
    prompt: str
    max_output_tokens: int


def build_decode_profile(
    answer_mode: str,
    base_prompt: str,
    short_max_output_tokens: int,
    long_max_output_tokens: int,
) -> DecodeProfile:
    normalized_mode = answer_mode.strip().lower()
    if normalized_mode == "short":
        prompt = f"{base_prompt.rstrip()} Answer briefly."
        return DecodeProfile(
            answer_mode="short",
            prompt=prompt,
            max_output_tokens=short_max_output_tokens,
        )
    if normalized_mode == "long":
        prompt = f"{base_prompt.rstrip()} Answer in detail."
        return DecodeProfile(
            answer_mode="long",
            prompt=prompt,
            max_output_tokens=long_max_output_tokens,
        )
    raise ValueError(f"Unsupported answer mode: {answer_mode}")
