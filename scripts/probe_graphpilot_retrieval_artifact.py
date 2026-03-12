#!/usr/bin/env python3
"""Probe for a real on-device GraphPilot retrieval embedder artifact."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
ARTIFACT_ROOT = ROOT_DIR / "artifacts" / "graphpilot_edge"
DEFAULT_OUTPUT_ROOT = ARTIFACT_ROOT / "experiments"
EXPECTED_VARIANT = "embeddinggemma_300m"
EXPECTED_MODEL_HINTS = (
    "embeddinggemma",
    "embedding-gemma",
    "embedding_gemma",
)
DEVICE_SEARCH_ROOTS = (
    "/data/local/tmp/graphpilot_edge",
    "/sdcard/Download",
    "/data/user/0/com.qidk.fastvlm/files/graphpilot_edge",
)
HOST_SEARCH_ROOTS = (
    ROOT_DIR / "models",
    ROOT_DIR / "artifacts",
    ROOT_DIR / "configs",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_candidate(path: str) -> str:
    return path.strip()


def is_retrieval_artifact(path: str) -> bool:
    lowered = path.lower()
    return any(hint in lowered for hint in EXPECTED_MODEL_HINTS)


def summarize_probe(host_matches: list[str], device_matches: list[str]) -> dict[str, Any]:
    normalized_host = [
        normalize_candidate(path) for path in host_matches if is_retrieval_artifact(path)
    ]
    normalized_device = [
        normalize_candidate(path) for path in device_matches if is_retrieval_artifact(path)
    ]
    verdict = "artifact_present" if normalized_host or normalized_device else "missing_artifact"
    return {
        "generated_at": utc_now(),
        "expected_stage": "retrieval.embedder.primary",
        "expected_variant": EXPECTED_VARIANT,
        "verdict": verdict,
        "host_matches": normalized_host,
        "device_matches": normalized_device,
        "usable_stage_variant": EXPECTED_VARIANT if verdict == "artifact_present" else None,
        "remediation": (
            None
            if verdict == "artifact_present"
            else "Missing EmbeddingGemma-300m retrieval artifact. Remediation: stage a real EmbeddingGemma-300m-compatible model under /data/local/tmp/graphpilot_edge/ or record retrieval workflow infeasibility explicitly."
        ),
    }


def host_probe() -> list[str]:
    matches: list[str] = []
    for root in HOST_SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and is_retrieval_artifact(str(path)):
                matches.append(str(path))
    return sorted(set(matches))


def parse_device_probe_result(returncode: int, stdout: str, stderr: str) -> list[str]:
    if returncode not in (0, 1):
        raise RuntimeError(f"adb device probe failed: {stderr.strip()}")
    if returncode == 1 and stderr.strip():
        raise RuntimeError(f"adb device probe failed: {stderr.strip()}")
    return sorted(
        {
            normalize_candidate(line)
            for line in stdout.splitlines()
            if normalize_candidate(line) and is_retrieval_artifact(line)
        }
    )


def device_probe() -> list[str]:
    quoted_roots = " ".join(DEVICE_SEARCH_ROOTS)
    result = subprocess.run(
        [
            "adb",
            "shell",
            f"find {quoted_roots} -maxdepth 4 -type f 2>/dev/null",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return parse_device_probe_result(result.returncode, result.stdout, result.stderr)


def write_summary(output_root: Path, summary: dict[str, Any]) -> Path:
    run_dir = output_root / (
        f"retrieval_artifact_probe_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize_probe(host_matches=host_probe(), device_matches=device_probe())
    summary_path = write_summary(args.output_root, summary)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
