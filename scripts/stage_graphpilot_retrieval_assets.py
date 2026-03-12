#!/usr/bin/env python3
"""Stage GraphPilot retrieval assets and binaries onto the connected device."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
LITERT_LM_DIR = ROOT_DIR / "third_party/litert-lm"
DEFAULT_HOST_BINARY = LITERT_LM_DIR / "bazel-bin/runtime/engine/graphpilot_retrieval_main"
DEFAULT_ANDROID_BINARY = LITERT_LM_DIR / "bazel-bin/runtime/engine/graphpilot_retrieval_main"
DEFAULT_LITERT_RUNTIME = LITERT_LM_DIR / "bazel-bin/external/litert/litert/c/libLiteRt.so"
DEFAULT_GPU_ACCELERATOR = LITERT_LM_DIR / "prebuilt/android_arm64/libLiteRtOpenClAccelerator.so"
DEFAULT_GPU_SAMPLER = LITERT_LM_DIR / "prebuilt/android_arm64/libLiteRtTopKOpenClSampler.so"
DEFAULT_GEMMA_CONSTRAINT_PROVIDER = (
    LITERT_LM_DIR / "prebuilt/android_arm64/libGemmaModelConstraintProvider.so"
)
DEFAULT_MODEL = ROOT_DIR / "artifacts/models/embeddinggemma-300M_seq512_mixed-precision.tflite"
DEFAULT_TOKENIZER = ROOT_DIR / "artifacts/models/tokenizer.model"
DEFAULT_KB = ROOT_DIR / "artifacts/graphpilot_edge/retrieval/retrieval_kb.json"
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "artifacts/graphpilot_edge/experiments"
DEFAULT_DEVICE_DIR = "/data/local/tmp/graphpilot_edge"


def require_file(path: Path, remediation: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file '{path}'. Remediation: {remediation}")


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def adb(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return run(["adb", *cmd], cwd=ROOT_DIR)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-binary", type=Path, default=DEFAULT_HOST_BINARY)
    parser.add_argument("--android-binary", type=Path, default=DEFAULT_ANDROID_BINARY)
    parser.add_argument("--litert-runtime-so", type=Path, default=DEFAULT_LITERT_RUNTIME)
    parser.add_argument("--gpu-accelerator-so", type=Path, default=DEFAULT_GPU_ACCELERATOR)
    parser.add_argument("--gpu-sampler-so", type=Path, default=DEFAULT_GPU_SAMPLER)
    parser.add_argument(
        "--gemma-constraint-provider-so",
        type=Path,
        default=DEFAULT_GEMMA_CONSTRAINT_PROVIDER,
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--kb-path", type=Path, default=DEFAULT_KB)
    parser.add_argument("--device-dir", default=DEFAULT_DEVICE_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require_file(args.host_binary, "build the host retrieval binary first.")
    require_file(
        args.android_binary,
        "build the android retrieval binary first: cd third_party/litert-lm && bazel build --config=android_arm64 --jobs=18 //runtime/engine:graphpilot_retrieval_main",
    )
    require_file(
        args.litert_runtime_so,
        "build the LiteRT runtime shared library first: cd third_party/litert-lm && bazel build --config=android_arm64 --jobs=18 @litert//litert/c:litert_runtime_c_api_so",
    )
    require_file(
        args.gpu_accelerator_so,
        "package the LiteRT OpenCL GPU accelerator under third_party/litert-lm/prebuilt/android_arm64/.",
    )
    require_file(
        args.gpu_sampler_so,
        "package the LiteRT OpenCL TopK sampler under third_party/litert-lm/prebuilt/android_arm64/.",
    )
    require_file(
        args.gemma_constraint_provider_so,
        "package libGemmaModelConstraintProvider.so under third_party/litert-lm/prebuilt/android_arm64/.",
    )
    require_file(args.model_path, "stage the real EmbeddingGemma retrieval model under artifacts/models/.")
    require_file(args.tokenizer_path, "stage tokenizer.model under artifacts/models/.")
    require_file(args.kb_path, "run scripts/build_graphpilot_retrieval_kb.py first.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"retrieval_stage_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    commands = []

    for cmd in (
        ["root"],
        ["wait-for-device"],
        ["shell", f"mkdir -p '{args.device_dir}'"],
        ["push", str(args.android_binary), f"{args.device_dir}/graphpilot_retrieval_main"],
        ["push", str(args.model_path), f"{args.device_dir}/{args.model_path.name}"],
        ["push", str(args.tokenizer_path), f"{args.device_dir}/{args.tokenizer_path.name}"],
        ["push", str(args.kb_path), f"{args.device_dir}/{args.kb_path.name}"],
        ["shell", f"mkdir -p '{args.device_dir}/gpu_libs_opencl_only'"],
        ["push", str(args.litert_runtime_so), f"{args.device_dir}/gpu_libs_opencl_only/libLiteRt.so"],
        [
            "push",
            str(args.gpu_accelerator_so),
            f"{args.device_dir}/gpu_libs_opencl_only/{args.gpu_accelerator_so.name}",
        ],
        [
            "push",
            str(args.gpu_sampler_so),
            f"{args.device_dir}/gpu_libs_opencl_only/{args.gpu_sampler_so.name}",
        ],
        [
            "push",
            str(args.gemma_constraint_provider_so),
            f"{args.device_dir}/gpu_libs_opencl_only/{args.gemma_constraint_provider_so.name}",
        ],
        ["shell", f"chmod 0755 '{args.device_dir}/graphpilot_retrieval_main'"],
    ):
        result = adb(cmd)
        commands.append(
            {
                "command": ["adb", *cmd],
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Failed to stage retrieval assets.\n"
                f"command={' '.join(['adb', *cmd])}\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
            )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "device_dir": args.device_dir,
        "host_binary": str(args.host_binary.resolve()),
        "android_binary": str(args.android_binary.resolve()),
        "litert_runtime_so": str(args.litert_runtime_so.resolve()),
        "gpu_accelerator_so": str(args.gpu_accelerator_so.resolve()),
        "gpu_sampler_so": str(args.gpu_sampler_so.resolve()),
        "gemma_constraint_provider_so": str(args.gemma_constraint_provider_so.resolve()),
        "model_path": str(args.model_path.resolve()),
        "tokenizer_path": str(args.tokenizer_path.resolve()),
        "kb_path": str(args.kb_path.resolve()),
        "commands": commands,
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
