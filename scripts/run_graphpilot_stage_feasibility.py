#!/usr/bin/env python3
"""Run GraphPilot stage-feasibility smoke tests on the connected Android device."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
ANDROID_APP_DIR = ROOT_DIR / "android-app"
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "artifacts" / "graphpilot_edge" / "experiments"
DEFAULT_APP_ID = "com.qidk.fastvlm"
DEFAULT_TEST_APP_ID = "com.qidk.fastvlm.test"
DEFAULT_MAX_WORKERS = 6
DEFAULT_ANDROID_SDK_ROOT = Path("/home/aryamavmurthy/android-sdk")
DEFAULT_ANDROID_NDK_HOME = DEFAULT_ANDROID_SDK_ROOT / "ndk" / "28.1.13356709"
DEFAULT_JNI_LIB_CANDIDATES = (
    ROOT_DIR
    / "third_party/litert-lm/bazel-bin/kotlin/java/com/google/ai/edge/litertlm/jni/liblitertlm_jni.so",
    ROOT_DIR / "android-app/app/build/generated/phase1/jniLibs/arm64-v8a/liblitertlm_jni.so",
)


@dataclass(frozen=True)
class StageTest:
    stage_id: str
    test_class: str
    description: str


STAGE_TESTS = {
    "asr": StageTest(
        stage_id="asr.primary",
        test_class="com.qidk.fastvlm.speech.WhisperSttInstrumentedTest",
        description="Whisper STT smoke test",
    ),
    "planner_cpu": StageTest(
        stage_id="planner.primary",
        test_class="com.qidk.fastvlm.text.LiteRtLmTextStageInstrumentedTest#plannerCpuPromptCompletes",
        description="Planner Gemma CPU smoke test",
    ),
    "planner_gpu": StageTest(
        stage_id="planner.primary",
        test_class="com.qidk.fastvlm.text.LiteRtLmTextStageInstrumentedTest#plannerGpuPromptCompletes",
        description="Planner Gemma GPU smoke test",
    ),
    "planner_npu": StageTest(
        stage_id="planner.primary",
        test_class="com.qidk.fastvlm.text.LiteRtLmTextStageInstrumentedTest#plannerNpuPromptCompletes",
        description="Planner Gemma NPU smoke test",
    ),
    "responder_cpu": StageTest(
        stage_id="responder.primary",
        test_class="com.qidk.fastvlm.text.LiteRtLmTextStageInstrumentedTest#responderCpuPromptCompletes",
        description="Responder Gemma CPU smoke test",
    ),
    "responder_gpu": StageTest(
        stage_id="responder.primary",
        test_class="com.qidk.fastvlm.text.LiteRtLmTextStageInstrumentedTest#responderGpuPromptCompletes",
        description="Responder Gemma GPU smoke test",
    ),
    "responder_npu": StageTest(
        stage_id="responder.primary",
        test_class="com.qidk.fastvlm.text.LiteRtLmTextStageInstrumentedTest#responderNpuPromptCompletes",
        description="Responder Gemma NPU smoke test",
    ),
    "tts": StageTest(
        stage_id="tts.primary",
        test_class="com.qidk.fastvlm.speech.AndroidTtsSpeakerInstrumentedTest",
        description="Android TTS smoke test",
    ),
    "retrieval_cpu": StageTest(
        stage_id="retrieval.embedder.primary",
        test_class="com.qidk.fastvlm.graphpilot.GraphPilotRetrievalBridgeInstrumentedTest#cpuQueryReturnsJfkHit",
        description="EmbeddingGemma retrieval CPU smoke test",
    ),
    "retrieval_gpu": StageTest(
        stage_id="retrieval.embedder.primary",
        test_class="com.qidk.fastvlm.graphpilot.GraphPilotRetrievalBridgeInstrumentedTest#gpuQueryReturnsJfkHit",
        description="EmbeddingGemma retrieval GPU smoke test",
    ),
    "retrieval_npu": StageTest(
        stage_id="retrieval.embedder.primary",
        test_class="com.qidk.fastvlm.graphpilot.GraphPilotRetrievalBridgeInstrumentedTest#npuQueryReturnsJfkHit",
        description="EmbeddingGemma retrieval NPU smoke test",
    ),
    "vlm_cpu": StageTest(
        stage_id="vlm.fastvlm.primary",
        test_class="com.qidk.fastvlm.bridge.FastVlmBridgeInstrumentedTest",
        description="FastVLM CPU bridge smoke test",
    ),
    "voice_pipeline": StageTest(
        stage_id="workflow_a_voice_only",
        test_class="com.qidk.fastvlm.pipeline.VoicePipelineE2EInstrumentedTest",
        description="Voice pipeline end-to-end smoke test covering STT, FastVLM CPU, and TTS",
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_instrumentation_output(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if "INSTRUMENTATION_FAILED" in stripped or "FAILURES!!!" in stripped:
        summary_line = next(
            (line.strip() for line in stripped.splitlines() if "Failure" in line or "FAILURES!!!" in line),
            "instrumentation_failed",
        )
        return {"verdict": "fail", "summary": summary_line, "test_count": None}
    match = re.search(r"OK \((\d+) test", stripped)
    if match:
        return {
            "verdict": "pass",
            "summary": match.group(0),
            "test_count": int(match.group(1)),
        }
    return {
        "verdict": "unknown",
        "summary": "Unable to determine instrumentation verdict from output.",
        "test_count": None,
    }


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, env=env, check=False, capture_output=True, text=True)


def require_file(path: Path, remediation: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file '{path}'. Remediation: {remediation}")


def resolve_jni_lib() -> Path:
    for candidate in DEFAULT_JNI_LIB_CANDIDATES:
        if candidate.is_file():
            return candidate
    bazel_out = ROOT_DIR / "third_party/litert-lm/bazel-out"
    for candidate in sorted(bazel_out.glob("**/bin/kotlin/java/com/google/ai/edge/litertlm/jni/liblitertlm_jni.so")):
        if candidate.is_file():
            return candidate
    searched = "\n".join(str(path) for path in DEFAULT_JNI_LIB_CANDIDATES)
    raise FileNotFoundError(
        "Missing patched JNI library. Searched:\n"
        f"{searched}\n"
        "and Bazel output globs under third_party/litert-lm/bazel-out/. "
        "Remediation: build the JNI target first with ANDROID_NDK_HOME set, for example: "
        "cd third_party/litert-lm && ANDROID_HOME=/home/aryamavmurthy/android-sdk "
        "ANDROID_SDK_ROOT=/home/aryamavmurthy/android-sdk "
        "ANDROID_NDK_HOME=/home/aryamavmurthy/android-sdk/ndk/28.1.13356709 "
        "bazel build --config=android_arm64 --jobs=18 //kotlin/java/com/google/ai/edge/litertlm/jni:litertlm_jni"
    )


def ensure_android_prereqs() -> None:
    _ = resolve_jni_lib()
    gradlew = ANDROID_APP_DIR / "gradlew"
    require_file(gradlew, "ensure the Android app project is present under android-app/")


def resolve_android_sdk_root(explicit: Path | None) -> Path:
    if explicit is not None:
        sdk_root = explicit
    else:
        env_value = os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT")
        sdk_root = Path(env_value) if env_value else DEFAULT_ANDROID_SDK_ROOT
    if not sdk_root.is_dir():
        raise FileNotFoundError(
            f"Missing Android SDK root '{sdk_root}'. Remediation: install the Android SDK and pass --android-sdk-root or set ANDROID_HOME."
        )
    adb_path = sdk_root / "platform-tools" / "adb"
    if not adb_path.exists():
        raise FileNotFoundError(
            f"Android SDK root '{sdk_root}' is missing platform-tools/adb. Remediation: install platform-tools under that SDK root."
        )
    return sdk_root


def resolve_android_ndk_home(explicit: Path | None, sdk_root: Path) -> Path:
    if explicit is not None:
        ndk_home = explicit
    else:
        env_value = os.environ.get("ANDROID_NDK_HOME")
        ndk_home = Path(env_value) if env_value else DEFAULT_ANDROID_NDK_HOME
    if not ndk_home.is_dir():
        raise FileNotFoundError(
            f"Missing Android NDK directory '{ndk_home}'. Remediation: install the Android NDK and pass --android-ndk-home or set ANDROID_NDK_HOME."
        )
    if sdk_root not in ndk_home.parents and ndk_home.parent.name != "ndk":
        # Only a sanity check; do not reject valid standalone installs that exist.
        pass
    return ndk_home


def build_gradle_env(android_sdk_root: Path, android_ndk_home: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["ANDROID_HOME"] = str(android_sdk_root)
    env["ANDROID_SDK_ROOT"] = str(android_sdk_root)
    env["ANDROID_NDK_HOME"] = str(android_ndk_home)
    return env


def install_app(max_workers: int, env: dict[str, str]) -> dict[str, Any]:
    cmd = [
        "./gradlew",
        "--no-daemon",
        f"--max-workers={max_workers}",
        ":app:installDebug",
        ":app:installDebugAndroidTest",
    ]
    result = run(cmd, cwd=ANDROID_APP_DIR, env=env)
    return {
        "command": " ".join(cmd),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def run_instrumentation(test_class: str) -> subprocess.CompletedProcess[str]:
    cmd = [
        "adb",
        "shell",
        "am",
        "instrument",
        "-w",
        "-e",
        "class",
        test_class,
        f"{DEFAULT_TEST_APP_ID}/androidx.test.runner.AndroidJUnitRunner",
    ]
    return run(cmd)


def clear_logcat() -> None:
    result = run(["adb", "logcat", "-c"])
    if result.returncode != 0:
        raise RuntimeError(f"Failed to clear logcat before instrumentation: {result.stderr.strip()}")


def capture_logcat() -> str:
    result = run(["adb", "logcat", "-d"])
    if result.returncode != 0:
        raise RuntimeError(f"Failed to capture logcat after instrumentation: {result.stderr.strip()}")
    return result.stdout


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_summary(output_dir: Path, payload: dict[str, Any]) -> Path:
    path = output_dir / "summary.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=sorted(STAGE_TESTS.keys()),
        default=[
            "asr",
            "planner_cpu",
            "planner_gpu",
            "planner_npu",
            "vlm_cpu",
            "responder_cpu",
            "responder_gpu",
            "responder_npu",
            "tts",
            "retrieval_cpu",
            "retrieval_gpu",
            "retrieval_npu",
            "voice_pipeline",
        ],
        help="Stage smoke tests to run",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Artifact root for stage-feasibility runs",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Gradle max workers for app/test install",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip app/test APK installation",
    )
    parser.add_argument(
        "--android-sdk-root",
        type=Path,
        default=None,
        help="Explicit Android SDK root for Gradle and adb tooling",
    )
    parser.add_argument(
        "--android-ndk-home",
        type=Path,
        default=None,
        help="Explicit Android NDK path for native Android prerequisites",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    ensure_android_prereqs()
    android_sdk_root = resolve_android_sdk_root(args.android_sdk_root)
    android_ndk_home = resolve_android_ndk_home(args.android_ndk_home, android_sdk_root)
    gradle_env = build_gradle_env(android_sdk_root, android_ndk_home)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"stage_feasibility_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    summary: dict[str, Any] = {
        "generated_at": utc_now(),
        "output_dir": str(output_dir),
        "stage_runs": [],
    }

    if not args.skip_install:
        install_result = install_app(args.max_workers, gradle_env)
        write_text(output_dir / "install.stdout.log", install_result["stdout"])
        write_text(output_dir / "install.stderr.log", install_result["stderr"])
        if install_result["returncode"] != 0:
            summary["install"] = install_result
            write_summary(output_dir, summary)
            raise RuntimeError(
                "Android app/test installation failed. Inspect install.stdout.log and install.stderr.log for root cause."
            )
        summary["install"] = {
            "command": install_result["command"],
            "returncode": 0,
            "android_sdk_root": str(android_sdk_root),
            "android_ndk_home": str(android_ndk_home),
        }

    for key in args.stages:
        stage = STAGE_TESTS[key]
        clear_logcat()
        result = run_instrumentation(stage.test_class)
        logcat = capture_logcat()
        stage_dir = output_dir / key
        stage_dir.mkdir(parents=True, exist_ok=True)
        write_text(stage_dir / "instrumentation.stdout.log", result.stdout)
        write_text(stage_dir / "instrumentation.stderr.log", result.stderr)
        write_text(stage_dir / "logcat.log", logcat)
        parsed = parse_instrumentation_output(result.stdout + "\n" + result.stderr)
        summary["stage_runs"].append(
            {
                "stage_key": key,
                "stage_id": stage.stage_id,
                "description": stage.description,
                "test_class": stage.test_class,
                "command": "adb shell am instrument -w -e class "
                + f"{stage.test_class} {DEFAULT_TEST_APP_ID}/androidx.test.runner.AndroidJUnitRunner",
                "returncode": result.returncode,
                "verdict": parsed["verdict"],
                "summary": parsed["summary"],
                "test_count": parsed["test_count"],
                "stdout_log": str(stage_dir / "instrumentation.stdout.log"),
                "stderr_log": str(stage_dir / "instrumentation.stderr.log"),
                "logcat_log": str(stage_dir / "logcat.log"),
            }
        )

    summary_path = write_summary(output_dir, summary)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
