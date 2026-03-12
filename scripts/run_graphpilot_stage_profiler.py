#!/usr/bin/env python3
"""Run GraphPilot stage profilers on device and record machine-readable metrics."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
ANDROID_APP_DIR = ROOT_DIR / "android-app"
ARTIFACT_ROOT = ROOT_DIR / "artifacts" / "graphpilot_edge"
DEFAULT_OUTPUT_ROOT = ARTIFACT_ROOT / "metrics"
PROFILER_REGISTRY_PATH = ARTIFACT_ROOT / "registries" / "profiler_registry.json"
DEFAULT_CANDIDATE_PLAN_REGISTRY = ARTIFACT_ROOT / "registries" / "candidate_plan_registry.json"
DEVICE_GRAPH_PILOT_DIR = "/data/local/tmp/graphpilot_edge"
DEVICE_CANDIDATE_PLAN_REGISTRY = f"{DEVICE_GRAPH_PILOT_DIR}/candidate_plan_registry.json"
DEFAULT_ANDROID_SDK_ROOT = Path("/home/aryamavmurthy/android-sdk")
DEFAULT_ANDROID_NDK_HOME = DEFAULT_ANDROID_SDK_ROOT / "ndk" / "28.1.13356709"


PROFILE_RUNS = {
    "asr_cpu": {
        "stage_id": "asr.primary",
        "backend": "cpu",
        "variant": "whisper_stt",
        "test_class": "com.qidk.fastvlm.speech.WhisperSttInstrumentedTest",
        "parser": "asr",
    },
    "planner_cpu": {
        "stage_id": "planner.primary",
        "backend": "cpu",
        "variant": "gemma3_1b_it",
        "test_class": "com.qidk.fastvlm.text.LiteRtLmTextStageInstrumentedTest#plannerCpuPromptCompletes",
        "parser": "text",
    },
    "responder_cpu": {
        "stage_id": "responder.primary",
        "backend": "cpu",
        "variant": "gemma3_1b_it",
        "test_class": "com.qidk.fastvlm.text.LiteRtLmTextStageInstrumentedTest#responderCpuPromptCompletes",
        "parser": "text",
    },
    "tts_cpu": {
        "stage_id": "tts.primary",
        "backend": "cpu",
        "variant": "android_tts",
        "test_class": "com.qidk.fastvlm.speech.AndroidTtsSpeakerInstrumentedTest",
        "parser": "tts",
    },
    "retrieval_cpu": {
        "stage_id": "retrieval.embedder.primary",
        "backend": "cpu",
        "variant": "embeddinggemma_300m",
        "test_class": "com.qidk.fastvlm.graphpilot.GraphPilotRetrievalBridgeInstrumentedTest#cpuQueryReturnsJfkHit",
        "parser": "retrieval",
    },
    "retrieval_gpu": {
        "stage_id": "retrieval.embedder.primary",
        "backend": "gpu",
        "variant": "embeddinggemma_300m",
        "test_class": "com.qidk.fastvlm.graphpilot.GraphPilotRetrievalBridgeInstrumentedTest#gpuQueryReturnsJfkHit",
        "parser": "retrieval",
    },
    "vlm_cpu": {
        "stage_id": "vlm.fastvlm.primary",
        "backend": "cpu",
        "variant": "fastvlm_cpu_bridge",
        "test_class": "com.qidk.fastvlm.bridge.FastVlmBridgeInstrumentedTest",
        "parser": "vlm",
    },
    "workflow_a_voice_only": {
        "stage_id": "workflow_a_voice_only",
        "backend": "mixed",
        "variant": "voice_pipeline_cpu",
        "test_class": "com.qidk.fastvlm.pipeline.VoicePipelineE2EInstrumentedTest",
        "parser": "voice_pipeline",
    },
    "graphpilot_workflow_a": {
        "stage_id": "workflow_a_voice_only",
        "backend": "mixed",
        "variant": "graphpilot_cpu_stack",
        "test_class": "com.qidk.fastvlm.graphpilot.GraphPilotCoordinatorInstrumentedTest#workflowAVoiceOnly_completes",
        "parser": "graphpilot",
    },
    "graphpilot_workflow_b": {
        "stage_id": "workflow_b_voice_vision",
        "backend": "mixed",
        "variant": "graphpilot_vlm_npu",
        "test_class": "com.qidk.fastvlm.graphpilot.GraphPilotCoordinatorInstrumentedTest#workflowBVoiceVision_completes",
        "parser": "graphpilot",
    },
    "graphpilot_workflow_c": {
        "stage_id": "workflow_c_voice_vision_retrieval",
        "backend": "mixed",
        "variant": "graphpilot_vlm_npu_retrieval_cpu",
        "test_class": "com.qidk.fastvlm.graphpilot.GraphPilotCoordinatorInstrumentedTest#workflowCVoiceVisionRetrieval_completes",
        "parser": "graphpilot",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, env=env, check=False, capture_output=True, text=True)


def build_gradle_env() -> dict[str, str]:
    env = os.environ.copy()
    env["ANDROID_HOME"] = str(DEFAULT_ANDROID_SDK_ROOT)
    env["ANDROID_SDK_ROOT"] = str(DEFAULT_ANDROID_SDK_ROOT)
    env["ANDROID_NDK_HOME"] = str(DEFAULT_ANDROID_NDK_HOME)
    return env


def install_app(max_workers: int) -> None:
    result = run(
        [
            "./gradlew",
            "--no-daemon",
            f"--max-workers={max_workers}",
            ":app:installDebug",
            ":app:installDebugAndroidTest",
        ],
        cwd=ANDROID_APP_DIR,
        env=build_gradle_env(),
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Android app/test installation failed for profiler run. "
            "Inspect Gradle stdout/stderr in the command output."
        )


def clear_logcat() -> None:
    result = run(["adb", "logcat", "-c"])
    if result.returncode != 0:
        raise RuntimeError(f"Failed to clear logcat: {result.stderr.strip()}")


def capture_logcat() -> str:
    result = run(["adb", "logcat", "-d"])
    if result.returncode != 0:
        raise RuntimeError(f"Failed to capture logcat: {result.stderr.strip()}")
    return result.stdout


def run_instrumentation(test_class: str) -> subprocess.CompletedProcess[str]:
    return run(
        [
            "adb",
            "shell",
            "am",
            "instrument",
            "-w",
            "-e",
            "class",
            test_class,
            "com.qidk.fastvlm.test/androidx.test.runner.AndroidJUnitRunner",
        ]
    )


def stage_candidate_plans(candidate_registry: Path) -> None:
    if not candidate_registry.is_file():
        raise FileNotFoundError(
            f"Missing candidate plan registry '{candidate_registry}'. Remediation: run scripts/generate_graphpilot_candidate_plans.py first."
        )
    mkdir = run(["adb", "shell", "mkdir", "-p", DEVICE_GRAPH_PILOT_DIR])
    if mkdir.returncode != 0:
        raise RuntimeError(f"Failed to create device GraphPilot dir: {mkdir.stderr.strip()}")
    push = run(["adb", "push", str(candidate_registry), DEVICE_CANDIDATE_PLAN_REGISTRY])
    if push.returncode != 0:
        raise RuntimeError(
            "Failed to stage candidate plan registry to device. "
            f"stderr={push.stderr.strip()}"
        )


def parse_asr_metrics(logcat: str) -> dict[str, Any]:
    match = re.search(
        r"first_elapsed_ms=(?P<cold>\d+)\s+second_elapsed_ms=(?P<warm>\d+)", logcat
    )
    if not match:
        raise ValueError("Failed to parse ASR profiler metrics from logcat.")
    cold = int(match.group("cold"))
    warm = int(match.group("warm"))
    return {
        "cold_latency_ms": cold,
        "warm_latency_ms": warm,
        "steady_state_latency_ms": warm,
        "prefill_latency_ms": None,
        "decode_latency_ms": None,
        "peak_memory_bytes": None,
        "transfer_bytes": None,
        "transfer_time_ms": None,
        "energy_mj_or_power_proxy": None,
        "thermal_bin": "unknown",
        "hidden_fallback_status": "not_applicable",
    }


def parse_text_metrics(logcat: str) -> dict[str, Any]:
    match = re.search(
        r"TEXT_STAGE role=(?P<role>\w+)\s+backend=(?P<backend>\w+)\s+model=(?P<model>\S+)\s+"
        r"init_elapsed_ms=(?P<init>\d+)\s+generate_elapsed_ms=(?P<generate>\d+)",
        logcat,
    )
    if not match:
        raise ValueError("Failed to parse LiteRT-LM text stage profiler metrics from logcat.")
    init_ms = int(match.group("init"))
    generate_ms = int(match.group("generate"))
    return {
        "cold_latency_ms": init_ms + generate_ms,
        "warm_latency_ms": generate_ms,
        "steady_state_latency_ms": generate_ms,
        "prefill_latency_ms": None,
        "decode_latency_ms": None,
        "peak_memory_bytes": None,
        "transfer_bytes": None,
        "transfer_time_ms": None,
        "energy_mj_or_power_proxy": None,
        "thermal_bin": "unknown",
        "hidden_fallback_status": "not_applicable",
        "model_path": match.group("model"),
        "role": match.group("role"),
        "backend_reported": match.group("backend"),
    }


def parse_tts_metrics(logcat: str) -> dict[str, Any]:
    match = re.search(
        r"TTS validated init_elapsed_ms=(?P<init>\d+)\s+speak_elapsed_ms=(?P<speak>\d+)",
        logcat,
    )
    if not match:
        raise ValueError("Failed to parse TTS profiler metrics from logcat.")
    init_ms = int(match.group("init"))
    speak_ms = int(match.group("speak"))
    return {
        "cold_latency_ms": init_ms + speak_ms,
        "warm_latency_ms": speak_ms,
        "steady_state_latency_ms": speak_ms,
        "prefill_latency_ms": None,
        "decode_latency_ms": None,
        "peak_memory_bytes": None,
        "transfer_bytes": None,
        "transfer_time_ms": None,
        "energy_mj_or_power_proxy": None,
        "thermal_bin": "unknown",
        "hidden_fallback_status": "not_applicable",
    }


def parse_retrieval_metrics(logcat: str) -> dict[str, Any]:
    match = re.search(
        r"GRAPH_PILOT_RETRIEVAL backend=(?P<backend>\w+)\s+elapsed_ms=(?P<elapsed>-?\d+)\s+top_doc_id=(?P<doc_id>\S+)\s+top_score=(?P<score>[-+0-9.eE]+)",
        logcat,
    )
    if not match:
        raise ValueError("Failed to parse retrieval profiler metrics from logcat.")
    elapsed_ms = int(match.group("elapsed"))
    return {
        "cold_latency_ms": elapsed_ms,
        "warm_latency_ms": elapsed_ms,
        "steady_state_latency_ms": elapsed_ms,
        "prefill_latency_ms": None,
        "decode_latency_ms": None,
        "peak_memory_bytes": None,
        "transfer_bytes": None,
        "transfer_time_ms": None,
        "energy_mj_or_power_proxy": None,
        "thermal_bin": "unknown",
        "hidden_fallback_status": "not_applicable",
        "backend_reported": match.group("backend"),
        "top_doc_id": match.group("doc_id"),
        "top_score": float(match.group("score")),
    }


def parse_vlm_metrics(logcat: str) -> dict[str, Any]:
    match = re.search(r"metrics=(\{.*\})", logcat)
    if not match:
        raise ValueError("Failed to parse VLM profiler metrics from logcat.")
    payload = json.loads(match.group(1))
    error = payload.get("error")
    if error is not None:
        raise ValueError(
            f"VLM profiler run returned error={error.get('code')}: {error.get('message')}"
        )
    stage_timings = payload["stage_timings"]
    token_stats = payload["token_stats"]
    backend_status = payload["backend_status"]
    return {
        "cold_latency_ms": stage_timings["total_ms"],
        "warm_latency_ms": stage_timings["total_ms"],
        "steady_state_latency_ms": stage_timings["total_ms"],
        "prefill_latency_ms": stage_timings["prefill_ms"],
        "decode_latency_ms": stage_timings["decode_ms"],
        "peak_memory_bytes": None,
        "transfer_bytes": None,
        "transfer_time_ms": None,
        "energy_mj_or_power_proxy": None,
        "thermal_bin": "unknown",
        "hidden_fallback_status": "no_hidden_fallback_observed",
        "ttft_ms": token_stats["ttft_ms"],
        "backend_config_actual": backend_status["backend_config_actual"],
    }


def parse_voice_pipeline_metrics(logcat: str) -> dict[str, Any]:
    match = re.search(
        r"PIPELINE_TIMINGS stt_init_ms=(?P<stt_init>\d+) stt_warmup_ms=(?P<stt_warmup>\d+) "
        r"stt_transcribe_ms=(?P<stt_transcribe>\d+) vlm_init_ms=(?P<vlm_init>\d+) "
        r"vlm_ttft_ms=(?P<vlm_ttft>-?\d+) vlm_prefill_ms=(?P<vlm_prefill>\d+) "
        r"vlm_decode_ms=(?P<vlm_decode>\d+) vlm_total_ms=(?P<vlm_total>\d+) "
        r"vlm_wall_ms=(?P<vlm_wall>\d+) tts_init_ms=(?P<tts_init>\d+) "
        r"tts_speak_ms=(?P<tts_speak>\d+)",
        logcat,
    )
    if not match:
        raise ValueError("Failed to parse voice pipeline profiler metrics from logcat.")
    return {
        "cold_latency_ms": int(match.group("stt_init"))
        + int(match.group("stt_transcribe"))
        + int(match.group("vlm_wall"))
        + int(match.group("tts_init"))
        + int(match.group("tts_speak")),
        "warm_latency_ms": int(match.group("stt_warmup"))
        + int(match.group("stt_transcribe"))
        + int(match.group("vlm_wall"))
        + int(match.group("tts_speak")),
        "steady_state_latency_ms": int(match.group("stt_transcribe"))
        + int(match.group("vlm_wall"))
        + int(match.group("tts_speak")),
        "prefill_latency_ms": int(match.group("vlm_prefill")),
        "decode_latency_ms": int(match.group("vlm_decode")),
        "peak_memory_bytes": None,
        "transfer_bytes": None,
        "transfer_time_ms": None,
        "energy_mj_or_power_proxy": None,
        "thermal_bin": "unknown",
        "hidden_fallback_status": "no_hidden_fallback_observed",
        "ttft_ms": int(match.group("vlm_ttft")),
        "tts_first_audio_ms": int(match.group("tts_init")) + int(match.group("tts_speak")),
    }


def parse_graphpilot_metrics(logcat: str) -> dict[str, Any]:
    match = re.search(
        r"GRAPHPILOT_METRICS workflow=(?P<workflow>\S+)\s+plan_id=(?P<plan_id>\S+)\s+state_id=(?P<state_id>\S+)\s+"
        r"request_id=(?P<request_id>\d+)\s+queue_depth_at_admission=(?P<queue_depth>\d+)\s+"
        r"queue_wait_ms=(?P<queue_wait>\d+)\s+memory_decision=(?P<memory_decision>\S+)\s+"
        r"memory_effective_required_bytes=(?P<memory_required>\d+)\s+"
        r"stage_backends=(?P<stage_backends>\S+)\s+"
        r"stage_timings_ms=(?P<stage_timings>\S+)\s+"
        r"total_ms=(?P<total>\d+)\s+"
        r"(?:planner_first_partial_ms=(?P<planner_first_partial>-?\d+)\s+)?"
        r"(?:asr_chunk_count=(?P<asr_chunk_count>\d+)\s+)?"
        r"ttft_ms=(?P<ttft>-?\d+)\s+"
        r"tts_first_chunk_queued_ms=(?P<tts_first_chunk_queued>-?\d+)\s+"
        r"tts_first_audio_ms=(?P<tts_first_audio>-?\d+)\s+"
        r"vlm_prefill_ms=(?P<vlm_prefill>\d+)\s+vlm_decode_ms=(?P<vlm_decode>\d+)",
        logcat,
    )
    if not match:
        raise ValueError("Failed to parse GraphPilot workflow metrics from logcat.")

    def parse_kv_map(raw: str, value_cast):
        result: dict[str, Any] = {}
        for item in raw.split(","):
            if not item:
                continue
            key, value = item.rsplit(":", 1)
            result[key] = value_cast(value)
        return result

    ttft_ms = int(match.group("ttft"))
    tts_first_chunk_queued_ms = int(match.group("tts_first_chunk_queued"))
    tts_first_audio_ms = int(match.group("tts_first_audio"))
    planner_first_partial = match.group("planner_first_partial")
    asr_chunk_count = match.group("asr_chunk_count")
    return {
        "workflow_id": match.group("workflow"),
        "plan_id": match.group("plan_id"),
        "state_id": match.group("state_id"),
        "request_id": int(match.group("request_id")),
        "queue_depth_at_admission": int(match.group("queue_depth")),
        "queue_wait_ms": int(match.group("queue_wait")),
        "memory_decision": match.group("memory_decision"),
        "memory_effective_required_bytes": int(match.group("memory_required")),
        "cold_latency_ms": int(match.group("total")),
        "warm_latency_ms": int(match.group("total")),
        "steady_state_latency_ms": int(match.group("total")),
        "prefill_latency_ms": int(match.group("vlm_prefill")),
        "decode_latency_ms": int(match.group("vlm_decode")),
        "peak_memory_bytes": None,
        "transfer_bytes": None,
        "transfer_time_ms": None,
        "energy_mj_or_power_proxy": None,
        "thermal_bin": "unknown",
        "hidden_fallback_status": "no_hidden_fallback_observed",
        "planner_first_partial_ms": None
        if planner_first_partial in (None, "")
        else int(planner_first_partial),
        "asr_chunk_count": None
        if asr_chunk_count in (None, "")
        else int(asr_chunk_count),
        "ttft_ms": None if ttft_ms < 0 else ttft_ms,
        "tts_first_chunk_queued_ms": None
        if tts_first_chunk_queued_ms < 0
        else tts_first_chunk_queued_ms,
        "tts_first_audio_ms": None if tts_first_audio_ms < 0 else tts_first_audio_ms,
        "stage_backends": parse_kv_map(match.group("stage_backends"), str),
        "stage_timings_ms": parse_kv_map(match.group("stage_timings"), int),
    }


PARSERS = {
    "asr": parse_asr_metrics,
    "text": parse_text_metrics,
    "tts": parse_tts_metrics,
    "retrieval": parse_retrieval_metrics,
    "vlm": parse_vlm_metrics,
    "voice_pipeline": parse_voice_pipeline_metrics,
    "graphpilot": parse_graphpilot_metrics,
}


def resolve_profile_variant(
    profile_key: str,
    profile: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    if profile_key == "graphpilot_workflow_b":
        stage_backends = metrics.get("stage_backends", {})
        vlm_backend = stage_backends.get("vlm.fastvlm.primary")
        if vlm_backend:
            return f"graphpilot_vlm_{vlm_backend}"
    if profile_key == "graphpilot_workflow_c":
        stage_backends = metrics.get("stage_backends", {})
        vlm_backend = stage_backends.get("vlm.fastvlm.primary")
        retrieval_backend = stage_backends.get("retrieval.embedder.primary")
        if vlm_backend and retrieval_backend:
            return f"graphpilot_vlm_{vlm_backend}_retrieval_{retrieval_backend}"
    return profile["variant"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def update_profiler_registry(entries: list[dict[str, Any]]) -> None:
    registry = load_json(PROFILER_REGISTRY_PATH)
    current_entries = registry.setdefault("entries", [])
    for entry in entries:
        current_entries.append(entry)
    registry["last_updated"] = datetime.now(timezone.utc).date().isoformat()
    write_json(PROFILER_REGISTRY_PATH, registry)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=sorted(PROFILE_RUNS.keys()),
        default=sorted(PROFILE_RUNS.keys()),
    )
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--candidate-plans", type=Path, default=DEFAULT_CANDIDATE_PLAN_REGISTRY)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not args.skip_install:
        install_app(args.max_workers)
    if any(PROFILE_RUNS[key]["parser"] == "graphpilot" for key in args.profiles):
        stage_candidate_plans(args.candidate_plans)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"profile_run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    registry_entries: list[dict[str, Any]] = []

    for profile_key in args.profiles:
        profile = PROFILE_RUNS[profile_key]
        clear_logcat()
        result = run_instrumentation(profile["test_class"])
        logcat = capture_logcat()
        run_dir = output_dir / profile_key
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "instrumentation.stdout.log").write_text(result.stdout, encoding="utf-8")
        (run_dir / "instrumentation.stderr.log").write_text(result.stderr, encoding="utf-8")
        (run_dir / "logcat.log").write_text(logcat, encoding="utf-8")
        if result.returncode != 0 or "FAILURES!!!" in result.stdout or "INSTRUMENTATION_FAILED" in result.stdout:
            raise RuntimeError(
                f"Profiler run '{profile_key}' failed. Inspect '{run_dir / 'instrumentation.stdout.log'}'."
            )
        metrics = PARSERS[profile["parser"]](logcat)
        registry_entries.append(
            {
                "profile_id": f"{timestamp}:{profile_key}",
                "recorded_at": utc_now(),
                "stage_id": profile["stage_id"],
                "backend": profile["backend"],
                "variant": resolve_profile_variant(profile_key, profile, metrics),
                "test_class": profile["test_class"],
                "command": "adb shell am instrument -w -e class "
                + f"{profile['test_class']} com.qidk.fastvlm.test/androidx.test.runner.AndroidJUnitRunner",
                "artifacts": {
                    "run_dir": str(run_dir),
                    "instrumentation_stdout": str(run_dir / "instrumentation.stdout.log"),
                    "instrumentation_stderr": str(run_dir / "instrumentation.stderr.log"),
                    "logcat": str(run_dir / "logcat.log"),
                },
                "metrics": metrics,
            }
        )

    update_profiler_registry(registry_entries)
    summary_path = output_dir / "summary.json"
    write_json(summary_path, {"generated_at": utc_now(), "profiles": registry_entries})
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
