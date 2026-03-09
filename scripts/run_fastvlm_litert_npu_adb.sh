#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/shell_quote_lib.sh"

LITERT_LM_DIR="${ROOT_DIR}/third_party/litert-lm"
LITERT_DIR="${ROOT_DIR}/third_party/litert"
QAIRT_ROOT="${QAIRT_ROOT:-/opt/qcom/aistack/qairt}"
LOCAL_QAIRT_REPO="${ROOT_DIR}/artifacts/local_qairt_repo"
MODEL_HOST_PATH="${ROOT_DIR}/artifacts/models/FastVLM-0.5B.qualcomm.sm8750.auxmaskrope_runtime.litertlm"

DEVICE_DIR="/data/local/tmp/vlm_phase1"
DEVICE_MODEL_PATH=""
DEVICE_IMAGE_PATH=""

PROMPT="Describe this image in one sentence."
MAX_NUM_TOKENS=512
MAX_OUTPUT_TOKENS=-1
MAX_VISUAL_TOKENS=0
VISUAL_TOKEN_PRUNING_STRATEGY="prompt_conditioned_v1"
CONSTRAINT_REGEX=""
BENCHMARK=0
EVENT_MODE=0
JOBS=6
SKIP_BUILD=0
SKIP_PUSH=0
IMAGE_HOST_PATH=""

usage() {
  cat <<'EOF'
Usage:
  run_fastvlm_litert_npu_adb.sh [options]

Options:
  --model PATH          Host .litertlm path to run on device
  --image PATH          Host image path to push to device as image.jpg
  --prompt TEXT         Prompt text (default: "Describe this image in one sentence.")
  --max-num-tokens N    Max context length for run (default: 512)
  --max-output-tokens N Max generated output tokens (-1 keeps runtime default)
  --max-visual-tokens N Max projected visual tokens to keep (default: 0 = off)
  --visual-token-pruning-strategy S
                       Visual token pruning strategy (default: prompt_conditioned_v1)
  --constraint-regex R Decode-time regex constraint passed to LiteRT-LM
  --benchmark 0|1       Emit benchmark info from runtime (default: 0)
  --event-mode 0|1      Emit structured VLM_EVENT lines (default: 0)
  --jobs N              Bazel jobs / local CPU cap (default: 6)
  --skip-build 0|1      Skip Bazel build steps (default: 0)
  --skip-push 0|1       Skip adb push steps (default: 0)
  --device-dir PATH     Device run directory (default: /data/local/tmp/vlm_phase1)
  -h, --help            Show help

Notes:
  - Defaults to:
    artifacts/models/FastVLM-0.5B.qualcomm.sm8750.auxmaskrope_runtime.litertlm
    Override with --model to run a different .litertlm artifact.
  - Requires QAIRT at /opt/qcom/aistack/qairt (override with QAIRT_ROOT env).
  - Uses hexagon-v79 DSP libs (required on this SM8750P setup).
EOF
}

refresh_device_paths() {
  DEVICE_MODEL_PATH="${DEVICE_DIR}/$(basename "${MODEL_HOST_PATH}")"
  DEVICE_IMAGE_PATH="${DEVICE_DIR}/image.jpg"
}

refresh_device_paths

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_HOST_PATH="${2:-}"
      refresh_device_paths
      shift 2
      ;;
    --image)
      IMAGE_HOST_PATH="${2:-}"
      shift 2
      ;;
    --prompt)
      PROMPT="${2:-}"
      shift 2
      ;;
    --max-num-tokens)
      MAX_NUM_TOKENS="${2:-}"
      shift 2
      ;;
    --max-output-tokens)
      MAX_OUTPUT_TOKENS="${2:-}"
      shift 2
      ;;
    --max-visual-tokens)
      MAX_VISUAL_TOKENS="${2:-}"
      shift 2
      ;;
    --visual-token-pruning-strategy)
      VISUAL_TOKEN_PRUNING_STRATEGY="${2:-}"
      shift 2
      ;;
    --constraint-regex)
      CONSTRAINT_REGEX="${2:-}"
      shift 2
      ;;
    --benchmark)
      BENCHMARK="${2:-}"
      shift 2
      ;;
    --event-mode)
      EVENT_MODE="${2:-}"
      shift 2
      ;;
    --jobs)
      JOBS="${2:-}"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD="${2:-}"
      shift 2
      ;;
    --skip-push)
      SKIP_PUSH="${2:-}"
      shift 2
      ;;
    --device-dir)
      DEVICE_DIR="${2:-}"
      refresh_device_paths
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! "${MAX_NUM_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "Invalid --max-num-tokens: ${MAX_NUM_TOKENS}" >&2
  exit 2
fi
if [[ ! "${MAX_OUTPUT_TOKENS}" =~ ^-?[0-9]+$ ]]; then
  echo "Invalid --max-output-tokens: ${MAX_OUTPUT_TOKENS}" >&2
  exit 2
fi
if (( MAX_OUTPUT_TOKENS < -1 )); then
  echo "Invalid --max-output-tokens (expected -1 or >= 0): ${MAX_OUTPUT_TOKENS}" >&2
  exit 2
fi
if [[ ! "${MAX_VISUAL_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "Invalid --max-visual-tokens: ${MAX_VISUAL_TOKENS}" >&2
  exit 2
fi
if [[ -z "${VISUAL_TOKEN_PRUNING_STRATEGY}" ]]; then
  echo "Invalid --visual-token-pruning-strategy: must be non-empty" >&2
  exit 2
fi
if [[ "${BENCHMARK}" != "0" && "${BENCHMARK}" != "1" ]]; then
  echo "Invalid --benchmark (expected 0 or 1): ${BENCHMARK}" >&2
  exit 2
fi
if [[ "${EVENT_MODE}" != "0" && "${EVENT_MODE}" != "1" ]]; then
  echo "Invalid --event-mode (expected 0 or 1): ${EVENT_MODE}" >&2
  exit 2
fi
if [[ ! "${JOBS}" =~ ^[0-9]+$ ]]; then
  echo "Invalid --jobs: ${JOBS}" >&2
  exit 2
fi
if [[ "${SKIP_BUILD}" != "0" && "${SKIP_BUILD}" != "1" ]]; then
  echo "Invalid --skip-build (expected 0 or 1): ${SKIP_BUILD}" >&2
  exit 2
fi
if [[ "${SKIP_PUSH}" != "0" && "${SKIP_PUSH}" != "1" ]]; then
  echo "Invalid --skip-push (expected 0 or 1): ${SKIP_PUSH}" >&2
  exit 2
fi

if [[ ! -f "${MODEL_HOST_PATH}" ]]; then
  echo "Missing model artifact: ${MODEL_HOST_PATH}" >&2
  echo "Remediation: place the pinned FastVLM .litertlm file at this path." >&2
  exit 3
fi
if [[ -n "${IMAGE_HOST_PATH}" && ! -f "${IMAGE_HOST_PATH}" ]]; then
  echo "Missing image file: ${IMAGE_HOST_PATH}" >&2
  exit 3
fi
if [[ ! -d "${QAIRT_ROOT}" ]]; then
  echo "Missing QAIRT root: ${QAIRT_ROOT}" >&2
  exit 3
fi
if [[ ! -d "${QAIRT_ROOT}/lib/aarch64-android" ]]; then
  echo "Missing QAIRT arm64 libs: ${QAIRT_ROOT}/lib/aarch64-android" >&2
  exit 3
fi
if [[ ! -d "${QAIRT_ROOT}/lib/hexagon-v79/unsigned" ]]; then
  echo "Missing QAIRT hexagon-v79 libs: ${QAIRT_ROOT}/lib/hexagon-v79/unsigned" >&2
  exit 3
fi

if ! command -v adb >/dev/null 2>&1; then
  echo "adb not found in PATH" >&2
  exit 4
fi
adb wait-for-device

if [[ "${SKIP_BUILD}" == "0" ]]; then
  mkdir -p "${LOCAL_QAIRT_REPO}"
  cp "${LITERT_DIR}/third_party/qairt/qairt.BUILD" "${LOCAL_QAIRT_REPO}/BUILD"
  cat > "${LOCAL_QAIRT_REPO}/WORKSPACE" <<'EOF'
workspace(name = "qairt")
EOF
  ln -sfn "${QAIRT_ROOT}/include" "${LOCAL_QAIRT_REPO}/include"

  NDK_PATH="${ANDROID_NDK_HOME:-/home/aryamavmurthy/android-sdk/ndk/28.1.13356709}"
  if [[ ! -d "${NDK_PATH}" ]]; then
    echo "Missing Android NDK path: ${NDK_PATH}" >&2
    echo "Remediation: set ANDROID_NDK_HOME to your NDK (r28+ recommended)." >&2
    exit 5
  fi

  (
    cd "${LITERT_LM_DIR}"
    ANDROID_NDK_HOME="${NDK_PATH}" ANDROID_NDK_ROOT="${NDK_PATH}" \
      bazel build --config=android_arm64 \
      //runtime/engine:litert_lm_advanced_main \
      --override_repository="litert=${LITERT_DIR}" \
      --jobs="${JOBS}" \
      --local_resources="cpu=${JOBS}" \
      --local_resources=memory=8192 \
      --loading_phase_threads=2

    android_runtime_targets=(
      "@litert//litert/c:litert_runtime_c_api_so"
      "@litert//litert/vendors/qualcomm/dispatch:dispatch_api_so"
      "@litert//litert/vendors/qualcomm/compiler:qnn_compiler_plugin_so"
    )
    ANDROID_NDK_HOME="${NDK_PATH}" ANDROID_NDK_ROOT="${NDK_PATH}" \
      bazel build --config=android_arm64 \
      "${android_runtime_targets[@]}" \
      --override_repository="litert=${LITERT_DIR}" \
      --override_repository="qairt=${LOCAL_QAIRT_REPO}" \
      --jobs="${JOBS}" \
      --local_resources="cpu=${JOBS}" \
      --local_resources=memory=8192 \
      --loading_phase_threads=2
  )
fi

ANDROID_BAZEL_BIN="${LITERT_LM_DIR}/bazel-out/arm64-v8a-opt/bin"
BINARY_PATH="${ANDROID_BAZEL_BIN}/runtime/engine/litert_lm_advanced_main"
LITERT_SO_PATH="${ANDROID_BAZEL_BIN}/external/litert/litert/c/libLiteRt.so"
DISPATCH_SO_PATH="${ANDROID_BAZEL_BIN}/external/litert/litert/vendors/qualcomm/dispatch/libLiteRtDispatch_Qualcomm.so"
COMPILER_PLUGIN_SO_PATH="${ANDROID_BAZEL_BIN}/external/litert/litert/vendors/qualcomm/compiler/libLiteRtCompilerPlugin_Qualcomm.so"
if [[ ! -f "${BINARY_PATH}" ]]; then
  echo "Missing built binary: ${BINARY_PATH}" >&2
  exit 6
fi
if command -v file >/dev/null 2>&1; then
  binary_file_info="$(file "${BINARY_PATH}")"
  if [[ "${binary_file_info}" != *"ARM aarch64"* ]]; then
    echo "Built binary is not an Android ARM64 executable: ${BINARY_PATH}" >&2
    echo "file(1) output: ${binary_file_info}" >&2
    echo "Remediation: rebuild with --config=android_arm64 and use the Android bazel-out path." >&2
    exit 6
  fi
fi
if [[ ! -f "${LITERT_SO_PATH}" ]]; then
  echo "Missing LiteRT shared library: ${LITERT_SO_PATH}" >&2
  exit 6
fi
if [[ ! -f "${DISPATCH_SO_PATH}" ]]; then
  echo "Missing dispatch library: ${DISPATCH_SO_PATH}" >&2
  exit 6
fi
if [[ ! -f "${COMPILER_PLUGIN_SO_PATH}" ]]; then
  echo "Missing compiler plugin library: ${COMPILER_PLUGIN_SO_PATH}" >&2
  exit 6
fi

if [[ "${SKIP_PUSH}" == "0" ]]; then
  DEVICE_DIR_QUOTED="$(shell_single_quote "${DEVICE_DIR}")"
  DEVICE_DISPATCH_DIR_QUOTED="$(shell_single_quote "${DEVICE_DIR}/dispatch_libs")"
  DEVICE_HEXAGON_DIR_QUOTED="$(shell_single_quote "${DEVICE_DIR}/hexagon-v79")"
  adb shell "mkdir -p ${DEVICE_DIR_QUOTED} ${DEVICE_DISPATCH_DIR_QUOTED} ${DEVICE_HEXAGON_DIR_QUOTED}"

  adb push "${BINARY_PATH}" "${DEVICE_DIR}/litert_lm_advanced_main" >/dev/null
  DEVICE_BINARY_QUOTED="$(shell_single_quote "${DEVICE_DIR}/litert_lm_advanced_main")"
  adb shell "chmod +x ${DEVICE_BINARY_QUOTED}"

  adb push "${MODEL_HOST_PATH}" "${DEVICE_MODEL_PATH}" >/dev/null
  adb push "${LITERT_LM_DIR}/prebuilt/android_arm64/." "${DEVICE_DIR}/" >/dev/null
  adb push "${LITERT_SO_PATH}" "${DEVICE_DIR}/dispatch_libs/libLiteRt.so" >/dev/null
  adb push "${DISPATCH_SO_PATH}" "${DEVICE_DIR}/dispatch_libs/libLiteRtDispatch_Qualcomm.so" >/dev/null
  adb push "${COMPILER_PLUGIN_SO_PATH}" "${DEVICE_DIR}/dispatch_libs/libLiteRtCompilerPlugin_Qualcomm.so" >/dev/null
  adb push "${QAIRT_ROOT}/lib/aarch64-android/." "${DEVICE_DIR}/dispatch_libs/" >/dev/null
  adb push "${QAIRT_ROOT}/lib/hexagon-v79/unsigned/." "${DEVICE_DIR}/hexagon-v79/" >/dev/null

  if [[ -n "${IMAGE_HOST_PATH}" ]]; then
    adb push "${IMAGE_HOST_PATH}" "${DEVICE_IMAGE_PATH}" >/dev/null
  fi
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/artifacts/logs"
mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/adb_npu_npu_run_${TIMESTAMP}.log"

set +e
EVENT_MODE_EXPORT=""
if [[ "${EVENT_MODE}" == "1" ]]; then
  EVENT_MODE_EXPORT="export LITERT_LM_EVENT_MODE=1 &&"
fi
DEVICE_DIR_QUOTED="$(shell_single_quote "${DEVICE_DIR}")"
DEVICE_MODEL_PATH_QUOTED="$(shell_single_quote "${DEVICE_MODEL_PATH}")"
DEVICE_INPUT_PROMPT_QUOTED="$(shell_single_quote "${PROMPT} [image:${DEVICE_IMAGE_PATH}]")"
DEVICE_DISPATCH_LIB_DIR_QUOTED="$(shell_single_quote "${DEVICE_DIR}/dispatch_libs")"
LD_LIBRARY_PATH_QUOTED="$(shell_single_quote "${DEVICE_DIR}:${DEVICE_DIR}/dispatch_libs")"
ADSP_LIBRARY_PATH_QUOTED="$(shell_single_quote "${DEVICE_DIR}/hexagon-v79;/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp")"
CONSTRAINT_REGEX_ARG=""
if [[ -n "${CONSTRAINT_REGEX}" ]]; then
  DEVICE_CONSTRAINT_REGEX_QUOTED="$(shell_single_quote "${CONSTRAINT_REGEX}")"
  CONSTRAINT_REGEX_ARG="    --constraint_regex=${DEVICE_CONSTRAINT_REGEX_QUOTED} \\"
fi
adb shell "cd ${DEVICE_DIR_QUOTED} && \
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH_QUOTED} && \
  export ADSP_LIBRARY_PATH=${ADSP_LIBRARY_PATH_QUOTED} && \
  ${EVENT_MODE_EXPORT} \
  ./litert_lm_advanced_main \
    --backend=npu \
    --vision_backend=npu \
    --model_path=${DEVICE_MODEL_PATH_QUOTED} \
    --input_prompt=${DEVICE_INPUT_PROMPT_QUOTED} \
    --max_num_tokens='${MAX_NUM_TOKENS}' \
    --max_output_tokens='${MAX_OUTPUT_TOKENS}' \
    --max_visual_tokens='${MAX_VISUAL_TOKENS}' \
    --visual_token_pruning_strategy='${VISUAL_TOKEN_PRUNING_STRATEGY}' \
${CONSTRAINT_REGEX_ARG}\
    --benchmark='${BENCHMARK}' \
    --litert_dispatch_lib_dir=${DEVICE_DISPATCH_LIB_DIR_QUOTED}" >"${RUN_LOG}" 2>&1
RUN_RC=$?
set -e

echo "Run exit code: ${RUN_RC}"
echo "Run log: ${RUN_LOG}"
echo
echo "Key runtime lines:"
rg -n "EncoderBackend|AdapterBackend|DispatchDelegate|context_binary_info|RunPrefillAsync status|RunDecodeAsync|Visual token pruning decision|ERROR|Invalid|Custom NPU execution latency stats|tokens per second" "${RUN_LOG}" | head -n 120 || true

exit "${RUN_RC}"
