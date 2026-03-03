#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LITERT_LM_DIR="${ROOT_DIR}/third_party/litert-lm"
LITERT_DIR="${ROOT_DIR}/third_party/litert"
QAIRT_ROOT="${QAIRT_ROOT:-/opt/qcom/aistack/qairt}"
LOCAL_QAIRT_REPO="${ROOT_DIR}/artifacts/local_qairt_repo"
MODEL_HOST_PATH="${ROOT_DIR}/artifacts/models/FastVLM-0.5B.qualcomm.sm8750.litertlm"

DEVICE_DIR="/data/local/tmp/vlm_phase1"
DEVICE_MODEL_PATH="${DEVICE_DIR}/FastVLM-0.5B.qualcomm.sm8750.litertlm"
DEVICE_IMAGE_PATH="${DEVICE_DIR}/image.jpg"

PROMPT="Describe this image in one sentence."
MAX_NUM_TOKENS=512
JOBS=6
SKIP_BUILD=0
SKIP_PUSH=0
IMAGE_HOST_PATH=""

usage() {
  cat <<'EOF'
Usage:
  run_fastvlm_litert_npu_adb.sh [options]

Options:
  --image PATH          Host image path to push to device as image.jpg
  --prompt TEXT         Prompt text (default: "Describe this image in one sentence.")
  --max-num-tokens N    Max context length for run (default: 512)
  --jobs N              Bazel jobs / local CPU cap (default: 6)
  --skip-build 0|1      Skip Bazel build steps (default: 0)
  --skip-push 0|1       Skip adb push steps (default: 0)
  --device-dir PATH     Device run directory (default: /data/local/tmp/vlm_phase1)
  -h, --help            Show help

Notes:
  - Uses FastVLM model artifact pinned at:
    artifacts/models/FastVLM-0.5B.qualcomm.sm8750.litertlm
  - Requires QAIRT at /opt/qcom/aistack/qairt (override with QAIRT_ROOT env).
  - Uses hexagon-v79 DSP libs (required on this SM8750P setup).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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
      DEVICE_MODEL_PATH="${DEVICE_DIR}/FastVLM-0.5B.qualcomm.sm8750.litertlm"
      DEVICE_IMAGE_PATH="${DEVICE_DIR}/image.jpg"
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
      --jobs="${JOBS}" \
      --local_resources="cpu=${JOBS}" \
      --local_resources=memory=8192 \
      --loading_phase_threads=2

    ANDROID_NDK_HOME="${NDK_PATH}" ANDROID_NDK_ROOT="${NDK_PATH}" \
      bazel build --config=android_arm64 \
      @litert//litert/vendors/qualcomm/dispatch:dispatch_api_so \
      --override_repository="qairt=${LOCAL_QAIRT_REPO}" \
      --jobs="${JOBS}" \
      --local_resources="cpu=${JOBS}" \
      --local_resources=memory=8192 \
      --loading_phase_threads=2
  )
fi

BINARY_PATH="${LITERT_LM_DIR}/bazel-bin/runtime/engine/litert_lm_advanced_main"
DISPATCH_SO_PATH="${LITERT_LM_DIR}/bazel-bin/external/litert/litert/vendors/qualcomm/dispatch/libLiteRtDispatch_Qualcomm.so"
if [[ ! -f "${BINARY_PATH}" ]]; then
  echo "Missing built binary: ${BINARY_PATH}" >&2
  exit 6
fi
if [[ ! -f "${DISPATCH_SO_PATH}" ]]; then
  echo "Missing dispatch library: ${DISPATCH_SO_PATH}" >&2
  exit 6
fi

if [[ "${SKIP_PUSH}" == "0" ]]; then
  adb shell "mkdir -p '${DEVICE_DIR}' '${DEVICE_DIR}/dispatch_libs' '${DEVICE_DIR}/hexagon-v79'"

  adb push "${BINARY_PATH}" "${DEVICE_DIR}/litert_lm_advanced_main" >/dev/null
  adb shell "chmod +x '${DEVICE_DIR}/litert_lm_advanced_main'"

  adb push "${MODEL_HOST_PATH}" "${DEVICE_MODEL_PATH}" >/dev/null
  adb push "${LITERT_LM_DIR}/prebuilt/android_arm64/." "${DEVICE_DIR}/" >/dev/null
  adb push "${DISPATCH_SO_PATH}" "${DEVICE_DIR}/dispatch_libs/libLiteRtDispatch_Qualcomm.so" >/dev/null
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
adb shell "cd '${DEVICE_DIR}' && \
  export LD_LIBRARY_PATH='${DEVICE_DIR}:${DEVICE_DIR}/dispatch_libs' && \
  export ADSP_LIBRARY_PATH='${DEVICE_DIR}/hexagon-v79;/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp' && \
  ./litert_lm_advanced_main \
    --backend=npu \
    --vision_backend=npu \
    --model_path='${DEVICE_MODEL_PATH}' \
    --input_prompt='${PROMPT} [image:${DEVICE_IMAGE_PATH}]' \
    --max_num_tokens='${MAX_NUM_TOKENS}' \
    --litert_dispatch_lib_dir='${DEVICE_DIR}/dispatch_libs'" >"${RUN_LOG}" 2>&1
RUN_RC=$?
set -e

echo "Run exit code: ${RUN_RC}"
echo "Run log: ${RUN_LOG}"
echo
echo "Key runtime lines:"
rg -n "EncoderBackend|AdapterBackend|DispatchDelegate|context_binary_info|RunPrefillAsync status|RunDecodeAsync|ERROR|Invalid|Custom NPU execution latency stats|tokens per second" "${RUN_LOG}" | head -n 120 || true

exit "${RUN_RC}"
