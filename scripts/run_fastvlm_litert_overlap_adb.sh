#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/shell_quote_lib.sh"

LITERT_LM_DIR="${ROOT_DIR}/third_party/litert-lm"
LITERT_DIR="${ROOT_DIR}/third_party/litert"
QAIRT_ROOT="${QAIRT_ROOT:-/opt/qcom/aistack/qairt}"
LOCAL_QAIRT_REPO="${ROOT_DIR}/artifacts/local_qairt_repo"
MODEL_HOST_PATH="${ROOT_DIR}/artifacts/models/FastVLM-0.5B.qualcomm.sm8750.auxmaskrope_runtime.litertlm"
DECODE_MODEL_HOST_PATH="${ROOT_DIR}/artifacts/models/FastVLM-0.5B.litertlm"

DEVICE_DIR="/data/local/tmp/vlm_overlap"
PROMPT="Describe this image in one sentence."
MAX_NUM_TOKENS=512
MAX_OUTPUT_TOKENS=128
MAX_VISUAL_TOKENS=0
VISUAL_TOKEN_PRUNING_STRATEGY="prompt_conditioned_v1"
PREPARE_QUEUE_SIZE=4
ADAPTIVE_VISUAL_TOKEN_BUDGETS=""
BUDGET_CONTROLLER_ANSWER_MODE="none"
CONSTRAINT_REGEX=""
JOBS=6
SKIP_BUILD=0
SKIP_PUSH=0
SKIP_INPUT_SYNC=0
EVENT_MODE=1
IMAGE_HOST_PATHS=()
REQUEST_MANIFEST_HOST_PATH=""
REQUEST_MANIFEST_DEVICE_PATH=""

usage() {
  cat <<'EOF'
Usage:
  run_fastvlm_litert_overlap_adb.sh [options] --image PATH --image PATH
  run_fastvlm_litert_overlap_adb.sh [options] --request-manifest PATH [--image PATH ...]

Options:
  --model PATH          Host .litertlm path to run on device
  --decode-model PATH   Host CPU decode .litertlm path (default: raw FastVLM bundle)
  --image PATH          Host image path to push to device (repeat at least twice)
  --request-manifest PATH
                       Host JSONL manifest with request_id, prompt, image_path
  --prompt TEXT         Prompt text
  --max-num-tokens N    Max context length
  --max-output-tokens N Max generated output tokens
  --max-visual-tokens N Max projected visual tokens to keep
  --visual-token-pruning-strategy S
                       Visual token pruning strategy
  --prepare-queue-size N
                       Number of prepared requests to buffer ahead of prefill
  --adaptive-visual-token-budgets CSV
                       Optional comma-separated adaptive budget buckets
  --budget-controller-answer-mode MODE
                       Adaptive controller hint: none, short, or long
  --constraint-regex R Decode-time regex constraint passed to LiteRT-LM
  --event-mode 0|1      Emit structured VLM_EVENT lines (default: 1)
  --jobs N              Bazel jobs / local CPU cap
  --skip-build 0|1      Skip Bazel build
  --skip-push 0|1       Skip pushing static runtime/model assets
  --skip-input-sync 0|1 Skip pushing the current manifest and images
  --device-dir PATH     Device run directory
  -h, --help            Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_HOST_PATH="${2:-}"
      shift 2
      ;;
    --decode-model)
      DECODE_MODEL_HOST_PATH="${2:-}"
      shift 2
      ;;
    --image)
      IMAGE_HOST_PATHS+=("${2:-}")
      shift 2
      ;;
    --request-manifest)
      REQUEST_MANIFEST_HOST_PATH="${2:-}"
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
    --prepare-queue-size)
      PREPARE_QUEUE_SIZE="${2:-}"
      shift 2
      ;;
    --adaptive-visual-token-budgets)
      ADAPTIVE_VISUAL_TOKEN_BUDGETS="${2:-}"
      shift 2
      ;;
    --budget-controller-answer-mode)
      BUDGET_CONTROLLER_ANSWER_MODE="${2:-}"
      shift 2
      ;;
    --constraint-regex)
      CONSTRAINT_REGEX="${2:-}"
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
    --skip-input-sync)
      SKIP_INPUT_SYNC="${2:-}"
      shift 2
      ;;
    --device-dir)
      DEVICE_DIR="${2:-}"
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

if [[ -n "${REQUEST_MANIFEST_HOST_PATH}" && ! -f "${REQUEST_MANIFEST_HOST_PATH}" ]]; then
  echo "Missing request manifest: ${REQUEST_MANIFEST_HOST_PATH}" >&2
  exit 3
fi
if [[ -n "${REQUEST_MANIFEST_HOST_PATH}" ]]; then
  REQUEST_MANIFEST_DEVICE_PATH="${DEVICE_DIR}/$(basename "${REQUEST_MANIFEST_HOST_PATH}")"
fi
if [[ -z "${REQUEST_MANIFEST_HOST_PATH}" && ${#IMAGE_HOST_PATHS[@]} -lt 2 ]]; then
  echo "At least 2 --image arguments are required for overlap when no request manifest is provided." >&2
  exit 2
fi
if [[ ! -f "${MODEL_HOST_PATH}" ]]; then
  echo "Missing model artifact: ${MODEL_HOST_PATH}" >&2
  exit 3
fi
if [[ ! -f "${DECODE_MODEL_HOST_PATH}" ]]; then
  echo "Missing decode model artifact: ${DECODE_MODEL_HOST_PATH}" >&2
  exit 3
fi
for image_path in "${IMAGE_HOST_PATHS[@]}"; do
  if [[ ! -f "${image_path}" ]]; then
    echo "Missing image file: ${image_path}" >&2
    exit 3
  fi
done
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
    exit 5
  fi

  (
    cd "${LITERT_LM_DIR}"
    ANDROID_NDK_HOME="${NDK_PATH}" ANDROID_NDK_ROOT="${NDK_PATH}" \
      bazel build --config=android_arm64 \
      //runtime/engine:litert_lm_overlap_main \
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
BINARY_PATH="${ANDROID_BAZEL_BIN}/runtime/engine/litert_lm_overlap_main"
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
    exit 6
  fi
fi
for required_file in "${LITERT_SO_PATH}" "${DISPATCH_SO_PATH}" "${COMPILER_PLUGIN_SO_PATH}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing runtime library: ${required_file}" >&2
    exit 6
  fi
done

DEVICE_MODEL_PATH="${DEVICE_DIR}/$(basename "${MODEL_HOST_PATH}")"
DEVICE_DECODE_MODEL_PATH="${DEVICE_DIR}/$(basename "${DECODE_MODEL_HOST_PATH}")"
DEVICE_IMAGE_PATHS=()
for idx in "${!IMAGE_HOST_PATHS[@]}"; do
  DEVICE_IMAGE_PATHS+=("${DEVICE_DIR}/image_$(printf '%03d' "${idx}").jpg")
done

if [[ "${SKIP_PUSH}" == "0" ]]; then
  DEVICE_DIR_QUOTED="$(shell_single_quote "${DEVICE_DIR}")"
  DEVICE_DISPATCH_DIR_QUOTED="$(shell_single_quote "${DEVICE_DIR}/dispatch_libs")"
  DEVICE_HEXAGON_DIR_QUOTED="$(shell_single_quote "${DEVICE_DIR}/hexagon-v79")"
  adb shell "mkdir -p ${DEVICE_DIR_QUOTED} ${DEVICE_DISPATCH_DIR_QUOTED} ${DEVICE_HEXAGON_DIR_QUOTED}"

  adb push "${BINARY_PATH}" "${DEVICE_DIR}/litert_lm_overlap_main" >/dev/null
  adb shell "chmod +x $(shell_single_quote "${DEVICE_DIR}/litert_lm_overlap_main")"
  adb push "${MODEL_HOST_PATH}" "${DEVICE_MODEL_PATH}" >/dev/null
  adb push "${DECODE_MODEL_HOST_PATH}" "${DEVICE_DECODE_MODEL_PATH}" >/dev/null
  adb push "${LITERT_LM_DIR}/prebuilt/android_arm64/." "${DEVICE_DIR}/" >/dev/null
  adb push "${LITERT_SO_PATH}" "${DEVICE_DIR}/dispatch_libs/libLiteRt.so" >/dev/null
  adb push "${DISPATCH_SO_PATH}" "${DEVICE_DIR}/dispatch_libs/libLiteRtDispatch_Qualcomm.so" >/dev/null
  adb push "${COMPILER_PLUGIN_SO_PATH}" "${DEVICE_DIR}/dispatch_libs/libLiteRtCompilerPlugin_Qualcomm.so" >/dev/null
  adb push "${QAIRT_ROOT}/lib/aarch64-android/." "${DEVICE_DIR}/dispatch_libs/" >/dev/null
  adb push "${QAIRT_ROOT}/lib/hexagon-v79/unsigned/." "${DEVICE_DIR}/hexagon-v79/" >/dev/null
fi

if [[ "${SKIP_INPUT_SYNC}" == "0" ]]; then
  DEVICE_DIR_QUOTED="$(shell_single_quote "${DEVICE_DIR}")"
  adb shell "mkdir -p ${DEVICE_DIR_QUOTED}"
  for idx in "${!IMAGE_HOST_PATHS[@]}"; do
    adb push "${IMAGE_HOST_PATHS[$idx]}" "${DEVICE_IMAGE_PATHS[$idx]}" >/dev/null
  done
  if [[ -n "${REQUEST_MANIFEST_HOST_PATH}" ]]; then
    adb push "${REQUEST_MANIFEST_HOST_PATH}" "${REQUEST_MANIFEST_DEVICE_PATH}" >/dev/null
  fi
fi

DEVICE_IMAGE_PATHS_CSV="$(IFS=,; echo "${DEVICE_IMAGE_PATHS[*]}")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/artifacts/logs"
mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/adb_overlap_run_${TIMESTAMP}.log"

EVENT_MODE_EXPORT=""
if [[ "${EVENT_MODE}" == "1" ]]; then
  EVENT_MODE_EXPORT="export LITERT_LM_EVENT_MODE=1 &&"
fi
PRUNING_ENV_EXPORT=""
for pruning_env in \
  LITERT_LM_PRUNING_PROMPT_SIMILARITY_WEIGHT \
  LITERT_LM_PRUNING_SALIENCE_WEIGHT \
  LITERT_LM_PRUNING_REDUNDANCY_PENALTY_WEIGHT \
  LITERT_LM_PRUNING_PROMPT_ATTENTION_LOGIT_SCALE \
  LITERT_LM_PRUNING_PROMPT_ATTENTION_TOP_K \
  LITERT_LM_PRUNING_LOCAL_REFINEMENT_MIN_PROMPT_GAIN \
  LITERT_LM_PRUNING_MAX_LOCAL_REFINEMENT_FRACTION \
  LITERT_LM_PRUNING_MAX_LOCAL_REFINEMENT_SALIENCE_DROP \
  LITERT_LM_PRUNING_MIN_GLOBAL_MEAN_PROMPT_SIMILARITY \
  LITERT_LM_PRUNING_MIN_GLOBAL_MEAN_SALIENCE \
  LITERT_LM_PRUNING_FUSE_PROJECTION_PRUNE_PACK; do
  if [[ -n "${!pruning_env:-}" ]]; then
    PRUNING_ENV_EXPORT+="export ${pruning_env}=$(shell_single_quote "${!pruning_env}") && "
  fi
done
REQUEST_MANIFEST_ARG=""
if [[ -n "${REQUEST_MANIFEST_DEVICE_PATH}" ]]; then
  REQUEST_MANIFEST_ARG="    --request_manifest_path=$(shell_single_quote "${REQUEST_MANIFEST_DEVICE_PATH}") \\"
fi
INPUT_PROMPT_ARG="    --input_prompt=$(shell_single_quote "${PROMPT}") \\"
IMAGE_PATHS_ARG="    --image_paths=$(shell_single_quote "${DEVICE_IMAGE_PATHS_CSV}") \\"
CONSTRAINT_REGEX_ARG=""
if [[ -n "${CONSTRAINT_REGEX}" ]]; then
  CONSTRAINT_REGEX_ARG="    --constraint_regex=$(shell_single_quote "${CONSTRAINT_REGEX}") \\"
fi
ADAPTIVE_VISUAL_TOKEN_BUDGETS_ARG=""
if [[ -n "${ADAPTIVE_VISUAL_TOKEN_BUDGETS}" ]]; then
  ADAPTIVE_VISUAL_TOKEN_BUDGETS_ARG="    --adaptive_visual_token_budgets=$(shell_single_quote "${ADAPTIVE_VISUAL_TOKEN_BUDGETS}") \\"
fi
BUDGET_CONTROLLER_ANSWER_MODE_ARG="    --budget_controller_answer_mode=$(shell_single_quote "${BUDGET_CONTROLLER_ANSWER_MODE}") \\"
if [[ -n "${REQUEST_MANIFEST_DEVICE_PATH}" ]]; then
  INPUT_PROMPT_ARG=""
  IMAGE_PATHS_ARG=""
fi

echo "Run log: ${RUN_LOG}"

set +e
adb shell "cd $(shell_single_quote "${DEVICE_DIR}") && \
  export LD_LIBRARY_PATH=$(shell_single_quote "${DEVICE_DIR}:${DEVICE_DIR}/dispatch_libs") && \
  export ADSP_LIBRARY_PATH=$(shell_single_quote "${DEVICE_DIR}/hexagon-v79;/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp") && \
  ${EVENT_MODE_EXPORT} \
  ${PRUNING_ENV_EXPORT} \
  ./litert_lm_overlap_main \
    --model_path=$(shell_single_quote "${DEVICE_MODEL_PATH}") \
    --decode_model_path=$(shell_single_quote "${DEVICE_DECODE_MODEL_PATH}") \
${INPUT_PROMPT_ARG}\
${REQUEST_MANIFEST_ARG}\
${IMAGE_PATHS_ARG}\
    --max_num_tokens='${MAX_NUM_TOKENS}' \
    --max_output_tokens='${MAX_OUTPUT_TOKENS}' \
    --max_visual_tokens='${MAX_VISUAL_TOKENS}' \
    --visual_token_pruning_strategy='${VISUAL_TOKEN_PRUNING_STRATEGY}' \
    --prepare_queue_size='${PREPARE_QUEUE_SIZE}' \
${ADAPTIVE_VISUAL_TOKEN_BUDGETS_ARG}\
${BUDGET_CONTROLLER_ANSWER_MODE_ARG}\
${CONSTRAINT_REGEX_ARG}\
    --litert_dispatch_lib_dir=$(shell_single_quote "${DEVICE_DIR}/dispatch_libs")" >"${RUN_LOG}" 2>&1
RUN_RC=$?
set -e

echo "Run exit code: ${RUN_RC}"
echo
echo "Key runtime lines:"
rg -n "OVERLAP_|VLM_EVENT|Applied visual token budget|Visual token pruning decision|ERROR|Invalid" "${RUN_LOG}" | head -n 200 || true

exit "${RUN_RC}"
