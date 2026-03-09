#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LITERT_DIR="${ROOT_DIR}/third_party/litert"
LITERT_LM_DIR="${ROOT_DIR}/third_party/litert-lm"
QAIRT_ROOT="${QAIRT_ROOT:-/opt/qcom/aistack/qairt}"
LOCAL_QAIRT_REPO="${ROOT_DIR}/artifacts/local_qairt_repo"

BASE_MODEL="${ROOT_DIR}/artifacts/models/FastVLM-0.5B.qualcomm.sm8750.litertlm"
SOURCE_MODEL="${ROOT_DIR}/artifacts/models/FastVLM-0.5B.litertlm"
OUTPUT_MODEL="${ROOT_DIR}/artifacts/models/FastVLM-0.5B.qualcomm.sm8750.prefill243.litertlm"
SOC_MODEL="SM8750"
JOBS=6
SKIP_BUILD=0
AUX_MODEL=""

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  regenerate_fastvlm_prefill_sm8750.sh [options]

Options:
  --base-model PATH     Base Qualcomm .litertlm artifact to copy non-prefill sections from
  --source-model PATH   Source non-Qualcomm .litertlm artifact to extract prefill from
  --output-model PATH   Output .litertlm artifact path
  --aux-model PATH      Optional replacement TF_LITE_AUX section to package
  --qairt-root PATH     QAIRT root (default: /opt/qcom/aistack/qairt or QAIRT_ROOT)
  --soc-model NAME      Qualcomm compiler target SoC (default: SM8750)
  --jobs N              Bazel jobs / local CPU cap (default: 6)
  --skip-build 0|1      Skip Bazel build steps (default: 0)
  -h, --help            Show help

Notes:
  - The Qualcomm compiler plugin currently targets SM8750 for SM8750P devices.
  - This script rebuilds the .litertlm archive by replacing only the
    tf_lite_prefill_decode section with a fresh Qualcomm-compiled section.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-model)
      BASE_MODEL="${2:-}"
      shift 2
      ;;
    --source-model)
      SOURCE_MODEL="${2:-}"
      shift 2
      ;;
    --output-model)
      OUTPUT_MODEL="${2:-}"
      shift 2
      ;;
    --aux-model)
      AUX_MODEL="${2:-}"
      shift 2
      ;;
    --qairt-root)
      QAIRT_ROOT="${2:-}"
      shift 2
      ;;
    --soc-model)
      SOC_MODEL="${2:-}"
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
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown option: $1"
      ;;
  esac
done

[[ "${JOBS}" =~ ^[0-9]+$ ]] || fail "Invalid --jobs: ${JOBS}"
[[ "${SKIP_BUILD}" == "0" || "${SKIP_BUILD}" == "1" ]] || fail "Invalid --skip-build (expected 0 or 1): ${SKIP_BUILD}"

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || fail "Missing file: ${path}"
}

require_dir() {
  local path="$1"
  [[ -d "${path}" ]] || fail "Missing directory: ${path}"
}

run_clean_python_env() {
  env -u PYTHONHOME -u PYTHONPATH "$@"
}

resolve_host_qairt_lib_dir() {
  local candidate
  for candidate in \
    "${QAIRT_ROOT}/lib/x86_64-linux-clang" \
    "${QAIRT_ROOT}/lib/x86_64-linux-gcc" \
    "${QAIRT_ROOT}/lib/x86_64-linux-gnu"; do
    if [[ -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

prepare_local_qairt_repo() {
  mkdir -p "${LOCAL_QAIRT_REPO}"
  cp "${LITERT_DIR}/third_party/qairt/qairt.BUILD" "${LOCAL_QAIRT_REPO}/BUILD"
  cat > "${LOCAL_QAIRT_REPO}/WORKSPACE" <<'EOF'
workspace(name = "qairt")
EOF
  ln -sfn "${QAIRT_ROOT}/include" "${LOCAL_QAIRT_REPO}/include"
  ln -sfn "${QAIRT_ROOT}/lib" "${LOCAL_QAIRT_REPO}/lib"
}

build_tools() {
  prepare_local_qairt_repo
  (
    cd "${LITERT_DIR}"
    bazel build \
      --override_repository="qairt=${LOCAL_QAIRT_REPO}" \
      --jobs="${JOBS}" \
      --local_resources="cpu=${JOBS}" \
      --local_resources=memory=8192 \
      --loading_phase_threads=2 \
      //litert/tools:apply_plugin_main \
      //litert/tools:analyze_model_main \
      //litert/vendors/qualcomm/compiler:qnn_compiler_plugin_so
  )
  (
    cd "${LITERT_LM_DIR}"
    bazel build \
      --override_repository="litert=${LITERT_DIR}" \
      --jobs="${JOBS}" \
      --local_resources="cpu=${JOBS}" \
      --local_resources=memory=8192 \
      --loading_phase_threads=2 \
      //schema/py:litertlm_peek_main \
      //schema/py:litertlm_builder_cli
  )
}

find_single_dump_file() {
  local dump_dir="$1"
  local pattern="$2"
  local matches=()
  while IFS= read -r match; do
    matches+=("${match}")
  done < <(find "${dump_dir}" -maxdepth 1 -type f -name "${pattern}" | sort)

  [[ "${#matches[@]}" -eq 1 ]] || fail "Expected exactly one file matching ${pattern} in ${dump_dir}, found ${#matches[@]}"
  printf '%s\n' "${matches[0]}"
}

require_dir "${ROOT_DIR}"
require_dir "${LITERT_DIR}"
require_dir "${LITERT_LM_DIR}"
require_dir "${QAIRT_ROOT}"
require_file "${BASE_MODEL}"
require_file "${SOURCE_MODEL}"
if [[ -n "${AUX_MODEL}" ]]; then
  require_file "${AUX_MODEL}"
fi

if [[ "${SKIP_BUILD}" == "0" ]]; then
  build_tools
fi

APPLY_PLUGIN_BIN="${LITERT_DIR}/bazel-bin/litert/tools/apply_plugin_main"
ANALYZE_MODEL_BIN="${LITERT_DIR}/bazel-bin/litert/tools/analyze_model_main"
COMPILER_PLUGIN_SO="${LITERT_DIR}/bazel-bin/litert/vendors/qualcomm/compiler/libLiteRtCompilerPlugin_Qualcomm.so"
PEEK_BIN="${LITERT_LM_DIR}/bazel-bin/schema/py/litertlm_peek_main"
BUILDER_BIN="${LITERT_LM_DIR}/bazel-bin/schema/py/litertlm_builder_cli"

require_file "${APPLY_PLUGIN_BIN}"
require_file "${ANALYZE_MODEL_BIN}"
require_file "${COMPILER_PLUGIN_SO}"
require_file "${PEEK_BIN}"
require_file "${BUILDER_BIN}"

HOST_QAIRT_LIB_DIR="$(resolve_host_qairt_lib_dir)" || fail "Missing host QAIRT libs under ${QAIRT_ROOT}/lib. Expected x86_64-linux-clang or x86_64-linux-gcc."
PLUGIN_LIB_DIR="$(dirname "${COMPILER_PLUGIN_SO}")"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
WORK_DIR="${ROOT_DIR}/artifacts/regeneration/prefill_${SOC_MODEL,,}_${TIMESTAMP}"
BASE_DUMP_DIR="${WORK_DIR}/base_dump"
SOURCE_DUMP_DIR="${WORK_DIR}/source_dump"
OUTPUT_DUMP_DIR="${WORK_DIR}/output_dump"
mkdir -p "${BASE_DUMP_DIR}" "${SOURCE_DUMP_DIR}" "${OUTPUT_DUMP_DIR}" "${WORK_DIR}/compiled"
mkdir -p "$(dirname "${OUTPUT_MODEL}")"

echo "Work directory: ${WORK_DIR}"

run_clean_python_env "${PEEK_BIN}" \
  --litertlm_file="${BASE_MODEL}" \
  --dump_files_dir="${BASE_DUMP_DIR}" \
  >"${WORK_DIR}/peek_base.log" 2>&1

run_clean_python_env "${PEEK_BIN}" \
  --litertlm_file="${SOURCE_MODEL}" \
  --dump_files_dir="${SOURCE_DUMP_DIR}" \
  >"${WORK_DIR}/peek_source.log" 2>&1

BASE_LLM_METADATA="${BASE_DUMP_DIR}/LlmMetadataProto.pbtext"
BASE_TOKENIZER="$(find_single_dump_file "${BASE_DUMP_DIR}" 'Section*_SP_Tokenizer*')"
BASE_EMBEDDER="$(find_single_dump_file "${BASE_DUMP_DIR}" 'Section*_tf_lite_embedder.tflite')"
BASE_AUX="$(find_single_dump_file "${BASE_DUMP_DIR}" 'Section*_tf_lite_aux.tflite')"
BASE_VISION_ADAPTER="$(find_single_dump_file "${BASE_DUMP_DIR}" 'Section*_tf_lite_vision_adapter.tflite')"
BASE_VISION_ENCODER="$(find_single_dump_file "${BASE_DUMP_DIR}" 'Section*_tf_lite_vision_encoder.tflite')"
SOURCE_PREFILL="$(find_single_dump_file "${SOURCE_DUMP_DIR}" 'Section*_tf_lite_prefill_decode.tflite')"

if [[ -n "${AUX_MODEL}" ]]; then
  BASE_AUX="${AUX_MODEL}"
fi

require_file "${BASE_LLM_METADATA}"
require_file "${BASE_TOKENIZER}"
require_file "${BASE_EMBEDDER}"
require_file "${BASE_AUX}"
require_file "${BASE_VISION_ADAPTER}"
require_file "${BASE_VISION_ENCODER}"
require_file "${SOURCE_PREFILL}"

COMPILED_PREFILL="${WORK_DIR}/compiled/Section5_TFLiteModel_tf_lite_prefill_decode.${SOC_MODEL}.tflite"
LD_LIBRARY_PATH_VALUE="${HOST_QAIRT_LIB_DIR}:${PLUGIN_LIB_DIR}"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  LD_LIBRARY_PATH_VALUE="${LD_LIBRARY_PATH_VALUE}:${LD_LIBRARY_PATH}"
fi

env LD_LIBRARY_PATH="${LD_LIBRARY_PATH_VALUE}" \
  "${APPLY_PLUGIN_BIN}" \
  --cmd=apply \
  --model="${SOURCE_PREFILL}" \
  --soc_manufacturer=Qualcomm \
  --soc_model="${SOC_MODEL}" \
  --libs="${PLUGIN_LIB_DIR}" \
  --o="${COMPILED_PREFILL}" \
  --qualcomm_log_level=info \
  --qualcomm_backend=htp \
  --qualcomm_optimization_level=O3 \
  --qualcomm_graph_priority=default \
  --qualcomm_use_conv_hmx=true \
  --qualcomm_use_fold_relu=true \
  >"${WORK_DIR}/apply_prefill.log" 2>&1

require_file "${COMPILED_PREFILL}"

"${ANALYZE_MODEL_BIN}" \
  --model_path="${COMPILED_PREFILL}" \
  >"${WORK_DIR}/analyze_prefill.log" 2>&1

run_clean_python_env "${BUILDER_BIN}" \
  system_metadata --str Authors "ODML team" \
  llm_metadata --path "${BASE_LLM_METADATA}" \
  tflite_model --path "${BASE_EMBEDDER}" --model_type embedder \
  tflite_model --path "${BASE_AUX}" --model_type aux \
  tflite_model --path "${BASE_VISION_ADAPTER}" --model_type vision_adapter \
  tflite_model --path "${BASE_VISION_ENCODER}" --model_type vision_encoder \
  tflite_model --path "${COMPILED_PREFILL}" --model_type prefill_decode \
  sp_tokenizer --path "${BASE_TOKENIZER}" \
  output --path "${OUTPUT_MODEL}" \
  >"${WORK_DIR}/rebuild.log" 2>&1

require_file "${OUTPUT_MODEL}"

run_clean_python_env "${PEEK_BIN}" \
  --litertlm_file="${OUTPUT_MODEL}" \
  --dump_files_dir="${OUTPUT_DUMP_DIR}" \
  >"${WORK_DIR}/peek_output.log" 2>&1

echo "Regenerated artifact: ${OUTPUT_MODEL}"
echo "Compiled prefill section: ${COMPILED_PREFILL}"
echo "Logs:"
echo "  ${WORK_DIR}/peek_base.log"
echo "  ${WORK_DIR}/peek_source.log"
echo "  ${WORK_DIR}/apply_prefill.log"
echo "  ${WORK_DIR}/analyze_prefill.log"
echo "  ${WORK_DIR}/rebuild.log"
echo "  ${WORK_DIR}/peek_output.log"
