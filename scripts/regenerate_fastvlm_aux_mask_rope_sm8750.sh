#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LITERT_DIR="${ROOT_DIR}/third_party/litert"
QAIRT_ROOT="${QAIRT_ROOT:-/opt/qcom/aistack/qairt}"
LOCAL_QAIRT_REPO="${ROOT_DIR}/artifacts/local_qairt_repo"

INPUT_AUX_MODEL="${ROOT_DIR}/artifacts/graph_inspect/extracted/Section2_TFLiteModel_tf_lite_aux.tflite"
OUTPUT_AUX_MODEL="${ROOT_DIR}/artifacts/models/FastVLM-0.5B.qualcomm.sm8750.aux_mask_rope.tflite"
SOC_MODEL="SM8750"
JOBS=6
SKIP_BUILD=0

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  regenerate_fastvlm_aux_mask_rope_sm8750.sh [options]

Options:
  --input-aux-model PATH   Input TF_LITE_AUX TFLite model
  --output-aux-model PATH  Output reduced compiled TF_LITE_AUX model
  --qairt-root PATH        QAIRT root (default: /opt/qcom/aistack/qairt or QAIRT_ROOT)
  --soc-model NAME         Qualcomm compiler target SoC (default: SM8750)
  --jobs N                 Bazel jobs / local CPU cap (default: 6)
  --skip-build 0|1         Skip Bazel build steps (default: 0)
  -h, --help               Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-aux-model)
      INPUT_AUX_MODEL="${2:-}"
      shift 2
      ;;
    --output-aux-model)
      OUTPUT_AUX_MODEL="${2:-}"
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
[[ -f "${INPUT_AUX_MODEL}" ]] || fail "Missing input aux model: ${INPUT_AUX_MODEL}"
[[ -d "${QAIRT_ROOT}" ]] || fail "Missing QAIRT root: ${QAIRT_ROOT}"

prepare_local_qairt_repo() {
  mkdir -p "${LOCAL_QAIRT_REPO}"
  cp "${LITERT_DIR}/third_party/qairt/qairt.BUILD" "${LOCAL_QAIRT_REPO}/BUILD"
  cat > "${LOCAL_QAIRT_REPO}/WORKSPACE" <<'EOF'
workspace(name = "qairt")
EOF
  ln -sfn "${QAIRT_ROOT}/include" "${LOCAL_QAIRT_REPO}/include"
  ln -sfn "${QAIRT_ROOT}/lib" "${LOCAL_QAIRT_REPO}/lib"
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

if [[ "${SKIP_BUILD}" == "0" ]]; then
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
      //litert/tools:subset_signatures_main \
      //litert/vendors/qualcomm/compiler:qnn_compiler_plugin_so
  )
fi

APPLY_PLUGIN_BIN="${LITERT_DIR}/bazel-bin/litert/tools/apply_plugin_main"
ANALYZE_MODEL_BIN="${LITERT_DIR}/bazel-bin/litert/tools/analyze_model_main"
SUBSET_SIGNATURES_BIN="${LITERT_DIR}/bazel-bin/litert/tools/subset_signatures_main"
COMPILER_PLUGIN_SO="${LITERT_DIR}/bazel-bin/litert/vendors/qualcomm/compiler/libLiteRtCompilerPlugin_Qualcomm.so"
[[ -f "${APPLY_PLUGIN_BIN}" ]] || fail "Missing apply_plugin_main: ${APPLY_PLUGIN_BIN}"
[[ -f "${ANALYZE_MODEL_BIN}" ]] || fail "Missing analyze_model_main: ${ANALYZE_MODEL_BIN}"
[[ -f "${SUBSET_SIGNATURES_BIN}" ]] || fail "Missing subset_signatures_main: ${SUBSET_SIGNATURES_BIN}"
[[ -f "${COMPILER_PLUGIN_SO}" ]] || fail "Missing compiler plugin: ${COMPILER_PLUGIN_SO}"

HOST_QAIRT_LIB_DIR="$(resolve_host_qairt_lib_dir)" || fail "Missing host QAIRT libs under ${QAIRT_ROOT}/lib"
PLUGIN_LIB_DIR="$(dirname "${COMPILER_PLUGIN_SO}")"
LD_LIBRARY_PATH_VALUE="${HOST_QAIRT_LIB_DIR}:${PLUGIN_LIB_DIR}"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  LD_LIBRARY_PATH_VALUE="${LD_LIBRARY_PATH_VALUE}:${LD_LIBRARY_PATH}"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
WORK_DIR="${ROOT_DIR}/artifacts/regeneration/aux_${SOC_MODEL,,}_${TIMESTAMP}"
mkdir -p "${WORK_DIR}" "$(dirname "${OUTPUT_AUX_MODEL}")"

SELECTIVE_AUX_MODEL="${WORK_DIR}/aux_mask_rope_selective_apply.tflite"

env LD_LIBRARY_PATH="${LD_LIBRARY_PATH_VALUE}" \
  "${APPLY_PLUGIN_BIN}" \
  --cmd=apply \
  --subgraphs=1,2,4,5 \
  --model="${INPUT_AUX_MODEL}" \
  --soc_manufacturer=Qualcomm \
  --soc_model="${SOC_MODEL}" \
  --libs="${PLUGIN_LIB_DIR}" \
  --o="${SELECTIVE_AUX_MODEL}" \
  --qualcomm_log_level=info \
  --qualcomm_backend=htp \
  --qualcomm_optimization_level=O3 \
  --qualcomm_graph_priority=default \
  --qualcomm_use_conv_hmx=true \
  --qualcomm_use_fold_relu=true \
  >"${WORK_DIR}/apply_aux.log" 2>&1

[[ -f "${SELECTIVE_AUX_MODEL}" ]] || fail "Selective aux apply did not produce: ${SELECTIVE_AUX_MODEL}"

"${SUBSET_SIGNATURES_BIN}" \
  --model_path="${SELECTIVE_AUX_MODEL}" \
  --output_path="${OUTPUT_AUX_MODEL}" \
  --signature_keys=decode_mask,decode_rope,prefill_mask_128,prefill_rope_128 \
  >"${WORK_DIR}/subset_aux.log" 2>&1

[[ -f "${OUTPUT_AUX_MODEL}" ]] || fail "Reduced aux generation did not produce: ${OUTPUT_AUX_MODEL}"

"${ANALYZE_MODEL_BIN}" \
  --model_path="${OUTPUT_AUX_MODEL}" \
  >"${WORK_DIR}/analyze_aux.log" 2>&1

echo "Reduced AUX model: ${OUTPUT_AUX_MODEL}"
echo "Logs:"
echo "  ${WORK_DIR}/apply_aux.log"
echo "  ${WORK_DIR}/subset_aux.log"
echo "  ${WORK_DIR}/analyze_aux.log"
