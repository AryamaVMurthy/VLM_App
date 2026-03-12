#!/system/bin/sh
set -eu

DEVICE_DIR="/data/local/tmp/graphpilot_edge"
BINARY="${DEVICE_DIR}/graphpilot_retrieval_main"
MODEL="${DEVICE_DIR}/embeddinggemma-300M_seq512_mixed-precision.tflite"
TOKENIZER="${DEVICE_DIR}/tokenizer.model"
KB="${DEVICE_DIR}/retrieval_kb.json"
GPU_RUNTIME_DIR="${DEVICE_DIR}/gpu_libs_opencl_only"

emit_error() {
  printf 'ERROR %s\n' "$1"
}

read -r proto || exit 1
if [ "${proto}" != "GRAPHPILOT_RETRIEVAL_V1" ]; then
  emit_error "invalid protocol header"
  exit 2
fi

read -r request_id || {
  emit_error "missing request id"
  exit 2
}
read -r query_b64 || {
  emit_error "missing query payload"
  exit 2
}
read -r top_k || {
  emit_error "missing top_k"
  exit 2
}

case "${top_k}" in
  ''|*[!0-9]*)
    emit_error "invalid top_k"
    exit 2
    ;;
esac

if [ ! -x "${BINARY}" ]; then
  emit_error "missing retrieval executable on device"
  exit 2
fi
if [ ! -f "${MODEL}" ] || [ ! -f "${TOKENIZER}" ] || [ ! -f "${KB}" ]; then
  emit_error "missing retrieval assets on device"
  exit 2
fi
if [ ! -d "${GPU_RUNTIME_DIR}" ]; then
  emit_error "missing gpu runtime directory"
  exit 2
fi

export LD_LIBRARY_PATH="${GPU_RUNTIME_DIR}:${DEVICE_DIR}"

"${BINARY}" \
  --mode=retrieve \
  --model_path="${MODEL}" \
  --tokenizer_path="${TOKENIZER}" \
  --kb_path="${KB}" \
  --query_b64="${query_b64}" \
  --accelerator=gpu \
  --runtime_library_dir="${GPU_RUNTIME_DIR}" \
  --top_k="${top_k}"

rc=$?
if [ "${rc}" -ne 0 ]; then
  emit_error "graphpilot_retrieval_main failed for request ${request_id}"
  exit "${rc}"
fi
