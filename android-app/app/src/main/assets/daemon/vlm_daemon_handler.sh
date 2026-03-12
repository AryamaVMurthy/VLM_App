#!/system/bin/sh
set -eu

DEVICE_DIR="/data/local/tmp/vlm_phase1"
BINARY="${DEVICE_DIR}/litert_lm_advanced_main"
NATIVE_LIB_DIR_DEFAULT="${DEVICE_DIR}/dispatch_libs"
MODEL_DEFAULT="${DEVICE_DIR}/FastVLM-0.5B.qualcomm.sm8750.litertlm"

emit_error() {
  msg="$1"
  printf 'VLM_EVENT {"type":"ERROR","message":"%s"}\n' "${msg}"
}

read -r proto || exit 1
if [ "${proto}" != "VLM_PHASE1" ]; then
  emit_error "invalid protocol header"
  exit 2
fi

read -r request_id || {
  emit_error "missing request id"
  exit 2
}
read -r question_b64 || {
  emit_error "missing question"
  exit 2
}
read -r image_path || {
  emit_error "missing image path"
  exit 2
}
read -r model_path || {
  model_path="${MODEL_DEFAULT}"
}
read -r max_num_tokens || {
  max_num_tokens="768"
}
read -r max_output_tokens || {
  max_output_tokens="160"
}
read -r requested_backend || {
  requested_backend="npu"
}

if [ -z "${model_path}" ]; then
  model_path="${MODEL_DEFAULT}"
fi

if [ -z "${max_num_tokens}" ]; then
  max_num_tokens="768"
fi
if [ -z "${max_output_tokens}" ]; then
  max_output_tokens="160"
fi
if [ -z "${requested_backend}" ]; then
  requested_backend="npu"
fi

case "${max_output_tokens}" in
  npu|NPU|cpu|CPU)
    requested_backend="${max_output_tokens}"
    max_output_tokens="160"
    ;;
esac

case "${requested_backend}" in
  npu|NPU)
    llm_backend="npu"
    vision_backend="npu"
    ;;
  cpu|CPU)
    llm_backend="cpu"
    vision_backend="cpu"
    ;;
  *)
    emit_error "invalid backend; expected npu or cpu"
    exit 2
    ;;
esac

NATIVE_LIB_DIR="${VLM_NATIVE_LIB_DIR:-${NATIVE_LIB_DIR_DEFAULT}}"

case "${max_num_tokens}" in
  ''|*[!0-9]*)
    emit_error "invalid max_num_tokens"
    exit 2
    ;;
esac

case "${max_output_tokens}" in
  ''|*[!0-9]*)
    emit_error "invalid max_output_tokens"
    exit 2
    ;;
esac

question="$(printf '%s' "${question_b64}" | base64 -d 2>/dev/null || true)"
if [ -z "${question}" ]; then
  emit_error "invalid base64 question payload"
  exit 2
fi

if [ ! -x "${BINARY}" ]; then
  emit_error "missing executable litert_lm_advanced_main"
  exit 2
fi
if [ ! -d "${NATIVE_LIB_DIR}" ]; then
  emit_error "missing native libs directory"
  exit 2
fi
if [ ! -f "${model_path}" ]; then
  emit_error "missing model artifact on device"
  exit 2
fi
if [ ! -f "${image_path}" ]; then
  emit_error "missing image file on device"
  exit 2
fi

export LD_LIBRARY_PATH="${DEVICE_DIR}:${NATIVE_LIB_DIR}"
export ADSP_LIBRARY_PATH="${DEVICE_DIR}/hexagon-v79;/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"
export LITERT_LM_EVENT_MODE=1
export LITERT_QNN_SIGNED_PD_SUPPORT=1
export LITERT_QNN_DISABLE_DEVICE_PLATFORM_INFO=1

"${BINARY}" \
  --backend="${llm_backend}" \
  --vision_backend="${vision_backend}" \
  --benchmark=true \
  --model_path="${model_path}" \
  --input_prompt="${question} [image:${image_path}]" \
  --max_num_tokens="${max_num_tokens}" \
  --max_output_tokens="${max_output_tokens}" \
  --litert_dispatch_lib_dir="${NATIVE_LIB_DIR}"

rc=$?
if [ "${rc}" -ne 0 ]; then
  emit_error "litert_lm_advanced_main failed"
  exit "${rc}"
fi
