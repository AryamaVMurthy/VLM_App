#!/usr/bin/env bash
set -euo pipefail

DEVICE_DIR="/data/local/tmp/vlm_phase1"
QUESTION="${1:-Describe this image in one sentence.}"
IMAGE_PATH="${2:-${DEVICE_DIR}/image.jpg}"
MODEL_PATH="${3:-${DEVICE_DIR}/FastVLM-0.5B.qualcomm.sm8750.litertlm}"
MAX_NUM_TOKENS="${4:-768}"
MAX_OUTPUT_TOKENS="${5:-160}"
BACKEND="${6:-npu}"

if ! command -v adb >/dev/null 2>&1; then
  echo "adb not found" >&2
  exit 1
fi

Q_B64="$(printf '%s' "${QUESTION}" | base64 -w 0)"
adb wait-for-device >/dev/null
adb shell "printf 'VLM_PHASE1\n1\n${Q_B64}\n${IMAGE_PATH}\n${MODEL_PATH}\n${MAX_NUM_TOKENS}\n${MAX_OUTPUT_TOKENS}\n${BACKEND}\n' | toybox nc -w 120 127.0.0.1 21909" \
  | tee /tmp/vlm_daemon_test_$(date +%Y%m%d_%H%M%S).log
