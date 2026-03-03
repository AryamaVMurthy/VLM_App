#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME="FastVLM-0.5B.qualcomm.sm8750.litertlm"
REV="74e5aa3fb2adc196ca24bc7fb561ec92fc165e81"
MODEL_LOCAL="${1:-${ROOT_DIR}/artifacts/models/${MODEL_NAME}}"
PKG="com.qidk.fastvlm"
TMP_REMOTE="/data/local/tmp/${MODEL_NAME}"

if [[ ! -f "${MODEL_LOCAL}" ]]; then
  echo "Model not found at ${MODEL_LOCAL}" >&2
  echo "Remediation: run ./scripts/download_pinned_model.sh first or pass explicit model path." >&2
  exit 1
fi

if ! command -v adb >/dev/null 2>&1; then
  echo "adb not found in PATH" >&2
  exit 1
fi

adb wait-for-device
adb push "${MODEL_LOCAL}" "${TMP_REMOTE}"
adb shell run-as "${PKG}" mkdir -p "files/models/${REV}"
adb shell run-as "${PKG}" cp "${TMP_REMOTE}" "files/models/${REV}/${MODEL_NAME}"
adb shell run-as "${PKG}" ls -lh "files/models/${REV}/${MODEL_NAME}"

echo "Provisioned model into app sandbox."
