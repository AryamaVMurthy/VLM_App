#!/usr/bin/env bash
set -euo pipefail

PKG="com.qidk.fastvlm"
OUT_DIR="${1:-./artifacts/metrics_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUT_DIR}"

if ! command -v adb >/dev/null 2>&1; then
  echo "adb not found in PATH" >&2
  exit 1
fi

adb wait-for-device

TMP_TGZ="/data/local/tmp/qidk_fastvlm_metrics.tgz"
adb shell "run-as ${PKG} sh -c 'cd files && tar -czf ${TMP_TGZ} metrics 2>/dev/null || true'"
adb shell "run-as ${PKG} cat ${TMP_TGZ}" > "${OUT_DIR}/metrics.tgz" || true

if [[ -s "${OUT_DIR}/metrics.tgz" ]]; then
  tar -xzf "${OUT_DIR}/metrics.tgz" -C "${OUT_DIR}" || true
  echo "Metrics exported to ${OUT_DIR}"
else
  echo "No metrics archive found. Ensure app has run at least one request."
fi
