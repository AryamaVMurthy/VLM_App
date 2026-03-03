#!/usr/bin/env bash
set -euo pipefail

PKG="com.qidk.fastvlm"
OUT_FILE="${1:-./artifacts/latest_benchmark.json}"

mkdir -p "$(dirname "${OUT_FILE}")"

if ! command -v adb >/dev/null 2>&1; then
  echo "adb not found in PATH" >&2
  exit 1
fi

adb wait-for-device

LATEST_FILE=$(adb shell "run-as ${PKG} sh -c 'ls -1t files/metrics/benchmarks/*.json 2>/dev/null | head -n 1'" | tr -d '\r')
if [[ -z "${LATEST_FILE}" ]]; then
  echo "No benchmark JSON found in app sandbox." >&2
  exit 2
fi

adb shell "run-as ${PKG} cat ${LATEST_FILE}" > "${OUT_FILE}"
echo "Pulled ${LATEST_FILE} -> ${OUT_FILE}"
