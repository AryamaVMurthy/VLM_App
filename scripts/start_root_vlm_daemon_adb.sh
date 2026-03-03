#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE_DIR="/data/local/tmp/vlm_phase1"
HANDLER_LOCAL="${ROOT_DIR}/scripts/vlm_daemon_handler.sh"
BINARY_LOCAL="${ROOT_DIR}/third_party/litert-lm/bazel-bin/runtime/engine/litert_lm_advanced_main"

if ! command -v adb >/dev/null 2>&1; then
  echo "adb not found in PATH" >&2
  exit 1
fi
if [[ ! -f "${HANDLER_LOCAL}" ]]; then
  echo "Missing handler script: ${HANDLER_LOCAL}" >&2
  exit 1
fi
if [[ ! -f "${BINARY_LOCAL}" ]]; then
  echo "Missing litert_lm_advanced_main binary: ${BINARY_LOCAL}" >&2
  echo "Remediation: build //runtime/engine:litert_lm_advanced_main first." >&2
  exit 1
fi

adb wait-for-device >/dev/null
adb root >/dev/null
adb wait-for-device >/dev/null

adb shell "mkdir -p '${DEVICE_DIR}'"
adb push "${HANDLER_LOCAL}" "${DEVICE_DIR}/vlm_daemon_handler.sh" >/dev/null
adb push "${BINARY_LOCAL}" "${DEVICE_DIR}/litert_lm_advanced_main" >/dev/null
adb shell "chmod 0755 '${DEVICE_DIR}/vlm_daemon_handler.sh' '${DEVICE_DIR}/litert_lm_advanced_main'"

APP_BASE_APK_PATH="$(adb shell "pm path com.qidk.fastvlm 2>/dev/null | sed -n 's/^package://p' | head -n1" | tr -d '\r')"
APP_NATIVE_LIB_DIR=""
if [[ -n "${APP_BASE_APK_PATH}" ]]; then
  APP_NATIVE_LIB_DIR="${APP_BASE_APK_PATH%/base.apk}/lib/arm64"
  if ! adb shell "[ -d '${APP_NATIVE_LIB_DIR}' ]"; then
    APP_NATIVE_LIB_DIR=""
  fi
fi

if adb shell "toybox netstat -tlpn 2>/dev/null | grep -q ':21909 '"; then
  echo "Root daemon already listening on 127.0.0.1:21909"
  exit 0
fi

if [[ -n "${APP_NATIVE_LIB_DIR}" ]]; then
  echo "Using app native lib directory for daemon runtime: ${APP_NATIVE_LIB_DIR}"
  adb shell "nohup env VLM_NATIVE_LIB_DIR='${APP_NATIVE_LIB_DIR}' toybox nc -s 127.0.0.1 -p 21909 -L ${DEVICE_DIR}/vlm_daemon_handler.sh >${DEVICE_DIR}/vlm_daemon_stdout.log 2>${DEVICE_DIR}/vlm_daemon_stderr.log </dev/null &"
else
  echo "App native lib directory unavailable; daemon will use handler default VLM_NATIVE_LIB_DIR."
  adb shell "nohup toybox nc -s 127.0.0.1 -p 21909 -L ${DEVICE_DIR}/vlm_daemon_handler.sh >${DEVICE_DIR}/vlm_daemon_stdout.log 2>${DEVICE_DIR}/vlm_daemon_stderr.log </dev/null &"
fi

sleep 1
if adb shell "toybox netstat -tlpn | grep -q ':21909'"; then
  echo "Root daemon is listening on 127.0.0.1:21909"
  exit 0
fi

echo "Failed to start root daemon. Inspect logs:" >&2
echo "  adb shell tail -n 200 ${DEVICE_DIR}/vlm_daemon_stderr.log" >&2
echo "  adb shell tail -n 200 ${DEVICE_DIR}/vlm_daemon_stdout.log" >&2
exit 2
