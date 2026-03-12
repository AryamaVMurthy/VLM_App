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

if adb shell "toybox netstat -tlpn 2>/dev/null | awk '\$4 == \"127.0.0.1:21909\" && \$6 == \"LISTEN\" { found=1 } END { exit found ? 0 : 1 }'"; then
  echo "Root daemon already listening on 127.0.0.1:21909"
  exit 0
fi

echo "Starting root daemon with device dispatch libs at ${DEVICE_DIR}/dispatch_libs"
adb shell "nohup toybox nc -s 127.0.0.1 -p 21909 -L ${DEVICE_DIR}/vlm_daemon_handler.sh >${DEVICE_DIR}/vlm_daemon_stdout.log 2>${DEVICE_DIR}/vlm_daemon_stderr.log </dev/null &"

sleep 1
if adb shell "toybox netstat -tlpn 2>/dev/null | awk '\$4 == \"127.0.0.1:21909\" && \$6 == \"LISTEN\" { found=1 } END { exit found ? 0 : 1 }'"; then
  echo "Root daemon is listening on 127.0.0.1:21909"
  exit 0
fi

echo "Failed to start root daemon. Inspect logs:" >&2
echo "  adb shell tail -n 200 ${DEVICE_DIR}/vlm_daemon_stderr.log" >&2
echo "  adb shell tail -n 200 ${DEVICE_DIR}/vlm_daemon_stdout.log" >&2
exit 2
