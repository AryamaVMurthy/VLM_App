#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE_DIR="/data/local/tmp/graphpilot_edge"
HANDLER_LOCAL="${ROOT_DIR}/scripts/graphpilot_retrieval_daemon_handler.sh"
HANDLER_REMOTE="${DEVICE_DIR}/graphpilot_retrieval_daemon_handler.sh"
PORT="21910"

adb wait-for-device >/dev/null
adb shell "mkdir -p '${DEVICE_DIR}'" >/dev/null
adb push "${HANDLER_LOCAL}" "${HANDLER_REMOTE}" >/dev/null
adb shell "chmod 0755 '${HANDLER_REMOTE}' '${DEVICE_DIR}/graphpilot_retrieval_main'" >/dev/null

if adb shell "toybox nc -z 127.0.0.1 ${PORT}" >/dev/null 2>&1; then
  echo "GraphPilot retrieval GPU daemon already listening on 127.0.0.1:${PORT}"
  exit 0
fi

adb shell "nohup toybox nc -s 127.0.0.1 -p ${PORT} -L ${HANDLER_REMOTE} >${DEVICE_DIR}/graphpilot_retrieval_daemon_stdout.log 2>${DEVICE_DIR}/graphpilot_retrieval_daemon_stderr.log </dev/null &" >/dev/null

for _ in $(seq 1 20); do
  if adb shell "toybox nc -z 127.0.0.1 ${PORT}" >/dev/null 2>&1; then
    echo "GraphPilot retrieval GPU daemon is listening on 127.0.0.1:${PORT}"
    exit 0
  fi
  sleep 1
done

echo "Failed to start GraphPilot retrieval GPU daemon." >&2
echo "Inspect logs:" >&2
echo "  adb shell tail -n 200 ${DEVICE_DIR}/graphpilot_retrieval_daemon_stderr.log" >&2
echo "  adb shell tail -n 200 ${DEVICE_DIR}/graphpilot_retrieval_daemon_stdout.log" >&2
exit 1
