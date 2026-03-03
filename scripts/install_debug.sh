#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="${ROOT_DIR}/android-app"
APK_PATH="${APP_DIR}/app/build/outputs/apk/debug/app-debug.apk"

"${ROOT_DIR}/scripts/build_android_app.sh"

if ! command -v adb >/dev/null 2>&1; then
  echo "adb not found in PATH" >&2
  exit 1
fi

adb wait-for-device
adb install -r "${APK_PATH}"

echo "Installed ${APK_PATH}"
