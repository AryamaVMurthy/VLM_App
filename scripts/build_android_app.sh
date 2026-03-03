#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="${ROOT_DIR}/android-app"

cd "${APP_DIR}"
./gradlew --no-daemon clean :app:assembleDebug

echo "APK: ${APP_DIR}/app/build/outputs/apk/debug/app-debug.apk"
