#!/usr/bin/env bash
set -euo pipefail

if ! command -v adb >/dev/null 2>&1; then
  echo "adb not found in PATH" >&2
  exit 1
fi

adb wait-for-device
SOC="$(adb shell getprop ro.soc.model | tr -d '\r')"
SDK="$(adb shell getprop ro.build.version.sdk | tr -d '\r')"
ABI="$(adb shell getprop ro.product.cpu.abi | tr -d '\r')"

if [[ "${SOC}" != "SM8750P" ]]; then
  echo "FAIL: unsupported SoC '${SOC}', expected 'SM8750P'" >&2
  exit 2
fi
if [[ "${SDK}" != "35" ]]; then
  echo "FAIL: unsupported SDK '${SDK}', expected '35'" >&2
  exit 3
fi
if [[ "${ABI}" != "arm64-v8a" ]]; then
  echo "FAIL: unsupported ABI '${ABI}', expected 'arm64-v8a'" >&2
  exit 4
fi

echo "PASS: device is compatible (SoC=${SOC}, SDK=${SDK}, ABI=${ABI})"
