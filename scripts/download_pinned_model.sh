#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/artifacts/models"
MODEL_NAME="FastVLM-0.5B.litertlm"
REV="74e5aa3fb2adc196ca24bc7fb561ec92fc165e81"
SHA256_EXPECTED="87114d1331e0543455d57d81ceacf28a0d80d49d98691e1b24792e51903cf8c4"
URL="https://huggingface.co/litert-community/FastVLM-0.5B/resolve/${REV}/${MODEL_NAME}?download=true"

mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

curl -fL --continue-at - -o "${MODEL_NAME}" "${URL}"
SHA256_ACTUAL="$(sha256sum "${MODEL_NAME}" | awk '{print $1}')"

if [[ "${SHA256_ACTUAL}" != "${SHA256_EXPECTED}" ]]; then
  echo "Checksum mismatch: expected=${SHA256_EXPECTED} actual=${SHA256_ACTUAL}" >&2
  exit 1
fi

echo "Downloaded and verified: ${OUT_DIR}/${MODEL_NAME}"
