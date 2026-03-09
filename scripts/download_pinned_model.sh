#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/artifacts/models"
MODEL_NAME="FastVLM-0.5B.litertlm"
REV="74e5aa3fb2adc196ca24bc7fb561ec92fc165e81"
SHA256_EXPECTED=""

usage() {
  cat <<'EOF'
Usage:
  download_pinned_model.sh [options]

Options:
  --model-name NAME     Hugging Face filename to download
  --revision REV        Revision / commit to pin (default: repo's pinned commit)
  --sha256 HASH         Override expected sha256 for verification
  -h, --help            Show help

Supported model names:
  - FastVLM-0.5B.litertlm
  - FastVLM-0.5B.qualcomm.sm8750.litertlm
  - FastVLM-0.5B.qualcomm.sm8850.litertlm
EOF
}

resolve_sha256() {
  case "${MODEL_NAME}" in
    FastVLM-0.5B.litertlm)
      printf '%s\n' '87114d1331e0543455d57d81ceacf28a0d80d49d98691e1b24792e51903cf8c4'
      ;;
    FastVLM-0.5B.qualcomm.sm8750.litertlm)
      printf '%s\n' '55d755e9058713581ce554591ea2bff4d69c600fe4c5384aa3e95380cc759634'
      ;;
    FastVLM-0.5B.qualcomm.sm8850.litertlm)
      printf '%s\n' 'c28a78dd0e991d2f9bcf0792213ca8c74d309b7c726ce4fa2da1be28bb8ad545'
      ;;
    *)
      echo "Unsupported --model-name: ${MODEL_NAME}" >&2
      echo "Remediation: choose one of the supported FastVLM files from litert-community/FastVLM-0.5B." >&2
      exit 2
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-name)
      MODEL_NAME="${2:-}"
      shift 2
      ;;
    --revision)
      REV="${2:-}"
      shift 2
      ;;
    --sha256)
      SHA256_EXPECTED="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${SHA256_EXPECTED}" ]]; then
  SHA256_EXPECTED="$(resolve_sha256)"
fi

URL="https://huggingface.co/litert-community/FastVLM-0.5B/resolve/${REV}/${MODEL_NAME}?download=true"
TMP_NAME="${MODEL_NAME}.download.$$"

cleanup() {
  rm -f "${TMP_NAME}"
}

mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

trap cleanup EXIT

curl -fL -o "${TMP_NAME}" "${URL}"
SHA256_ACTUAL="$(sha256sum "${TMP_NAME}" | awk '{print $1}')"

if [[ "${SHA256_ACTUAL}" != "${SHA256_EXPECTED}" ]]; then
  echo "Checksum mismatch: expected=${SHA256_EXPECTED} actual=${SHA256_ACTUAL}" >&2
  exit 1
fi

mv -f "${TMP_NAME}" "${MODEL_NAME}"
trap - EXIT

echo "Downloaded and verified: ${OUT_DIR}/${MODEL_NAME}"
