#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
mkdir -p "${THIRD_PARTY_DIR}"

clone_or_reset() {
  local repo_url="$1"
  local target_dir="$2"
  local commit_sha="$3"

  if [[ -d "${target_dir}/.git" ]]; then
    echo "Updating ${target_dir}"
    git -C "${target_dir}" fetch --all --tags --prune
  else
    echo "Cloning ${repo_url} -> ${target_dir}"
    git clone "${repo_url}" "${target_dir}"
  fi

  git -C "${target_dir}" checkout "${commit_sha}"
}

clone_or_reset \
  "https://github.com/google-ai-edge/LiteRT" \
  "${THIRD_PARTY_DIR}/litert" \
  "04cc55e848fe21f483d00be761a27a88aaa74cd0"

clone_or_reset \
  "https://github.com/google-ai-edge/LiteRT-LM" \
  "${THIRD_PARTY_DIR}/litert-lm" \
  "00e1342dae7223006e06917bbd76928385e85dea"

echo "Third-party dependencies pinned successfully."
