#!/usr/bin/env bash
# Fetch a Piper TTS voice model (.onnx + .onnx.json) from Hugging Face.
#
# The .onnx weights file is gitignored (~63MB binary); the matching .onnx.json
# is committed so we know exactly which voice the codebase targets.
#
# Usage:
#   thVoice/scripts/fetch_voice.sh                       # default voice
#   thVoice/scripts/fetch_voice.sh en_US-lessac-medium   # any rhasspy voice
#
# Voice catalogue: https://huggingface.co/rhasspy/piper-voices
set -euo pipefail

VOICE="${1:-en_GB-cori-medium}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/piper/models"

mkdir -p "${MODEL_DIR}"

IFS='-' read -r LANG_CODE NAME QUALITY <<< "${VOICE}"
LANG_PREFIX="${LANG_CODE%%_*}"
BASE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/${LANG_PREFIX}/${LANG_CODE}/${NAME}/${QUALITY}"

ONNX_PATH="${MODEL_DIR}/${VOICE}.onnx"
JSON_PATH="${MODEL_DIR}/${VOICE}.onnx.json"

echo "Fetching Piper voice: ${VOICE}"
echo "  -> ${ONNX_PATH}"

if [[ -f "${ONNX_PATH}" ]]; then
  echo "  weights already present, skipping download"
else
  curl -L --fail --progress-bar \
    -o "${ONNX_PATH}" \
    "${BASE_URL}/${VOICE}.onnx"
fi

if [[ -f "${JSON_PATH}" ]]; then
  echo "  config already present, skipping download"
else
  curl -L --fail --progress-bar \
    -o "${JSON_PATH}" \
    "${BASE_URL}/${VOICE}.onnx.json"
fi

echo "Done. Set PIPER_VOICE_PATH=${ONNX_PATH} (or leave default if it matches)."
