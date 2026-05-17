#!/usr/bin/env bash
# Download Whisper ggml weights for pywhispercpp (talkingHead STT).
#
# The .bin is large (~142MB for base.en); not committed. talkingHead expects:
#   thVoice/models/ggml-base.en.bin
#
# Usage:
#   thVoice/scripts/fetch_whisper.sh
#   thVoice/scripts/fetch_whisper.sh ggml-small.en.bin   # optional model name
set -euo pipefail

MODEL="${1:-ggml-base.en.bin}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/models"
DEST="${DEST_DIR}/${MODEL}"

mkdir -p "${DEST_DIR}"

# Same canonical tree whisper.cpp / pywhispercpp docs use.
URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/${MODEL}"

echo "Fetching Whisper model: ${MODEL}"
echo "  -> ${DEST}"

if [[ -f "${DEST}" ]] && [[ "$(stat -c%s "${DEST}" 2>/dev/null || stat -f%z "${DEST}" 2>/dev/null)" -gt 1000000 ]]; then
  echo "  model already present (>1MB), skipping download"
  exit 0
fi

# Remove stale Git LFS pointer or partial file
rm -f "${DEST}"

curl -L --fail --progress-bar -o "${DEST}" "${URL}"

echo "Done."
wc -c "${DEST}"
