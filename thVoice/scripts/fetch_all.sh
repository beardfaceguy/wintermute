#!/usr/bin/env bash
# Fetch all thVoice binaries required by talkingHead (TTS + STT).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/fetch_voice.sh"
bash "${SCRIPT_DIR}/fetch_whisper.sh"
