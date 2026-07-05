#!/usr/bin/env bash
# Create a Python 3.11 venv and install titans-trainer (from our fork's fix
# branch) + a CUDA torch. titans-trainer wheels do not build on newer system
# interpreters, so 3.11 is required.
#
# Usage: bash install.sh [VENV_DIR]   (default: ./titan-venv)
set -euo pipefail

VENV="${1:-./titan-venv}"
PY="${TITAN_PY:-python3.11}"

command -v "$PY" >/dev/null || { echo "need $PY on PATH (or set TITAN_PY)"; exit 1; }

"$PY" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip setuptools wheel

# Fork branch fixes save_pretrained/from_pretrained full-config round-trip
# (upstream PR pafos-ai/titans-trainer#1). datasets+tokenizers for the data stage.
"$VENV/bin/pip" install \
  "git+https://github.com/beardfaceguy/titans-trainer.git@fix/save-pretrained-full-config" \
  datasets tokenizers

"$VENV/bin/python" - <<'PY'
import torch, titans_trainer as tt
print("titans_trainer", getattr(tt, "__version__", "?"))
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-only")
PY
echo "venv ready: $VENV"
