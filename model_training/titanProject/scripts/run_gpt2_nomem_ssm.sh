#!/usr/bin/env bash
set -euo pipefail

LOG=/tmp/hf_gpt2_nomem.log
TRAIN_TXT=/mnt/data/hf_data/train.txt
VAL_TXT=/mnt/data/hf_data/val.txt
OUT=/mnt/data/hf_runs/gpt2_nomem
TMP_VENV=/tmp/hftrain_venv

echo "[start] $(date -Iseconds)" | tee "$LOG"
mkdir -p /mnt/data/hf_data
aws s3 cp s3://alix-ai-ml-staging-data/titan/data/tinystories_sampled/train_sample.txt "$TRAIN_TXT" | tee -a "$LOG"
aws s3 cp s3://alix-ai-ml-staging-data/titan/data/tinystories_sampled/val_sample.txt "$VAL_TXT" | tee -a "$LOG"

rm -rf "$TMP_VENV"
python3 -m venv "$TMP_VENV"
source "$TMP_VENV/bin/activate"
export PIP_NO_CACHE_DIR=1
pip install --upgrade pip | tee -a "$LOG"
# Install CPU-only torch and pin numpy<2 to avoid ABI issues
pip install "numpy<2" | tee -a "$LOG"
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.2.2 | tee -a "$LOG"
pip install "transformers<4.41" datasets accelerate sentencepiece | tee -a "$LOG"

aws s3 cp s3://alix-ai-ml-staging-data/titan/scripts/train_gpt2_nomem.py /tmp/train_gpt2_nomem.py | tee -a "$LOG"
TRAIN_TXT=$TRAIN_TXT VAL_TXT=$VAL_TXT OUT=$OUT python /tmp/train_gpt2_nomem.py \
  --train "$TRAIN_TXT" \
  --val "$VAL_TXT" \
  --out "$OUT" \
  --epochs 1 | tee -a "$LOG"

aws s3 sync "$OUT" s3://alix-ai-ml-staging-data/titan/hf_exports/gpt2_nomem/ --delete | tee -a "$LOG"
echo "[done] $(date -Iseconds)" | tee -a "$LOG"
tail -n 80 "$LOG"
