#!/usr/bin/env bash
# Cheap on-instance bootstrap for Dixie SFT path validation (single GPU).
#
# Use on g5.xlarge, g6.xlarge, or similar (~$1–2/hr) instead of p4d.24xlarge.
# Pulls the same pentest JSONL from S3 as the full Mistral run, but trains
# gpt2 for a handful of steps to prove torchrun + finetune_sft + AMP/GC work.
#
# Required env (set by SSM / parent): S3_CODE_URI, S3_TRAIN_URI, S3_VAL_URI
# Optional: TRAIN_STEPS, RUN_ID, TIMESTAMP, AWS_LIFECYCLE_MODE, etc.
#
# Controller: sync titanProject to NVMe runner_stage; entry (see launch script).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d%H%M%S)}"

export HF_MODEL_REPO="${HF_MODEL_REPO:-gpt2}"
export CONFIG_REL_PATH="${CONFIG_REL_PATH:-configs/config_dixie_gpt2_smoke_aws.yaml}"
export TORCH_NPROC="${TORCH_NPROC:-1}"
export TRAIN_STEPS="${TRAIN_STEPS:-10}"
export SEQ_LEN="${SEQ_LEN:-256}"
export SKIP_MISTRAL_IMPORT="${SKIP_MISTRAL_IMPORT:-1}"
export USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-0}"
export LOG_EVERY="${LOG_EVERY:-1}"
export EVAL_EVERY="${EVAL_EVERY:-5}"
export EVAL_BATCHES="${EVAL_BATCHES:-5}"
export SAVE_EVERY="${SAVE_EVERY:-10}"
# Shorter seq + smaller model → more truncation; pentest JSONL often lands ~0.86–0.89.
export SMOKE_MIN_KEEP="${SMOKE_MIN_KEEP:-0.82}"
export RUN_ID="${RUN_ID:-dixie_gpt2_smoke_${TIMESTAMP}}"

exec bash "${SCRIPT_DIR}/run_dixie_mistral_sft_ssm.sh"
