#!/usr/bin/env bash
# Dixie / Titan SFT runner — on-instance entry after `launch_dixie_sft_ssm.sh`.
#
# Canonical controller: model_training/titanProject/scripts/launch_dixie_sft_ssm.sh
# Cheap smoke entry:     scripts/run_dixie_sft_smoke_cheap_ssm.sh (gpt2, TORCH_NPROC=1).
#
# Required env: S3_CODE_URI, S3_TRAIN_URI, S3_VAL_URI. HF_TOKEN for gated models.
# Optional: RUN_ID, HF_MODEL_REPO, CONFIG_REL_PATH, TRAIN_STEPS, SEQ_LEN,
# SMOKE_MIN_KEEP, TORCH_NPROC, SKIP_MISTRAL_IMPORT, USE_CHAT_TEMPLATE,
# LOG_EVERY, EVAL_*, SAVE_EVERY.
# When using a public HF model without a token, set DIXIE_ALLOW_MISSING_HF_TOKEN=1.
#
# Flow: cleanup trap → code sync (skipped if launcher set DIXIE_CODE_PRE_SYNCED=1) →
# NVMe venv + deps → import smoke → data + HF weights → dataset smoke → torch.distributed.run.
#
# See launch_dixie_sft_ssm.sh header for monitoring, security (HF_TOKEN in SSM), and lifecycle.

set -euo pipefail

# ---------------------------------------------------------------------------
# Run identity.
# ---------------------------------------------------------------------------
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d%H%M%S)}"
S3_BUCKET="${S3_BUCKET:-alix-ai-ml-staging-data}"
S3_CKPT_PREFIX="s3://${S3_BUCKET}/titan/checkpoints"
RUN_ID="${RUN_ID:-dixie_mistral_full_${TIMESTAMP}}"
S3_CKPT_URI="${S3_CKPT_PREFIX}/${RUN_ID}/"
HF_MODEL_REPO="${HF_MODEL_REPO:-mistralai/Mistral-7B-Instruct-v0.3}"
CONFIG_REL_PATH="${CONFIG_REL_PATH:-configs/config_dixie_mistral_full.yaml}"
TRAIN_STEPS="${TRAIN_STEPS:-3000}"
SEQ_LEN="${SEQ_LEN:-2048}"
SMOKE_MIN_KEEP="${SMOKE_MIN_KEEP:-0.90}"
TORCH_NPROC="${TORCH_NPROC:-8}"
SKIP_MISTRAL_IMPORT="${SKIP_MISTRAL_IMPORT:-0}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-1}"
LOG_EVERY="${LOG_EVERY:-20}"
EVAL_EVERY="${EVAL_EVERY:-250}"
EVAL_BATCHES="${EVAL_BATCHES:-20}"
SAVE_EVERY="${SAVE_EVERY:-500}"

for var in S3_CODE_URI S3_TRAIN_URI S3_VAL_URI; do
  if [[ -z "${!var:-}" ]]; then
    echo "[runner] FATAL: ${var} is required" >&2
    exit 64
  fi
done

# ---------------------------------------------------------------------------
# Local layout — DLAMI exposes ephemeral NVMe at /opt/dlami/nvme.
# ---------------------------------------------------------------------------
WORKDIR=/opt/dlami/nvme/dixie
CODEDIR="${WORKDIR}/code/titanProject"
DATADIR="${WORKDIR}/data"
MODELDIR="${WORKDIR}/models/$(basename "${HF_MODEL_REPO}")"
CKPTDIR="${WORKDIR}/ckpts"

mkdir -p "${WORKDIR}/logs" "${DATADIR}" "${CKPTDIR}"
# Pip wheel unpack + HTTP cache default to /tmp on root EBS; g5/smaller DLAMI
# SKUs can hit ENOSPC before training starts (dixie_gpt2_smoke 2026-05-16).
RUNNER_TMP="${WORKDIR}/.pip_tmp"
RUNNER_PIP_CACHE="${WORKDIR}/.pip_cache"
mkdir -p "${RUNNER_TMP}" "${RUNNER_PIP_CACHE}"
export TMPDIR="${RUNNER_TMP}"
export PIP_CACHE_DIR="${RUNNER_PIP_CACHE}"
GLOBAL_LOG="${WORKDIR}/logs/runner.log"
TRAIN_LOG="${WORKDIR}/logs/train.log"
exec > >(tee -a "${GLOBAL_LOG}") 2>&1

echo "[runner] starting at $(date -Iseconds)"
echo "[runner] RUN_ID=${RUN_ID}"
echo "[runner] code from ${S3_CODE_URI}"
echo "[runner] data train=${S3_TRAIN_URI}"
echo "[runner] data val  =${S3_VAL_URI}"
echo "[runner] hf model  =${HF_MODEL_REPO}"
echo "[runner] torch_nproc=${TORCH_NPROC} USE_CHAT_TEMPLATE=${USE_CHAT_TEMPLATE} SKIP_MISTRAL_IMPORT=${SKIP_MISTRAL_IMPORT}"
echo "[runner] checkpoints -> ${S3_CKPT_URI}"
echo "[runner] pip scratch TMPDIR=${TMPDIR} PIP_CACHE_DIR=${PIP_CACHE_DIR}"
df -h / /tmp "${WORKDIR}" 2>/dev/null || df -h || true

# ---------------------------------------------------------------------------
# Self-terminate trap (installed FIRST so any setup failure still tears the
# instance down — keeps p4d.24xlarge costs bounded if anything goes wrong).
# The 2026-05-15 disaster cost ~$525 because the hand-rolled launch had no
# such trap; not repeating that mistake.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=model_training/titanProject/scripts/lib/aws_lifecycle.sh
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/aws_lifecycle.sh"

_upload_runner_log() {
  aws s3 cp "${GLOBAL_LOG}" "${S3_CKPT_URI}runner.log" --quiet \
    || { echo "[runner] WARN: failed to upload runner.log" >&2; return 1; }
}
_upload_train_log() {
  [[ -f "${TRAIN_LOG}" ]] || return 0
  aws s3 cp "${TRAIN_LOG}" "${S3_CKPT_URI}train.log" --quiet \
    || { echo "[runner] WARN: failed to upload train.log" >&2; return 1; }
}
CLEANUP_PRE_TERMINATE_HOOKS+=(_upload_train_log _upload_runner_log)

resolve_instance_metadata
install_cleanup_trap

# ---------------------------------------------------------------------------
# Pull code from S3 — use sync so we get the WHOLE titanProject directory,
# not a hand-picked tarball that might be missing data.py (see 2026-05-15).
# Skipped when launch_dixie_sft_ssm.sh already synced to this CODEDIR (single download).
# ---------------------------------------------------------------------------
mkdir -p "${CODEDIR}"
if [[ "${DIXIE_CODE_PRE_SYNCED:-}" == "1" ]]; then
  echo "[runner] skipping S3 code sync (DIXIE_CODE_PRE_SYNCED=1, launcher populated ${CODEDIR})"
else
  aws s3 sync "${S3_CODE_URI}/" "${CODEDIR}/" --no-progress
fi

# ---------------------------------------------------------------------------
# Extra deps: use an NVMe-backed venv.  DLAMI CUDA torch lives in /opt/pytorch
# on the (small) root EBS; g5/smaller SKUs often have root ~97% full before we
# start.  Pip installing titans+transformers into /opt/pytorch then hits ENOSPC
# even when TMPDIR is on NVMe (dixie_gpt2_smoke 2026-05-17).
# ---------------------------------------------------------------------------
PYTORCH_BASE=/opt/pytorch
BASE_PY="${PYTORCH_BASE}/bin/python"
if [[ ! -x "${BASE_PY}" ]]; then
  echo "[runner] FATAL: /opt/pytorch venv not found on this AMI" >&2
  exit 64
fi
VENV="${WORKDIR}/.venv"
if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "[runner] creating NVMe venv (system-site-packages -> DLAMI torch)"
  "${BASE_PY}" -m venv --system-site-packages "${VENV}"
fi
PY="${VENV}/bin/python"
PIP="${VENV}/bin/pip"
echo "[runner] training python: ${PY}"
"${PIP}" install --no-input --quiet --no-cache-dir \
  "titans-pytorch==0.5.3" \
  "transformers>=4.45,<5" \
  "accelerate>=0.34" \
  "peft>=0.13" \
  "bitsandbytes>=0.44" \
  "huggingface_hub>=0.25" \
  "sentencepiece" "boto3" "pyyaml" "tqdm" "numpy<2"

# ---------------------------------------------------------------------------
# Torchvision must match the DLAMI torch wheel (same +cu* channel).
# `transformers` ≥4.45 imports `torchvision` while resolving Mistral/Llama
# modules (image_utils → torchvision.transforms). A mismatched torchvision
# wheel produces:
#   RuntimeError: operator torchvision::nms does not exist
# and surfaces as a bogus MistralForCausalLM import error (2026-05-16).
# Install/upgrade torchvision *after* titans-pytorch so deps cannot pin an
# incompatible stub.
# ---------------------------------------------------------------------------
TORCH_CUDA_TAG="$("${PY}" -c "import torch
v = torch.__version__
if '+' in v:
    print(v.split('+', 1)[1])
elif torch.version.cuda:
    p = torch.version.cuda.split('.')
    print('cu' + p[0] + p[1])
else:
    print('cpu')")"
if [[ "${TORCH_CUDA_TAG}" == "cpu" ]]; then
  echo "[runner] FATAL: torch has no CUDA build; Dixie SFT requires CUDA" >&2
  exit 64
fi
PYTORCH_WHL_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"
echo "[runner] torchvision: upgrading to match torch (${TORCH_CUDA_TAG}) via ${PYTORCH_WHL_INDEX}"
"${PIP}" install --no-input --quiet --no-cache-dir --upgrade \
  torchvision --extra-index-url "${PYTORCH_WHL_INDEX}"

# ---------------------------------------------------------------------------
# Import smoke — every dep we use must import cleanly before we proceed.
# This catches "missing titans-pytorch in pip line" type bugs in <2 seconds.
# ---------------------------------------------------------------------------
echo "[runner] import smoke:"
"${PY}" -c "import torch; print('  torch=', torch.__version__, 'cuda=', torch.cuda.is_available(), 'gpus=', torch.cuda.device_count())"
"${PY}" -c "import torchvision; print('  torchvision=', torchvision.__version__)"
"${PY}" -c "import titans_pytorch; print('  titans_pytorch=ok')"
"${PY}" -c "import transformers; print('  transformers=', transformers.__version__)"
if [[ "${SKIP_MISTRAL_IMPORT}" != "1" ]]; then
  "${PY}" -c "from transformers.models.mistral.modeling_mistral import MistralForCausalLM; print('  MistralForCausalLM=ok')"
else
  echo "  MistralForCausalLM=skipped (SKIP_MISTRAL_IMPORT=1)"
fi
# finetune_sft.py imports model.py imports titans_pytorch, plus train_utils
# imports data — both of which were missing in the failed run.
( cd "${CODEDIR}" && "${PY}" -c "import finetune_sft; print('  finetune_sft=ok')" )

# ---------------------------------------------------------------------------
# Download training data.
# ---------------------------------------------------------------------------
echo "[runner] downloading training data..."
aws s3 cp "${S3_TRAIN_URI}" "${DATADIR}/train.jsonl" --no-progress
aws s3 cp "${S3_VAL_URI}"   "${DATADIR}/val.jsonl"   --no-progress

# ---------------------------------------------------------------------------
# HF token cascade. Gated repos (default Mistral) need auth for snapshot_download.
# Token is normally set by the SSM env from launch_dixie_sft_ssm.sh.
# The cascade mirrors hf_sft_cloudwatch.sh so any of the three common var
# names work.
# ---------------------------------------------------------------------------
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}}"
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}}"
if [[ -z "${HF_TOKEN}" ]]; then
  echo "[runner] WARN: no HF_TOKEN / HUGGINGFACE_TOKEN / HUGGINGFACE_HUB_TOKEN in env." >&2
  if [[ "${DIXIE_ALLOW_MISSING_HF_TOKEN:-}" == "1" ]]; then
    echo "[runner] WARN: continuing (DIXIE_ALLOW_MISSING_HF_TOKEN=1)." >&2
  elif [[ "${SKIP_MISTRAL_IMPORT:-0}" == "1" ]]; then
    echo "[runner] WARN: continuing (SKIP_MISTRAL_IMPORT=1; public or local model path)." >&2
  else
    echo "[runner] FATAL: gated HF repos need a token. Set HF_TOKEN (SSM env), use a cheap smoke path, or set DIXIE_ALLOW_MISSING_HF_TOKEN=1 for known-public models." >&2
    exit 64
  fi
else
  echo "[runner] HF auth: token present"
fi

# HF hub/transformers default cache is ~/.cache on root EBS — keep on NVMe.
export HF_HOME="${WORKDIR}/.hf_hub"
mkdir -p "${HF_HOME}"
echo "[runner] HF_HOME=${HF_HOME}"

# ---------------------------------------------------------------------------
# Download HF model weights to NVMe so we don't compete with HF cache eviction.
# huggingface_hub picks up HF_TOKEN / HUGGINGFACE_HUB_TOKEN automatically.
# ---------------------------------------------------------------------------
echo "[runner] downloading HF model weights: ${HF_MODEL_REPO}"
mkdir -p "${MODELDIR}"
"${PY}" -c "
from huggingface_hub import snapshot_download
snapshot_download('${HF_MODEL_REPO}', local_dir='${MODELDIR}')
print('  model staged at ${MODELDIR}')
"

# ---------------------------------------------------------------------------
# Dataset smoke test — the *new* line of defense. Build MaskedSFTDataset on a
# 5k-line head sample of train.jsonl using the same Mistral tokenizer the real
# run will use. If keep-rate falls below SMOKE_MIN_KEEP, abort BEFORE
# allocating 8 GPUs for ~3h. The 2026-05-15 run discovered an empty dataset
# AFTER 10 min of model load + ~$5 of wasted GPU time per attempt; this catches
# the same class of bug in <30s.
# ---------------------------------------------------------------------------
echo "[runner] dataset smoke test (this is the line of defense added after the 2026-05-15 disaster)..."
cd "${CODEDIR}"
SMOKE_ARGS=(
  --data "${DATADIR}/train.jsonl"
  --hf-model "${MODELDIR}"
  --seq-len "${SEQ_LEN}"
  --max-lines 5000
  --min-keep-rate "${SMOKE_MIN_KEEP}"
)
if [[ "${USE_CHAT_TEMPLATE}" != "1" ]]; then
  SMOKE_ARGS+=(--no-chat-template)
fi
"${PY}" scripts/smoke_sft_data.py "${SMOKE_ARGS[@]}"

# ---------------------------------------------------------------------------
# Launch training. -u so log tail-following sees lines live.
# Final command mirrors the one documented at the top of
# configs/config_dixie_mistral_full.yaml (multi-GPU); cheap smoke uses TORCH_NPROC=1.
# ---------------------------------------------------------------------------
FINETUNE_ARGS=(
  --config "${CONFIG_REL_PATH}"
  --hf-model "${MODELDIR}"
  --no-lora
  --device cuda
  --steps "${TRAIN_STEPS}"
  --log-every "${LOG_EVERY}"
  --eval-every "${EVAL_EVERY}"
  --eval-batches "${EVAL_BATCHES}"
  --save-every "${SAVE_EVERY}"
  --checkpoint-dir "${CKPTDIR}"
  --s3-checkpoint-uri "${S3_CKPT_URI}"
  --min-free-gb 5
  --aws-bin aws
)
if [[ "${USE_CHAT_TEMPLATE}" == "1" ]]; then
  FINETUNE_ARGS+=(--chat-template)
fi

echo "[runner] launching training: ${RUN_ID}"
PYTHONUNBUFFERED=1 "${PY}" -m torch.distributed.run \
  --nproc_per_node="${TORCH_NPROC}" finetune_sft.py \
  "${FINETUNE_ARGS[@]}" \
  2>&1 | tee -a "${TRAIN_LOG}"

# torchrun's exit code is the rightmost (tee=0); pipefail surfaces non-zero.
TORCHRUN_RC="${PIPESTATUS[0]}"
echo "[runner] torchrun rc=${TORCHRUN_RC}"

aws s3 cp "${TRAIN_LOG}" "${S3_CKPT_URI}train.log" --no-progress || true

echo "[runner] complete at $(date -Iseconds)"
echo "[runner] artifacts at ${S3_CKPT_URI}"
exit "${TORCHRUN_RC}"
