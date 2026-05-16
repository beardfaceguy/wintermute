#!/usr/bin/env bash
# Dixie Flatline SFT runner (Mistral 7B Instruct v0.3, full fine-tune).
#
# Designed to run unattended on a p4d.24xlarge via SSM RunCommand
# (systemd-run --collect for clean detachment from SSM's process group).
#
# Successor to the hand-rolled SSM blob that powered the failed
# dixie_mistral_full_20260515035945 run. The 2026-05-15 failure cost ~$525 of
# GPU-hours to discover three packaging bugs (missing titans-pytorch, missing
# data.py in the bundle, stale finetune_sft.py). All three are guarded by the
# pre-flight checks below — the box never gets within 10 minutes of torchrun
# unless the env is sane.
#
# Sequence:
#   1. Install cleanup trap (uploads logs to S3, terminates instance).
#   2. aws s3 sync the entire titanProject directory to NVMe.
#   3. pip install pinned deps INCLUDING titans-pytorch into /opt/pytorch.
#   4. Import smoke: torch/cuda, transformers, titans_pytorch.
#   5. Download training data from S3 to NVMe.
#   6. Download HF model weights (or rely on huggingface_hub cache).
#   7. Dataset smoke test: cap-5k sample build of MaskedSFTDataset on
#      train.jsonl. Aborts the run if keep-rate < threshold.
#   8. torchrun --nproc_per_node=8 finetune_sft.py with live S3 checkpoint sync.
#   9. Self-terminate.
#
# Required env (on the instance, set by the SSM RunCommand env block):
#   TIMESTAMP        Run identity, e.g. 20260515140000. (Default: now, UTC.)
#   S3_CODE_URI      Where titanProject/ was synced before launch.
#                    Example: s3://alix-ai-ml-staging-data/titan/code/dixie_20260515140000
#   S3_TRAIN_URI     Where train.jsonl lives.
#                    Example: s3://alix-ai-ml-staging-data/titan/data/dixie_pentest/train.jsonl
#   S3_VAL_URI       Where val.jsonl lives.
#   HF_TOKEN         HuggingFace API token for gated repos. REQUIRED for the
#                    default HF_MODEL_REPO (mistralai/Mistral-7B-Instruct-v0.3
#                    is gated). Source on the controller:
#                      grep ^accessToken= ~/work/wintermute/model_training/hf.env \
#                        | cut -d= -f2-
#                    The cascade below also accepts HUGGINGFACE_TOKEN or
#                    HUGGINGFACE_HUB_TOKEN if one of those is already set.
#
# Optional env:
#   RUN_ID           Defaults to dixie_mistral_full_${TIMESTAMP}.
#   S3_BUCKET        Defaults to alix-ai-ml-staging-data.
#   HF_MODEL_REPO    Defaults to mistralai/Mistral-7B-Instruct-v0.3.
#   CONFIG_REL_PATH  Config path relative to titanProject/. Default
#                    configs/config_dixie_mistral_full.yaml.
#   TRAIN_STEPS      Default 3000. Overrides config max_steps for the CLI.
#   SEQ_LEN          Default 2048. Must match config.train.seq_len for the
#                    smoke check to be representative.
#   SMOKE_MIN_KEEP   Default 0.90. Smoke-test failure threshold.
#
# Controller-side kickoff (run from ~/work/wintermute on the dev box):
#
#   export AWS_PROFILE=experimental-admin
#   export AWS_DEFAULT_REGION=us-east-1
#   TIMESTAMP="$(date -u +%Y%m%d%H%M%S)"
#   RUN_ID="dixie_mistral_full_${TIMESTAMP}"
#   S3_CODE_URI="s3://alix-ai-ml-staging-data/titan/code/dixie_${TIMESTAMP}"
#   S3_TRAIN_URI="s3://alix-ai-ml-staging-data/titan/data/dixie_pentest/train.jsonl"
#   S3_VAL_URI="s3://alix-ai-ml-staging-data/titan/data/dixie_pentest/val.jsonl"
#   HF_TOKEN="$(grep ^accessToken= model_training/hf.env | cut -d= -f2-)"
#   INSTANCE_ID="i-XXXXXXXX"  # already-running p4d.24xlarge with SSM agent
#
#   # 1) Pre-flight smoke on the controller (cheap; avoids burning GPU time).
#   python model_training/titanProject/scripts/smoke_sft_data.py \
#       --data "${S3_TRAIN_URI}" \
#       --hf-model mistralai/Mistral-7B-Instruct-v0.3 \
#       --seq-len 2048 --max-lines 5000 --min-keep-rate 0.90
#
#   # 2) Sync code (whole dir, not a tarball — that's how 2026-05-15 broke).
#   aws s3 sync model_training/titanProject/ "${S3_CODE_URI}/" \
#       --delete --exclude '__pycache__/*' --exclude 'results/*' \
#       --exclude 'logs/*' --exclude 'saved_models/*'
#
#   # 3) Submit SSM RunCommand. HF_TOKEN goes in via Parameters (NOT echoed
#   #    to the runner log — SSM redacts the inline env block in CloudTrail).
#   aws ssm send-command --instance-ids "${INSTANCE_ID}" \
#       --document-name AWS-RunShellScript \
#       --timeout-seconds 43200 \
#       --parameters "executionTimeout=43200,commands=[
#         \"export TIMESTAMP=${TIMESTAMP}\",
#         \"export S3_CODE_URI=${S3_CODE_URI}\",
#         \"export S3_TRAIN_URI=${S3_TRAIN_URI}\",
#         \"export S3_VAL_URI=${S3_VAL_URI}\",
#         \"export HF_TOKEN=${HF_TOKEN}\",
#         \"aws s3 sync ${S3_CODE_URI}/ /tmp/dixie_runner/ --no-progress\",
#         \"bash /tmp/dixie_runner/scripts/run_dixie_mistral_sft_ssm.sh\"
#       ]"
#
#   # 4) Arm the watcher (detached) so we get a ntfy push when the box stops.
#   #    Cheap insurance in case the cleanup trap fails silently — p4d.24xlarge
#   #    is $32/hr, the watcher costs nothing.
#   nohup ~/.local/bin/aws-instance-watcher \
#       "${INSTANCE_ID}" "dixie ${RUN_ID}" wintermute \
#       > /tmp/watcher-${RUN_ID}.log 2>&1 & disown
#
#   # 5) Tail progress (from S3 — train.log syncs every save_every steps).
#   aws s3 cp s3://alix-ai-ml-staging-data/titan/checkpoints/${RUN_ID}/train.log -
#
#   # 6) Optional: SSM probe of on-instance logs (same paths the runner uses).
#   From repo root: REMOTE_LAYOUT=dixie_sft RUN_ID="${RUN_ID}" INSTANCE_ID="${INSTANCE_ID}" \\
#     bash scripts/aws_commands/check_detached_training_status.sh
#
# All paths are deliberate; this script is the source of truth for the run.

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
GLOBAL_LOG="${WORKDIR}/logs/runner.log"
TRAIN_LOG="${WORKDIR}/logs/train.log"
exec > >(tee -a "${GLOBAL_LOG}") 2>&1

echo "[runner] starting at $(date -Iseconds)"
echo "[runner] RUN_ID=${RUN_ID}"
echo "[runner] code from ${S3_CODE_URI}"
echo "[runner] data train=${S3_TRAIN_URI}"
echo "[runner] data val  =${S3_VAL_URI}"
echo "[runner] hf model  =${HF_MODEL_REPO}"
echo "[runner] checkpoints -> ${S3_CKPT_URI}"

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
# ---------------------------------------------------------------------------
mkdir -p "${CODEDIR}"
aws s3 sync "${S3_CODE_URI}/" "${CODEDIR}/" --no-progress

# ---------------------------------------------------------------------------
# Install deps into the DLAMI's pre-built /opt/pytorch venv. Pinned versions
# only; titans-pytorch is REQUIRED (model.py imports it at module load).
# ---------------------------------------------------------------------------
PYTORCH_BASE=/opt/pytorch
PY="${PYTORCH_BASE}/bin/python"
PIP="${PYTORCH_BASE}/bin/pip"
if [[ ! -x "${PY}" ]]; then
  echo "[runner] FATAL: /opt/pytorch venv not found on this AMI" >&2
  exit 64
fi
"${PIP}" install --no-input --quiet \
  "titans-pytorch==0.5.3" \
  "transformers>=4.45,<5" \
  "accelerate>=0.34" \
  "peft>=0.13" \
  "bitsandbytes>=0.44" \
  "huggingface_hub>=0.25" \
  "sentencepiece" "boto3" "pyyaml" "tqdm" "numpy<2"

# ---------------------------------------------------------------------------
# Import smoke — every dep we use must import cleanly before we proceed.
# This catches "missing titans-pytorch in pip line" type bugs in <2 seconds.
# ---------------------------------------------------------------------------
echo "[runner] import smoke:"
"${PY}" -c "import torch; print('  torch=', torch.__version__, 'cuda=', torch.cuda.is_available(), 'gpus=', torch.cuda.device_count())"
"${PY}" -c "import titans_pytorch; print('  titans_pytorch=ok')"
"${PY}" -c "import transformers; print('  transformers=', transformers.__version__)"
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
# HF token cascade. mistralai/Mistral-7B-Instruct-v0.3 (default HF_MODEL_REPO)
# is gated — snapshot_download will 401 without auth. Token must be supplied
# by the SSM submit env (see the kickoff snippet in this script's header).
# The cascade mirrors hf_sft_cloudwatch.sh so any of the three common var
# names work.
# ---------------------------------------------------------------------------
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}}"
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}}"
if [[ -z "${HF_TOKEN}" ]]; then
  echo "[runner] WARN: no HF_TOKEN / HUGGINGFACE_TOKEN / HUGGINGFACE_HUB_TOKEN in env." >&2
  echo "[runner] WARN: gated HF repos (incl. ${HF_MODEL_REPO}) will 401 on download." >&2
  echo "[runner] WARN: pass the token via the SSM RunCommand env. See script header." >&2
else
  echo "[runner] HF auth: token present (len=${#HF_TOKEN})"
fi

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
"${PY}" scripts/smoke_sft_data.py \
  --data "${DATADIR}/train.jsonl" \
  --hf-model "${MODELDIR}" \
  --seq-len "${SEQ_LEN}" \
  --max-lines 5000 \
  --min-keep-rate "${SMOKE_MIN_KEEP}"

# ---------------------------------------------------------------------------
# Launch training. -u so log tail-following sees lines live.
# Final command mirrors the one documented at the top of
# configs/config_dixie_mistral_full.yaml.
# ---------------------------------------------------------------------------
echo "[runner] launching training: ${RUN_ID}"
PYTHONUNBUFFERED=1 "${PYTORCH_BASE}/bin/torchrun" \
  --nproc_per_node=8 finetune_sft.py \
  --config "${CONFIG_REL_PATH}" \
  --hf-model "${MODELDIR}" \
  --no-lora --chat-template \
  --device cuda \
  --steps "${TRAIN_STEPS}" \
  --log-every 20 --eval-every 250 --eval-batches 20 --save-every 500 \
  --checkpoint-dir "${CKPTDIR}" \
  --s3-checkpoint-uri "${S3_CKPT_URI}" \
  --min-free-gb 5 --aws-bin aws \
  2>&1 | tee -a "${TRAIN_LOG}"

# torchrun's exit code is the rightmost (tee=0); pipefail surfaces non-zero.
TORCHRUN_RC="${PIPESTATUS[0]}"
echo "[runner] torchrun rc=${TORCHRUN_RC}"

aws s3 cp "${TRAIN_LOG}" "${S3_CKPT_URI}train.log" --no-progress || true

echo "[runner] complete at $(date -Iseconds)"
echo "[runner] artifacts at ${S3_CKPT_URI}"
exit "${TORCHRUN_RC}"
