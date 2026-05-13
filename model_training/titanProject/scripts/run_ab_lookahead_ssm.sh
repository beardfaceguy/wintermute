#!/usr/bin/env bash
# A/B comparison runner: lookahead-on vs baseline (same MAC arch, lookahead-off).
# Designed to run unattended on a g6.2xlarge spot via SSM RunCommand.
#
# Sequence:
#   1. Bootstrap: mount NVMe, fetch code+configs from S3, build venv.
#   2. Run A (lookahead enabled), 25k steps, checkpoints synced live to S3.
#   3. Run B (baseline, lookahead disabled), same hyperparams.
#   4. Self-terminate the EC2 instance.
#
# Tail progress:
#   aws s3 cp s3://alix-ai-ml-staging-data/titan/checkpoints/<RUN>/train.log -
#
# All paths are deliberate; this script is the source of truth for the run.

set -euo pipefail

# ---------------------------------------------------------------------------
# Run identity (caller can override TIMESTAMP for reproducibility).
# ---------------------------------------------------------------------------
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d%H%M%S)}"
RUN_A="mac_lookahead_25k_${TIMESTAMP}"
RUN_B="mac_baseline_25k_${TIMESTAMP}"
S3_BUCKET="${S3_BUCKET:-alix-ai-ml-staging-data}"
S3_CKPT_PREFIX="s3://${S3_BUCKET}/titan/checkpoints"
S3_CODE_URI="${S3_CODE_URI:-s3://${S3_BUCKET}/titan/code/titanProject_ab_lookahead_${TIMESTAMP}}"

# ---------------------------------------------------------------------------
# Local layout — DLAMI exposes ephemeral NVMe at /opt/dlami/nvme.
# ---------------------------------------------------------------------------
WORKDIR=/opt/dlami/nvme/titan
CODEDIR="${WORKDIR}/code/titanProject"
VENV="${WORKDIR}/venv"

mkdir -p "${WORKDIR}/logs"
GLOBAL_LOG="${WORKDIR}/logs/ab_runner.log"
exec > >(tee -a "${GLOBAL_LOG}") 2>&1

echo "[runner] starting at $(date -Iseconds)"
echo "[runner] RUN_A=${RUN_A}"
echo "[runner] RUN_B=${RUN_B}"
echo "[runner] code from ${S3_CODE_URI}"

# ---------------------------------------------------------------------------
# Self-terminate trap (installed FIRST so any setup failure still shuts the
# instance down — keeps spot costs bounded if anything goes wrong).
# ---------------------------------------------------------------------------
TOKEN="$(curl -s -X PUT 'http://169.254.169.254/latest/api/token' -H 'X-aws-ec2-metadata-token-ttl-seconds: 600')"
INSTANCE_ID="$(curl -s -H "X-aws-ec2-metadata-token: ${TOKEN}" \
  http://169.254.169.254/latest/meta-data/instance-id || echo '')"
REGION="$(curl -s -H "X-aws-ec2-metadata-token: ${TOKEN}" \
  http://169.254.169.254/latest/meta-data/placement/region || echo 'us-east-1')"

cleanup() {
  rc=$?
  echo "[runner] exiting with rc=${rc} at $(date -Iseconds)"
  if [[ -n "${INSTANCE_ID}" ]]; then
    aws s3 cp "${GLOBAL_LOG}" "${S3_CKPT_PREFIX}/${RUN_A}/ab_runner.log" --quiet || true
    aws s3 cp "${GLOBAL_LOG}" "${S3_CKPT_PREFIX}/${RUN_B}/ab_runner.log" --quiet || true
    echo "[runner] terminating ${INSTANCE_ID} in ${REGION}"
    aws ec2 terminate-instances --instance-ids "${INSTANCE_ID}" --region "${REGION}" || true
  fi
  exit "${rc}"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Pull code from S3.
# ---------------------------------------------------------------------------
mkdir -p "${CODEDIR}"
aws s3 sync "${S3_CODE_URI}/" "${CODEDIR}/" --no-progress

# ---------------------------------------------------------------------------
# DLAMI ships a pre-built pytorch venv at /opt/pytorch (torch + CUDA, NCCL).
# Its python lacks ensurepip, so creating a child venv fails. The instance
# is ephemeral (we self-terminate at the end), so we just pip-install the
# extra deps directly into /opt/pytorch.
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
  "sentencepiece" "boto3" "pyyaml" "tqdm" "numpy<2"
echo "[runner] env summary:"
"${PY}" -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())"
"${PY}" -c "import titans_pytorch; print('titans=ok')"

# ---------------------------------------------------------------------------
# Run helper: invoke train.py for one config arm with live S3 checkpoint sync.
# ---------------------------------------------------------------------------
run_arm() {
  local arm_name="$1"
  local config_path="$2"
  local arm_log="${WORKDIR}/logs/${arm_name}.log"
  local s3_ckpt="${S3_CKPT_PREFIX}/${arm_name}/"
  local local_ckpt="${WORKDIR}/ckpts/${arm_name}"
  mkdir -p "${local_ckpt}"

  echo "[arm:${arm_name}] starting at $(date -Iseconds)"
  echo "[arm:${arm_name}] config=${config_path}"
  echo "[arm:${arm_name}] checkpoints -> ${s3_ckpt}"

  cd "${CODEDIR}"
  # -u so log rotation/tail-following sees lines as they happen.
  "${PY}" -u train.py \
    --config "${config_path}" \
    --device cuda \
    --log-every 50 \
    --checkpoint-dir "${local_ckpt}" \
    --s3-checkpoint-uri "${s3_ckpt}" \
    --aws-bin aws \
    2>&1 | tee "${arm_log}"

  echo "[arm:${arm_name}] finished at $(date -Iseconds)"
  aws s3 cp "${arm_log}" "${s3_ckpt}train.log" --no-progress
}

# ---------------------------------------------------------------------------
# A — lookahead ON (the candidate).
# ---------------------------------------------------------------------------
run_arm "${RUN_A}" "configs/config_mac_lookahead_25k.yaml"

# ---------------------------------------------------------------------------
# B — lookahead OFF (the control). Identical model + train hyperparams.
# ---------------------------------------------------------------------------
run_arm "${RUN_B}" "configs/config_mac_baseline_25k.yaml"

echo "[runner] both arms complete at $(date -Iseconds)"
echo "[runner] artifacts:"
echo "  ${S3_CKPT_PREFIX}/${RUN_A}/"
echo "  ${S3_CKPT_PREFIX}/${RUN_B}/"
