#!/usr/bin/env bash
# Generic N-arm Titan training runner. Designed to run unattended on a
# g6.2xlarge spot via SSM RunCommand (systemd-run --collect for clean
# detachment from SSM's process group).
#
# Successor to scripts/run_ab_lookahead_ssm.sh: takes an arbitrary list of
# (arm_name, config_path) pairs in the ARMS env var so the same script can
# drive the original A/B, the 3-arm 0.5.3 ablation, or any future sweep.
#
# Sequence (per arm, in order):
#   1. Bootstrap (once): mount NVMe, fetch code from S3, install deps into
#      /opt/pytorch.
#   2. For each arm: run train.py with live S3 checkpoint sync.
#   3. Self-terminate the EC2 instance.
#
# Required env:
#   TIMESTAMP    Run identity, e.g. 20260513192012.
#   S3_CODE_URI  Where train.py + configs were synced before launch.
#   ARMS         Colon-separated list of "arm_name=config_relpath" pairs.
#                Example:
#                  ARMS="mac_lookahead_only_25k_${TIMESTAMP}=configs/config_mac_lookahead_only_25k.yaml:..."
#
# Optional env:
#   S3_BUCKET    Defaults to alix-ai-ml-staging-data.
#
# All paths are deliberate; this script is the source of truth for the run.

set -euo pipefail

# ---------------------------------------------------------------------------
# Run identity / arm list.
# ---------------------------------------------------------------------------
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d%H%M%S)}"
S3_BUCKET="${S3_BUCKET:-alix-ai-ml-staging-data}"
S3_CKPT_PREFIX="s3://${S3_BUCKET}/titan/checkpoints"
S3_CODE_URI="${S3_CODE_URI:-}"
if [[ -z "${S3_CODE_URI}" ]]; then
  echo "[runner] FATAL: S3_CODE_URI is required" >&2
  exit 64
fi

# Default ARMS = the original A/B for back-compat with existing kickoffs.
ARMS="${ARMS:-mac_lookahead_25k_${TIMESTAMP}=configs/config_mac_lookahead_25k.yaml:mac_baseline_25k_${TIMESTAMP}=configs/config_mac_baseline_25k.yaml}"

ARM_NAMES=()
ARM_CONFIGS=()
IFS=':' read -ra ARM_LIST <<< "${ARMS}"
for spec in "${ARM_LIST[@]}"; do
  IFS='=' read -r arm_name arm_config <<< "${spec}"
  if [[ -z "${arm_name}" || -z "${arm_config}" ]]; then
    echo "[runner] FATAL: malformed ARMS entry: '${spec}'" >&2
    exit 64
  fi
  ARM_NAMES+=("${arm_name}")
  ARM_CONFIGS+=("${arm_config}")
done

# ---------------------------------------------------------------------------
# Local layout — DLAMI exposes ephemeral NVMe at /opt/dlami/nvme.
# ---------------------------------------------------------------------------
WORKDIR=/opt/dlami/nvme/titan
CODEDIR="${WORKDIR}/code/titanProject"

mkdir -p "${WORKDIR}/logs"
GLOBAL_LOG="${WORKDIR}/logs/runner.log"
exec > >(tee -a "${GLOBAL_LOG}") 2>&1

echo "[runner] starting at $(date -Iseconds)"
echo "[runner] timestamp=${TIMESTAMP}"
echo "[runner] code from ${S3_CODE_URI}"
echo "[runner] arms (${#ARM_NAMES[@]}):"
for i in "${!ARM_NAMES[@]}"; do
  echo "  ${i}: ${ARM_NAMES[$i]}  <-  ${ARM_CONFIGS[$i]}"
done

# ---------------------------------------------------------------------------
# Self-terminate trap (installed FIRST so any setup failure still tears the
# instance down — keeps spot costs bounded).
#
# Lesson from the 20260513 A/B: the previous version of this script lost
# IMDSv2 lookups on a single curl failure and silently no-op'd, leaving
# the instance idle for ~2hr after rc=0. Now we:
#   1. retry IMDSv2 token + field lookups,
#   2. fall back to /var/lib/cloud/data/instance-id (cloud-init writes it),
#   3. if even that fails, fall back to `shutdown -h now` (spot terminates
#      on shutdown — InstanceInitiatedShutdownBehavior=terminate by default).
# ---------------------------------------------------------------------------
get_imds_field() {
  local field="$1" token result
  for attempt in 1 2 3; do
    token="$(curl -fsS --max-time 3 \
      -X PUT 'http://169.254.169.254/latest/api/token' \
      -H 'X-aws-ec2-metadata-token-ttl-seconds: 600' 2>/dev/null || true)"
    if [[ -n "${token}" ]]; then
      result="$(curl -fsS --max-time 3 \
        -H "X-aws-ec2-metadata-token: ${token}" \
        "http://169.254.169.254/latest/meta-data/${field}" 2>/dev/null || true)"
      if [[ -n "${result}" ]]; then
        echo "${result}"
        return 0
      fi
    fi
    sleep 1
  done
  return 1
}

INSTANCE_ID="$(get_imds_field instance-id || true)"
if [[ -z "${INSTANCE_ID}" && -r /var/lib/cloud/data/instance-id ]]; then
  INSTANCE_ID="$(cat /var/lib/cloud/data/instance-id)"
  echo "[runner] IMDSv2 failed; using cloud-init instance-id=${INSTANCE_ID}"
fi
REGION="$(get_imds_field placement/region || true)"
REGION="${REGION:-us-east-1}"
echo "[runner] resolved instance_id='${INSTANCE_ID}' region='${REGION}'"

cleanup() {
  rc=$?
  echo "[runner] exiting with rc=${rc} at $(date -Iseconds)"
  for arm in "${ARM_NAMES[@]}"; do
    aws s3 cp "${GLOBAL_LOG}" "${S3_CKPT_PREFIX}/${arm}/runner.log" --quiet || true
  done
  if [[ -n "${INSTANCE_ID}" ]]; then
    echo "[runner] terminating ${INSTANCE_ID} in ${REGION}"
    if ! aws ec2 terminate-instances --instance-ids "${INSTANCE_ID}" --region "${REGION}"; then
      echo "[runner] terminate-instances API call failed; falling back to shutdown -h"
      sync; sleep 2
      shutdown -h now || /sbin/shutdown -h now || true
    fi
  else
    echo "[runner] WARNING: instance-id still unresolved; falling back to shutdown -h"
    sync; sleep 2
    shutdown -h now || /sbin/shutdown -h now || true
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
# Run helper: invoke train.py for one arm with live S3 checkpoint sync.
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
# Run all arms sequentially.
# ---------------------------------------------------------------------------
for i in "${!ARM_NAMES[@]}"; do
  run_arm "${ARM_NAMES[$i]}" "${ARM_CONFIGS[$i]}"
done

echo "[runner] all ${#ARM_NAMES[@]} arms complete at $(date -Iseconds)"
echo "[runner] artifacts:"
for arm in "${ARM_NAMES[@]}"; do
  echo "  ${S3_CKPT_PREFIX}/${arm}/"
done
