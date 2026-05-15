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
# Lifecycle helpers (instance-id resolution, terminate_self, EXIT trap) live
# in scripts/lib/aws_lifecycle.sh so they can be unit-tested in isolation.
# Order-of-operations safety property: every pre-terminate hook completes
# (success OR failure) before terminate_self is called. This is what keeps
# us from torching runner.log + checkpoints by terminating too early.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=model_training/titanProject/scripts/lib/aws_lifecycle.sh
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/aws_lifecycle.sh"

# Hook: upload the global runner.log into each arm's S3 prefix BEFORE we
# terminate the instance. Returning rc!=0 is logged but does not block
# termination (we'd rather lose a log than strand a billing instance).
_runner_upload_runner_log() {
  local arm rc=0
  for arm in "${ARM_NAMES[@]}"; do
    if ! aws s3 cp "${GLOBAL_LOG}" "${S3_CKPT_PREFIX}/${arm}/runner.log" --quiet; then
      echo "[runner] WARN: failed to upload runner.log for arm=${arm}" >&2
      rc=1
    fi
  done
  return "${rc}"
}
CLEANUP_PRE_TERMINATE_HOOKS+=(_runner_upload_runner_log)

resolve_instance_metadata
install_cleanup_trap

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
