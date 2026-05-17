#!/usr/bin/env bash
# =============================================================================
# Dixie SFT — controller launcher for AWS SSM (bash-safe, zsh-safe)
# =============================================================================
#
# Builds SSM parameters with `jq` + a remote shell script. Hugging Face tokens
# and paths are passed with `printf '%q'` so special characters never trip
# zsh/bash mismatch (avoids ${VAR@Q}, which breaks under zsh).
#
# Usage (from Wintermute repo root, after: cd model_training/titanProject):
#
#   export INSTANCE_ID=i-0xxxxxxxxxxxxxxxxx
#   export AWS_PROFILE=experimental-admin
#   export AWS_SDK_LOAD_CONFIG=1
#
#   # Optional: full Mistral run on p4d (default entry script)
#   scripts/launch_dixie_sft_ssm.sh
#
#   # Optional: cheap gpt2 smoke on g5.xlarge (set DIXIE_CHEAP=1)
#   DIXIE_CHEAP=1 INSTANCE_ID=i-... scripts/launch_dixie_sft_ssm.sh
#
# Steps performed:
#   1. aws s3 sync ./  -> ${S3_CODE_URI}/   (whole titanProject tree)
#   2. aws ssm send-command -> remote: sync code + bash entry script
#
# Monitor:
#   aws s3 cp "s3://${S3_BUCKET}/titan/checkpoints/${RUN_ID}/runner.log" -
#   AWS_PROFILE=... aws logs tail "${CW_LOG_GROUP}" --follow --region "${REGION}"
#
# Security — HF_TOKEN
#   The token is embedded in the SSM RunShellScript payload (API + console
#   history) and in the generated remote script. Do not enable shell xtrace on
#   the remote wrapper (would leak into CloudWatch). Local mktemp files below
#   hold the same secret until EXIT; run on a trusted workstation. Prefer
#   rotating the token if exposure is suspected. For stricter policy, fetch a
#   secret on-instance (Secrets Manager / SSM Parameter SecureString) instead
#   of inline export — not implemented here.
#
# Lifecycle — AWS_LIFECYCLE_MODE=stop (cheap smoke default)
#   Stops the instance on EXIT (saves EBS); you still pay for storage and must
#   clean up stopped instances / volumes per your account hygiene. Use
#   terminate (default for long runs) when you want the instance gone.
#
# S3 code URI — sync --delete
#   S3_CODE_URI must contain .../dixie_YYYYMMDDHHMMSS/... unless
#   DIXIE_ALLOW_NONSTANDARD_S3_CODE_URI=1 (see scripts/lib/dixie_s3_code_uri.sh).
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TITAN_PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# wintermute repo root (…/model_training/titanProject/scripts -> …/wintermute)
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

REGION="${REGION:-us-east-1}"
AWS_PROFILE="${AWS_PROFILE:-experimental-admin}"
export AWS_PROFILE
export AWS_SDK_LOAD_CONFIG="${AWS_SDK_LOAD_CONFIG:-1}"

S3_BUCKET="${S3_BUCKET:-alix-ai-ml-staging-data}"
S3_TRAIN_URI="${S3_TRAIN_URI:-s3://${S3_BUCKET}/titan/data/dixie_pentest/train.jsonl}"
S3_VAL_URI="${S3_VAL_URI:-s3://${S3_BUCKET}/titan/data/dixie_pentest/val.jsonl}"

SSM_EXEC_TIMEOUT_SECONDS="${SSM_EXEC_TIMEOUT_SECONDS:-43200}"
SSM_DELIVERY_TIMEOUT_SECONDS="${SSM_DELIVERY_TIMEOUT_SECONDS:-43200}"
CW_LOG_GROUP="${CW_LOG_GROUP:-/aws/ssm/titan-llm-training}"
LOG_PREFIX="ssm-logs/dixie-sft/$(date -u +%Y%m%d%H%M%S)"

INSTANCE_ID="${INSTANCE_ID:-}"
if [[ -z "${INSTANCE_ID}" || "${INSTANCE_ID}" == i-REPLACE_ME ]]; then
  echo "Set INSTANCE_ID to a GPU instance with SSM (e.g. p4d or g5.xlarge)." >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required (e.g. apt install jq / brew install jq)." >&2
  exit 1
fi

TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d%H%M%S)}"
S3_CODE_URI="${S3_CODE_URI:-s3://${S3_BUCKET}/titan/code/dixie_${TIMESTAMP}}"

# shellcheck source=model_training/titanProject/scripts/lib/dixie_s3_code_uri.sh
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/dixie_s3_code_uri.sh"
dixie_validate_s3_code_uri_for_delete_sync "${S3_CODE_URI}" || exit 1

if [[ "${DIXIE_CHEAP:-0}" == "1" ]]; then
  RUN_ID="${RUN_ID:-dixie_gpt2_smoke_${TIMESTAMP}}"
  ENTRY_SCRIPT="run_dixie_sft_smoke_cheap_ssm.sh"
  : "${AWS_LIFECYCLE_MODE:=stop}"
else
  RUN_ID="${RUN_ID:-dixie_mistral_full_${TIMESTAMP}}"
  ENTRY_SCRIPT="run_dixie_mistral_sft_ssm.sh"
fi

HF_ENV_FILE="${HF_ENV_FILE:-${REPO_ROOT}/model_training/hf.env}"
if [[ -z "${HF_TOKEN:-}" && -f "${HF_ENV_FILE}" ]]; then HF_TOKEN="$(grep '^accessToken=' "${HF_ENV_FILE}" | cut -d= -f2-)" || true; fi
HF_TOKEN="${HF_TOKEN:-}"

if [[ "${ENTRY_SCRIPT}" == run_dixie_mistral_sft_ssm.sh ]] && [[ -z "${HF_TOKEN}" ]]; then
  echo "[launch] FATAL: HF_TOKEN is empty (Mistral is gated). Set HF_TOKEN or point HF_ENV_FILE" >&2
  echo "[launch] FATAL: at ${HF_ENV_FILE} with accessToken=... or run with DIXIE_CHEAP=1 (gpt2 smoke)." >&2
  exit 1
fi

REMOTE_SCRIPT="$(mktemp)"
PARAMS_JSON="$(mktemp)"
chmod 600 "${REMOTE_SCRIPT}" "${PARAMS_JSON}" 2>/dev/null || true
trap 'rm -f "${REMOTE_SCRIPT}" "${PARAMS_JSON}"' EXIT

echo "[launch] sync titanProject -> ${S3_CODE_URI}/"
aws s3 sync "${TITAN_PROJECT}/" "${S3_CODE_URI}/" \
  --delete \
  --exclude '__pycache__/*' \
  --exclude '.pytest_cache/*' \
  --exclude 'results/*' \
  --exclude 'logs/*' \
  --exclude 'saved_models/*' \
  --exclude 'checkpoints_sft/*' \
  --exclude '.git/*' \
  --region "${REGION}"

# Remote script: every dynamic value passed through printf '%q' for safe shell words.
# No xtrace: would leak HF_TOKEN into SSM / CloudWatch.
{
  echo '#!/bin/bash'
  echo 'set -euo pipefail'
  echo "export TIMESTAMP=$(printf '%q' "${TIMESTAMP}")"
  echo "export RUN_ID=$(printf '%q' "${RUN_ID}")"
  echo "export S3_CODE_URI=$(printf '%q' "${S3_CODE_URI}")"
  echo "export S3_TRAIN_URI=$(printf '%q' "${S3_TRAIN_URI}")"
  echo "export S3_VAL_URI=$(printf '%q' "${S3_VAL_URI}")"
  echo "export AWS_DEFAULT_REGION=$(printf '%q' "${REGION}")"
  echo "export HF_TOKEN=$(printf '%q' "${HF_TOKEN}")"
  if [[ -n "${AWS_LIFECYCLE_MODE:-}" ]]; then
    echo "export AWS_LIFECYCLE_MODE=$(printf '%q' "${AWS_LIFECYCLE_MODE}")"
  fi
  # Same path as run_dixie_mistral_sft_ssm.sh CODEDIR — avoids a second full S3 sync on the instance.
  echo 'CODEDIR_PRE=/opt/dlami/nvme/dixie/code/titanProject'
  echo 'mkdir -p "${CODEDIR_PRE}"'
  echo 'aws s3 sync "${S3_CODE_URI}/" "${CODEDIR_PRE}/" --no-progress'
  echo 'export DIXIE_CODE_PRE_SYNCED=1'
  echo 'chmod +x "${CODEDIR_PRE}/scripts/"*.sh 2>/dev/null || true'
  echo "exec bash \"\${CODEDIR_PRE}/scripts/${ENTRY_SCRIPT}\""
} > "${REMOTE_SCRIPT}"

jq -n \
  --rawfile script "${REMOTE_SCRIPT}" \
  --arg et "${SSM_EXEC_TIMEOUT_SECONDS}" \
  '{commands: [$script], executionTimeout: [$et]}' > "${PARAMS_JSON}"

echo "[launch] S3_CODE_URI=${S3_CODE_URI}"
echo "[launch] RUN_ID=${RUN_ID} ENTRY=${ENTRY_SCRIPT} INSTANCE=${INSTANCE_ID}"

CMD_ID="$(aws ssm send-command \
  --region "${REGION}" \
  --document-name "AWS-RunShellScript" \
  --comment "Dixie SFT (${ENTRY_SCRIPT}) RUN_ID=${RUN_ID}" \
  --timeout-seconds "${SSM_DELIVERY_TIMEOUT_SECONDS}" \
  --instance-ids "${INSTANCE_ID}" \
  --cloud-watch-output-config "CloudWatchLogGroupName=${CW_LOG_GROUP},CloudWatchOutputEnabled=true" \
  --parameters "file://${PARAMS_JSON}" \
  --output-s3-bucket-name "${S3_BUCKET}" \
  --output-s3-key-prefix "${LOG_PREFIX}" \
  --query "Command.CommandId" \
  --output text)"

echo ""
echo "Command ID: ${CMD_ID}"
echo "CloudWatch: ${CW_LOG_GROUP}"
echo "  aws logs tail ${CW_LOG_GROUP} --follow --region ${REGION}"
echo "Artifacts:  s3://${S3_BUCKET}/titan/checkpoints/${RUN_ID}/"
echo "  aws s3 cp s3://${S3_BUCKET}/titan/checkpoints/${RUN_ID}/runner.log -"
echo ""
