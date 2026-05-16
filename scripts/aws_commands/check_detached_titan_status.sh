#!/usr/bin/env bash
# Generic detached LLM training job probe (SSM → EC2).
#
# Originally built for Titan GPT long-runs under /mnt/data/ssm_runs/${RUN_ID}.
# Extended for other pipelines (Dixie SFT on NVMe, custom layouts) via
# REMOTE_LAYOUT and path overrides — see scripts/aws_commands/README.md.
#
# Usage (backward compatible):
#   RUN_ID=gpt_small_pretrain_YYYYMMDD... INSTANCE_ID=i-... \\
#     bash scripts/aws_commands/check_detached_titan_status.sh
#
# Dixie SFT (run_dixie_mistral_sft_ssm.sh):
#   REMOTE_LAYOUT=dixie_sft RUN_ID=dixie_mistral_full_... INSTANCE_ID=i-... \\
#     bash scripts/aws_commands/check_detached_titan_status.sh
#
# Custom on-instance paths:
#   REMOTE_LAYOUT=custom RUN_ID=my_run INSTANCE_ID=i-... \\
#     REMOTE_RUN_WORK_DIR=/opt/myproject/run1 \\
#     TRAIN_LOG=/opt/myproject/run1/logs/train.log \\
#     RUNNER_LOG=/opt/myproject/run1/logs/runner.log \\
#     bash scripts/aws_commands/check_detached_titan_status.sh
#
# Optional:
#   CMD_ID=...                    Original bootstrap SSM command id
#   DETACHED_TRAINING_PROBE_CONFIG  Path to JSON config (default: repo config/detached_training_probe.json)
#   REGION, AWS_PROFILE, LOG_TAIL_LINES  Default from config if unset
#   REMOTE_LAYOUT=titan_detached | dixie_sft | custom  (must match config known_layouts)
#   REMOTE_RUN_ROOT=...            (titan_detached only; default in config)
#   REMOTE_RUN_WORK_DIR=...        Override run directory (any layout)
#   TRAIN_LOG=... RUNNER_LOG=...   Absolute paths on the instance
#   RUN_STATUS_JSON=... RUNNER_PID_FILE=...
#   PROBE_CHECK_PID=0|1
#   S3_PREFIX=...                  Default from config template + RUN_ID

set -euo pipefail

: "${RUN_ID:?set RUN_ID first}"
: "${INSTANCE_ID:?set INSTANCE_ID first}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/aws_commands/lib/remote_training_probe_paths.sh
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib/remote_training_probe_paths.sh"

CFG="$(detached_training_probe_config_path)"
if [[ ! -f "${CFG}" ]]; then
  echo "[check_detached_titan_status] FATAL: missing config ${CFG}" >&2
  exit 3
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "[check_detached_titan_status] FATAL: jq is required (install jq or set PATH)" >&2
  exit 3
fi

REGION="${REGION:-$(jq -r '.aws.region' "${CFG}")}"
AWS_PROFILE="${AWS_PROFILE:-$(jq -r '.aws.profile' "${CFG}")}"
LOG_TAIL_LINES="${LOG_TAIL_LINES:-$(jq -r '.probe.log_tail_lines' "${CFG}")}"

_s3_tmpl="$(jq -r '.s3.checkpoint_uri_template' "${CFG}")"
S3_PREFIX="${S3_PREFIX:-${_s3_tmpl//\{run_id\}/${RUN_ID}}}"

remote_training_probe_set_paths

EC2_Q="$(jq -r '.aws.ec2_describe_instance_query' "${CFG}")"
SSM_TO="$(jq -r '.probe.ssm_execution_timeout_seconds' "${CFG}")"
SSM_DOC="$(jq -r '.probe.ssm_document_name' "${CFG}")"
SSM_POLL="$(jq -r '.probe.ssm_poll_interval_seconds' "${CFG}")"

echo "== Instance state =="
AWS_PROFILE="${AWS_PROFILE}" AWS_SDK_LOAD_CONFIG=1 aws ec2 describe-instances \
  --instance-ids "${INSTANCE_ID}" \
  --region "${REGION}" \
  --query "${EC2_Q}" \
  --output json

echo
echo "== Remote layout (${REMOTE_LAYOUT:-titan_detached}) =="
echo "RUN_WORK_DIR=${RUN_WORK_DIR}"
echo "TRAIN_LOG=${TRAIN_LOG}"
echo "RUNNER_LOG=${RUNNER_LOG:-}"
echo "RUN_STATUS_JSON=${RUN_STATUS_JSON:-}"
echo "RUNNER_PID_FILE=${RUNNER_PID_FILE:-}"
echo "PROBE_CHECK_PID=${PROBE_CHECK_PID}"

if [[ -n "${CMD_ID:-}" ]]; then
  echo
  echo "== Bootstrap SSM command =="
  AWS_PROFILE="${AWS_PROFILE}" AWS_SDK_LOAD_CONFIG=1 aws ssm get-command-invocation \
    --command-id "${CMD_ID}" \
    --instance-id "${INSTANCE_ID}" \
    --region "${REGION}" \
    --query '{Status:Status,ResponseCode:ResponseCode,StatusDetails:StatusDetails,ExecutionElapsedTime:ExecutionElapsedTime}' \
    --output json
fi

echo
echo "== Detached runner probe via SSM =="

PROBE_SCRIPT="$(mktemp)"
PARAMS_JSON="$(mktemp)"
trap 'rm -f "${PROBE_SCRIPT}" "${PARAMS_JSON}"' EXIT

# Embed resolved paths so the probe runs with no extra env on the instance.
# shellcheck disable=SC2089,SC2090
{
  printf '%s\n' '#!/bin/bash'
  printf '%s\n' 'set -euo pipefail'
  printf 'RUN_WORK_DIR=%q\n' "${RUN_WORK_DIR}"
  printf 'TRAIN_LOG=%q\n' "${TRAIN_LOG}"
  printf 'RUNNER_LOG=%q\n' "${RUNNER_LOG:-}"
  printf 'RUN_STATUS_JSON=%q\n' "${RUN_STATUS_JSON:-}"
  printf 'RUNNER_PID_FILE=%q\n' "${RUNNER_PID_FILE:-}"
  printf 'PROBE_CHECK_PID=%q\n' "${PROBE_CHECK_PID}"
  printf 'LOG_TAIL_LINES=%q\n' "${LOG_TAIL_LINES}"
  cat <<'PROBE_BODY'
echo "RUN_WORK_DIR=${RUN_WORK_DIR}"

if [[ "${PROBE_CHECK_PID}" == "1" ]]; then
  if [[ -n "${RUNNER_PID_FILE}" && -f "${RUNNER_PID_FILE}" ]]; then
    RUNNER_PID="$(cat "${RUNNER_PID_FILE}")"
    echo "RUNNER_PID=${RUNNER_PID}"
    if kill -0 "${RUNNER_PID}" 2>/dev/null; then
      echo "RUNNER_PROCESS_STATUS=running"
    else
      echo "RUNNER_PROCESS_STATUS=not-running"
    fi
  else
    echo "RUNNER_PROCESS_STATUS=pid-file-missing"
  fi
else
  echo "RUNNER_PROCESS_STATUS=skipped (PROBE_CHECK_PID=0)"
fi

if [[ -n "${RUN_STATUS_JSON}" && -f "${RUN_STATUS_JSON}" ]]; then
  echo "--- run_status.json ---"
  cat "${RUN_STATUS_JSON}"
elif [[ -n "${RUN_STATUS_JSON}" ]]; then
  echo "RUN_STATUS_JSON_STATUS=missing (${RUN_STATUS_JSON})"
fi

if [[ -n "${RUNNER_LOG}" && -f "${RUNNER_LOG}" ]]; then
  echo "--- runner log tail ---"
  tail -n "${LOG_TAIL_LINES}" "${RUNNER_LOG}"
elif [[ -n "${RUNNER_LOG}" ]]; then
  echo "RUNNER_LOG_STATUS=missing (${RUNNER_LOG})"
fi

if [[ -f "${TRAIN_LOG}" ]]; then
  echo "--- train.log tail ---"
  tail -n "${LOG_TAIL_LINES}" "${TRAIN_LOG}"
else
  echo "TRAIN_LOG_STATUS=missing (${TRAIN_LOG})"
fi
PROBE_BODY
} >"${PROBE_SCRIPT}"

jq -n --arg ex "${SSM_TO}" --rawfile script "${PROBE_SCRIPT}" \
  '{commands: [$script], executionTimeout: [$ex]}' >"${PARAMS_JSON}"

PROBE_CMD_ID="$(
  AWS_PROFILE="${AWS_PROFILE}" AWS_SDK_LOAD_CONFIG=1 aws ssm send-command \
    --region "${REGION}" \
    --document-name "${SSM_DOC}" \
    --instance-ids "${INSTANCE_ID}" \
    --parameters "file://${PARAMS_JSON}" \
    --query "Command.CommandId" \
    --output text
)"

echo "Probe command: ${PROBE_CMD_ID}"

while true; do
  STATUS_JSON="$(
    AWS_PROFILE="${AWS_PROFILE}" AWS_SDK_LOAD_CONFIG=1 aws ssm get-command-invocation \
      --command-id "${PROBE_CMD_ID}" \
      --instance-id "${INSTANCE_ID}" \
      --region "${REGION}" \
      --output json
  )"
  STATUS="$(printf '%s' "${STATUS_JSON}" | jq -r '.Status')"
  if [[ "${STATUS}" == "Pending" || "${STATUS}" == "InProgress" || "${STATUS}" == "Delayed" ]]; then
    sleep "${SSM_POLL}"
    continue
  fi
  printf '%s\n' "${STATUS_JSON}" | jq '{Status: .Status, ResponseCode: .ResponseCode, StatusDetails: .StatusDetails}'
  echo
  printf '%s\n' "${STATUS_JSON}" | jq -r '.StandardOutputContent'
  STDERR_CONTENT="$(printf '%s' "${STATUS_JSON}" | jq -r '.StandardErrorContent')"
  if [[ -n "${STDERR_CONTENT}" ]]; then
    echo
    echo "--- probe stderr ---"
    printf '%s\n' "${STDERR_CONTENT}"
  fi
  break
done

echo
echo "== Synced checkpoints in S3 =="
AWS_PROFILE="${AWS_PROFILE}" AWS_SDK_LOAD_CONFIG=1 aws s3 ls "${S3_PREFIX}" --recursive
