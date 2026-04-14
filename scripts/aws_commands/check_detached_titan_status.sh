#!/usr/bin/env bash
# Check status of a detached Titan long-run training job.
# Usage:
#   RUN_ID=gpt_small_pretrain_YYYYMMDDHHMMSS INSTANCE_ID=i-... bash scripts/aws_commands/check_detached_titan_status.sh
# Optional:
#   CMD_ID=...                    Original bootstrap SSM command id
#   REGION=us-east-1
#   AWS_PROFILE=experimental-admin
#   REMOTE_RUN_ROOT=/mnt/data/ssm_runs
#   S3_PREFIX=s3://...            Override inferred checkpoint prefix

set -euo pipefail

: "${RUN_ID:?set RUN_ID first}"
: "${INSTANCE_ID:?set INSTANCE_ID first}"

REGION="${REGION:-us-east-1}"
AWS_PROFILE="${AWS_PROFILE:-experimental-admin}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-/mnt/data/ssm_runs}"
RUN_WORK_DIR="${REMOTE_RUN_ROOT}/${RUN_ID}"
TRAIN_LOG="${RUN_WORK_DIR}/train.log"
RUN_STATUS_JSON="${RUN_WORK_DIR}/run_status.json"
RUNNER_PID_FILE="${RUN_WORK_DIR}/runner.pid"
S3_PREFIX="${S3_PREFIX:-s3://alix-ai-ml-staging-data/titan/checkpoints/${RUN_ID}/}"

echo "== Instance state =="
AWS_PROFILE="${AWS_PROFILE}" AWS_SDK_LOAD_CONFIG=1 aws ec2 describe-instances \
  --instance-ids "${INSTANCE_ID}" \
  --region "${REGION}" \
  --query 'Reservations[0].Instances[0].{State:State.Name,InstanceType:InstanceType,LaunchTime:LaunchTime,PublicIp:PublicIpAddress}' \
  --output json

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

cat >"${PROBE_SCRIPT}" <<EOF
#!/bin/bash
set -euo pipefail
RUN_WORK_DIR="${RUN_WORK_DIR}"
TRAIN_LOG="${TRAIN_LOG}"
RUN_STATUS_JSON="${RUN_STATUS_JSON}"
RUNNER_PID_FILE="${RUNNER_PID_FILE}"

echo "RUN_WORK_DIR=\${RUN_WORK_DIR}"

if [[ -f "\${RUNNER_PID_FILE}" ]]; then
  RUNNER_PID="\$(cat "\${RUNNER_PID_FILE}")"
  echo "RUNNER_PID=\${RUNNER_PID}"
  if kill -0 "\${RUNNER_PID}" 2>/dev/null; then
    echo "RUNNER_PROCESS_STATUS=running"
  else
    echo "RUNNER_PROCESS_STATUS=not-running"
  fi
else
  echo "RUNNER_PROCESS_STATUS=pid-file-missing"
fi

if [[ -f "\${RUN_STATUS_JSON}" ]]; then
  echo "--- run_status.json ---"
  cat "\${RUN_STATUS_JSON}"
fi

if [[ -f "\${TRAIN_LOG}" ]]; then
  echo "--- train.log tail ---"
  tail -n 40 "\${TRAIN_LOG}"
else
  echo "TRAIN_LOG_STATUS=missing"
fi
EOF

jq -n --rawfile script "${PROBE_SCRIPT}" '{commands: [$script], executionTimeout: ["600"]}' >"${PARAMS_JSON}"

PROBE_CMD_ID="$(
  AWS_PROFILE="${AWS_PROFILE}" AWS_SDK_LOAD_CONFIG=1 aws ssm send-command \
    --region "${REGION}" \
    --document-name "AWS-RunShellScript" \
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
    sleep 2
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
