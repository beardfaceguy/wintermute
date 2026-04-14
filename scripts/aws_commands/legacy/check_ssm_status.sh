#!/usr/bin/env bash
# OBSOLETE: legacy helper for foreground SSM-run commands.
# Prefer scripts/aws_commands/check_detached_titan_status.sh for the current detached Titan long-run flow.
# Kept only for older one-shot / timeout-debug command paths.
#
# Usage: CMD_ID=... scripts/aws_commands/legacy/check_ssm_status.sh
# Optional: INSTANCE_ID=... to include per-instance detail.

set -euo pipefail

: "${CMD_ID:?set CMD_ID first}"
CW_LOG_GROUP="${CW_LOG_GROUP:-/aws/ssm/titan-llm-training}"
REGION="${REGION:-us-east-1}"
AWS_PROFILE="${AWS_PROFILE:-experimental-admin}"

echo "== get-command-invocation =="
if [[ -n "${INSTANCE_ID:-}" ]]; then
  AWS_PROFILE="${AWS_PROFILE}" aws ssm get-command-invocation \
    --region "${REGION}" \
    --command-id "${CMD_ID}" \
    --instance-id "${INSTANCE_ID}" \
    --no-cli-pager \
    --query '{Status:Status, ResponseCode:ResponseCode, StatusDetails:StatusDetails, StandardOutputUrl:StandardOutputUrl, StandardErrorUrl:StandardErrorUrl}'
else
  AWS_PROFILE="${AWS_PROFILE}" aws ssm get-command-invocation \
    --region "${REGION}" \
    --command-id "${CMD_ID}" \
    --no-cli-pager \
    --query '{Status:Status, ResponseCode:ResponseCode, StatusDetails:StatusDetails, StandardOutputUrl:StandardOutputUrl, StandardErrorUrl:StandardErrorUrl}'
fi

echo
echo "== list-command-invocations (summary) =="
AWS_PROFILE="${AWS_PROFILE}" aws ssm list-command-invocations \
  --region "${REGION}" \
  --command-id "${CMD_ID}" \
  --no-cli-pager \
  --query 'CommandInvocations[].{InstanceId:InstanceId, Status:Status, Start:RequestedDateTime, End:FinishedDateTime}'

if [[ -n "${INSTANCE_ID:-}" ]]; then
  echo
  echo "== CloudWatch (if send-command used --cloud-watch-output-config) =="
  echo "Log group: ${CW_LOG_GROUP}"
  echo "Stream pattern: ${CMD_ID}/${INSTANCE_ID}/aws-runShellScript/stdout"
  echo "Tail: AWS_PROFILE=${AWS_PROFILE} aws logs tail ${CW_LOG_GROUP} --follow --region ${REGION}"
fi
