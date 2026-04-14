#!/usr/bin/env bash
# Poll SSM until command invocation leaves InProgress (or max wait).
#
# Usage:
#   CMD_ID=... INSTANCE_ID=... ./scripts/aws_commands/ssm_timeout_wait_for_command.sh
#
# Env:
#   POLL_INTERVAL_SEC  (default 15)
#   MAX_WAIT_SEC       (default 200000) ~55h; long pretrain + margin
#   REGION, AWS_PROFILE (same defaults as other scripts)

set -euo pipefail

: "${CMD_ID:?set CMD_ID}"
: "${INSTANCE_ID:?set INSTANCE_ID}"

REGION="${REGION:-us-east-1}"
AWS_PROFILE="${AWS_PROFILE:-experimental-admin}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-15}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-200000}"

deadline=$(( $(date +%s) + MAX_WAIT_SEC ))

while [[ $(date +%s) -lt "${deadline}" ]]; do
  st=$(AWS_PROFILE="${AWS_PROFILE}" aws ssm get-command-invocation \
    --region "${REGION}" \
    --command-id "${CMD_ID}" \
    --instance-id "${INSTANCE_ID}" \
    --query Status \
    --output text 2>/dev/null || echo "QueryError")

  echo "$(date -Iseconds) status=${st}"

  if [[ "${st}" != "InProgress" && "${st}" != "Pending" && "${st}" != "Delayed" ]]; then
    AWS_PROFILE="${AWS_PROFILE}" aws ssm get-command-invocation \
      --region "${REGION}" \
      --command-id "${CMD_ID}" \
      --instance-id "${INSTANCE_ID}" \
      --output json | jq '{Status, StatusDetails, ResponseCode, StandardOutputContent: (.StandardOutputContent | if length > 2000 then .[0:2000] + "..." else . end)}'
    exit 0
  fi

  sleep "${POLL_INTERVAL_SEC}"
done

echo "Timed out waiting for terminal status after ${MAX_WAIT_SEC}s" >&2
exit 1
