#!/usr/bin/env bash
# Long sleep via SSM to validate execution timeout behavior (AWS-RunShellScript).
#
# Modes (first match wins):
#   SSM_QUICK_TEST=1     ~2 min — executionTimeout=120s, sleep 180s → expect ExecutionTimedOut
#   SSM_LONG_VERIFY=1    ~70 min — sleep 4200s, executionTimeout≥7200, delivery 12h (proves fix vs 3600s default)
#   (else)               SLEEP_SEC default 4000 (~1h07), or set explicitly
#
# Usage:
#   AWS_PROFILE=experimental-admin INSTANCE_ID=i-... ./scripts/aws_commands/ssm_timeout_sleep_test.sh
#
# Env:
#   INSTANCE_ID       (required) target EC2 managed instance
#   REGION            (default us-east-1)
#   AWS_PROFILE       (default experimental-admin)
#   SLEEP_SEC         total sleep seconds on the instance (see defaults per mode)
#   SSM_SLEEP_MODE    heartbeat (default): 60s chunks + echo; single: one sleep, minimal output
#   INCLUDE_EXEC_TIMEOUT  true/false — if false, omit executionTimeout (default 3600s document cap)
#   SSM_EXEC_TIMEOUT_SECONDS  override; default derived from SLEEP_SEC (+margin, min 7200 for long jobs)
#   SSM_DELIVERY_TIMEOUT_SECONDS  passed to send-command --timeout-seconds (default 43200)
#   SSM_OUTPUT_S3_BUCKET / SSM_OUTPUT_S3_KEY_PREFIX  if both set, send-command saves stdout/stderr to S3
#   SSM_CLOUDWATCH_LOG_GROUP  if set, e.g. /aws/ssm/titan-llm-training, enables CloudWatch output
#
# After run:
#   CMD_ID=... INSTANCE_ID=... ./scripts/aws_commands/legacy/check_ssm_status.sh
#   CMD_ID=... INSTANCE_ID=... ./scripts/aws_commands/ssm_timeout_wait_for_command.sh
#
# Does plain sleep with no output cause timeout? No — limits are wall-clock (executionTimeout).

set -euo pipefail

INSTANCE_ID="${INSTANCE_ID:-}"
REGION="${REGION:-us-east-1}"
AWS_PROFILE="${AWS_PROFILE:-experimental-admin}"
INCLUDE_EXEC_TIMEOUT="${INCLUDE_EXEC_TIMEOUT:-true}"
SSM_DELIVERY_TIMEOUT_SECONDS="${SSM_DELIVERY_TIMEOUT_SECONDS:-43200}"
SSM_SLEEP_MODE="${SSM_SLEEP_MODE:-heartbeat}"

# Apply defaults after mode flags so SSM_LONG_VERIFY is not stuck at a generic default.
if [[ "${SSM_QUICK_TEST:-}" == "1" ]]; then
  SLEEP_SEC=180
  SSM_EXEC_TIMEOUT_SECONDS=120
  INCLUDE_EXEC_TIMEOUT=true
  SSM_DELIVERY_TIMEOUT_SECONDS=600
  echo "[ssm_sleep_test] SSM_QUICK_TEST=1: sleep 180s, executionTimeout=120s → expect ExecutionTimedOut ~2m"
elif [[ "${SSM_LONG_VERIFY:-}" == "1" ]]; then
  # >1h wall clock to exceed legacy 3600s default. Default 4200s (~70m). Override: SLEEP_SEC=4500 SSM_LONG_VERIFY=1
  SLEEP_SEC="${SLEEP_SEC:-${LONG_VERIFY_SLEEP_SEC:-4200}}"
  INCLUDE_EXEC_TIMEOUT=true
  SSM_DELIVERY_TIMEOUT_SECONDS="${SSM_DELIVERY_TIMEOUT_SECONDS:-43200}"
  echo "[ssm_sleep_test] SSM_LONG_VERIFY=1: sleep ${SLEEP_SEC}s (~$(( (SLEEP_SEC + 59) / 60 ))m), executionTimeout≥7200 → expect Success"
else
  SLEEP_SEC="${SLEEP_SEC:-4000}"
fi

if [[ -z "${INSTANCE_ID}" || "${INSTANCE_ID}" == "i-REPLACE_ME" ]]; then
  echo "Set INSTANCE_ID to a managed instance id." >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required (brew install jq)." >&2
  exit 1
fi

# Execution timeout must exceed sleep; at least sleep+10min, min 7200 for sub-2h jobs
MIN_EXEC=$((SLEEP_SEC + 600))
if [[ "${MIN_EXEC}" -lt 7200 ]]; then
  MIN_EXEC=7200
fi
DEFAULT_EXEC="${MIN_EXEC}"
if [[ "${DEFAULT_EXEC}" -gt 172800 ]]; then
  DEFAULT_EXEC=172800
fi
SSM_EXEC_TIMEOUT_SECONDS="${SSM_EXEC_TIMEOUT_SECONDS:-${DEFAULT_EXEC}}"

if [[ "${SSM_EXEC_TIMEOUT_SECONDS}" -gt 172800 ]]; then
  echo "SSM_EXEC_TIMEOUT_SECONDS max is 172800" >&2
  exit 1
fi

REMOTE_SCRIPT="$(mktemp)"
PARAMS_JSON="$(mktemp)"
trap 'rm -f "${REMOTE_SCRIPT}" "${PARAMS_JSON}"' EXIT

# Remote script: exact wall-clock sleep (fixes non-divisible-by-60 durations)
if [[ "${SSM_SLEEP_MODE}" == "single" ]]; then
  cat >"${REMOTE_SCRIPT}" <<EOF
#!/bin/bash
set -euo pipefail
echo "[ssm_sleep_test] start \$(date -Iseconds) sleep_sec=${SLEEP_SEC} mode=single"
sleep "${SLEEP_SEC}"
echo "[ssm_sleep_test] done \$(date -Iseconds)"
EOF
else
  cat >"${REMOTE_SCRIPT}" <<EOF
#!/bin/bash
set -euo pipefail
TOTAL=${SLEEP_SEC}
remaining=\${TOTAL}
echo "[ssm_sleep_test] start \$(date -Iseconds) sleep_sec=\${TOTAL} mode=heartbeat"
while [[ "\$remaining" -gt 0 ]]; do
  chunk=\$(( remaining > 60 ? 60 : remaining ))
  sleep "\$chunk"
  remaining=\$((remaining - chunk))
  elapsed=\$((TOTAL - remaining))
  echo "[ssm_sleep_test] elapsed \${elapsed}s /\ ${SLEEP_SEC} \$(date -Iseconds)"
done
echo "[ssm_sleep_test] done \$(date -Iseconds)"
EOF
fi

if [[ "${INCLUDE_EXEC_TIMEOUT}" == "true" ]]; then
  jq -n --rawfile script "${REMOTE_SCRIPT}" \
    --arg et "${SSM_EXEC_TIMEOUT_SECONDS}" \
    '{commands: [$script], executionTimeout: [$et]}' >"${PARAMS_JSON}"
  echo "Parameters: executionTimeout=${SSM_EXEC_TIMEOUT_SECONDS} (delivery --timeout-seconds=${SSM_DELIVERY_TIMEOUT_SECONDS})"
else
  jq -n --rawfile script "${REMOTE_SCRIPT}" '{commands: [$script]}' >"${PARAMS_JSON}"
  echo "Parameters: NO executionTimeout (document default 3600s) — expect ExecutionTimedOut if SLEEP_SEC>3600"
fi

SEND_ARGS=(
  aws ssm send-command
  --region "${REGION}"
  --document-name "AWS-RunShellScript"
  --comment "ssm_timeout_sleep_test sleep=${SLEEP_SEC}s exec=${SSM_EXEC_TIMEOUT_SECONDS} long_verify=${SSM_LONG_VERIFY:-0}"
  --timeout-seconds "${SSM_DELIVERY_TIMEOUT_SECONDS}"
  --instance-ids "${INSTANCE_ID}"
  --parameters "file://${PARAMS_JSON}"
  --query "Command.CommandId"
  --output text
)

if [[ -n "${SSM_OUTPUT_S3_BUCKET:-}" && -n "${SSM_OUTPUT_S3_KEY_PREFIX:-}" ]]; then
  SEND_ARGS+=(--output-s3-bucket-name "${SSM_OUTPUT_S3_BUCKET}" --output-s3-key-prefix "${SSM_OUTPUT_S3_KEY_PREFIX}")
  echo "S3 output: s3://${SSM_OUTPUT_S3_BUCKET}/${SSM_OUTPUT_S3_KEY_PREFIX}"
fi

if [[ -n "${SSM_CLOUDWATCH_LOG_GROUP:-}" ]]; then
  SEND_ARGS+=(--cloud-watch-output-config "CloudWatchLogGroupName=${SSM_CLOUDWATCH_LOG_GROUP},CloudWatchOutputEnabled=true")
  echo "CloudWatch log group: ${SSM_CLOUDWATCH_LOG_GROUP}"
fi

CMD_ID=$(AWS_PROFILE="${AWS_PROFILE}" "${SEND_ARGS[@]}")

echo "Command ID: ${CMD_ID}"
STATE_FILE="${SSM_TIMEOUT_TEST_STATE:-/tmp/ssm_timeout_sleep_test_last.env}"
{
  echo "export CMD_ID=${CMD_ID}"
  echo "export INSTANCE_ID=${INSTANCE_ID}"
  echo "export REGION=${REGION}"
  echo "export AWS_PROFILE=${AWS_PROFILE}"
} >"${STATE_FILE}"
echo "Wrote ${STATE_FILE} (source it to poll)"

echo "Poll: CMD_ID=${CMD_ID} INSTANCE_ID=${INSTANCE_ID} ./scripts/aws_commands/legacy/check_ssm_status.sh"
echo "Wait: CMD_ID=${CMD_ID} INSTANCE_ID=${INSTANCE_ID} ./scripts/aws_commands/ssm_timeout_wait_for_command.sh"
