#!/usr/bin/env bash
# Reusable AWS instance lifecycle helpers for Wintermute runners.
#
# Designed to be `source`d at the top of a bash runner script (e.g.
# scripts/run_titan_arms_ssm.sh). Provides:
#
#   get_imds_field <field>          Echo an IMDSv2 metadata field, with retries.
#                                   Returns rc=1 (and empty stdout) on total
#                                   failure.
#
#   resolve_instance_metadata        Populate globals INSTANCE_ID and REGION
#                                   from IMDSv2 -> /var/lib/cloud/data/instance-id
#                                   -> empty (caller decides what to do).
#                                   Idempotent — safe to call multiple times.
#
#   terminate_self                  Terminate the EC2 instance whose ID is in
#                                   INSTANCE_ID, falling back to `shutdown -h`
#                                   if the API call fails or INSTANCE_ID is
#                                   empty. Always tries hard to actually power
#                                   the box off so we never strand a spot
#                                   instance billing while idle.
#
#   install_cleanup_trap            Install an EXIT trap that runs every
#                                   function name in CLEANUP_PRE_TERMINATE_HOOKS
#                                   (in registration order) and THEN calls
#                                   terminate_self.
#
# Caller responsibilities (set BEFORE sourcing this lib or before calling
# install_cleanup_trap):
#   - CLEANUP_PRE_TERMINATE_HOOKS  Bash array of function names. Each is
#                                  called in order during the EXIT trap. Use
#                                  these to upload artifacts (logs,
#                                  checkpoints, results) to S3 BEFORE the
#                                  instance is torn down.
#
# Order-of-operations safety property (this is the whole point of the lib):
#
#   ALL pre-terminate hooks run to completion (success OR failure) BEFORE
#   any termination/shutdown call is made. A hook returning rc!=0 logs a
#   warning but does NOT block the next hook or termination — we never want
#   a flaky log-upload to leave the instance running and burning money.
#
# Environment / overrides (mostly for tests):
#   IMDS_BASE_URL                  Default http://169.254.169.254. Override
#                                  in tests to point at a stub HTTP server.
#   IMDS_TOKEN_TTL_SEC             Default 600.
#   IMDS_RETRY_ATTEMPTS            Default 3.
#   IMDS_TIMEOUT_SEC               Default 3.
#   CLOUD_INIT_INSTANCE_ID_PATH    Default /var/lib/cloud/data/instance-id.
#   AWS_BIN                        Default 'aws'. Override to mock in tests.
#   SHUTDOWN_BIN                   Default 'shutdown'. Override to mock in tests.
#
# This file should NOT use `set -euo pipefail` itself — that's the runner's
# decision. We do use `local`, defensive defaults, and explicit rc handling.

# ---------------------------------------------------------------------------
# Configuration with safe defaults.
# ---------------------------------------------------------------------------
: "${IMDS_BASE_URL:=http://169.254.169.254}"
: "${IMDS_TOKEN_TTL_SEC:=600}"
: "${IMDS_RETRY_ATTEMPTS:=3}"
: "${IMDS_TIMEOUT_SEC:=3}"
: "${CLOUD_INIT_INSTANCE_ID_PATH:=/var/lib/cloud/data/instance-id}"
: "${AWS_BIN:=aws}"
: "${SHUTDOWN_BIN:=shutdown}"

# Globals populated by resolve_instance_metadata.
INSTANCE_ID="${INSTANCE_ID:-}"
REGION="${REGION:-us-east-1}"

# Caller-registered pre-terminate hooks.
# Bash arrays inside `:-` are tricky; declare empty if unset.
if ! declare -p CLEANUP_PRE_TERMINATE_HOOKS >/dev/null 2>&1; then
  CLEANUP_PRE_TERMINATE_HOOKS=()
fi

# ---------------------------------------------------------------------------
# get_imds_field <field>
# ---------------------------------------------------------------------------
get_imds_field() {
  local field="$1" token result
  local _attempt
  for _attempt in $(seq 1 "${IMDS_RETRY_ATTEMPTS}"); do
    token="$(curl -fsS --max-time "${IMDS_TIMEOUT_SEC}" \
      -X PUT "${IMDS_BASE_URL}/latest/api/token" \
      -H "X-aws-ec2-metadata-token-ttl-seconds: ${IMDS_TOKEN_TTL_SEC}" \
      2>/dev/null || true)"
    if [[ -n "${token}" ]]; then
      result="$(curl -fsS --max-time "${IMDS_TIMEOUT_SEC}" \
        -H "X-aws-ec2-metadata-token: ${token}" \
        "${IMDS_BASE_URL}/latest/meta-data/${field}" \
        2>/dev/null || true)"
      if [[ -n "${result}" ]]; then
        echo "${result}"
        return 0
      fi
    fi
    sleep 1
  done
  return 1
}

# ---------------------------------------------------------------------------
# resolve_instance_metadata
#
# Populates globals INSTANCE_ID and REGION. Sources, in order:
#   1. IMDSv2 (preferred — works on EC2 with IMDSv2 enforced)
#   2. /var/lib/cloud/data/instance-id (cloud-init writes this on first boot,
#      survives if IMDSv2 is unreachable)
#   3. (no-op) leaves INSTANCE_ID="" — caller / terminate_self will fall
#      back to `shutdown -h now`
#
# Idempotent: skips work if INSTANCE_ID is already set.
# ---------------------------------------------------------------------------
resolve_instance_metadata() {
  if [[ -n "${INSTANCE_ID}" ]]; then
    return 0
  fi

  local imds_id imds_region
  imds_id="$(get_imds_field instance-id || true)"
  if [[ -n "${imds_id}" ]]; then
    INSTANCE_ID="${imds_id}"
  elif [[ -r "${CLOUD_INIT_INSTANCE_ID_PATH}" ]]; then
    INSTANCE_ID="$(cat "${CLOUD_INIT_INSTANCE_ID_PATH}")"
    echo "[aws_lifecycle] IMDSv2 unreachable; using cloud-init instance-id=${INSTANCE_ID}" >&2
  fi

  imds_region="$(get_imds_field placement/region || true)"
  if [[ -n "${imds_region}" ]]; then
    REGION="${imds_region}"
  fi
  # If IMDSv2 region failed, REGION keeps its prior value (default us-east-1
  # or whatever the caller set).

  echo "[aws_lifecycle] resolved instance_id='${INSTANCE_ID}' region='${REGION}'"
}

# ---------------------------------------------------------------------------
# terminate_self
#
# Tries `aws ec2 terminate-instances` first; falls back to `shutdown -h now`
# if that fails or INSTANCE_ID is empty. The fallback is critical: we'd
# rather force a hard power-off (and let EBS detach normally) than leave
# the instance running and burning spot $/hr.
# ---------------------------------------------------------------------------
terminate_self() {
  if [[ -n "${INSTANCE_ID}" ]]; then
    echo "[aws_lifecycle] terminating ${INSTANCE_ID} in ${REGION}"
    if "${AWS_BIN}" ec2 terminate-instances \
        --instance-ids "${INSTANCE_ID}" \
        --region "${REGION}"; then
      return 0
    fi
    echo "[aws_lifecycle] terminate-instances API call failed; falling back to shutdown -h" >&2
  else
    echo "[aws_lifecycle] WARNING: instance-id unresolved; falling back to shutdown -h" >&2
  fi
  sync
  sleep 2
  "${SHUTDOWN_BIN}" -h now || /sbin/shutdown -h now || true
}

# ---------------------------------------------------------------------------
# install_cleanup_trap
#
# Installs an EXIT trap that:
#   1. Records the script's rc.
#   2. Resolves instance metadata if not already done.
#   3. Runs every function in CLEANUP_PRE_TERMINATE_HOOKS in order. A hook
#      returning rc!=0 logs a warning but does NOT abort the rest — we
#      always want to get to terminate_self.
#   4. Calls terminate_self.
#   5. Exits with the original rc.
# ---------------------------------------------------------------------------
install_cleanup_trap() {
  # shellcheck disable=SC2317  # called via trap, not directly
  __aws_lifecycle_cleanup() {
    local rc=$?
    echo "[aws_lifecycle] EXIT trap firing with rc=${rc} at $(date -Iseconds)"
    resolve_instance_metadata

    local hook
    for hook in "${CLEANUP_PRE_TERMINATE_HOOKS[@]:-}"; do
      [[ -z "${hook}" ]] && continue
      echo "[aws_lifecycle] running pre-terminate hook: ${hook}"
      if ! "${hook}"; then
        echo "[aws_lifecycle] WARN: hook '${hook}' returned non-zero — continuing anyway" >&2
      fi
    done

    terminate_self
    exit "${rc}"
  }
  trap __aws_lifecycle_cleanup EXIT
}
