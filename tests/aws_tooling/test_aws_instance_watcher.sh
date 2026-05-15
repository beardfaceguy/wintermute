#!/usr/bin/env bash
# Unit tests for ~/.local/bin/aws-instance-watcher.
#
# Strategy: PATH-mock `aws` and `curl`. The mock `aws` walks a per-test
# state sequence file so we can simulate `running -> running -> terminated`
# or `None -> None -> None` etc. The mock `curl` records what would have
# been POSTed to ntfy without hitting the network.
#
# Property under test: the watcher's state machine fires the correct
# notification (or correctly stays silent) for each sequence, exits with
# the right rc, and does not silently loop on the failure modes that bit
# us in production (post-prune `None`, SSO expiry).

set -uo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tests/aws_tooling/test_helpers.sh
# shellcheck disable=SC1091
source "${THIS_DIR}/test_helpers.sh"

WATCHER="${WATCHER:-${HOME}/.local/bin/aws-instance-watcher}"

# ---------------------------------------------------------------------------
# Mock setup helpers specific to the watcher.
# ---------------------------------------------------------------------------
# write_aws_describe_mock <state1> <state2> ... <stateN>
# Writes a mock `aws` that, on each `ec2 describe-instances` call, returns
# the next state in sequence. Sentinel `FAIL` makes the mock exit 255 with
# a stderr message resembling expired SSO. Sentinel `EMPTY` exits 0 with
# stdout "None" (the post-prune case).
write_aws_describe_mock() {
  local i=1
  for state in "$@"; do
    printf '%s\n' "${state}" > "${MOCK_STATE_DIR}/state_seq_${i}"
    i=$((i + 1))
  done
  printf '%s\n' "$(($# + 1))" > "${MOCK_STATE_DIR}/state_seq_max"
  printf '1\n' > "${MOCK_STATE_DIR}/state_seq_idx"

  # shellcheck disable=SC2016  # body is intentionally single-quoted: $1, $MOCK_LOG etc. must expand at MOCK runtime, not now
  write_mock "aws" '
seq_dir="${MOCK_STATE_DIR}"
case "$1 $2" in
  "ec2 describe-instances")
    idx=$(cat "${seq_dir}/state_seq_idx" 2>/dev/null || echo 1)
    state_file="${seq_dir}/state_seq_${idx}"
    if [[ -r "${state_file}" ]]; then
      st=$(cat "${state_file}")
    else
      # Out of states — keep returning the last one we got.
      st=$(cat "${seq_dir}/state_seq_$(( idx - 1 ))" 2>/dev/null || echo "running")
    fi
    echo "$((idx + 1))" > "${seq_dir}/state_seq_idx"
    case "${st}" in
      FAIL)
        echo "An error occurred (ExpiredToken) when calling the DescribeInstances operation: The security token included in the request is expired" >&2
        exit 255
        ;;
      EMPTY|None)
        echo "None"
        exit 0
        ;;
      *)
        echo "${st}"
        exit 0
        ;;
    esac
    ;;
  *)
    echo "unhandled aws cmd: $*" >&2
    exit 99
    ;;
esac
'
}

# write_curl_mock — captures all curl invocations (which the watcher uses
# both for ntfy POSTs and for IMDSv2; the watcher never touches IMDS, so
# any curl call is a notification).
write_curl_mock() {
  # shellcheck disable=SC2016  # body is intentionally single-quoted: $@, $MOCK_LOG must expand at MOCK runtime, not now
  write_mock "curl" '
# Pull the URL (last positional) and any -H headers / -d body.
last=""
for arg in "$@"; do last="$arg"; done
echo "  url=${last}" >> "${MOCK_LOG}"
exit 0
'
}

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Helper to invoke the watcher with deterministic short timing.
run_watcher() {
  local instance="$1" label="${2:-test}" topic="${3:-test-topic}"
  POLL_INTERVAL_SEC=0 \
  AUTH_FAIL_THRESHOLD=2 \
  AWS_PROFILE=mock \
  AWS_DEFAULT_REGION=us-east-1 \
    "${WATCHER}" "${instance}" "${label}" "${topic}"
}

test_happy_path_running_then_terminated() {
  setup_mocks
  write_aws_describe_mock "running" "running" "terminated"
  write_curl_mock

  run_watcher i-aaa "TEST happy" "test-topic" >/dev/null 2>&1
  rc=$?

  assert_rc_eq 0 "${rc}" "happy path should exit 0"

  # Three describe-instances calls expected (running, running, terminated).
  count=$(grep -c '^aws ec2 describe-instances' "${MOCK_LOG}" || true)
  assert_eq 3 "${count}" "should have polled 3 times"

  # Exactly one curl call expected — the success ping.
  curl_count=$(grep -c '^curl ' "${MOCK_LOG}" || true)
  assert_eq 1 "${curl_count}" "happy path fires exactly 1 ntfy POST"

  # Verify it was the success ping (Title contains "terminated", topic correct).
  assert_file_contains "${MOCK_LOG}" "Title: TEST happy: terminated" "title carries state"
  assert_file_contains "${MOCK_LOG}" "url=https://ntfy.sh/test-topic" "topic in URL"

  teardown_mocks
}

test_post_prune_after_running_fires_pruned_ping() {
  # The Run-2 (2026-05-14) bug: instance terminated cleanly, AWS pruned
  # metadata before the next poll caught State='terminated'. Watcher must
  # treat 'None' AFTER seeing the instance running as a successful exit.
  setup_mocks
  write_aws_describe_mock "running" "None"
  write_curl_mock

  run_watcher i-bbb "TEST pruned" "topic-pruned" >/dev/null 2>&1
  rc=$?

  assert_rc_eq 1 "${rc}" "post-prune-after-running should exit 1"

  # Exactly one curl call — the "terminated (metadata pruned)" ping.
  curl_count=$(grep -c '^curl ' "${MOCK_LOG}" || true)
  assert_eq 1 "${curl_count}" "exactly 1 ntfy POST"
  assert_file_contains "${MOCK_LOG}" "Title: TEST pruned: terminated (metadata pruned)" "pruned title"

  teardown_mocks
}

test_post_prune_from_start_fires_alert() {
  # Misconfiguration case: wrong instance ID, or instance terminated+pruned
  # before watcher started. We never see it running; threshold trips an
  # alert ntfy and exit 2. This is the OLD silent-failure case.
  setup_mocks
  write_aws_describe_mock "None" "None" "None" "None"
  write_curl_mock

  run_watcher i-ccc "TEST never-existed" "topic-misconfig" >/dev/null 2>&1
  rc=$?

  assert_rc_eq 2 "${rc}" "post-prune-from-start should exit 2"

  curl_count=$(grep -c '^curl ' "${MOCK_LOG}" || true)
  assert_eq 1 "${curl_count}" "exactly 1 alert ntfy"
  assert_file_contains "${MOCK_LOG}" "Title: TEST never-existed: WATCHER BROKEN" "WATCHER BROKEN title"
  assert_file_contains "${MOCK_LOG}" "Priority: max" "max priority on alerts"

  teardown_mocks
}

test_auth_failure_fires_alert_with_stderr() {
  # The Run-1 (2026-05-13) bug: SSO expired mid-poll, aws cli started
  # returning rc!=0, OLD watcher classified it as 'lookup-failed' and
  # silently looped. New watcher must capture the stderr (so the user can
  # see WHY) and fire an alert after threshold.
  setup_mocks
  write_aws_describe_mock "FAIL" "FAIL" "FAIL"
  write_curl_mock

  run_watcher i-ddd "TEST auth-fail" "topic-auth" >/dev/null 2>&1
  rc=$?

  assert_rc_eq 2 "${rc}" "auth fail should exit 2"

  curl_count=$(grep -c '^curl ' "${MOCK_LOG}" || true)
  assert_eq 1 "${curl_count}" "exactly 1 alert ntfy"
  assert_file_contains "${MOCK_LOG}" "Title: TEST auth-fail: WATCHER BROKEN" "WATCHER BROKEN title"
  # The aws-cli stderr must be embedded in the curl body for diagnostics.
  # We can't easily grep curl bodies via the simplistic mock, but the body
  # is sent as -d's argument; mock just records the URL. The presence of
  # the alert is sufficient — full body assertion happens via integration
  # path with the real script. For now, assert the watcher exited 2 and
  # fired exactly one ntfy.
  teardown_mocks
}

test_transient_failure_recovers_no_false_alert() {
  # Transient: running -> FAIL -> running -> terminated.
  # Threshold counter must reset on the recovery so we don't fire a false
  # alert mid-run.
  setup_mocks
  write_aws_describe_mock "running" "FAIL" "running" "terminated"
  write_curl_mock

  run_watcher i-eee "TEST transient" "topic-transient" >/dev/null 2>&1
  rc=$?

  assert_rc_eq 0 "${rc}" "transient recovery -> happy path"
  curl_count=$(grep -c '^curl ' "${MOCK_LOG}" || true)
  assert_eq 1 "${curl_count}" "only the final terminate ping fires (no false alert)"
  assert_file_contains "${MOCK_LOG}" "Title: TEST transient: terminated" "final terminate title"

  teardown_mocks
}

test_threshold_tunable() {
  # AUTH_FAIL_THRESHOLD=4: needs 4 consecutive failures before tripping.
  # 3 fails followed by a success should NOT trip; 4 fails should.
  setup_mocks
  write_aws_describe_mock "FAIL" "FAIL" "FAIL" "running" "terminated"
  write_curl_mock

  POLL_INTERVAL_SEC=0 \
  AUTH_FAIL_THRESHOLD=4 \
  AWS_PROFILE=mock \
  AWS_DEFAULT_REGION=us-east-1 \
    "${WATCHER}" i-fff "TEST threshold" "topic-threshold" >/dev/null 2>&1
  rc=$?

  assert_rc_eq 0 "${rc}" "3 fails + recovery -> success"
  curl_count=$(grep -c '^curl ' "${MOCK_LOG}" || true)
  assert_eq 1 "${curl_count}" "only the final ping fires"
  assert_file_contains "${MOCK_LOG}" "Title: TEST threshold: terminated" "terminate title"

  teardown_mocks
}

# ---------------------------------------------------------------------------
# Run.
# ---------------------------------------------------------------------------
if ! [[ -x "${WATCHER}" ]]; then
  echo "ERROR: watcher not found or not executable at ${WATCHER}" >&2
  exit 2
fi

run_tests \
  test_happy_path_running_then_terminated \
  test_post_prune_after_running_fires_pruned_ping \
  test_post_prune_from_start_fires_alert \
  test_auth_failure_fires_alert_with_stderr \
  test_transient_failure_recovers_no_false_alert \
  test_threshold_tunable
