#!/usr/bin/env bash
# Unit tests for model_training/titanProject/scripts/lib/aws_lifecycle.sh.
#
# These tests verify the safety-critical property of the cleanup trap:
#
#   ALL pre-terminate hooks complete (success OR failure) BEFORE any
#   terminate-instance / shutdown call is made.
#
# A regression here would either:
#   (a) terminate the instance before logs/checkpoints are uploaded —
#       LOST DATA, expensive, hard to redo, or
#   (b) skip the terminate when a hook fails — IDLE BILLING, the bug we
#       hit during the 2026-05-13 A/B run.
#
# Both classes of bug are covered below.

set -uo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${THIS_DIR}/../.." && pwd)"
LIB="${REPO_ROOT}/model_training/titanProject/scripts/lib/aws_lifecycle.sh"

# shellcheck source=tests/aws_tooling/test_helpers.sh
# shellcheck disable=SC1091
source "${THIS_DIR}/test_helpers.sh"

# ---------------------------------------------------------------------------
# Mock builders specific to aws_lifecycle.
# ---------------------------------------------------------------------------

# Mock `aws` that handles both `s3 cp` and `ec2 terminate-instances`.
# Behaviour:
#   - `aws s3 cp ...`              -> rc=$AWS_S3_CP_RC (default 0), echo "[mock-aws-s3-cp]"
#   - `aws ec2 terminate-instances`-> rc=$AWS_TERMINATE_RC (default 0), echo "[mock-aws-terminate]"
write_aws_mock() {
  local s3_rc="${1:-0}" terminate_rc="${2:-0}"
  write_mock "aws" "
case \"\$1 \$2\" in
  's3 cp')
    echo '[mock-aws-s3-cp]'
    exit ${s3_rc}
    ;;
  'ec2 terminate-instances')
    echo '[mock-aws-terminate]'
    exit ${terminate_rc}
    ;;
  *)
    echo \"unhandled aws cmd: \$*\" >&2
    exit 99
    ;;
esac
"
}

# Mock `shutdown` — always succeeds, just records.
write_shutdown_mock() {
  write_mock "shutdown" '
echo "[mock-shutdown]"
exit 0
'
}

# Mock `curl` for IMDSv2. The lib hits IMDSv2 with two flavors:
#   PUT  /latest/api/token
#   GET  /latest/meta-data/<field>
# Optional env: IMDS_TOKEN_RC, IMDS_FIELD_RC, IMDS_FIELD_VALUE.
write_imds_curl_mock() {
  local token_rc="${1:-0}" field_rc="${2:-0}" field_value="${3:-i-mocked-1234}"
  write_mock "curl" "
# Decide whether this is a token PUT, a field GET, or something else.
is_token_put=0
is_field_get=0
url=\"\"
field_path=\"\"
for arg in \"\$@\"; do
  if [[ \"\${arg}\" == *'/latest/api/token' ]]; then
    is_token_put=1
    url=\"\${arg}\"
  elif [[ \"\${arg}\" == *'/latest/meta-data/'* ]]; then
    is_field_get=1
    url=\"\${arg}\"
    field_path=\"\${arg#*/latest/meta-data/}\"
  fi
done
if (( is_token_put )); then
  if [[ '${token_rc}' != '0' ]]; then exit ${token_rc}; fi
  echo 'mock-imds-token'
  exit 0
fi
if (( is_field_get )); then
  if [[ '${field_rc}' != '0' ]]; then exit ${field_rc}; fi
  case \"\${field_path}\" in
    'instance-id')      echo '${field_value}' ;;
    'placement/region') echo 'us-mock-1' ;;
    *)                  echo '${field_value}' ;;
  esac
  exit 0
fi
echo 'unhandled curl' >&2
exit 99
"
}

# Run the lib's cleanup trap inside a fresh subshell with a controlled
# environment. The provided <hook_body> defines the pre-terminate hook.
# Returns the subshell rc.
#
# Args:
#   hook_body         bash code that runs INSIDE the hook function
#   pre_set_instance  if non-empty, pre-set INSTANCE_ID="${pre_set_instance}"
#                     so resolve_instance_metadata is a no-op
#   exit_rc           rc to exit the subshell with (default 0)
run_cleanup_subshell() {
  local hook_body="$1"
  local pre_set_instance="${2:-}"
  local exit_rc="${3:-0}"

  # Build the subshell script.
  local script="${MOCK_STATE_DIR}/run_cleanup.sh"
  cat > "${script}" <<EOF
#!/usr/bin/env bash
set -uo pipefail

export AWS_BIN=aws
export SHUTDOWN_BIN=shutdown
export CLOUD_INIT_INSTANCE_ID_PATH=/nonexistent-cloud-init
export IMDS_BASE_URL=http://imds-mock.invalid
export IMDS_TIMEOUT_SEC=1
export IMDS_RETRY_ATTEMPTS=1

source "${LIB}"

INSTANCE_ID="${pre_set_instance}"
REGION="us-mock-1"

my_hook() {
  ${hook_body}
}
CLEANUP_PRE_TERMINATE_HOOKS+=(my_hook)
install_cleanup_trap

exit ${exit_rc}
EOF
  chmod +x "${script}"
  bash "${script}" 2>&1
}

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

test_order_on_success_hook_runs_before_terminate() {
  # Property: artifact upload (hook) MUST complete before terminate-instances.
  setup_mocks
  write_aws_mock 0 0   # s3 ok, terminate ok
  write_shutdown_mock
  write_imds_curl_mock # not used here since INSTANCE_ID is preset

  run_cleanup_subshell \
    'echo "[hook] uploading artifact"; aws s3 cp /tmp/fake s3://b/x; return 0' \
    "i-pre-set-aaa" \
    0 >/dev/null 2>&1
  rc=$?

  assert_rc_eq 0 "${rc}" "happy-path trap must exit 0"
  assert_calls_in_order "success-path order" \
    '^aws s3 cp ' \
    '^aws ec2 terminate-instances'
  shutdown_count=$(grep -c '^shutdown' "${MOCK_LOG}" || true)
  assert_eq 0 "${shutdown_count}" "no shutdown fallback when terminate succeeds"
  teardown_mocks
}

test_order_on_failure_hook_runs_before_terminate() {
  # Property: same ordering when the script exits with rc!=0, AND the
  # trap must propagate that rc (so callers / SSM see the real failure
  # code, not a swallowed 0 from terminate_self).
  setup_mocks
  write_aws_mock 0 0
  write_shutdown_mock
  write_imds_curl_mock

  run_cleanup_subshell \
    'aws s3 cp /tmp/fake s3://b/x; return 0' \
    "i-pre-set-bbb" \
    7 >/dev/null 2>&1
  rc=$?

  assert_rc_eq 7 "${rc}" "trap must propagate the script's exit rc"
  assert_calls_in_order "failure-path order" \
    '^aws s3 cp ' \
    '^aws ec2 terminate-instances'
  teardown_mocks
}

test_terminate_succeeds_does_not_call_shutdown() {
  # Property: when the terminate-instances API succeeds, we must NOT also
  # invoke the `shutdown -h` fallback (it'd be wasted shell ops, but more
  # importantly it'd interfere with the orderly EC2 termination).
  setup_mocks
  write_aws_mock 0 0
  write_shutdown_mock
  write_imds_curl_mock

  run_cleanup_subshell 'return 0' "i-pre-ccc" 0 >/dev/null 2>&1

  shutdown_count=$(grep -c '^shutdown' "${MOCK_LOG}" || true)
  assert_eq 0 "${shutdown_count}" "shutdown should NOT be called when terminate API succeeds"

  terminate_count=$(grep -c '^aws ec2 terminate-instances' "${MOCK_LOG}" || true)
  assert_eq 1 "${terminate_count}" "terminate-instances called exactly once"
  teardown_mocks
}

test_terminate_api_failure_falls_back_to_shutdown() {
  # Property: if `aws ec2 terminate-instances` returns rc!=0, we MUST fall
  # back to `shutdown -h now`. Stranding a billing instance is the expensive
  # failure mode, not a cosmetic one.
  setup_mocks
  write_aws_mock 0 7   # s3 ok, terminate FAILS rc=7
  write_shutdown_mock
  write_imds_curl_mock

  run_cleanup_subshell 'return 0' "i-pre-ddd" 0 >/dev/null 2>&1

  shutdown_count=$(grep -c '^shutdown' "${MOCK_LOG}" || true)
  assert_eq 1 "${shutdown_count}" "shutdown -h fallback fires when terminate API fails"
  teardown_mocks
}

test_no_instance_id_falls_back_to_shutdown() {
  # Property: if INSTANCE_ID can't be resolved (IMDSv2 + cloud-init both
  # missing), terminate_self must fall back to `shutdown -h now`. This is
  # the path the OLD runner failed silently — it just exited rc=0 and
  # left the box running.
  setup_mocks
  write_aws_mock 0 0
  write_shutdown_mock
  # IMDSv2 mock that fails BOTH token and field calls -> empty result.
  write_imds_curl_mock 1 1 ""

  run_cleanup_subshell 'return 0' "" 0 >/dev/null 2>&1

  # No terminate-instances API call (we have no instance id).
  terminate_count=$(grep -c '^aws ec2 terminate-instances' "${MOCK_LOG}" || true)
  assert_eq 0 "${terminate_count}" "no terminate-instances when ID unresolved"

  # MUST fall through to shutdown.
  shutdown_count=$(grep -c '^shutdown' "${MOCK_LOG}" || true)
  assert_eq 1 "${shutdown_count}" "shutdown -h fallback fires when ID unresolved"
  teardown_mocks
}

test_hook_failure_does_not_block_termination() {
  # Property: a hook returning rc!=0 (e.g. transient S3 outage) MUST NOT
  # block subsequent hooks OR termination. Lose the log; don't strand
  # the instance.
  setup_mocks
  write_aws_mock 0 0
  write_shutdown_mock
  write_imds_curl_mock

  run_cleanup_subshell 'aws s3 cp /tmp/fake s3://b/x; return 42' "i-pre-eee" 0 >/dev/null 2>&1

  # Hook still ran (we see the s3 cp) AND terminate-instances ran after.
  assert_calls_in_order "hook-fail order" \
    '^aws s3 cp ' \
    '^aws ec2 terminate-instances'
  teardown_mocks
}

test_imds_to_cloud_init_fallback_resolves_instance_id() {
  # Property: when IMDSv2 is unreachable but /var/lib/cloud/data/instance-id
  # exists, the cleanup trap must use the cloud-init id and successfully
  # call terminate-instances against it.
  setup_mocks
  write_aws_mock 0 0
  write_shutdown_mock
  # IMDSv2 fails entirely.
  write_imds_curl_mock 1 1 ""

  cloud_init_path="${MOCK_STATE_DIR}/cloud_init_id"
  echo "i-from-cloud-init" > "${cloud_init_path}"

  # Run with a hand-rolled subshell so we can override CLOUD_INIT_INSTANCE_ID_PATH.
  script="${MOCK_STATE_DIR}/run_cloud_init.sh"
  cat > "${script}" <<EOF
#!/usr/bin/env bash
set -uo pipefail

export AWS_BIN=aws
export SHUTDOWN_BIN=shutdown
export CLOUD_INIT_INSTANCE_ID_PATH=${cloud_init_path}
export IMDS_BASE_URL=http://imds-mock.invalid
export IMDS_TIMEOUT_SEC=1
export IMDS_RETRY_ATTEMPTS=1

source "${LIB}"

INSTANCE_ID=""
my_hook() { return 0; }
CLEANUP_PRE_TERMINATE_HOOKS+=(my_hook)
install_cleanup_trap
EOF
  chmod +x "${script}"
  bash "${script}" >/dev/null 2>&1

  # The terminate call must reference the cloud-init instance id.
  assert_file_contains "${MOCK_LOG}" "ec2 terminate-instances --instance-ids i-from-cloud-init" \
    "terminate uses cloud-init id"
  teardown_mocks
}

# ---------------------------------------------------------------------------
# Run.
# ---------------------------------------------------------------------------
if [[ ! -r "${LIB}" ]]; then
  echo "ERROR: lib not found at ${LIB}" >&2
  exit 2
fi

run_tests \
  test_order_on_success_hook_runs_before_terminate \
  test_order_on_failure_hook_runs_before_terminate \
  test_terminate_succeeds_does_not_call_shutdown \
  test_terminate_api_failure_falls_back_to_shutdown \
  test_no_instance_id_falls_back_to_shutdown \
  test_hook_failure_does_not_block_termination \
  test_imds_to_cloud_init_fallback_resolves_instance_id
