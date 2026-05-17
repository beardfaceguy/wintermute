#!/usr/bin/env bash
# Regression tests for dixie_s3_code_uri.sh (S3_CODE_URI guard before sync --delete).

set -uo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${THIS_DIR}/../.." && pwd)"
LIB="${REPO_ROOT}/model_training/titanProject/scripts/lib/dixie_s3_code_uri.sh"

# shellcheck source=tests/aws_tooling/test_helpers.sh
# shellcheck disable=SC1091
source "${THIS_DIR}/test_helpers.sh"

# shellcheck source=model_training/titanProject/scripts/lib/dixie_s3_code_uri.sh
# shellcheck disable=SC1091
source "${LIB}"

test_valid_per_run_uri_accepted() {
  dixie_validate_s3_code_uri_for_delete_sync "s3://b/titan/code/dixie_20260115120000" \
    || __fail "expected default uri pattern to pass"
  dixie_validate_s3_code_uri_for_delete_sync "s3://bucket/prefix/dixie_20260115120000/trailing" \
    || __fail "expected segment + trailing slash path to pass"
}

test_nonstandard_uri_rejected() {
  if dixie_validate_s3_code_uri_for_delete_sync "s3://b/titan/code/latest" 2>/dev/null; then
    __fail "shared prefix without dixie_ ts must be rejected"
  fi
  if dixie_validate_s3_code_uri_for_delete_sync "s3://b/prefix/my_dixie_20260115120000" 2>/dev/null; then
    __fail "suffix my_dixie_... must not satisfy segment guard"
  fi
  if dixie_validate_s3_code_uri_for_delete_sync "s3://b/dixie_20260115120000extra" 2>/dev/null; then
    __fail "timestamp must be exactly 14 digits as a segment boundary"
  fi
}

test_short_timestamp_rejected() {
  if dixie_validate_s3_code_uri_for_delete_sync "s3://b/dixie_2026011512000" 2>/dev/null; then
    __fail "13-digit suffix must not pass"
  fi
}

test_override_env_allows_nonstandard() {
  DIXIE_ALLOW_NONSTANDARD_S3_CODE_URI=1 \
    dixie_validate_s3_code_uri_for_delete_sync "s3://b/titan/code/latest" \
    || __fail "override must allow nonstandard uri"
}

# ---------------------------------------------------------------------------
if [[ ! -r "${LIB}" ]]; then
  echo "ERROR: lib not found at ${LIB}" >&2
  exit 2
fi

run_tests \
  test_valid_per_run_uri_accepted \
  test_nonstandard_uri_rejected \
  test_short_timestamp_rejected \
  test_override_env_allows_nonstandard
