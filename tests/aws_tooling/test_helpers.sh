#!/usr/bin/env bash
# Tiny shell test framework for the aws_tooling test suite.
#
# Usage in a test file:
#   source "$(dirname "$0")/test_helpers.sh"
#
#   test_thing_one() {
#     setup_mocks
#     # ... arrange ...
#     # ... act ...
#     assert_eq "expected" "$actual" "thing one stdout"
#     teardown_mocks
#   }
#
#   run_tests test_thing_one test_thing_two
#
# Provides:
#   setup_mocks                Create $MOCK_BIN dir, prepend to PATH, point
#                              $MOCK_LOG at a fresh log file. Sets $TEST_TMP.
#   teardown_mocks             Clean up.
#   write_mock <name> <body>   Write a mock binary to $MOCK_BIN that runs
#                              <body>. Body has access to $MOCK_LOG.
#   assert_eq                  expected, actual, label
#   assert_ne                  not_expected, actual, label
#   assert_contains            haystack, needle, label
#   assert_not_contains        haystack, needle, label
#   assert_file_exists         path, label
#   assert_file_contains       path, needle, label
#   assert_rc_eq               expected_rc, actual_rc, label
#   assert_calls_in_order      label, call1, call2, ...
#                              Each call is a regex; asserts MOCK_LOG
#                              contains lines matching call1..callN in order
#                              (other lines may be interleaved).
#   run_tests <funcname>...    Run each test function in a fresh subshell.
#                              Reports per-test pass/fail and aggregate.
#
# Test functions return rc=0 on success, rc=1 on assertion failure (the
# assert helpers exit 1 on first failure, so a test fn either falls
# through to the end with rc=0 or aborts at the failure).

set -uo pipefail

# ---------------------------------------------------------------------------
# Mock harness
# ---------------------------------------------------------------------------
setup_mocks() {
  TEST_TMP="$(mktemp -d -t aws_tooling_test.XXXXXX)"
  MOCK_BIN="${TEST_TMP}/mock_bin"
  MOCK_LOG="${TEST_TMP}/mock_calls.log"
  MOCK_STATE_DIR="${TEST_TMP}/mock_state"
  mkdir -p "${MOCK_BIN}" "${MOCK_STATE_DIR}"
  : > "${MOCK_LOG}"

  # Save original PATH so teardown can restore it. Each test gets its own
  # mock dir at the front of PATH.
  __ORIG_PATH="${PATH}"
  export PATH="${MOCK_BIN}:${PATH}"
  export MOCK_LOG MOCK_STATE_DIR
}

teardown_mocks() {
  export PATH="${__ORIG_PATH:-${PATH}}"
  if [[ -n "${TEST_TMP:-}" && -d "${TEST_TMP}" ]]; then
    rm -rf "${TEST_TMP}"
  fi
  unset TEST_TMP MOCK_BIN MOCK_LOG MOCK_STATE_DIR __ORIG_PATH
}

# write_mock <name> <body>
# Writes a #!/usr/bin/env bash script at $MOCK_BIN/<name>. The body should
# log all argv lines via 'log_call' (provided in the prelude), then emit
# stdout/stderr and exit appropriately.
write_mock() {
  local name="$1" body="$2"
  cat > "${MOCK_BIN}/${name}" <<EOF
#!/usr/bin/env bash
# Mock for ${name} — writes invocations to MOCK_LOG.
log_call() {
  printf '%s\\n' "${name} \$*" >> "\${MOCK_LOG}"
}
log_call "\$@"
${body}
EOF
  chmod +x "${MOCK_BIN}/${name}"
}

# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------
__fail() {
  echo "  ASSERTION FAILED: $1" >&2
  if [[ -n "${MOCK_LOG:-}" && -r "${MOCK_LOG}" ]]; then
    echo "  --- mock call log so far ---" >&2
    sed 's/^/  | /' "${MOCK_LOG}" >&2
  fi
  exit 1
}

assert_eq() {
  local expected="$1" actual="$2" label="${3:-eq}"
  if [[ "${expected}" != "${actual}" ]]; then
    __fail "${label}: expected '${expected}', got '${actual}'"
  fi
}

assert_ne() {
  local notexpected="$1" actual="$2" label="${3:-ne}"
  if [[ "${notexpected}" == "${actual}" ]]; then
    __fail "${label}: expected != '${notexpected}', got '${actual}'"
  fi
}

assert_contains() {
  local haystack="$1" needle="$2" label="${3:-contains}"
  if [[ "${haystack}" != *"${needle}"* ]]; then
    __fail "${label}: '${haystack}' does not contain '${needle}'"
  fi
}

assert_not_contains() {
  local haystack="$1" needle="$2" label="${3:-not_contains}"
  if [[ "${haystack}" == *"${needle}"* ]]; then
    __fail "${label}: '${haystack}' should not contain '${needle}'"
  fi
}

assert_file_exists() {
  local path="$1" label="${2:-file_exists}"
  if [[ ! -e "${path}" ]]; then
    __fail "${label}: file '${path}' does not exist"
  fi
}

assert_file_contains() {
  local path="$1" needle="$2" label="${3:-file_contains}"
  assert_file_exists "${path}" "${label} (file)"
  if ! grep -qF -- "${needle}" "${path}"; then
    __fail "${label}: file '${path}' does not contain '${needle}'"
  fi
}

assert_rc_eq() {
  local expected="$1" actual="$2" label="${3:-rc}"
  if [[ "${expected}" != "${actual}" ]]; then
    __fail "${label}: expected rc=${expected}, got rc=${actual}"
  fi
}

# assert_calls_in_order <label> <regex1> <regex2> ...
# Reads MOCK_LOG and verifies that lines matching each regex appear in the
# given order. Lines may have other entries between them.
assert_calls_in_order() {
  local label="$1"; shift
  local -a wanted=("$@")
  local -i wi=0
  local n=${#wanted[@]}
  if [[ ! -r "${MOCK_LOG:-/nonexistent}" ]]; then
    __fail "${label}: MOCK_LOG not readable"
  fi
  while IFS= read -r line; do
    (( wi >= n )) && break
    if [[ "${line}" =~ ${wanted[$wi]} ]]; then
      wi=$((wi + 1))
    fi
  done < "${MOCK_LOG}"
  if (( wi < n )); then
    __fail "${label}: expected calls in order: ${wanted[*]} — only matched ${wi}/${n}"
  fi
}

# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
# Run each named function in a fresh subshell. Print pass/fail per test
# and a final summary. Sets exit code: 0 if all passed, 1 otherwise.
run_tests() {
  local -a tests=("$@")
  local -i passed=0 failed=0
  local t out rc
  echo "Running ${#tests[@]} test(s)..."
  echo
  for t in "${tests[@]}"; do
    out="$( ( "${t}" ) 2>&1 )"
    rc=$?
    if (( rc == 0 )); then
      echo "  ✓ ${t}"
      passed=$((passed + 1))
    else
      echo "  ✗ ${t}"
      printf '%s\n' "${out}" | sed 's/^/    /'
      failed=$((failed + 1))
    fi
  done
  echo
  echo "Results: ${passed} passed, ${failed} failed"
  return "$(( failed == 0 ? 0 : 1 ))"
}
