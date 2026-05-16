#!/usr/bin/env bash
# Unit tests for scripts/aws_commands/lib/remote_training_probe_paths.sh
# Expected paths come from config/detached_training_probe.json (single source of truth).

set -uo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${THIS_DIR}/../.." && pwd)"
CFG="${REPO_ROOT}/config/detached_training_probe.json"

# shellcheck source=scripts/aws_commands/lib/remote_training_probe_paths.sh
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/aws_commands/lib/remote_training_probe_paths.sh"

# shellcheck source=tests/aws_tooling/test_helpers.sh
# shellcheck disable=SC1091
source "${THIS_DIR}/test_helpers.sh"

clear_probe_env() {
  unset REMOTE_LAYOUT REMOTE_RUN_ROOT REMOTE_RUN_WORK_DIR TRAIN_LOG RUNNER_LOG
  unset RUN_STATUS_JSON RUNNER_PID_FILE PROBE_CHECK_PID RUN_WORK_DIR RUN_ID
  unset DETACHED_TRAINING_PROBE_CONFIG
}

test_titan_default_paths() {
  clear_probe_env
  export REMOTE_LAYOUT=titan_detached
  export RUN_ID=myrun123
  remote_training_probe_set_paths || return 1
  local eroot tlog sjson pidf
  eroot="$(jq -r '.layouts.titan_detached.remote_run_root_default' "${CFG}")"
  tlog="$(jq -r '.layouts.titan_detached.train_log_basename' "${CFG}")"
  sjson="$(jq -r '.layouts.titan_detached.run_status_json_basename' "${CFG}")"
  pidf="$(jq -r '.layouts.titan_detached.runner_pid_basename' "${CFG}")"
  def_pid="$(jq -r '.layouts.titan_detached.default_probe_check_pid' "${CFG}")"
  assert_eq "${eroot}/myrun123" "${RUN_WORK_DIR}" "titan RUN_WORK_DIR"
  assert_eq "${eroot}/myrun123/${tlog}" "${TRAIN_LOG}" "titan TRAIN_LOG"
  assert_eq "${eroot}/myrun123/${sjson}" "${RUN_STATUS_JSON}" "titan status json"
  assert_eq "${eroot}/myrun123/${pidf}" "${RUNNER_PID_FILE}" "titan pid file"
  assert_eq "${def_pid}" "${PROBE_CHECK_PID}" "titan PROBE_CHECK_PID"
}

test_titan_remote_run_root_override() {
  clear_probe_env
  export REMOTE_LAYOUT=titan_detached
  export RUN_ID=abc
  export REMOTE_RUN_ROOT=/alt/root
  remote_training_probe_set_paths || return 1
  local tlog
  tlog="$(jq -r '.layouts.titan_detached.train_log_basename' "${CFG}")"
  assert_eq "/alt/root/abc" "${RUN_WORK_DIR}" "titan REMOTE_RUN_ROOT override"
  assert_eq "/alt/root/abc/${tlog}" "${TRAIN_LOG}" "titan TRAIN_LOG override"
}

test_titan_user_train_log_override() {
  clear_probe_env
  export REMOTE_LAYOUT=titan_detached
  export RUN_ID=x
  export TRAIN_LOG=/custom/train.log
  remote_training_probe_set_paths || return 1
  assert_eq "/custom/train.log" "${TRAIN_LOG}" "preserve user TRAIN_LOG"
}

test_dixie_paths() {
  clear_probe_env
  export REMOTE_LAYOUT=dixie_sft
  export RUN_ID=ignored_for_resolution
  remote_training_probe_set_paths || return 1
  local dd tlr rlr dpid
  dd="$(jq -r '.layouts.dixie_sft.default_work_dir' "${CFG}")"
  tlr="$(jq -r '.layouts.dixie_sft.train_log_relative' "${CFG}")"
  rlr="$(jq -r '.layouts.dixie_sft.runner_log_relative' "${CFG}")"
  dpid="$(jq -r '.layouts.dixie_sft.default_probe_check_pid' "${CFG}")"
  assert_eq "${dd}" "${RUN_WORK_DIR}" "dixie RUN_WORK_DIR"
  assert_eq "${dd}/${tlr}" "${TRAIN_LOG}" "dixie TRAIN_LOG"
  assert_eq "${dd}/${rlr}" "${RUNNER_LOG}" "dixie RUNNER_LOG"
  assert_eq "" "${RUN_STATUS_JSON}" "dixie no status json"
  assert_eq "" "${RUNNER_PID_FILE}" "dixie no pid file"
  assert_eq "${dpid}" "${PROBE_CHECK_PID}" "dixie PROBE_CHECK_PID"
}

test_dixie_remote_run_work_dir_override() {
  clear_probe_env
  export REMOTE_LAYOUT=dixie_sft
  export REMOTE_RUN_WORK_DIR=/mnt/nvme/other
  remote_training_probe_set_paths || return 1
  local tlr
  tlr="$(jq -r '.layouts.dixie_sft.train_log_relative' "${CFG}")"
  assert_eq "/mnt/nvme/other" "${RUN_WORK_DIR}" "dixie REMOTE_RUN_WORK_DIR"
  assert_eq "/mnt/nvme/other/${tlr}" "${TRAIN_LOG}" "dixie TRAIN_LOG under override"
}

test_custom_paths() {
  clear_probe_env
  export REMOTE_LAYOUT=custom
  export RUN_ID=run9
  export REMOTE_RUN_WORK_DIR=/proj/run9
  export TRAIN_LOG=/proj/run9/out/train.log
  export RUNNER_LOG=/proj/run9/out/runner.log
  remote_training_probe_set_paths || return 1
  cdef="$(jq -r '.layouts.custom.default_probe_check_pid' "${CFG}")"
  assert_eq "/proj/run9" "${RUN_WORK_DIR}" "custom RUN_WORK_DIR"
  assert_eq "/proj/run9/out/train.log" "${TRAIN_LOG}" "custom TRAIN_LOG"
  assert_eq "/proj/run9/out/runner.log" "${RUNNER_LOG}" "custom RUNNER_LOG"
  assert_eq "${cdef}" "${PROBE_CHECK_PID}" "custom default PROBE_CHECK_PID"
}

test_unknown_layout_fails() {
  clear_probe_env
  export REMOTE_LAYOUT=not_a_real_layout
  export RUN_ID=r
  if remote_training_probe_set_paths; then
    echo "expected nonzero exit" >&2
    return 1
  fi
  return 0
}

test_missing_config_fails() {
  clear_probe_env
  export REMOTE_LAYOUT=titan_detached
  export RUN_ID=x
  export DETACHED_TRAINING_PROBE_CONFIG=/this/path/does/not/exist_probe_cfg.json
  remote_training_probe_set_paths
  local rc=$?
  assert_eq 3 "${rc}" "missing config should exit 3"
}

run_tests \
  test_titan_default_paths \
  test_titan_remote_run_root_override \
  test_titan_user_train_log_override \
  test_dixie_paths \
  test_dixie_remote_run_work_dir_override \
  test_custom_paths \
  test_unknown_layout_fails \
  test_missing_config_fails
