#!/usr/bin/env bash
# Resolve on-instance paths for detached LLM training probes.
#
# Defaults are loaded from config/detached_training_probe.json (override path via
# DETACHED_TRAINING_PROBE_CONFIG). Requires jq.
#
# Sourced by check_detached_titan_status.sh (generic detached-training monitor).
# Sets shell variables (no subshell): RUN_WORK_DIR, TRAIN_LOG, RUNNER_LOG,
# RUN_STATUS_JSON, RUNNER_PID_FILE, PROBE_CHECK_PID.
#
# Env:
#   DETACHED_TRAINING_PROBE_CONFIG   Path to JSON config (default: <repo>/config/...)
#   REMOTE_LAYOUT                    Key in known_layouts / layouts (default from config)
#   RUN_ID                            Required for titan_detached and custom; used for S3 defaults.
#
# Overrides (any layout): if TRAIN_LOG / RUNNER_LOG / etc. are already non-empty
# before calling remote_training_probe_set_paths, they are left unchanged.

# Repo root: this file lives at scripts/aws_commands/lib/
detached_training_probe_repo_root() {
  (cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
}

detached_training_probe_config_path() {
  local root
  root="$(detached_training_probe_repo_root)"
  printf '%s\n' "${DETACHED_TRAINING_PROBE_CONFIG:-${root}/config/detached_training_probe.json}"
}

remote_training_probe_set_paths() {
  local cfg layout known
  cfg="$(detached_training_probe_config_path)"
  if [[ ! -f "${cfg}" ]]; then
    echo "[remote_training_probe] FATAL: missing config: ${cfg}" >&2
    return 3
  fi
  if ! command -v jq >/dev/null 2>&1; then
    echo "[remote_training_probe] FATAL: jq is required to load ${cfg}" >&2
    return 3
  fi

  layout="${REMOTE_LAYOUT:-$(jq -r '.default_layout' "${cfg}")}"
  if ! jq -e --arg L "${layout}" '.known_layouts | index($L) != null' "${cfg}" >/dev/null 2>&1; then
    known="$(jq -r '.known_layouts | join(" | ")' "${cfg}")"
    echo "remote_training_probe_set_paths: unknown REMOTE_LAYOUT='${layout}'" >&2
    echo "expected: ${known}" >&2
    return 2
  fi

  case "${layout}" in
    titan_detached)
      : "${RUN_ID:?set RUN_ID first}"
      local root tlog sjson pidf def_root
      def_root="$(jq -r '.layouts.titan_detached.remote_run_root_default' "${cfg}")"
      root="${REMOTE_RUN_ROOT:-${def_root}}"
      if [[ -z "${REMOTE_RUN_WORK_DIR:-}" ]]; then
        RUN_WORK_DIR="${root}/${RUN_ID}"
      else
        RUN_WORK_DIR="${REMOTE_RUN_WORK_DIR}"
      fi
      tlog="$(jq -r '.layouts.titan_detached.train_log_basename' "${cfg}")"
      sjson="$(jq -r '.layouts.titan_detached.run_status_json_basename' "${cfg}")"
      pidf="$(jq -r '.layouts.titan_detached.runner_pid_basename' "${cfg}")"
      [[ -n "${TRAIN_LOG:-}" ]] || TRAIN_LOG="${RUN_WORK_DIR}/${tlog}"
      [[ -n "${RUNNER_LOG:-}" ]] || RUNNER_LOG=""
      [[ -n "${RUN_STATUS_JSON:-}" ]] || RUN_STATUS_JSON="${RUN_WORK_DIR}/${sjson}"
      [[ -n "${RUNNER_PID_FILE:-}" ]] || RUNNER_PID_FILE="${RUN_WORK_DIR}/${pidf}"
      [[ -n "${PROBE_CHECK_PID:-}" ]] || PROBE_CHECK_PID="$(jq -r '.layouts.titan_detached.default_probe_check_pid' "${cfg}")"
      ;;
    dixie_sft)
      local dd tlr rlr
      dd="$(jq -r '.layouts.dixie_sft.default_work_dir' "${cfg}")"
      if [[ -z "${REMOTE_RUN_WORK_DIR:-}" ]]; then
        RUN_WORK_DIR="${dd}"
      else
        RUN_WORK_DIR="${REMOTE_RUN_WORK_DIR}"
      fi
      tlr="$(jq -r '.layouts.dixie_sft.train_log_relative' "${cfg}")"
      rlr="$(jq -r '.layouts.dixie_sft.runner_log_relative' "${cfg}")"
      [[ -n "${TRAIN_LOG:-}" ]] || TRAIN_LOG="${RUN_WORK_DIR}/${tlr}"
      [[ -n "${RUNNER_LOG:-}" ]] || RUNNER_LOG="${RUN_WORK_DIR}/${rlr}"
      [[ -n "${RUN_STATUS_JSON:-}" ]] || RUN_STATUS_JSON=""
      [[ -n "${RUNNER_PID_FILE:-}" ]] || RUNNER_PID_FILE=""
      [[ -n "${PROBE_CHECK_PID:-}" ]] || PROBE_CHECK_PID="$(jq -r '.layouts.dixie_sft.default_probe_check_pid' "${cfg}")"
      ;;
    custom)
      : "${RUN_ID:?set RUN_ID first}"
      : "${REMOTE_RUN_WORK_DIR:?REMOTE_RUN_WORK_DIR required for REMOTE_LAYOUT=custom}"
      : "${TRAIN_LOG:?TRAIN_LOG required for REMOTE_LAYOUT=custom}"
      RUN_WORK_DIR="${REMOTE_RUN_WORK_DIR}"
      [[ -n "${RUNNER_LOG:-}" ]] || RUNNER_LOG=""
      [[ -n "${RUN_STATUS_JSON:-}" ]] || RUN_STATUS_JSON=""
      [[ -n "${RUNNER_PID_FILE:-}" ]] || RUNNER_PID_FILE=""
      [[ -n "${PROBE_CHECK_PID:-}" ]] || PROBE_CHECK_PID="$(jq -r '.layouts.custom.default_probe_check_pid' "${cfg}")"
      ;;
    *)
      # Defensive — known_layouts check should have caught this.
      return 2
      ;;
  esac
}
