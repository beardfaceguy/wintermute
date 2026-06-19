"""
Tests for detached training path resolution and status script entrypoints.

Path defaults live in config/detached_training_probe.json. The bash library
scripts/aws_commands/lib/remote_training_probe_paths.sh loads that file via jq.
Shell unit tests: tests/aws_tooling/test_remote_training_probe_paths.sh.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "detached_training_probe.json"


@pytest.fixture(scope="module")
def probe_config() -> dict:
    assert CONFIG_PATH.is_file(), f"missing {CONFIG_PATH}"
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _bash_export_script(lib: Path, config_path: Path) -> str:
    """Return bash snippet: source lib, run remote_training_probe_set_paths, print resolved vars."""
    return f"""
set -euo pipefail
export DETACHED_TRAINING_PROBE_CONFIG='{config_path}'
source '{lib}'
remote_training_probe_set_paths
printf 'RUN_WORK_DIR=%s\\n' "$RUN_WORK_DIR"
printf 'TRAIN_LOG=%s\\n' "$TRAIN_LOG"
printf 'RUNNER_LOG=%s\\n' "${{RUNNER_LOG:-}}"
printf 'RUN_STATUS_JSON=%s\\n' "${{RUN_STATUS_JSON:-}}"
printf 'RUNNER_PID_FILE=%s\\n' "${{RUNNER_PID_FILE:-}}"
printf 'PROBE_CHECK_PID=%s\\n' "$PROBE_CHECK_PID"
"""


def _run_probe_paths(env: dict[str, str], config_path: Path | None = None) -> dict[str, str]:
    """Invoke remote_training_probe_set_paths via bash; return key/value dict from stdout."""
    lib = REPO_ROOT / "scripts/aws_commands/lib/remote_training_probe_paths.sh"
    assert lib.is_file(), f"missing {lib}"
    cfg = config_path or CONFIG_PATH
    merged = os.environ.copy()
    merged.update(env)
    proc = subprocess.run(
        ["bash", "-c", _bash_export_script(lib, cfg)],
        env=merged,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"bash failed rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )
    out: dict[str, str] = {}
    for line in proc.stdout.strip().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k] = v
    return out


def test_probe_config_has_required_keys(probe_config: dict) -> None:
    assert "aws" in probe_config and "region" in probe_config["aws"]
    assert "layouts" in probe_config
    for name in ("titan_detached", "dixie_sft", "custom"):
        assert name in probe_config["layouts"]
    assert set(probe_config["known_layouts"]) == set(probe_config["layouts"].keys())


def test_probe_paths_titan_detached_defaults(probe_config: dict) -> None:
    L = probe_config["layouts"]["titan_detached"]
    root = L["remote_run_root_default"]
    run_id = "run_xyz"
    got = _run_probe_paths({"REMOTE_LAYOUT": "titan_detached", "RUN_ID": run_id})
    assert got["RUN_WORK_DIR"] == f"{root}/{run_id}"
    assert got["TRAIN_LOG"] == f"{root}/{run_id}/{L['train_log_basename']}"
    assert got["RUN_STATUS_JSON"] == f"{root}/{run_id}/{L['run_status_json_basename']}"
    assert got["RUNNER_PID_FILE"] == f"{root}/{run_id}/{L['runner_pid_basename']}"
    assert got["PROBE_CHECK_PID"] == str(L["default_probe_check_pid"])


def test_probe_paths_titan_remote_run_root_override(probe_config: dict) -> None:
    L = probe_config["layouts"]["titan_detached"]
    got = _run_probe_paths(
        {
            "REMOTE_LAYOUT": "titan_detached",
            "RUN_ID": "r1",
            "REMOTE_RUN_ROOT": "/vol/ssm",
        }
    )
    assert got["RUN_WORK_DIR"] == "/vol/ssm/r1"
    assert got["TRAIN_LOG"] == f"/vol/ssm/r1/{L['train_log_basename']}"


def test_probe_paths_dixie_sft_defaults(probe_config: dict) -> None:
    L = probe_config["layouts"]["dixie_sft"]
    dd = L["default_work_dir"]
    got = _run_probe_paths({"REMOTE_LAYOUT": "dixie_sft", "RUN_ID": "unused_for_paths"})
    assert got["RUN_WORK_DIR"] == dd
    assert got["TRAIN_LOG"] == f"{dd}/{L['train_log_relative']}"
    assert got["RUNNER_LOG"] == f"{dd}/{L['runner_log_relative']}"
    assert got["RUN_STATUS_JSON"] == ""
    assert got["RUNNER_PID_FILE"] == ""
    assert got["PROBE_CHECK_PID"] == str(L["default_probe_check_pid"])


def test_probe_paths_custom_layout(probe_config: dict) -> None:
    got = _run_probe_paths(
        {
            "REMOTE_LAYOUT": "custom",
            "RUN_ID": "job42",
            "REMOTE_RUN_WORK_DIR": "/opt/acme/train_job42",
            "TRAIN_LOG": "/opt/acme/train_job42/log/train.log",
            "RUNNER_LOG": "/opt/acme/train_job42/log/runner.log",
        }
    )
    assert got["RUN_WORK_DIR"] == "/opt/acme/train_job42"
    assert got["TRAIN_LOG"] == "/opt/acme/train_job42/log/train.log"
    assert got["RUNNER_LOG"] == "/opt/acme/train_job42/log/runner.log"
    assert got["PROBE_CHECK_PID"] == str(
        probe_config["layouts"]["custom"]["default_probe_check_pid"]
    )


def test_probe_paths_unknown_layout_exits_nonzero() -> None:
    lib = REPO_ROOT / "scripts/aws_commands/lib/remote_training_probe_paths.sh"
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f"set -euo pipefail; export DETACHED_TRAINING_PROBE_CONFIG='{CONFIG_PATH}'; "
            f"source '{lib}'; REMOTE_LAYOUT=not_valid RUN_ID=x remote_training_probe_set_paths",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2


def test_probe_paths_missing_config_exits_nonzero(tmp_path: Path) -> None:
    lib = REPO_ROOT / "scripts/aws_commands/lib/remote_training_probe_paths.sh"
    bogus = tmp_path / "nope.json"
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f"set +e; export DETACHED_TRAINING_PROBE_CONFIG='{bogus}'; source '{lib}'; "
            "REMOTE_LAYOUT=titan_detached RUN_ID=x remote_training_probe_set_paths; echo rc=$?",
        ],
        capture_output=True,
        text=True,
    )
    assert "rc=3" in proc.stdout


@pytest.mark.parametrize(
    "script_name",
    [
        "check_detached_titan_status.sh",
        "check_detached_training_status.sh",
    ],
)
def test_status_script_bash_syntax_clean(script_name: str) -> None:
    script = REPO_ROOT / "scripts/aws_commands" / script_name
    assert script.is_file()
    proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_status_script_requires_run_id() -> None:
    titan = REPO_ROOT / "scripts/aws_commands/check_detached_titan_status.sh"
    env = {k: v for k, v in os.environ.items() if k not in ("RUN_ID", "INSTANCE_ID")}
    proc = subprocess.run(["bash", str(titan)], env=env, capture_output=True, text=True)
    assert proc.returncode != 0
    combined = (proc.stdout + proc.stderr).lower()
    assert "run_id" in combined


def test_status_script_requires_instance_id_when_run_id_set() -> None:
    titan = REPO_ROOT / "scripts/aws_commands/check_detached_titan_status.sh"
    env = {k: v for k, v in os.environ.items() if k != "INSTANCE_ID"}
    env["RUN_ID"] = "probe_test_run"
    proc = subprocess.run(["bash", str(titan)], env=env, capture_output=True, text=True)
    assert proc.returncode != 0
    combined = (proc.stdout + proc.stderr).lower()
    assert "instance_id" in combined


def test_training_status_alias_execs_canonical() -> None:
    alias_path = REPO_ROOT / "scripts/aws_commands/check_detached_training_status.sh"
    text = alias_path.read_text(encoding="utf-8")
    assert "check_detached_titan_status.sh" in text
    assert "exec " in text
