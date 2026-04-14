#!/usr/bin/env python3
"""
Run a small local sanity-overfit sweep around the current passing baseline.

The script:
- writes generated config variants under `configs/generated_sanity_sweep/`
- runs each variant sequentially with the current Python interpreter
- streams stdout live to the terminal and a per-run log file
- appends summary rows to `sanity_experiments.csv`

Expected usage:
    source .venv_docs/bin/activate
    export AWS_PROFILE=experimental-admin
    export AWS_SDK_LOAD_CONFIG=1
    python model_training/titanProject/scripts/run_sanity_sweep.py
"""

from __future__ import annotations

import csv
import copy
import datetime as dt
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


SCRIPT_PATH = Path(__file__).resolve()
TITAN_DIR = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
BASE_CONFIG = TITAN_DIR / "configs" / "config_gpt_small_sanity_overfit.yaml"
GENERATED_CONFIG_DIR = TITAN_DIR / "configs" / "generated_sanity_sweep"
LOG_DIR = TITAN_DIR / "logs" / "sanity_sweep"
CSV_PATH = TITAN_DIR / "sanity_experiments.csv"
TRAIN_SCRIPT = TITAN_DIR / "train.py"

CSV_COLUMNS = [
    "run_id",
    "date",
    "purpose",
    "status",
    "log_path",
    "device_resolved",
    "config_variant",
    "seq_len",
    "batch_size",
    "grad_accum_steps",
    "lr",
    "lr_min",
    "weight_decay",
    "warmup_steps",
    "max_steps",
    "target_tokens",
    "cosine_decay",
    "save_every",
    "eval_every",
    "max_tokens",
    "max_tokens_val",
    "final_train_loss",
    "final_eval_loss",
    "final_eval_ppl",
    "best_eval_loss",
    "best_eval_ppl",
    "best_eval_step",
    "notes",
]

VARIANTS = [
    {
        "name": "lr_4e4",
        "overrides": {
            "train": {"lr": 0.0004, "lr_min": 0.0004},
        },
        "notes": "Lower constant learning rate than the current passing baseline.",
    },
    {
        "name": "lr_8e4",
        "overrides": {
            "train": {"lr": 0.0008, "lr_min": 0.0008},
        },
        "notes": "Higher constant learning rate than the current passing baseline.",
    },
    {
        "name": "seq_128",
        "overrides": {
            "train": {"seq_len": 128},
        },
        "notes": "Shorter sequence length to increase repeat exposure per context window.",
    },
    {
        "name": "warmup_0",
        "overrides": {
            "train": {"warmup_steps": 0},
        },
        "notes": "Removes the remaining warmup to test fully aggressive early fitting.",
    },
]


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def read_existing_run_ids() -> set[str]:
    if not CSV_PATH.exists():
        return set()
    with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f)
        return {row["run_id"] for row in rows if row.get("run_id")}


def parse_metrics(log_text: str) -> dict[str, Any]:
    train_matches = re.findall(r"\[train\] step=(\d+).*?loss=([0-9.]+)", log_text)
    eval_matches = re.findall(r"\[eval\] step (\d+) loss ([0-9.]+) ppl ([0-9.]+)", log_text)
    device_match = re.search(r"\[init\] using device=([a-zA-Z0-9_:-]+)", log_text)

    final_train_step = None
    final_train_loss = None
    if train_matches:
        final_train_step = int(train_matches[-1][0])
        final_train_loss = float(train_matches[-1][1])

    final_eval_step = None
    final_eval_loss = None
    final_eval_ppl = None
    best_eval_step = None
    best_eval_loss = None
    best_eval_ppl = None
    if eval_matches:
        final_eval_step = int(eval_matches[-1][0])
        final_eval_loss = float(eval_matches[-1][1])
        final_eval_ppl = float(eval_matches[-1][2])

        best = min(eval_matches, key=lambda x: float(x[1]))
        best_eval_step = int(best[0])
        best_eval_loss = float(best[1])
        best_eval_ppl = float(best[2])

    return {
        "device_resolved": device_match.group(1) if device_match else "",
        "final_train_step": final_train_step,
        "final_train_loss": final_train_loss,
        "final_eval_step": final_eval_step,
        "final_eval_loss": final_eval_loss,
        "final_eval_ppl": final_eval_ppl,
        "best_eval_step": best_eval_step,
        "best_eval_loss": best_eval_loss,
        "best_eval_ppl": best_eval_ppl,
    }


def classify_status(final_eval_loss: float | None) -> str:
    if final_eval_loss is None:
        return "failed"
    if final_eval_loss <= 1.5:
        return "pass"
    if final_eval_loss <= 3.5:
        return "promising"
    return "inconclusive"


def append_row(row: dict[str, Any]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def within_time_window(stop_before: str) -> bool:
    now = dt.datetime.now().astimezone()
    hour, minute = (int(part) for part in stop_before.split(":"))
    cutoff = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return now < cutoff


def run_variant(base_cfg: dict[str, Any], variant: dict[str, Any], stop_before: str) -> None:
    if not within_time_window(stop_before):
        print(f"[sweep] stop time {stop_before} reached; skipping remaining runs", flush=True)
        raise SystemExit(0)

    timestamp = dt.datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    run_id = f"sanity_sweep_{variant['name']}_{timestamp}"
    config_variant = f"sweep_{variant['name']}"
    config = deep_update(base_cfg, variant["overrides"])

    config_path = GENERATED_CONFIG_DIR / f"{run_id}.yaml"
    log_path = LOG_DIR / f"{run_id}.log"
    write_yaml(config_path, config)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        str(TRAIN_SCRIPT),
        "--config",
        str(config_path),
        "--device",
        "auto",
        "--log-every",
        "20",
        "--data-log-every-lines",
        "10000",
    ]

    env = os.environ.copy()
    env.setdefault("AWS_PROFILE", "experimental-admin")
    env.setdefault("AWS_SDK_LOAD_CONFIG", "1")

    print(f"[sweep] starting {run_id}", flush=True)
    print(f"[sweep] config={config_path.relative_to(REPO_ROOT)}", flush=True)
    print(f"[sweep] log={log_path.relative_to(REPO_ROOT)}", flush=True)

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines: list[str] = []
    with open(log_path, "w", encoding="utf-8") as log_file:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_file.write(line)
            output_lines.append(line)

    exit_code = proc.wait()
    log_text = "".join(output_lines)
    metrics = parse_metrics(log_text)
    status = classify_status(metrics["final_eval_loss"]) if exit_code == 0 else "failed"

    row = {
        "run_id": run_id,
        "date": dt.date.today().isoformat(),
        "purpose": "sanity_overfit",
        "status": status,
        "log_path": str(log_path.relative_to(REPO_ROOT)),
        "device_resolved": metrics["device_resolved"],
        "config_variant": config_variant,
        "seq_len": config["train"]["seq_len"],
        "batch_size": config["train"]["batch_size"],
        "grad_accum_steps": config["train"]["grad_accum_steps"],
        "lr": config["train"]["lr"],
        "lr_min": config["train"]["lr_min"],
        "weight_decay": config["train"]["weight_decay"],
        "warmup_steps": config["train"]["warmup_steps"],
        "max_steps": config["train"]["max_steps"],
        "target_tokens": config["train"]["target_tokens"],
        "cosine_decay": str(config["train"]["cosine_decay"]).lower(),
        "save_every": config["train"]["save_every"],
        "eval_every": config["train"]["eval_every"],
        "max_tokens": config["data"]["max_tokens"],
        "max_tokens_val": config["data"]["max_tokens_val"],
        "final_train_loss": metrics["final_train_loss"],
        "final_eval_loss": metrics["final_eval_loss"],
        "final_eval_ppl": metrics["final_eval_ppl"],
        "best_eval_loss": metrics["best_eval_loss"],
        "best_eval_ppl": metrics["best_eval_ppl"],
        "best_eval_step": metrics["best_eval_step"],
        "notes": variant["notes"] if exit_code == 0 else f"Run failed with exit code {exit_code}. {variant['notes']}",
    }
    append_row(row)

    print(
        "[sweep] completed "
        f"{run_id} status={status} "
        f"best_eval_loss={metrics['best_eval_loss']} best_eval_ppl={metrics['best_eval_ppl']}",
        flush=True,
    )


def main() -> int:
    stop_before = os.environ.get("TITAN_SWEEP_STOP_BEFORE", "11:45")
    base_cfg = load_yaml(BASE_CONFIG)
    existing_ids = read_existing_run_ids()

    print(f"[sweep] local time: {dt.datetime.now().astimezone().isoformat()}", flush=True)
    print(f"[sweep] stop before: {stop_before}", flush=True)
    print(f"[sweep] using base config: {BASE_CONFIG.relative_to(REPO_ROOT)}", flush=True)

    for variant in VARIANTS:
        prefix = f"sanity_sweep_{variant['name']}_"
        if any(run_id.startswith(prefix) for run_id in existing_ids):
            print(f"[sweep] skipping {variant['name']} because a prior row already exists", flush=True)
            continue
        run_variant(base_cfg, variant, stop_before)

    print("[sweep] finished all requested variants", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
