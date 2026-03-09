"""
SFT pilot finetuning loop for Titans checkpoint on instruction/chat text.
"""

import argparse
import math
import subprocess
import time
from pathlib import Path
from typing import Iterable, Tuple

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW

from data import build_dataloader
from model import ModelConfig, build_model
from train import get_tokenizer, load_config, resolve_path


def cycle_batches(loader: Iterable[Tuple[torch.Tensor, torch.Tensor]]):
    while True:
        for batch in loader:
            yield batch


def sync_checkpoints_to_s3(checkpoint_dir: Path, s3_uri: str, aws_bin: str, log_fn) -> None:
    cmd = [
        aws_bin,
        "s3",
        "sync",
        str(checkpoint_dir),
        s3_uri,
        "--exclude",
        "*",
        "--include",
        "ckpt_sft_step_*.pt",
        "--only-show-errors",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        log_fn(f"[sync] warning: aws binary not found: {aws_bin}")
        return
    if proc.returncode != 0:
        log_fn(f"[sync] warning: command failed (exit={proc.returncode})")
        if proc.stderr.strip():
            log_fn(f"[sync] stderr: {proc.stderr.strip()}")
        elif proc.stdout.strip():
            log_fn(f"[sync] stdout: {proc.stdout.strip()}")
        return
    log_fn(f"[sync] checkpoints synced to {s3_uri}")


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description="SFT pilot finetune loop")
    parser.add_argument("--config", type=str, default="configs/config_sft_pilot_oasst1_dolly.yaml")
    parser.add_argument("--ckpt", type=str, default="ckpt_step_4000.pt")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=40)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--s3-checkpoint-uri", type=str, default=None)
    parser.add_argument("--aws-bin", type=str, default="aws")
    parser.add_argument("--lr", type=float, default=None, help="Override LR from config")
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be > 0")

    started = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - started:7.1f}s] {msg}")

    cfg = load_config(resolve_path(args.config))
    mcfg = ModelConfig(**cfg["model"])
    device = pick_device(args.device)
    amp_enabled = device.type == "cuda"
    amp_device = "cuda" if amp_enabled else None

    tokenizer_path = resolve_path(cfg["data"]["tokenizer_path"])
    tokenizer = get_tokenizer(str(tokenizer_path))
    train_path = resolve_path(cfg["data"]["train_path"])
    val_path = resolve_path(cfg["data"]["val_path"])

    train_loader = build_dataloader(
        str(train_path),
        tokenizer,
        seq_len=cfg["train"]["seq_len"],
        batch_size=cfg["train"]["batch_size"],
        shuffle_buffer=cfg["data"].get("shuffle_buffer", 100000),
        shuffle=True,
    )
    val_loader = build_dataloader(
        str(val_path),
        tokenizer,
        seq_len=cfg["train"]["seq_len"],
        batch_size=cfg["train"]["batch_size"],
        shuffle_buffer=cfg["data"].get("shuffle_buffer", 100000),
        shuffle=False,
    )
    log(
        f"[init] train_windows={len(train_loader.dataset)} val_windows={len(val_loader.dataset)} "
        f"seq_len={cfg['train']['seq_len']} batch={cfg['train']['batch_size']}"
    )

    model = build_model(mcfg).to(device)
    ckpt_path = resolve_path(args.ckpt)
    state = torch.load(ckpt_path, map_location=device)
    if "model" in state:
        model.load_state_dict(state["model"], strict=True)
    else:
        model.load_state_dict(state, strict=True)

    lr = args.lr if args.lr is not None else float(cfg["train"]["lr"])
    opt = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=cfg["train"]["weight_decay"],
        betas=tuple(cfg["train"]["betas"]),
        eps=cfg["train"]["eps"],
    )
    scaler = GradScaler(amp_device) if amp_device else None

    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir).expanduser()
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = (Path.cwd() / checkpoint_dir).resolve()
    else:
        checkpoint_dir = Path(__file__).parent / "checkpoints_sft"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log(f"[init] device={device} amp={amp_enabled} ckpt={ckpt_path}")
    log(f"[init] checkpoint_dir={checkpoint_dir}")
    if args.s3_checkpoint_uri:
        log(f"[init] periodic checkpoint sync enabled -> {args.s3_checkpoint_uri}")

    model.train()
    train_iter = cycle_batches(train_loader)

    for step in range(1, args.steps + 1):
        x, y = next(train_iter)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)

        if amp_device:
            with autocast(device_type=amp_device, dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()

        if step % args.log_every == 0 or step == 1:
            log(f"[train] step {step}/{args.steps} loss {loss.item():.4f}")

        if step % args.eval_every == 0:
            model.eval()
            total = 0.0
            count = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    vlogits = model(bx)
                    vloss = F.cross_entropy(vlogits.view(-1, vlogits.size(-1)), by.view(-1))
                    total += vloss.item()
                    count += 1
                    if count >= args.eval_batches:
                        break
            avg = total / max(count, 1)
            ppl = math.exp(avg)
            log(f"[eval] step {step} loss {avg:.4f} ppl {ppl:.2f} batches={count}")
            model.train()

        if step % args.save_every == 0 or step == args.steps:
            ckpt_out = checkpoint_dir / f"ckpt_sft_step_{step}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "step": step,
                    "source_ckpt": str(ckpt_path),
                },
                ckpt_out,
            )
            log(f"[save] {ckpt_out}")
            if args.s3_checkpoint_uri:
                sync_checkpoints_to_s3(checkpoint_dir, args.s3_checkpoint_uri, args.aws_bin, log)

    log("[done] SFT pilot finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
