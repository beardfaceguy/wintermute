"""
Unified training loop for Titans models.

Supports single-GPU, multi-GPU (DDP), and CPU/MPS development:

    # Single GPU
    python train.py --config configs/config_gpt_medium.yaml

    # Multi-GPU via torchrun
    torchrun --nproc_per_node=4 train.py --config configs/config_gpt_medium.yaml

    # CPU / Apple Silicon dev
    python train.py --config config_small.yaml --device mps
"""

import argparse
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from data import build_dataloader
from model import ModelConfig, build_model, load_model_source
from torch.amp import GradScaler
from torch.amp import autocast as amp_autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from train_utils import (
    build_distributed_dataloader,
    cleanup_distributed,
    cosine_lr,
    get_tokenizer,
    has_min_free_space,
    is_main,
    load_config,
    pick_device,
    reduce_scalar,
    resolve_checkpoint_dir,
    resolve_path,
    save_checkpoint,
    setup_distributed,
    sync_checkpoints_to_s3,
)


def main():
    parser = argparse.ArgumentParser(description="Titans model trainer (single & multi-GPU)")
    parser.add_argument(
        "--config",
        type=str,
        default="config_small.yaml",
        help="Path to YAML config (relative or absolute)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device override (ignored under torchrun, which forces CUDA)",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Override max_steps")
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help="Override gradient accumulation steps (auto-scaled by world_size unless --no-accum-scale)",
    )
    parser.add_argument(
        "--no-accum-scale",
        action="store_true",
        help="Disable automatic grad_accum_steps scaling by world_size",
    )
    parser.add_argument("--log-every", type=int, default=50, help="Log training loss every N steps")
    parser.add_argument("--debug", action="store_true", help="Verbose debug logging")
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="Optional cap on tokens to load (per split)"
    )
    parser.add_argument(
        "--data-log-every-lines",
        type=int,
        default=200000,
        help="Emit dataset tokenization heartbeat every N input lines (set 0 to disable)",
    )
    parser.add_argument(
        "--debug-every", type=int, default=1, help="If --debug, log every N steps (default 1)"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint (.pt) to resume from"
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Optional model init source (checkpoint path or Hugging Face ref like hf://gpt2)",
    )
    parser.add_argument(
        "--save-every", type=int, default=None, help="Override checkpoint save interval (steps)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to write checkpoints (default: model_training/titanProject)",
    )
    parser.add_argument(
        "--s3-checkpoint-uri",
        type=str,
        default=None,
        help="Optional S3 URI (s3://bucket/prefix/) to sync checkpoints after each save",
    )
    parser.add_argument(
        "--aws-bin", type=str, default="aws", help="AWS CLI binary for checkpoint sync"
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=None,
        help="Optional token budget target for progress logging (e.g., 2300000000)",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=20.0,
        help="Minimum free disk space (GiB) required to write a checkpoint",
    )
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", help="Force enable AMP")
    amp_group.add_argument(
        "--no-amp", dest="amp", action="store_false", help="Force disable AMP (default on MPS/CPU)"
    )
    parser.set_defaults(amp=None)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Determine whether we are in DDP mode or single-process mode
    # ------------------------------------------------------------------
    import os

    use_ddp = "RANK" in os.environ

    if use_ddp:
        rank, local_rank, world_size = setup_distributed()
        device = torch.device("cuda", local_rank)
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = pick_device(args.device)

    start_time = time.time()

    def log(msg: str):
        if not is_main(rank):
            return
        elapsed = time.time() - start_time
        print(f"[{elapsed:7.1f}s] {msg}")

    # ------------------------------------------------------------------
    # Config, tokenizer, paths
    # ------------------------------------------------------------------
    cfg = load_config(resolve_path(args.config))
    mcfg = ModelConfig(**cfg["model"])

    tokenizer_path = resolve_path(cfg["data"]["tokenizer_path"])
    tokenizer = get_tokenizer(str(tokenizer_path))
    tokenizer_fingerprint = getattr(tokenizer, "tokenizer_fingerprint", str(tokenizer_path))

    train_path = resolve_path(cfg["data"]["train_path"])
    val_path = resolve_path(cfg["data"]["val_path"])
    train_max_tokens = args.max_tokens or cfg["data"].get("max_tokens")
    val_max_tokens = args.max_tokens or cfg["data"].get(
        "max_tokens_val", cfg["data"].get("max_tokens")
    )

    log(f"[init] world_size={world_size} rank={rank} local_rank={local_rank} device={device}")
    log(f"[init] tokenizer={tokenizer_path}")
    log(f"[init] train_path={train_path}")
    log(f"[init] val_path={val_path}")
    log(f"[data] tokenization heartbeat every {args.data_log_every_lines} lines")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    if world_size > 1:
        if is_main(rank):
            log("[data] rank 0 building token caches (other ranks waiting) ...")
        if not is_main(rank):
            dist.barrier()

    if world_size > 1:
        log("[data] building train dataloader")
        train_loader, train_sampler = build_distributed_dataloader(
            str(train_path),
            tokenizer,
            tokenizer_fingerprint,
            seq_len=cfg["train"]["seq_len"],
            batch_size=cfg["train"]["batch_size"],
            rank=rank,
            world_size=world_size,
            shuffle=True,
            max_tokens=train_max_tokens,
            log_fn=log if is_main(rank) else None,
            progress_every_lines=args.data_log_every_lines,
            progress_label="train",
        )
        log("[data] building val dataloader")
        val_loader, _ = build_distributed_dataloader(
            str(val_path),
            tokenizer,
            tokenizer_fingerprint,
            seq_len=cfg["train"]["seq_len"],
            batch_size=cfg["train"]["batch_size"],
            rank=rank,
            world_size=world_size,
            shuffle=False,
            max_tokens=val_max_tokens,
            log_fn=log if is_main(rank) else None,
            progress_every_lines=args.data_log_every_lines,
            progress_label="val",
        )
    else:
        train_sampler = None
        log("[data] building train dataloader (tokenization can take a while)")
        train_loader = build_dataloader(
            str(train_path),
            tokenizer,
            tokenizer_fingerprint=tokenizer_fingerprint,
            seq_len=cfg["train"]["seq_len"],
            batch_size=cfg["train"]["batch_size"],
            shuffle_buffer=cfg["data"]["shuffle_buffer"],
            max_tokens=train_max_tokens,
            log_fn=log,
            progress_every_lines=args.data_log_every_lines,
            progress_label="train",
        )
        log("[data] building val dataloader (tokenization can take a while)")
        val_loader = build_dataloader(
            str(val_path),
            tokenizer,
            tokenizer_fingerprint=tokenizer_fingerprint,
            seq_len=cfg["train"]["seq_len"],
            batch_size=cfg["train"]["batch_size"],
            shuffle_buffer=cfg["data"]["shuffle_buffer"],
            shuffle=False,
            max_tokens=val_max_tokens,
            log_fn=log,
            progress_every_lines=args.data_log_every_lines,
            progress_label="val",
        )

    if world_size > 1 and is_main(rank):
        dist.barrier()
        log("[data] all ranks have loaded data")

    log(
        f"[init] train_dataset_size={len(train_loader.dataset)} windows, "
        f"tokens={train_loader.dataset.num_tokens}"
    )
    log(
        f"[init] val_dataset_size={len(val_loader.dataset)} windows, tokens={val_loader.dataset.num_tokens}"
    )
    if train_max_tokens:
        log(f"[init] train max_tokens cap: {train_max_tokens}")
    if val_max_tokens:
        log(f"[init] val max_tokens cap: {val_max_tokens}")

    # ------------------------------------------------------------------
    # AMP configuration
    # ------------------------------------------------------------------
    if args.amp is not None:
        amp_enabled = args.amp
    elif device.type == "cuda":
        amp_enabled = True
    else:
        amp_enabled = False

    amp_device = device.type if amp_enabled and device.type in ("cuda", "mps") else None
    log(f"[init] amp_enabled={amp_enabled}, amp_device={amp_device}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(mcfg).to(device)

    if args.resume:
        ckpt_path = resolve_path(args.resume)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        log(f"[resume] loaded weights from {ckpt_path}")
    elif args.init_from:
        init_source = load_model_source(
            model, resolve_path(args.init_from), map_location=device, strict=True
        )
        log(f"[init] loaded model weights from {init_source}")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    raw_model = model.module if world_size > 1 else model

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    opt = AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        betas=tuple(cfg["train"]["betas"]),
        eps=cfg["train"]["eps"],
    )
    scaler = GradScaler(enabled=amp_device in ("cuda", "mps"))

    global_step = 0
    if args.resume:
        opt.load_state_dict(ckpt["opt"])
        if scaler and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        global_step = ckpt.get("step", 0)
        log(f"[resume] optimizer restored at step {global_step}")

    # ------------------------------------------------------------------
    # Training params
    # ------------------------------------------------------------------
    max_steps = args.max_steps or cfg["train"]["max_steps"]
    grad_accum_steps = args.grad_accum_steps or cfg["train"].get("grad_accum_steps", 1)
    if grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be > 0")

    if world_size > 1 and not args.no_accum_scale:
        original_accum = grad_accum_steps
        grad_accum_steps = max(1, grad_accum_steps // world_size)
        log(
            f"[init] auto-scaled grad_accum_steps: {original_accum} -> {grad_accum_steps} (world_size={world_size})"
        )

    warmup = cfg["train"]["warmup_steps"]
    lr_min = cfg["train"].get("lr_min", 0.0)
    target_tokens = args.target_tokens or cfg["train"].get("target_tokens")
    log_every = args.debug_every if args.debug else args.log_every
    save_every = args.save_every or cfg["train"]["save_every"]
    if save_every <= 0:
        raise ValueError("--save-every must be > 0")

    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_dir, Path(__file__).parent)

    micro_batch_tokens = cfg["train"]["batch_size"] * cfg["train"]["seq_len"]
    effective_tokens_per_step = micro_batch_tokens * grad_accum_steps * world_size
    log(
        f"[init] max_steps={max_steps}, warmup={warmup}, log_every={log_every}, save_every={save_every}, "
        f"batch_size={cfg['train']['batch_size']}, seq_len={cfg['train']['seq_len']}, "
        f"grad_accum_steps={grad_accum_steps}, world_size={world_size}, "
        f"effective_tokens/step={effective_tokens_per_step:,}"
    )
    if target_tokens:
        est_steps_to_target = math.ceil(target_tokens / max(effective_tokens_per_step, 1))
        log(
            f"[init] target_tokens={target_tokens:,} -> estimated optimizer steps={est_steps_to_target:,} "
            f"at current effective batch"
        )
    log(f"[init] checkpoint_dir={checkpoint_dir}")
    if args.s3_checkpoint_uri:
        log(f"[init] periodic checkpoint sync enabled -> {args.s3_checkpoint_uri}")
    if args.debug:
        log(f"[debug] model cfg: {mcfg}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    if args.debug:
        xb, yb = next(iter(train_loader))
        log(f"[debug] first batch shapes x={xb.shape}, y={yb.shape}")

    model.train()
    total_tokens_seen = global_step * effective_tokens_per_step
    accum_in_step = 0
    accum_loss_sum = 0.0
    opt.zero_grad(set_to_none=True)
    epoch = 0

    while global_step < max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch += 1

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            total_tokens_seen += int(x.numel()) * world_size

            lr = (
                cosine_lr(global_step, warmup, max_steps, cfg["train"]["lr"], lr_min)
                if cfg["train"]["cosine_decay"]
                else cfg["train"]["lr"]
            )
            for pg in opt.param_groups:
                pg["lr"] = lr

            is_last_accum = (accum_in_step + 1) == grad_accum_steps
            sync_context = nullcontext() if (world_size <= 1 or is_last_accum) else model.no_sync()

            with sync_context:
                if amp_device:
                    with amp_autocast(device_type=amp_device, dtype=torch.float16):
                        out = model(x, return_loss=False)
                        logits = out if not isinstance(out, dict) else out.get("logits", out)
                        raw_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                        loss = raw_loss / grad_accum_steps
                    scaler.scale(loss).backward()
                else:
                    out = model(x, return_loss=False)
                    logits = out if not isinstance(out, dict) else out.get("logits", out)
                    raw_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss = raw_loss / grad_accum_steps
                    loss.backward()

            accum_in_step += 1
            accum_loss_sum += raw_loss.item()
            if accum_in_step < grad_accum_steps:
                continue

            # --- Optimizer step ---
            if amp_device:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            if amp_device:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)

            global_step += 1
            opt_step_loss = accum_loss_sum / max(accum_in_step, 1)
            accum_in_step = 0
            accum_loss_sum = 0.0

            if global_step % log_every == 0:
                avg_loss = reduce_scalar(opt_step_loss, world_size)
                if is_main(rank):
                    elapsed = max(time.time() - start_time, 1e-6)
                    tok_per_sec = total_tokens_seen / elapsed
                    token_msg = f" tokens_seen={total_tokens_seen:,} tok/s={tok_per_sec:,.0f}"
                    if target_tokens:
                        token_pct = 100.0 * total_tokens_seen / max(target_tokens, 1)
                        token_msg += f" target_progress={token_pct:.2f}%"
                    log(
                        f"[train] step={global_step} batch={batch_idx} loss={avg_loss:.4f} "
                        f"lr={lr:.6f}{token_msg}"
                    )
            if args.debug:
                log(
                    f"[debug] step={global_step} lr={lr:.6f} loss={opt_step_loss:.4f} "
                    f"accum={grad_accum_steps}"
                )

            # --- Eval ---
            if global_step % cfg["train"]["eval_every"] == 0:
                model.eval()
                total_loss = 0.0
                count = 0
                log(f"[eval] running eval at step {global_step} ...")
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(device, non_blocking=True), vy.to(device, non_blocking=True)
                        if amp_device:
                            with amp_autocast(device_type=amp_device, dtype=torch.float16):
                                out = model(vx, return_loss=False)
                                logits = (
                                    out if not isinstance(out, dict) else out.get("logits", out)
                                )
                                val_loss = F.cross_entropy(
                                    logits.view(-1, logits.size(-1)), vy.view(-1)
                                )
                        else:
                            out = model(vx, return_loss=False)
                            logits = out if not isinstance(out, dict) else out.get("logits", out)
                            val_loss = F.cross_entropy(
                                logits.view(-1, logits.size(-1)), vy.view(-1)
                            )
                        total_loss += val_loss.item()
                        count += 1

                avg_val_loss = reduce_scalar(total_loss / max(count, 1), world_size)
                if is_main(rank):
                    ppl = math.exp(avg_val_loss)
                    log(f"[eval] step {global_step} loss {avg_val_loss:.4f} ppl {ppl:.2f}")
                model.train()

            # --- Checkpoint (rank 0 only) ---
            if global_step % save_every == 0 and is_main(rank):
                if has_min_free_space(checkpoint_dir, args.min_free_gb, log):
                    ckpt_path = checkpoint_dir / f"ckpt_step_{global_step}.pt"
                    save_checkpoint(
                        ckpt_path,
                        raw_model.state_dict(),
                        opt.state_dict(),
                        global_step,
                        scaler.state_dict() if scaler else None,
                    )
                    log(f"Saved {ckpt_path}")
                    if args.s3_checkpoint_uri:
                        sync_checkpoints_to_s3(
                            checkpoint_dir, args.s3_checkpoint_uri, args.aws_bin, log
                        )

            if global_step >= max_steps:
                break

    # --- Final checkpoint + eval (if not already on a save boundary) ---
    if global_step > 0 and global_step % save_every != 0:
        log(
            f"[final] training ended at step {global_step} (not on save_every={save_every} boundary)"
        )
        model.eval()
        total_loss = 0.0
        count = 0
        log(f"[final-eval] running final eval at step {global_step} ...")
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device, non_blocking=True), vy.to(device, non_blocking=True)
                if amp_device:
                    with amp_autocast(device_type=amp_device, dtype=torch.float16):
                        out = model(vx, return_loss=False)
                        logits = out if not isinstance(out, dict) else out.get("logits", out)
                        val_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), vy.view(-1))
                else:
                    out = model(vx, return_loss=False)
                    logits = out if not isinstance(out, dict) else out.get("logits", out)
                    val_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), vy.view(-1))
                total_loss += val_loss.item()
                count += 1

        avg_val_loss = reduce_scalar(total_loss / max(count, 1), world_size)
        if is_main(rank):
            ppl = math.exp(avg_val_loss)
            log(f"[final-eval] step {global_step} loss {avg_val_loss:.4f} ppl {ppl:.2f}")

            if has_min_free_space(checkpoint_dir, args.min_free_gb, log):
                ckpt_path = checkpoint_dir / f"ckpt_step_{global_step}.pt"
                save_checkpoint(
                    ckpt_path,
                    raw_model.state_dict(),
                    opt.state_dict(),
                    global_step,
                    scaler.state_dict() if scaler else None,
                )
                log(f"[final] Saved {ckpt_path}")
                if args.s3_checkpoint_uri:
                    sync_checkpoints_to_s3(
                        checkpoint_dir, args.s3_checkpoint_uri, args.aws_bin, log
                    )
        model.train()
    elif global_step > 0:
        log(f"[final] training ended at step {global_step} (on save boundary, already saved)")

    log(f"[done] training complete at step {global_step}, tokens_seen={total_tokens_seen:,}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
