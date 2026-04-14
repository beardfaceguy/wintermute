"""
Training loop scaffold for Titans small model.
Fill in tokenizer init and paths; adjust config as needed.
"""

import argparse
import hashlib
import math
import shutil
import time
import subprocess
from contextlib import nullcontext
from pathlib import Path
from urllib.parse import urlparse
import yaml
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast as amp_autocast, GradScaler
import sentencepiece as spm

try:
    import boto3  # type: ignore
except Exception:
    boto3 = None

from data import build_dataloader
from model import ModelConfig, build_model, is_hf_source, load_model_source, normalize_hf_source


def load_config(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class TokenizerAdapter:
    def __init__(
        self,
        *,
        encode_fn,
        decode_fn,
        tokenizer_fingerprint: str,
        tokenizer_source_path: str,
        eos_id: int,
        pad_id: int,
    ):
        self._encode_fn = encode_fn
        self._decode_fn = decode_fn
        self.tokenizer_fingerprint = tokenizer_fingerprint
        self.tokenizer_source_path = tokenizer_source_path
        self.eos_id = eos_id
        self.pad_id = pad_id

    def __call__(self, text: str):
        return self.encode(text)

    def encode(self, text: str):
        return list(self._encode_fn(text))

    def decode(self, ids):
        return self._decode_fn(list(ids))


def _hf_tokenizer_fingerprint(tokenizer) -> str:
    digest = hashlib.sha256()
    if hasattr(tokenizer, "backend_tokenizer"):
        digest.update(tokenizer.backend_tokenizer.to_str().encode("utf-8"))
    else:
        for token, token_id in sorted(tokenizer.get_vocab().items(), key=lambda kv: kv[1]):
            digest.update(f"{token_id}:{token}\n".encode("utf-8"))
    return digest.hexdigest()


def get_tokenizer(tokenizer_path: str):
    if is_hf_source(tokenizer_path):
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers is required to load Hugging Face tokenizers. "
                "Install it in the active Python environment first."
            ) from e

        hf_name = normalize_hf_source(tokenizer_path)
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        return TokenizerAdapter(
            encode_fn=lambda text: tokenizer.encode(text, add_special_tokens=False),
            decode_fn=lambda ids: tokenizer.decode(
                ids,
                clean_up_tokenization_spaces=False,
                skip_special_tokens=True,
            ),
            tokenizer_fingerprint=_hf_tokenizer_fingerprint(tokenizer),
            tokenizer_source_path=tokenizer_path,
            eos_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1,
            pad_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1,
        )

    # Allow tokenizer on S3: download to a local temp path first.
    if tokenizer_path.startswith("s3://"):
        if boto3 is None:
            raise RuntimeError("boto3 is required to load tokenizer from s3:// paths")
        parsed = urlparse(tokenizer_path)
        local_path = Path("/tmp") / Path(parsed.path).name
        if not local_path.exists():
            client = boto3.client("s3")
            client.download_file(parsed.netloc, parsed.path.lstrip("/"), str(local_path))
        tokenizer_path = str(local_path)

    sp = spm.SentencePieceProcessor()
    if not sp.load(tokenizer_path):
        raise RuntimeError(f"Failed to load tokenizer at {tokenizer_path}")

    tokenizer_fingerprint = sha256_file(Path(tokenizer_path))
    return TokenizerAdapter(
        encode_fn=lambda text: sp.encode(text, out_type=int),
        decode_fn=lambda ids: sp.decode(ids),
        tokenizer_fingerprint=tokenizer_fingerprint,
        tokenizer_source_path=tokenizer_path,
        eos_id=sp.eos_id(),
        pad_id=sp.pad_id(),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cosine_lr(step, warmup, max_steps, base_lr, min_lr=0.0):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(1, max_steps - warmup)
    cos_val = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cos_val


def resolve_path(path_str: str) -> Path:
    if path_str.startswith("s3://") or path_str.startswith("hf://"):
        # For remote URIs we keep the raw string and let downstream loaders handle them.
        return path_str  # type: ignore[return-value]
    p = Path(path_str)
    if p.is_absolute():
        return p
    # first try relative to current working directory
    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return cwd_candidate
    # then try relative to this script's directory
    script_dir = Path(__file__).resolve().parent
    script_candidate = script_dir / p
    if script_candidate.exists():
        return script_candidate
    # fallback: repo root (wintermute) is two levels up from this file now that titanProject lives in model_training/
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / p


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
        "ckpt_step_*.pt",
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


def has_min_free_space(path: Path, min_free_gb: float, log_fn) -> bool:
    try:
        usage = shutil.disk_usage(path)
    except FileNotFoundError:
        return True

    free_gb = usage.free / (1024**3)
    if free_gb < min_free_gb:
        log_fn(
            f"[disk] free space low at {free_gb:.1f} GiB (< {min_free_gb:.1f} GiB); "
            "skipping checkpoint to avoid filling disk"
        )
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Titans MAC small-model trainer")
    parser.add_argument("--config", type=str, default="config_small.yaml", help="Path to YAML config (relative or absolute)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Device override")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max_steps")
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help="Override gradient accumulation steps (effective batch = batch_size * grad_accum_steps)",
    )
    parser.add_argument("--log-every", type=int, default=50, help="Log training loss every N steps")
    parser.add_argument("--debug", action="store_true", help="Verbose debug logging")
    parser.add_argument("--max-tokens", type=int, default=None, help="Optional cap on tokens to load (per split)")
    parser.add_argument(
        "--data-log-every-lines",
        type=int,
        default=200000,
        help="Emit dataset tokenization heartbeat every N input lines (set 0 to disable line-based heartbeats)",
    )
    parser.add_argument("--debug-every", type=int, default=1, help="If --debug, log every N steps (default 1)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.pt) to resume from")
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Optional model init source (checkpoint path or Hugging Face ref like hf://gpt2)",
    )
    parser.add_argument("--save-every", type=int, default=None, help="Override checkpoint save interval (steps)")
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
    parser.add_argument("--aws-bin", type=str, default="aws", help="AWS CLI binary for checkpoint sync")
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
        help="Minimum free disk space (GiB) required to write a checkpoint; below this, saves are skipped",
    )
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", help="Force enable AMP")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false", help="Force disable AMP (default on MPS/CPU)")
    parser.set_defaults(amp=None)
    args = parser.parse_args()

    start_time = time.time()

    def log(msg: str):
        elapsed = time.time() - start_time
        print(f"[{elapsed:7.1f}s] {msg}")

    cfg = load_config(resolve_path(args.config))
    mcfg = ModelConfig(**cfg["model"])

    tokenizer_path = resolve_path(cfg["data"]["tokenizer_path"])
    tokenizer = get_tokenizer(str(tokenizer_path))
    tokenizer_fingerprint = getattr(tokenizer, "tokenizer_fingerprint", str(tokenizer_path))

    train_path = resolve_path(cfg["data"]["train_path"])
    val_path = resolve_path(cfg["data"]["val_path"])
    train_max_tokens = args.max_tokens or cfg["data"].get("max_tokens")
    val_max_tokens = args.max_tokens or cfg["data"].get("max_tokens_val", cfg["data"].get("max_tokens"))

    log(f"[init] device_pref={args.device} | tokenizer={tokenizer_path}")
    log(f"[init] train_path={train_path}")
    log(f"[init] val_path={val_path}")
    log(f"[data] tokenization heartbeat every {args.data_log_every_lines} lines")

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

    log(
        f"[init] train_dataset_size={len(train_loader.dataset)} windows, tokens={train_loader.dataset.num_tokens}"
    )
    log(f"[init] val_dataset_size={len(val_loader.dataset)} windows, tokens={val_loader.dataset.num_tokens}")
    if train_max_tokens:
        log(f"[init] train max_tokens cap: {train_max_tokens}")
    if val_max_tokens:
        log(f"[init] val max_tokens cap: {val_max_tokens}")

    # device selection
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    # Decide AMP: default on CUDA, off on MPS/CPU unless forced on.
    amp_enabled = args.amp if args.amp is not None else device.type == "cuda"
    amp_device = device.type if amp_enabled and device.type in ("cuda", "mps") else None

    log(f"[init] using device={device}, amp_enabled={amp_enabled}, amp_device={amp_device}")

    model = build_model(mcfg).to(device)

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
        ckpt_path = resolve_path(args.resume)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        opt.load_state_dict(ckpt["opt"])
        if scaler and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        global_step = ckpt.get("step", 0)
        log(f"[resume] loaded {ckpt_path} at step {global_step}")
    elif args.init_from:
        init_source = load_model_source(model, resolve_path(args.init_from), map_location=device, strict=True)
        log(f"[init] loaded model weights from {init_source}")

    max_steps = args.max_steps or cfg["train"]["max_steps"]
    grad_accum_steps = args.grad_accum_steps or cfg["train"].get("grad_accum_steps", 1)
    if grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be > 0")
    warmup = cfg["train"]["warmup_steps"]
    lr_min = cfg["train"].get("lr_min", 0.0)
    target_tokens = args.target_tokens or cfg["train"].get("target_tokens")
    log_every = args.debug_every if args.debug else args.log_every
    save_every = args.save_every or cfg["train"]["save_every"]
    if save_every <= 0:
        raise ValueError("--save-every must be > 0")

    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir).expanduser()
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = (Path.cwd() / checkpoint_dir).resolve()
    else:
        checkpoint_dir = Path(__file__).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    micro_batch_tokens = cfg["train"]["batch_size"] * cfg["train"]["seq_len"]
    effective_tokens_per_step = micro_batch_tokens * grad_accum_steps
    log(
        f"[init] max_steps={max_steps}, warmup={warmup}, log_every={log_every}, save_every={save_every}, "
        f"batch_size={cfg['train']['batch_size']}, seq_len={cfg['train']['seq_len']}, "
        f"grad_accum_steps={grad_accum_steps}, effective_tokens/step={effective_tokens_per_step:,}"
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

    if args.debug:
        # grab a tiny batch to inspect shape
        xb, yb = next(iter(train_loader))
        log(f"[debug] first batch shapes x={xb.shape}, y={yb.shape}")
    model.train()
    total_tokens_seen = global_step * effective_tokens_per_step
    accum_in_step = 0
    accum_loss_sum = 0.0
    opt.zero_grad(set_to_none=True)
    while global_step < max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            total_tokens_seen += int(x.numel())
            lr = (
                cosine_lr(global_step, warmup, max_steps, cfg["train"]["lr"], lr_min)
                if cfg["train"]["cosine_decay"]
                else cfg["train"]["lr"]
            )
            for pg in opt.param_groups:
                pg["lr"] = lr

            if amp_device:
                if amp_device == "cuda":
                    amp_ctx = amp_autocast(device_type="cuda", dtype=torch.float16)
                elif amp_device == "mps":
                    amp_ctx = amp_autocast(device_type="mps", dtype=torch.float16)
                else:
                    amp_ctx = nullcontext()

                with amp_ctx:
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
                elapsed = max(time.time() - start_time, 1e-6)
                tok_per_sec = total_tokens_seen / elapsed
                token_msg = f" tokens_seen={total_tokens_seen:,} tok/s={tok_per_sec:,.0f}"
                if target_tokens:
                    token_pct = 100.0 * total_tokens_seen / max(target_tokens, 1)
                    token_msg += f" target_progress={token_pct:.2f}%"
                log(
                    f"[train] step={global_step} batch={batch_idx} loss={opt_step_loss:.4f} "
                    f"lr={lr:.6f}{token_msg}"
                )
            if args.debug:
                log(
                    f"[debug] step={global_step} lr={lr:.6f} loss={opt_step_loss:.4f} "
                    f"accum={grad_accum_steps}"
                )

            # Step-based eval/checkpoint hooks so long epochs do not delay persistence.
            if global_step % cfg["train"]["eval_every"] == 0:
                model.eval()
                total_loss = 0.0
                count = 0
                log(f"[eval] running eval at step {global_step} ...")
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
                        out = model(x, return_loss=False)
                        logits = out if not isinstance(out, dict) else out.get("logits", out)
                        val_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                        total_loss += val_loss.item()
                        count += 1
                ppl = math.exp(total_loss / max(count, 1))
                log(f"[eval] step {global_step} loss {total_loss/max(count,1):.4f} ppl {ppl:.2f}")
                model.train()

            if global_step % save_every == 0:
                if not has_min_free_space(checkpoint_dir, args.min_free_gb, log):
                    continue
                ckpt_path = checkpoint_dir / f"ckpt_step_{global_step}.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "step": global_step,
                        "scaler": scaler.state_dict() if scaler else None,
                    },
                    ckpt_path,
                )
                log(f"Saved {ckpt_path}")
                if args.s3_checkpoint_uri:
                    sync_checkpoints_to_s3(checkpoint_dir, args.s3_checkpoint_uri, args.aws_bin, log)

            if global_step >= max_steps:
                break


if __name__ == "__main__":
    main()

