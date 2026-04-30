"""
Shared training utilities used by train.py and finetune_sft.py.

This module is the single source of truth for config loading, tokenizer setup,
path resolution, LR scheduling, checkpoint I/O, S3 sync, disk space checks,
device selection, and distributed (DDP) helpers.
"""

import hashlib
import math
import os
import shutil
import subprocess
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse

import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import sentencepiece as spm

try:
    import boto3  # type: ignore
except Exception:
    boto3 = None

from data import TextWindowDataset
from model import is_hf_source, normalize_hf_source


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_path(path_str: str) -> Path:
    if path_str.startswith("s3://") or path_str.startswith("hf://"):
        return path_str  # type: ignore[return-value]
    p = Path(path_str)
    if p.is_absolute():
        return p
    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return cwd_candidate
    script_dir = Path(__file__).resolve().parent
    script_candidate = script_dir / p
    if script_candidate.exists():
        return script_candidate
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / p


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# LR scheduling
# ---------------------------------------------------------------------------

def cosine_lr(step, warmup, max_steps, base_lr, min_lr=0.0):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(1, max_steps - warmup)
    cos_val = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cos_val


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Disk space
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def resolve_checkpoint_dir(args_checkpoint_dir: str | None, fallback_dir: Path) -> Path:
    if args_checkpoint_dir:
        checkpoint_dir = Path(args_checkpoint_dir).expanduser()
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = (Path.cwd() / checkpoint_dir).resolve()
    else:
        checkpoint_dir = fallback_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def save_checkpoint(
    path: Path,
    model_state_dict: dict,
    opt_state_dict: dict,
    step: int,
    scaler_state_dict: dict | None = None,
    extra: dict | None = None,
) -> None:
    payload = {
        "model": model_state_dict,
        "opt": opt_state_dict,
        "step": step,
        "scaler": scaler_state_dict,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# S3 sync
# ---------------------------------------------------------------------------

def sync_checkpoints_to_s3(
    checkpoint_dir: Path,
    s3_uri: str,
    aws_bin: str,
    log_fn,
    glob_pattern: str = "ckpt_step_*.pt",
) -> None:
    cmd = [
        aws_bin, "s3", "sync",
        str(checkpoint_dir), s3_uri,
        "--exclude", "*",
        "--include", glob_pattern,
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


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    """Initialize the process group. Returns (rank, local_rank, world_size).

    When launched via torchrun, env vars RANK, LOCAL_RANK, WORLD_SIZE, and
    MASTER_ADDR/MASTER_PORT are set automatically. When launched with plain
    ``python``, we fall back to a single-process group so the rest of the code
    doesn't need special-casing.
    """
    if "RANK" in os.environ:
        # Token cache builds can take 60-80 min on first run; the default
        # 600s store timeout kills waiting ranks before rank 0 finishes.
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
    else:
        rank = 0
        local_rank = 0
        world_size = 1
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def reduce_scalar(value: float, world_size: int) -> float:
    """All-reduce a scalar value (mean) across ranks."""
    if world_size <= 1:
        return value
    t = torch.tensor(value, dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item() / world_size


# ---------------------------------------------------------------------------
# Distributed dataloader builder
# ---------------------------------------------------------------------------

def build_distributed_dataloader(
    path: str,
    tokenizer,
    tokenizer_fingerprint: str,
    seq_len: int,
    batch_size: int,
    rank: int,
    world_size: int,
    shuffle: bool = True,
    max_tokens=None,
    log_fn=None,
    progress_every_lines: int = 200000,
    progress_label: str = "dataset",
    num_workers: int = 2,
) -> tuple:
    """Build dataset + distributed-aware dataloader. Returns (loader, sampler)."""
    ds = TextWindowDataset(
        path,
        tokenizer,
        tokenizer_fingerprint,
        seq_len,
        max_tokens=max_tokens,
        log_fn=log_fn,
        progress_every_lines=progress_every_lines,
        progress_label=progress_label,
    )
    if world_size > 1:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=shuffle)
        loader = DataLoader(
            ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
    else:
        sampler = None
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
    return loader, sampler
