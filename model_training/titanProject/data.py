"""
Dataset/dataloader scaffold for Titans LM.
Tokenization is pluggable; here we assume a text file with one sample per line.
"""

import hashlib
import json
import os
import shutil
import time
import uuid
from array import array
from bisect import bisect_right
from collections.abc import Callable, Iterable
from pathlib import Path
from urllib.parse import urlparse

try:
    import boto3
except Exception:  # boto3 is available on DLAMI; keep optional for local CPU runs
    boto3 = None

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

DEFAULT_CACHE_VERSION = 3
DEFAULT_SHARD_SIZE_TOKENS = 1_750_000


class TextWindowDataset(Dataset):
    """
    Disk-backed dataset that tokenizes the corpus once into a cache file and
    slices fixed windows lazily without keeping the whole token buffer in RAM.
    """

    def __init__(
        self,
        path: str,
        tokenizer: Callable[[str], list[int]],
        tokenizer_fingerprint: str,
        seq_len: int,
        max_tokens: int | None = None,
        log_fn: Callable[[str], None] | None = None,
        progress_every_lines: int = 200000,
        progress_label: str = "dataset",
    ):
        self.seq_len = seq_len
        self.token_cache = self._load_tokens(
            path,
            tokenizer,
            tokenizer_fingerprint,
            max_tokens,
            log_fn=log_fn,
            progress_every_lines=progress_every_lines,
            progress_label=progress_label,
        )
        if self.token_cache.num_tokens <= seq_len:
            raise ValueError(
                f"Not enough tokens ({self.token_cache.num_tokens}) for seq_len={seq_len}"
            )
        self.num_tokens = self.token_cache.num_tokens
        self.num_windows = (self.num_tokens - 1) // seq_len

    def _load_tokens(
        self,
        path: str,
        tokenizer: Callable[[str], list[int]],
        tokenizer_fingerprint: str,
        max_tokens: int | None,
        log_fn: Callable[[str], None] | None,
        progress_every_lines: int,
        progress_label: str,
    ) -> "TokenCache":
        def emit(msg: str) -> None:
            if log_fn is not None:
                log_fn(msg)
            else:
                print(msg, flush=True)

        cache_root = get_cache_root(path)
        shard_size_tokens = get_shard_size_tokens()
        cache_key = build_cache_key(path, tokenizer_fingerprint, max_tokens, shard_size_tokens)
        cache_dir = cache_root / cache_key
        manifest_path = cache_dir / "manifest.json"
        trust_existing = os.environ.get("TITAN_TOKEN_CACHE_TRUST_EXISTING", "").strip() == "1"
        source_fingerprint = None if trust_existing else get_source_fingerprint(path)

        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                version_ok = manifest.get("cache_version") == DEFAULT_CACHE_VERSION
                tokenizer_ok = manifest.get("tokenizer_fingerprint") == tokenizer_fingerprint
                shard_ok = int(manifest.get("shard_size_tokens", 0)) == shard_size_tokens
                source_ok = (
                    trust_existing or manifest.get("source_fingerprint") == source_fingerprint
                )
                if version_ok and tokenizer_ok and shard_ok and source_ok:
                    trust_note = " (trust_existing)" if trust_existing else ""
                    emit(
                        f"[data] [{progress_label}] reusing token cache dir={cache_dir} "
                        f"tokens={manifest.get('num_tokens', 0):,} shards={len(manifest.get('shards', []))}"
                        f"{trust_note}"
                    )
                    return TokenCache(
                        cache_dir=cache_dir, manifest_path=manifest_path, manifest=manifest
                    )
            except Exception:
                emit(f"[data] [{progress_label}] cache manifest unreadable; rebuilding {cache_dir}")

        emit(
            f"[data] [{progress_label}] token cache build start path={path}"
            + (f" max_tokens={max_tokens}" if max_tokens is not None else "")
        )
        cache_root.mkdir(parents=True, exist_ok=True)
        tmp_cache_dir = cache_root / f".{cache_key}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        tmp_cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            start_ts = time.time()
            last_log_ts = start_ts
            line_count = 0
            nonempty_line_count = 0
            token_count = 0
            token_buffer = array("I")
            shards = []
            shard_index = 0
            tokens_written = 0

            def flush_pending(force: bool = False) -> None:
                nonlocal token_buffer, shard_index, tokens_written
                while len(token_buffer) >= shard_size_tokens or (force and token_buffer):
                    if len(token_buffer) > shard_size_tokens:
                        chunk = token_buffer[:shard_size_tokens]
                        token_buffer = token_buffer[shard_size_tokens:]
                    else:
                        chunk = token_buffer
                        token_buffer = array("I")

                    shard_path = tmp_cache_dir / f"tokens-{shard_index:05d}.uint32.bin"
                    with open(shard_path, "wb") as shard_file:
                        chunk.tofile(shard_file)
                    shards.append(
                        {
                            "filename": shard_path.name,
                            "num_tokens": len(chunk),
                            "start_token": tokens_written,
                        }
                    )
                    tokens_written += len(chunk)
                    shard_index += 1

            for line in _iter_lines(path):
                line_count += 1
                line = line.strip()
                if not line:
                    if progress_every_lines > 0 and line_count % progress_every_lines == 0:
                        elapsed = max(time.time() - start_ts, 1e-6)
                        emit(
                            f"[data] [{progress_label}] lines={line_count:,} nonempty={nonempty_line_count:,} "
                            f"tokens={token_count:,} elapsed={elapsed:.1f}s tok/s={token_count / elapsed:,.0f}"
                        )
                    continue

                nonempty_line_count += 1
                line_tokens = tokenizer(line)
                if max_tokens is not None and token_count + len(line_tokens) > max_tokens:
                    line_tokens = line_tokens[: max_tokens - token_count]

                token_buffer.extend(line_tokens)
                token_count += len(line_tokens)
                now = time.time()
                should_log = progress_every_lines > 0 and line_count % progress_every_lines == 0
                if should_log or (now - last_log_ts) >= 120:
                    elapsed = max(now - start_ts, 1e-6)
                    emit(
                        f"[data] [{progress_label}] lines={line_count:,} nonempty={nonempty_line_count:,} "
                        f"tokens={token_count:,} elapsed={elapsed:.1f}s tok/s={token_count / elapsed:,.0f}"
                    )
                    last_log_ts = now

                flush_pending()

                if max_tokens is not None and token_count >= max_tokens:
                    elapsed = max(time.time() - start_ts, 1e-6)
                    emit(
                        f"[data] [{progress_label}] reached max_tokens={max_tokens:,} "
                        f"at lines={line_count:,} elapsed={elapsed:.1f}s"
                    )
                    break

            flush_pending(force=True)
            elapsed = max(time.time() - start_ts, 1e-6)
            emit(
                f"[data] [{progress_label}] token cache build done lines={line_count:,} "
                f"nonempty={nonempty_line_count:,} tokens={token_count:,} shards={len(shards):,} "
                f"elapsed={elapsed:.1f}s tok/s={token_count / elapsed:,.0f}"
            )
            if source_fingerprint is None:
                source_fingerprint = get_source_fingerprint(path)
            manifest = {
                "cache_version": DEFAULT_CACHE_VERSION,
                "path": path,
                "tokenizer_fingerprint": tokenizer_fingerprint,
                "max_tokens": max_tokens,
                "num_tokens": token_count,
                "line_count": line_count,
                "nonempty_line_count": nonempty_line_count,
                "dtype": "uint32",
                "shard_size_tokens": shard_size_tokens,
                "shards": shards,
                "source_fingerprint": source_fingerprint,
                "created_at_epoch_s": time.time(),
            }
            (tmp_cache_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True)
            )
            shutil.rmtree(cache_dir, ignore_errors=True)
            os.replace(tmp_cache_dir, cache_dir)
            return TokenCache(cache_dir=cache_dir, manifest_path=manifest_path, manifest=manifest)
        except Exception:
            shutil.rmtree(tmp_cache_dir, ignore_errors=True)
            raise

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        x = self.token_cache.slice(start, end)
        y = self.token_cache.slice(start + 1, end + 1)
        return x, y


class TokenCache:
    def __init__(self, cache_dir: Path, manifest_path: Path, manifest: dict):
        self.cache_dir = cache_dir
        self.manifest_path = manifest_path
        self.meta = manifest
        self.num_tokens = int(manifest["num_tokens"])
        self.shards = manifest.get("shards", [])
        self._shard_starts = [int(shard["start_token"]) for shard in self.shards]
        self._memmaps: list[np.memmap | None] = [None] * len(self.shards)

    @property
    def token_path(self) -> Path:
        if not self.shards:
            raise ValueError("Token cache has no shards.")
        return self.cache_dir / self.shards[0]["filename"]

    @property
    def meta_path(self) -> Path:
        return self.manifest_path

    def _get_memmap(self, shard_idx: int) -> np.memmap:
        memmap = self._memmaps[shard_idx]
        if memmap is None:
            shard = self.shards[shard_idx]
            shard_path = self.cache_dir / shard["filename"]
            memmap = np.memmap(
                shard_path, dtype=np.uint32, mode="r", shape=(int(shard["num_tokens"]),)
            )
            self._memmaps[shard_idx] = memmap
        return memmap

    def _find_shard_index(self, token_offset: int) -> int:
        shard_idx = bisect_right(self._shard_starts, token_offset) - 1
        if shard_idx < 0:
            raise IndexError(f"Token offset {token_offset} is before the first shard.")
        return shard_idx

    def slice(self, start: int, end: int) -> torch.Tensor:
        pieces = []
        pos = start
        while pos < end:
            shard_idx = self._find_shard_index(pos)
            shard = self.shards[shard_idx]
            shard_start = int(shard["start_token"])
            local_start = pos - shard_start
            available = int(shard["num_tokens"]) - local_start
            take = min(end - pos, available)
            memmap = self._get_memmap(shard_idx)
            pieces.append(np.asarray(memmap[local_start : local_start + take], dtype=np.int64))
            pos += take
        if len(pieces) == 1:
            return torch.from_numpy(pieces[0])
        return torch.from_numpy(np.concatenate(pieces, axis=0))


def build_dataloader(
    path: str,
    tokenizer: Callable[[str], list[int]],
    tokenizer_fingerprint: str,
    seq_len: int,
    batch_size: int,
    shuffle_buffer: int = 100000,  # kept for API compatibility; unused
    num_workers: int = 0,
    shuffle: bool = True,
    max_tokens: int | None = None,
    log_fn: Callable[[str], None] | None = None,
    progress_every_lines: int = 200000,
    progress_label: str = "dataset",
) -> DataLoader:
    # Accept local paths or s3://bucket/key. For S3, we stream lines without a full local copy.
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
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True
    )


def _iter_lines(path: str) -> Iterable[str]:
    if path.startswith("s3://"):
        if boto3 is None:
            raise RuntimeError("boto3 is required for s3:// paths but is not installed")
        parsed = urlparse(path)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        client = boto3.client("s3")
        obj = client.get_object(Bucket=bucket, Key=key)
        body = obj["Body"]
        for raw in body.iter_lines():
            if raw is None:
                continue
            yield raw.decode("utf-8", errors="ignore")
    else:
        with open(Path(path), encoding="utf-8") as f:
            for line in f:
                yield line


def get_cache_root(path: str) -> Path:
    if path.startswith("s3://"):
        return Path.home() / ".cache" / "wintermute" / "token_cache"
    return Path(path).resolve().parent / ".titan_token_cache"


def build_cache_key(
    path: str,
    tokenizer_fingerprint: str,
    max_tokens: int | None,
    shard_size_tokens: int,
) -> str:
    # Use basename so the cache is portable across data root paths
    # (e.g. /mnt/data/datasets/train.txt vs /opt/dlami/nvme/datasets/train.txt).
    # Source fingerprint (file content hash) guarantees correctness.
    canonical = os.path.basename(path) if not path.startswith("s3://") else path
    digest = hashlib.sha256()
    digest.update(canonical.encode("utf-8"))
    digest.update(b"\0")
    digest.update(tokenizer_fingerprint.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(max_tokens).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(shard_size_tokens).encode("utf-8"))
    return digest.hexdigest()[:24]


def get_source_fingerprint(path: str) -> str:
    if path.startswith("s3://"):
        if boto3 is None:
            return path
        parsed = urlparse(path)
        client = boto3.client("s3")
        try:
            head = client.head_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
        except Exception:
            return path
        etag = str(head.get("ETag", "")).strip('"')
        size = str(head.get("ContentLength", ""))
        last_modified = str(head.get("LastModified", ""))
        return f"s3:{parsed.netloc}/{parsed.path.lstrip('/')}:{etag}:{size}:{last_modified}"

    stat = Path(path).resolve().stat()
    return f"file:{Path(path).resolve()}:{stat.st_size}:{stat.st_mtime_ns}"


def get_shard_size_tokens() -> int:
    raw = os.environ.get("TITAN_TOKEN_CACHE_SHARD_SIZE_TOKENS")
    if raw is None:
        return DEFAULT_SHARD_SIZE_TOKENS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("TITAN_TOKEN_CACHE_SHARD_SIZE_TOKENS must be an integer") from exc
    if value <= 0:
        raise ValueError("TITAN_TOKEN_CACHE_SHARD_SIZE_TOKENS must be > 0")
    return value
