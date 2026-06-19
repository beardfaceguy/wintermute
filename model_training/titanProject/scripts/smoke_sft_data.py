#!/usr/bin/env python3
"""Pre-flight SFT data smoke test.

Loads the HF tokenizer, builds ``MaskedSFTDataset`` on a head slice of the
training corpus, and asserts that the keep-rate is high enough to be worth
spending GPU-hours on. Designed to be cheap (CPU-only, no model weights) so it
can run on the controller before submitting an AWS SSM training command, and
again on the GPU host before launching torchrun.

Catches the three classes of bug that wasted ~$525 of p4d.24xlarge time on
2026-05-15:
  1. Missing python dependency (``titans-pytorch``) — import smoke fails fast.
  2. Stale or incomplete code bundle (e.g. missing ``data.py``) — same.
  3. Data/code shape mismatch (e.g. trainer rejects every sample) — dataset
     build produces a measurable keep-rate that we can threshold.

Exit codes:
  0  smoke passed
  1  smoke failed (low keep-rate)
  2  pre-condition failed (bad args, missing file, import failure)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Make the titanProject root importable when invoked as a script from anywhere.
TITAN_ROOT = Path(__file__).resolve().parent.parent
if str(TITAN_ROOT) not in sys.path:
    sys.path.insert(0, str(TITAN_ROOT))


def _download_s3_head(uri: str, max_bytes: int, dest_dir: Path) -> Path:
    """Fetch the head of an S3 object to a local file. Uses the aws CLI to
    avoid pulling in boto3 just for a smoke test."""
    if shutil.which("aws") is None:
        raise RuntimeError("aws CLI is required to fetch s3:// inputs")
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3 URI: {uri}")
    without_scheme = uri[len("s3://") :]
    bucket, _, key = without_scheme.partition("/")
    if not bucket or not key:
        raise ValueError(f"malformed s3 URI: {uri}")
    dest = dest_dir / "sample.jsonl"
    cmd = [
        "aws",
        "s3api",
        "get-object",
        "--bucket",
        bucket,
        "--key",
        key,
        "--range",
        f"bytes=0-{max_bytes - 1}",
    ]
    profile = os.environ.get("AWS_PROFILE")
    if profile:
        cmd[1:1] = ["--profile", profile]
    cmd.append(str(dest))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"aws s3api get-object failed (rc={proc.returncode}): {proc.stderr.strip()}"
        )
    return dest


def _trim_to_complete_lines(path: Path, max_lines: int) -> Path:
    """Drop the final (possibly truncated) line of a ranged S3 download and
    cap to ``max_lines`` complete lines. Writes to a sibling file."""
    out = path.with_suffix(".trimmed.jsonl")
    kept = 0
    last_line: str | None = None
    with (
        path.open("r", encoding="utf-8", errors="replace") as src,
        out.open("w", encoding="utf-8") as dst,
    ):
        for raw in src:
            if last_line is not None:
                dst.write(last_line)
                kept += 1
                if kept >= max_lines:
                    break
            last_line = raw if raw.endswith("\n") else None
    return out


def _load_tokenizer(hf_model: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise RuntimeError("transformers is required for the smoke test") from e
    tok = AutoTokenizer.from_pretrained(hf_model)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok


def run_smoke(
    data_path: Path,
    *,
    hf_model: str,
    seq_len: int,
    max_lines: int | None,
    chat_template: bool,
    min_keep_rate: float,
    min_kept: int,
) -> int:
    """Run the smoke check. Returns process exit code."""
    try:
        from finetune_sft import MaskedSFTDataset, _hf_tokenizer_to_adapter
    except ImportError as exc:
        print(f"[smoke] FAIL: cannot import titanProject modules: {exc}", file=sys.stderr)
        print(
            "[smoke] hint: ensure pip install titans-pytorch (and the rest of the bundle)",
            file=sys.stderr,
        )
        return 2

    if not data_path.exists():
        print(f"[smoke] FAIL: data file not found: {data_path}", file=sys.stderr)
        return 2

    sample_path = data_path
    if max_lines is not None:
        sample_path = data_path.with_suffix(".smoke-head.jsonl")
        with (
            data_path.open("r", encoding="utf-8") as src,
            sample_path.open("w", encoding="utf-8") as dst,
        ):
            for i, raw in enumerate(src):
                if i >= max_lines:
                    break
                dst.write(raw)

    print(f"[smoke] loading tokenizer: {hf_model}")
    hf_tokenizer = _load_tokenizer(hf_model)
    adapter = _hf_tokenizer_to_adapter(hf_tokenizer)

    print(
        f"[smoke] building MaskedSFTDataset path={sample_path} seq_len={seq_len} "
        f"chat_template={chat_template}"
    )
    log_lines: list[str] = []
    try:
        ds = MaskedSFTDataset(
            path=str(sample_path),
            tokenizer=adapter,
            seq_len=seq_len,
            log_fn=log_lines.append,
            progress_label="smoke",
            chat_template_tokenizer=hf_tokenizer if chat_template else None,
        )
    except ValueError as e:
        # MaskedSFTDataset raises this when 0 samples survived. Surface it.
        print(f"[smoke] FAIL: dataset rejected every sample: {e}", file=sys.stderr)
        for ln in log_lines[-5:]:
            print(f"[smoke]   {ln}", file=sys.stderr)
        return 1

    kept = len(ds)
    total_seen: int | None = None
    for ln in log_lines:
        if "total=" in ln:
            for chunk in ln.split():
                if chunk.startswith("total="):
                    try:
                        total_seen = int(chunk[len("total=") :].rstrip(",").replace(",", ""))
                    except ValueError:
                        pass
        print(f"[smoke] {ln}")

    keep_rate = kept / total_seen if total_seen else 0.0
    print(f"[smoke] kept={kept} total={total_seen} keep_rate={keep_rate:.3f}")

    if kept < min_kept:
        print(
            f"[smoke] FAIL: kept={kept} below min_kept={min_kept}",
            file=sys.stderr,
        )
        return 1
    if total_seen and keep_rate < min_keep_rate:
        print(
            f"[smoke] FAIL: keep_rate={keep_rate:.3f} below threshold={min_keep_rate:.3f}",
            file=sys.stderr,
        )
        return 1

    print("[smoke] PASS")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--data",
        required=True,
        help="Local path or s3:// URI to a JSONL training file",
    )
    p.add_argument(
        "--hf-model",
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="HF model id or local path used to load the tokenizer",
    )
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument(
        "--max-lines",
        type=int,
        default=5000,
        help="Cap on number of lines to evaluate (None for full file)",
    )
    p.add_argument(
        "--s3-head-mb",
        type=int,
        default=16,
        help="When --data is s3://, fetch this many MB from the head",
    )
    p.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable chat-template tokenization (test the plain-text path)",
    )
    p.add_argument("--min-keep-rate", type=float, default=0.90)
    p.add_argument("--min-kept", type=int, default=100)
    args = p.parse_args()

    cleanup_dir: tempfile.TemporaryDirectory | None = None
    try:
        if args.data.startswith("s3://"):
            cleanup_dir = tempfile.TemporaryDirectory(prefix="smoke_sft_")
            head_bytes = args.s3_head_mb * 1024 * 1024
            local = _download_s3_head(args.data, head_bytes, Path(cleanup_dir.name))
            trim_lines = args.max_lines if args.max_lines is not None else 1_000_000
            data_path = _trim_to_complete_lines(local, trim_lines)
            max_lines = None  # already capped by the trimmer
        else:
            data_path = Path(args.data)
            max_lines = args.max_lines

        return run_smoke(
            data_path,
            hf_model=args.hf_model,
            seq_len=args.seq_len,
            max_lines=max_lines,
            chat_template=not args.no_chat_template,
            min_keep_rate=args.min_keep_rate,
            min_kept=args.min_kept,
        )
    except Exception as e:
        print(f"[smoke] ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    finally:
        if cleanup_dir is not None:
            cleanup_dir.cleanup()


if __name__ == "__main__":
    sys.exit(main())
