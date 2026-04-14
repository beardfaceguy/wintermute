"""
Preprocess fineweb + stack-smol into newline text and train a fresh tokenizer.

Intended to run on the EC2 training box (DLAMI). Assumes AWS CLI is configured
and boto3/pyarrow/sentencepiece are available (install if needed).

Steps:
1) Sync source datasets from S3 to local workspace (small manageable slices).
2) Convert fineweb parquet shards and stack-smol JSON files to newline text.
3) Write combined train/val text locally and (optionally) sync back to S3.
4) Train a SentencePiece BPE tokenizer on the combined text.
5) Upload tokenizer artifacts to S3.
"""

import argparse
import json
import random
import subprocess
from pathlib import Path
from typing import Iterable, List

import pyarrow.parquet as pq
import sentencepiece as spm


def run(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, check=True)


def sync_from_s3(prefix: str, dest: Path, aws_bin: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    run([aws_bin, "s3", "sync", prefix, str(dest)])


def sync_to_s3(src: Path, dest_prefix: str, aws_bin: str) -> None:
    run([aws_bin, "s3", "sync", str(src), dest_prefix])


def iter_fineweb_parquet(parquet_dir: Path, text_column: str = "text") -> Iterable[str]:
    for pfile in sorted(parquet_dir.glob("*.parquet")):
        try:
            pf = pq.ParquetFile(pfile)
            for batch in pf.iter_batches(columns=[text_column]):
                col = batch.column(text_column)
                for val in col:
                    if val is None:
                        continue
                    txt = str(val.as_py()).strip()
                    if txt:
                        yield txt
        except Exception as e:
            print(f"[warn] skipping {pfile} due to read error: {e}")
            continue


def iter_stack_smol_json(data_dir: Path, content_field: str = "content") -> Iterable[str]:
    for lang_dir in sorted(data_dir.glob("*")):
        jf = lang_dir / "data.json"
        if not jf.exists():
            continue
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                txt = obj.get(content_field) or obj.get("text")
                if txt:
                    txt = str(txt).strip()
                    if txt:
                        yield txt


def split_and_sample(
    sources: Iterable[str],
    train_path: Path,
    val_path: Path,
    val_ratio: float,
    spm_sample_lines: int,
    seed: int = 42,
) -> Path:
    """
    Streams sources -> train/val files without holding everything in memory.
    Reservoir-samples up to spm_sample_lines from the train portion for tokenizer training.
    """
    random.seed(seed)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    spm_sample: List[str] = []
    seen = 0

    with open(train_path, "w", encoding="utf-8") as f_train, open(val_path, "w", encoding="utf-8") as f_val:
        for raw in sources:
            line = raw.replace("\n", " ").strip()
            if not line:
                continue
            if random.random() < val_ratio:
                f_val.write(line + "\n")
            else:
                f_train.write(line + "\n")
                seen += 1
                if len(spm_sample) < spm_sample_lines:
                    spm_sample.append(line)
                else:
                    r = random.randrange(seen)
                    if r < spm_sample_lines:
                        spm_sample[r] = line

    spm_sample_path = train_path.parent / "spm_sample.txt"
    with open(spm_sample_path, "w", encoding="utf-8") as f:
        for line in spm_sample:
            f.write(line + "\n")
    return spm_sample_path


def train_spm(input_path: Path, model_prefix: Path, vocab_size: int) -> None:
    spm.SentencePieceTrainer.Train(
        input=str(input_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        byte_fallback=True,
        split_digits=True,
        allow_whitespace_only_pieces=False,
        normalization_rule_name="nfkc",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fineweb-s3", required=True, help="S3 prefix with parquet shards (e.g., s3://.../sample/100BT/)")
    ap.add_argument("--stack-smol-s3", required=True, help="S3 prefix with stack-smol data/ folder")
    ap.add_argument("--aws-bin", default="aws", help="AWS CLI binary")
    ap.add_argument("--work-dir", default="/mnt/data/preproc", help="Local working dir")
    ap.add_argument("--text-out-s3", required=True, help="S3 prefix to upload text outputs")
    ap.add_argument("--tokenizer-out-s3", required=True, help="S3 prefix to upload tokenizer artifacts")
    ap.add_argument("--spm-sample-lines", type=int, default=2_000_000, help="Lines to sample for tokenizer training")
    ap.add_argument("--vocab-size", type=int, default=50_000, help="SentencePiece vocab size")
    ap.add_argument("--val-ratio", type=float, default=0.05, help="Validation split ratio")
    args = ap.parse_args()

    work = Path(args.work_dir)
    fineweb_dir = work / "fineweb_parquet"
    stack_dir = work / "stack_smol"
    text_dir = work / "text"
    tok_dir = work / "tokenizer"

    print("Syncing fineweb...")
    sync_from_s3(args.fineweb_s3, fineweb_dir, args.aws_bin)
    print("Syncing stack-smol...")
    sync_from_s3(args.stack_smol_s3, stack_dir, args.aws_bin)

    train_path = text_dir / "train.txt"
    val_path = text_dir / "val.txt"

    print("Streaming convert + split + sample...")
    def sources():
        yield from iter_fineweb_parquet(fineweb_dir)
        yield from iter_stack_smol_json(stack_dir / "data")

    spm_sample = split_and_sample(
        sources(),
        train_path,
        val_path,
        val_ratio=args.val_ratio,
        spm_sample_lines=args.spm_sample_lines,
        seed=42,
    )
    tok_prefix = tok_dir / "bpe_50k_fw_stack"
    train_spm(spm_sample, tok_prefix, args.vocab_size)

    # Sync outputs to S3
    print("Uploading text and tokenizer to S3...")
    sync_to_s3(text_dir, args.text_out_s3, args.aws_bin)
    sync_to_s3(tok_dir, args.tokenizer_out_s3, args.aws_bin)
    print("Done.")


if __name__ == "__main__":
    main()
