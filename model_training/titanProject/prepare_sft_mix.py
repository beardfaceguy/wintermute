"""
Prepare an instruction/chat fine-tuning corpus from OASST1 + Dolly.

Output format is one training sample per line:
    User: <prompt> Assistant: <response>
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset


def normalize_text(value: str) -> str:
    text = str(value or "").strip()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def format_pair(user_text: str, assistant_text: str) -> str:
    user_clean = normalize_text(user_text)
    assistant_clean = normalize_text(assistant_text)
    if not user_clean or not assistant_clean:
        return ""
    return f"User: {user_clean} Assistant: {assistant_clean}"


def _is_english(lang_value: object) -> bool:
    lang = str(lang_value or "").lower()
    if not lang:
        return True
    return lang.startswith("en")


def collect_oasst_pairs(split: str) -> List[str]:
    ds = load_dataset("OpenAssistant/oasst1", split=split)
    rows = [dict(row) for row in ds]
    by_id: Dict[str, Dict[str, object]] = {}
    for row in rows:
        mid = str(row.get("message_id", "")).strip()
        if mid:
            by_id[mid] = row

    pairs: List[str] = []
    seen = set()
    for row in rows:
        role = str(row.get("role", "")).lower()
        if role != "assistant":
            continue
        if row.get("deleted", False):
            continue
        if not _is_english(row.get("lang")):
            continue
        parent_id = str(row.get("parent_id", "")).strip()
        if not parent_id or parent_id not in by_id:
            continue
        parent = by_id[parent_id]
        parent_role = str(parent.get("role", "")).lower()
        if parent_role not in {"prompter", "user", "human"}:
            continue
        if parent.get("deleted", False):
            continue
        if not _is_english(parent.get("lang")):
            continue

        sample = format_pair(str(parent.get("text", "")), str(row.get("text", "")))
        if not sample:
            continue
        if sample in seen:
            continue
        seen.add(sample)
        pairs.append(sample)
    return pairs


def collect_dolly_pairs() -> List[str]:
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    pairs: List[str] = []
    seen = set()
    for row in ds:
        instruction = normalize_text(row.get("instruction", ""))
        context = normalize_text(row.get("context", ""))
        response = normalize_text(row.get("response", ""))
        if not instruction or not response:
            continue
        user_text = instruction if not context else f"{instruction} Context: {context}"
        sample = format_pair(user_text, response)
        if not sample:
            continue
        if sample in seen:
            continue
        seen.add(sample)
        pairs.append(sample)
    return pairs


def sample_up_to(items: List[str], n: int, rng: random.Random) -> List[str]:
    if n <= 0 or n >= len(items):
        return list(items)
    return rng.sample(items, n)


def write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build OASST1 + Dolly SFT mix data.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model_training/LLM/data/sft_mix",
        help="Directory for train/val output files",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--oasst-train-pairs", type=int, default=24000)
    parser.add_argument("--oasst-val-pairs", type=int, default=2000)
    parser.add_argument("--dolly-train-pairs", type=int, default=10000)
    parser.add_argument("--dolly-val-pairs", type=int, default=1000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)

    print("[prep] loading OASST1 train/validation ...")
    oasst_train_all = collect_oasst_pairs("train")
    oasst_val_all = collect_oasst_pairs("validation")
    print(f"[prep] OASST1 pairs available: train={len(oasst_train_all)}, val={len(oasst_val_all)}")

    print("[prep] loading Dolly-15k ...")
    dolly_all = collect_dolly_pairs()
    rng.shuffle(dolly_all)
    print(f"[prep] Dolly pairs available: {len(dolly_all)}")

    oasst_train = sample_up_to(oasst_train_all, args.oasst_train_pairs, rng)
    oasst_val = sample_up_to(oasst_val_all, args.oasst_val_pairs, rng)

    dolly_val_n = min(args.dolly_val_pairs, len(dolly_all))
    dolly_val = dolly_all[:dolly_val_n]
    dolly_train_pool = dolly_all[dolly_val_n:]
    dolly_train = sample_up_to(dolly_train_pool, args.dolly_train_pairs, rng)

    train_lines = list(oasst_train) + list(dolly_train)
    val_lines = list(oasst_val) + list(dolly_val)
    rng.shuffle(train_lines)
    rng.shuffle(val_lines)

    train_path = out_dir / "train_sft_mix.txt"
    val_path = out_dir / "val_sft_mix.txt"
    meta_path = out_dir / "meta.json"

    write_lines(train_path, train_lines)
    write_lines(val_path, val_lines)

    meta = {
        "seed": args.seed,
        "counts": {
            "train_total": len(train_lines),
            "val_total": len(val_lines),
            "oasst_train_used": len(oasst_train),
            "oasst_val_used": len(oasst_val),
            "dolly_train_used": len(dolly_train),
            "dolly_val_used": len(dolly_val),
        },
        "paths": {"train": str(train_path), "val": str(val_path)},
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[prep] wrote train: {train_path} ({len(train_lines)} lines)")
    print(f"[prep] wrote val:   {val_path} ({len(val_lines)} lines)")
    print(f"[prep] wrote meta:  {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
