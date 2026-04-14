import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build a deterministic local instruction-smoke variant.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for train/val JSONL and meta.json",
    )
    parser.add_argument(
        "--variant-name",
        type=str,
        default="custom_smoke_variant",
        help="Variant name recorded in meta.json",
    )
    parser.add_argument(
        "--base-train",
        type=Path,
        default=Path("model_training/LLM/data/sft_smoke_instruction_curated/train_sft_instruction.jsonl"),
    )
    parser.add_argument(
        "--base-val",
        type=Path,
        default=Path("model_training/LLM/data/sft_smoke_instruction_story_curated/val_sft_instruction.jsonl"),
    )
    parser.add_argument(
        "--intro-boosters",
        type=Path,
        default=Path("data/smoke_instruction_intro_boosters.jsonl"),
    )
    parser.add_argument(
        "--story-boosters",
        type=Path,
        default=Path("data/smoke_instruction_story_exact_boosters.jsonl"),
    )
    parser.add_argument(
        "--arith-boosters",
        type=Path,
        default=Path("data/smoke_instruction_arith_boosters.jsonl"),
    )
    parser.add_argument("--intro-count", type=int, default=8)
    parser.add_argument("--story-count", type=int, default=8)
    parser.add_argument("--arith-count", type=int, default=4)
    args = parser.parse_args()

    base_train = read_jsonl(args.base_train)
    base_val = read_jsonl(args.base_val)
    intro_rows = read_jsonl(args.intro_boosters)[: args.intro_count]
    story_rows = read_jsonl(args.story_boosters)[: args.story_count]
    arith_rows = read_jsonl(args.arith_boosters)[: args.arith_count]

    train_rows = base_train + intro_rows + story_rows + arith_rows

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_sft_instruction.jsonl"
    val_path = output_dir / "val_sft_instruction.jsonl"
    meta_path = output_dir / "meta.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, base_val)

    meta = {
        "source": args.variant_name,
        "base_train": len(base_train),
        "base_val": len(base_val),
        "intro_train_added": len(intro_rows),
        "story_train_added": len(story_rows),
        "arith_train_added": len(arith_rows),
        "train_total": len(train_rows),
        "val_total": len(base_val),
        "base_train_path": str(args.base_train),
        "base_val_path": str(args.base_val),
        "intro_boosters_path": str(args.intro_boosters),
        "story_boosters_path": str(args.story_boosters),
        "arith_boosters_path": str(args.arith_boosters),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"train_path": str(train_path), "val_path": str(val_path), "meta_path": str(meta_path)}, indent=2))


if __name__ == "__main__":
    main()
