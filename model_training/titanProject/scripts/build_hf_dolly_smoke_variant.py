import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

DEFAULT_CATEGORIES = [
    "classification",
    "open_qa",
    "closed_qa",
    "information_extraction",
    "creative_writing",
]

REJECT_RESPONSE_SUBSTRINGS = (
    "as an ai",
    "language model",
    "open assistant",
    "i do not have access",
    "i don't have access",
    "i do not have personal",
    "i don't have personal",
)


def normalize_block_text(value: str) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = [line.rstrip() for line in text.split("\n")]
    while raw_lines and not raw_lines[0].strip():
        raw_lines.pop(0)
    while raw_lines and not raw_lines[-1].strip():
        raw_lines.pop()

    lines: list[str] = []
    prev_blank = False
    for line in raw_lines:
        line = " ".join(line.split())
        if not line:
            if not prev_blank:
                lines.append("")
            prev_blank = True
            continue
        lines.append(line)
        prev_blank = False
    return "\n".join(lines).strip()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def as_instruction_row(instruction: str, input_text: str, response: str) -> dict[str, str]:
    return {
        "format": "raschka_instruction",
        "instruction": instruction,
        "input": input_text,
        "response": response,
    }


def row_passes_filters(
    row: dict[str, str],
    *,
    max_instruction_chars: int,
    max_input_chars: int,
    min_response_chars: int,
    max_response_chars: int,
    max_response_newlines: int,
) -> bool:
    instruction = normalize_block_text(row.get("instruction", ""))
    input_text = normalize_block_text(row.get("context", ""))
    response = normalize_block_text(row.get("response", ""))

    if not instruction or not response:
        return False
    if len(instruction) > max_instruction_chars:
        return False
    if len(input_text) > max_input_chars:
        return False
    if len(response) < min_response_chars or len(response) > max_response_chars:
        return False
    if response.count("\n") > max_response_newlines:
        return False
    lowered = response.lower()
    if "http://" in lowered or "https://" in lowered or "```" in response:
        return False
    return not any(token in lowered for token in REJECT_RESPONSE_SUBSTRINGS)


def select_rows(
    *,
    dataset_id: str,
    split: str,
    categories: list[str],
    limit: int,
    max_instruction_chars: int,
    max_input_chars: int,
    min_response_chars: int,
    max_response_chars: int,
    max_response_newlines: int,
) -> list[dict[str, object]]:
    ds = load_dataset(dataset_id, split=split)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)

    for row in ds:
        category = str(row.get("category", "")).strip()
        if category not in categories:
            continue
        if not row_passes_filters(
            row,
            max_instruction_chars=max_instruction_chars,
            max_input_chars=max_input_chars,
            min_response_chars=min_response_chars,
            max_response_chars=max_response_chars,
            max_response_newlines=max_response_newlines,
        ):
            continue
        grouped[category].append(
            {
                "category": category,
                "instruction": normalize_block_text(row.get("instruction", "")),
                "input": normalize_block_text(row.get("context", "")),
                "response": normalize_block_text(row.get("response", "")),
            }
        )

    selected: list[dict[str, object]] = []
    positions = {category: 0 for category in categories}
    while len(selected) < limit:
        advanced = False
        for category in categories:
            idx = positions[category]
            if idx >= len(grouped[category]):
                continue
            selected.append(grouped[category][idx])
            positions[category] += 1
            advanced = True
            if len(selected) >= limit:
                break
        if not advanced:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deterministic Dolly-backed instruction-smoke variant."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for train/val JSONL and meta.json",
    )
    parser.add_argument(
        "--variant-name",
        type=str,
        default="cla99_micro_balanced_dolly_short",
        help="Variant name recorded in meta.json",
    )
    parser.add_argument(
        "--base-train",
        type=Path,
        default=Path(
            "model_training/LLM/data/sft_smoke_instruction_micro_balanced/train_sft_instruction.jsonl"
        ),
    )
    parser.add_argument(
        "--base-val",
        type=Path,
        default=Path(
            "model_training/LLM/data/sft_smoke_instruction_micro_balanced/val_sft_instruction.jsonl"
        ),
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="databricks/databricks-dolly-15k",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--categories",
        type=str,
        default=",".join(DEFAULT_CATEGORIES),
        help="Comma-separated Dolly categories to consider",
    )
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--max-instruction-chars", type=int, default=180)
    parser.add_argument("--max-input-chars", type=int, default=220)
    parser.add_argument("--min-response-chars", type=int, default=20)
    parser.add_argument("--max-response-chars", type=int, default=180)
    parser.add_argument("--max-response-newlines", type=int, default=3)
    args = parser.parse_args()

    categories = [part.strip() for part in args.categories.split(",") if part.strip()]
    if not categories:
        raise ValueError("At least one category is required")

    base_train = read_jsonl(args.base_train)
    base_val = read_jsonl(args.base_val)
    selected_rows = select_rows(
        dataset_id=args.dataset_id,
        split=args.split,
        categories=categories,
        limit=args.limit,
        max_instruction_chars=args.max_instruction_chars,
        max_input_chars=args.max_input_chars,
        min_response_chars=args.min_response_chars,
        max_response_chars=args.max_response_chars,
        max_response_newlines=args.max_response_newlines,
    )
    imported_rows = [
        as_instruction_row(row["instruction"], row["input"], row["response"])  # type: ignore[index]
        for row in selected_rows
    ]
    train_rows = base_train + imported_rows

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_sft_instruction.jsonl"
    val_path = output_dir / "val_sft_instruction.jsonl"
    hf_slice_path = output_dir / "hf_slice_instruction.jsonl"
    meta_path = output_dir / "meta.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, base_val)
    write_jsonl(hf_slice_path, imported_rows)

    category_counts = Counter(row["category"] for row in selected_rows)
    meta = {
        "source": args.variant_name,
        "base_train": len(base_train),
        "base_val": len(base_val),
        "hf_dataset_id": args.dataset_id,
        "hf_split": args.split,
        "hf_categories": categories,
        "hf_added": len(imported_rows),
        "hf_category_counts": dict(category_counts),
        "filters": {
            "max_instruction_chars": args.max_instruction_chars,
            "max_input_chars": args.max_input_chars,
            "min_response_chars": args.min_response_chars,
            "max_response_chars": args.max_response_chars,
            "max_response_newlines": args.max_response_newlines,
            "rejected_response_substrings": list(REJECT_RESPONSE_SUBSTRINGS),
        },
        "train_total": len(train_rows),
        "val_total": len(base_val),
        "base_train_path": str(args.base_train),
        "base_val_path": str(args.base_val),
        "hf_slice_path": str(hf_slice_path),
        "selected_preview": selected_rows[:5],
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "train_path": str(train_path),
                "val_path": str(val_path),
                "hf_slice_path": str(hf_slice_path),
                "meta_path": str(meta_path),
                "hf_added": len(imported_rows),
                "hf_category_counts": dict(category_counts),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
