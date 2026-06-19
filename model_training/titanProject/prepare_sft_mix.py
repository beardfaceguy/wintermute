"""
Prepare an instruction/chat fine-tuning corpus from multiple sources.

Current sources:
- OASST1 (English-only pairs)
- OpenHermes (ShareGPT-style conversations)
- SlimOrca (ShareGPT-style conversations)
- Logic/math boosters (GSM8K style)

Output formats:
- `chat_text`: one training sample per line as `User: <prompt> Assistant: <response>`
- `instruction_jsonl`: one JSON object per line with `instruction`, optional `input`, and `response`
"""

import argparse
import json
import random
import re
from collections.abc import Iterable
from pathlib import Path

from datasets import load_dataset

SamplePair = tuple[str, str]


def normalize_text(value: str) -> str:
    text = str(value or "").strip()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_block_text(value: str) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    while raw_lines and not raw_lines[0]:
        raw_lines.pop(0)
    while raw_lines and not raw_lines[-1]:
        raw_lines.pop()

    lines: list[str] = []
    prev_blank = False
    for line in raw_lines:
        if not line:
            if not prev_blank:
                lines.append("")
            prev_blank = True
            continue
        lines.append(line)
        prev_blank = False
    return "\n".join(lines).strip()


def make_pair(user_text: str, assistant_text: str) -> SamplePair | None:
    user_clean = normalize_block_text(user_text)
    assistant_clean = normalize_block_text(assistant_text)
    if not user_clean or not assistant_clean:
        return None
    return user_clean, assistant_clean


def format_chat_pair(user_text: str, assistant_text: str) -> str:
    user_clean = normalize_text(user_text)
    assistant_clean = normalize_text(assistant_text)
    if not user_clean or not assistant_clean:
        return ""
    return f"User: {user_clean} Assistant: {assistant_clean}"


def render_instruction_prompt(instruction_text: str, input_text: str = "") -> str:
    instruction_clean = normalize_block_text(instruction_text)
    input_clean = normalize_block_text(input_text)
    if not instruction_clean:
        return ""
    prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction_clean}"
    )
    if input_clean:
        prompt += f"\n\n### Input:\n{input_clean}"
    prompt += "\n\n### Response:\n"
    return prompt


def serialize_pair(user_text: str, assistant_text: str, *, output_format: str) -> str:
    if output_format == "chat_text":
        return format_chat_pair(user_text, assistant_text)
    if output_format == "instruction_jsonl":
        instruction_clean = normalize_block_text(user_text)
        response_clean = normalize_block_text(assistant_text)
        if not instruction_clean or not response_clean:
            return ""
        return json.dumps(
            {
                "format": "raschka_instruction",
                "instruction": instruction_clean,
                "input": "",
                "response": response_clean,
            },
            ensure_ascii=True,
        )
    raise ValueError(f"Unsupported output format: {output_format}")


def _digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    digits = sum(ch.isdigit() for ch in text)
    return digits / max(1, len(text))


def _label_value(row: dict[str, object], label_name: str) -> float | None:
    labels = row.get("labels")
    if not isinstance(labels, dict):
        return None
    names = labels.get("name")
    values = labels.get("value")
    if not isinstance(names, list) or not isinstance(values, list):
        return None
    for name, value in zip(names, values, strict=False):
        if name == label_name:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def filter_pair(
    pair: SamplePair,
    *,
    max_user_chars: int,
    max_assistant_chars: int,
    max_digit_ratio: float,
    allow_http: bool,
    reject_role_markers: bool,
) -> bool:
    """
    Returns True if pair passes filters.
    """
    if not pair:
        return False
    user_part, assistant_part = pair
    user_part = normalize_block_text(user_part)
    assistant_part = normalize_block_text(assistant_part)

    if len(user_part) == 0 or len(assistant_part) == 0:
        return False
    if reject_role_markers:
        lowered_user = user_part.lower()
        lowered_assistant = assistant_part.lower()
        if "user:" in lowered_user or "assistant:" in lowered_user:
            return False
        if "user:" in lowered_assistant or "assistant:" in lowered_assistant:
            return False
    if len(user_part) > max_user_chars or len(assistant_part) > max_assistant_chars:
        return False
    if not allow_http and (
        "http://" in user_part
        or "https://" in user_part
        or "http://" in assistant_part
        or "https://" in assistant_part
    ):
        return False
    if _digit_ratio(assistant_part) > max_digit_ratio and len(assistant_part) > 50:
        return False
    return True


def _is_english(lang_value: object) -> bool:
    lang = str(lang_value or "").lower()
    if not lang:
        return True
    return lang.startswith("en")


def collect_oasst_pairs(
    split: str,
    *,
    best_only: bool = False,
    min_quality: float | None = None,
    min_helpfulness: float | None = None,
    max_fails_task: float | None = None,
    max_spam: float | None = None,
) -> list[SamplePair]:
    ds = load_dataset("OpenAssistant/oasst1", split=split)
    rows = [dict(row) for row in ds]
    by_id: dict[str, dict[str, object]] = {}
    for row in rows:
        mid = str(row.get("message_id", "")).strip()
        if mid:
            by_id[mid] = row

    pairs: list[SamplePair] = []
    seen = set()
    for row in rows:
        role = str(row.get("role", "")).lower()
        if role != "assistant":
            continue
        if row.get("deleted", False):
            continue
        if not _is_english(row.get("lang")):
            continue
        if best_only and row.get("rank") != 0:
            continue
        quality = _label_value(row, "quality")
        helpfulness = _label_value(row, "helpfulness")
        fails_task = _label_value(row, "fails_task")
        spam = _label_value(row, "spam")
        if min_quality is not None and (quality is None or quality < min_quality):
            continue
        if min_helpfulness is not None and (helpfulness is None or helpfulness < min_helpfulness):
            continue
        if max_fails_task is not None and fails_task is not None and fails_task > max_fails_task:
            continue
        if max_spam is not None and spam is not None and spam > max_spam:
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

        pair = make_pair(str(parent.get("text", "")), str(row.get("text", "")))
        if not pair:
            continue
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    return pairs


def _iter_sharegpt_pairs(
    rows: Iterable[dict[str, object]],
    *,
    conv_key: str = "conversations",
) -> list[SamplePair]:
    """
    Extract user->assistant pairs from ShareGPT-style conversation lists.
    Each conversation item is expected to have keys like {"from": "human"/"assistant", "value": "..."}.
    """
    pairs: list[SamplePair] = []
    seen = set()

    for row in rows:
        convs = row.get(conv_key) or row.get("conversation") or row.get("messages")
        if not convs:
            continue
        last_user: str | None = None
        for msg in convs:
            if not isinstance(msg, dict):
                continue
            role_raw = str(msg.get("from") or msg.get("role") or "").lower()
            text_raw = msg.get("value") or msg.get("content") or msg.get("text") or ""
            text = normalize_block_text(text_raw)
            if not text:
                continue
            if role_raw in {"human", "user", "prompter"}:
                last_user = text
                continue
            if role_raw in {"assistant", "gpt", "bot"} and last_user:
                pair = make_pair(last_user, text)
                if pair and pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
                last_user = None
        # If conversation ended without assistant after last_user, drop it silently.
    return pairs


def collect_openhermes_pairs() -> list[SamplePair]:
    ds = load_dataset("teknium/OpenHermes-2.5", split="train")
    return _iter_sharegpt_pairs(ds)


def collect_slimorca_pairs() -> list[SamplePair]:
    ds = load_dataset("Open-Orca/SlimOrca", split="train")
    return _iter_sharegpt_pairs(ds)


def collect_logic_pairs(split: str, max_len: int = 0) -> list[SamplePair]:
    """
    Logic/math booster from GSM8K. Uses question/answer pairs directly.
    """
    ds = load_dataset("gsm8k", "main", split=split)
    items: list[SamplePair] = []
    for row in ds:
        pair = make_pair(row.get("question", ""), row.get("answer", ""))
        if pair:
            items.append(pair)
        if max_len and len(items) >= max_len:
            break
    return items


def sample_up_to(items: list[SamplePair], n: int, rng: random.Random) -> list[SamplePair]:
    if n <= 0:
        return []
    if n >= len(items):
        return list(items)
    return rng.sample(items, n)


def filter_pairs(
    items: list[SamplePair],
    *,
    max_user_chars: int,
    max_assistant_chars: int,
    max_digit_ratio: float,
    allow_http: bool,
    reject_role_markers: bool,
) -> list[SamplePair]:
    keep: list[SamplePair] = []
    for pair in items:
        if filter_pair(
            pair,
            max_user_chars=max_user_chars,
            max_assistant_chars=max_assistant_chars,
            max_digit_ratio=max_digit_ratio,
            allow_http=allow_http,
            reject_role_markers=reject_role_markers,
        ):
            keep.append(pair)
    return keep


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SFT mix data.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model_training/LLM/data/sft_mix",
        help="Directory for train/val output files",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-format",
        type=str,
        default="chat_text",
        choices=["chat_text", "instruction_jsonl"],
        help="Output sample format; keep chat_text for legacy path or use instruction_jsonl for Branch A Raschka-style prep",
    )
    parser.add_argument("--oasst-train-pairs", type=int, default=24000)
    parser.add_argument("--oasst-val-pairs", type=int, default=2000)
    parser.add_argument("--openhermes-train-pairs", type=int, default=0)
    parser.add_argument("--openhermes-val-pairs", type=int, default=0)
    parser.add_argument("--slimorca-train-pairs", type=int, default=0)
    parser.add_argument("--slimorca-val-pairs", type=int, default=0)
    parser.add_argument("--logic-train-pairs", type=int, default=0)
    parser.add_argument("--logic-val-pairs", type=int, default=0)
    parser.add_argument("--max-user-chars", type=int, default=512)
    parser.add_argument("--max-assistant-chars", type=int, default=512)
    parser.add_argument("--max-digit-ratio", type=float, default=0.25)
    parser.add_argument(
        "--allow-http", action="store_true", help="Allow samples containing http/https"
    )
    parser.add_argument(
        "--reject-role-markers",
        action="store_true",
        help="Drop samples whose user/assistant text already contains literal User:/Assistant: markers",
    )
    parser.add_argument(
        "--oasst-best-only",
        action="store_true",
        help="Keep only rank-0 OASST1 assistant replies",
    )
    parser.add_argument(
        "--oasst-min-quality",
        type=float,
        default=None,
        help="Minimum OASST1 quality label required for assistant replies",
    )
    parser.add_argument(
        "--oasst-min-helpfulness",
        type=float,
        default=None,
        help="Minimum OASST1 helpfulness label required for assistant replies",
    )
    parser.add_argument(
        "--oasst-max-fails-task",
        type=float,
        default=None,
        help="Maximum OASST1 fails_task label allowed for assistant replies",
    )
    parser.add_argument(
        "--oasst-max-spam",
        type=float,
        default=None,
        help="Maximum OASST1 spam label allowed for assistant replies",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)

    oasst_train_all: list[SamplePair] = []
    oasst_val_all: list[SamplePair] = []
    if args.oasst_train_pairs > 0 or args.oasst_val_pairs > 0:
        print("[prep] loading OASST1 train/validation ...")
        oasst_train_all = collect_oasst_pairs(
            "train",
            best_only=args.oasst_best_only,
            min_quality=args.oasst_min_quality,
            min_helpfulness=args.oasst_min_helpfulness,
            max_fails_task=args.oasst_max_fails_task,
            max_spam=args.oasst_max_spam,
        )
        oasst_val_all = collect_oasst_pairs(
            "validation",
            best_only=args.oasst_best_only,
            min_quality=args.oasst_min_quality,
            min_helpfulness=args.oasst_min_helpfulness,
            max_fails_task=args.oasst_max_fails_task,
            max_spam=args.oasst_max_spam,
        )
        print(
            f"[prep] OASST1 pairs available: train={len(oasst_train_all)}, val={len(oasst_val_all)}"
        )

    openhermes_all: list[SamplePair] = []
    if args.openhermes_train_pairs > 0 or args.openhermes_val_pairs > 0:
        print("[prep] loading OpenHermes ...")
        openhermes_all = collect_openhermes_pairs()
        rng.shuffle(openhermes_all)
        print(f"[prep] OpenHermes pairs available: {len(openhermes_all)}")

    slimorca_all: list[SamplePair] = []
    if args.slimorca_train_pairs > 0 or args.slimorca_val_pairs > 0:
        print("[prep] loading SlimOrca ...")
        slimorca_all = collect_slimorca_pairs()
        rng.shuffle(slimorca_all)
        print(f"[prep] SlimOrca pairs available: {len(slimorca_all)}")

    logic_all: list[SamplePair] = []
    if args.logic_train_pairs > 0 or args.logic_val_pairs > 0:
        print("[prep] loading GSM8K ...")
        # max_len here is just a mild guard; sampling happens later.
        logic_all = collect_logic_pairs("train")
        rng.shuffle(logic_all)
        print(f"[prep] Logic pairs available: {len(logic_all)}")

    oasst_train = sample_up_to(
        filter_pairs(
            oasst_train_all,
            max_user_chars=args.max_user_chars,
            max_assistant_chars=args.max_assistant_chars,
            max_digit_ratio=args.max_digit_ratio,
            allow_http=args.allow_http,
            reject_role_markers=args.reject_role_markers,
        ),
        args.oasst_train_pairs,
        rng,
    )
    oasst_val = sample_up_to(
        filter_pairs(
            oasst_val_all,
            max_user_chars=args.max_user_chars,
            max_assistant_chars=args.max_assistant_chars,
            max_digit_ratio=args.max_digit_ratio,
            allow_http=args.allow_http,
            reject_role_markers=args.reject_role_markers,
        ),
        args.oasst_val_pairs,
        rng,
    )

    openhermes_val_n = min(args.openhermes_val_pairs, len(openhermes_all))
    openhermes_filtered = filter_pairs(
        openhermes_all,
        max_user_chars=args.max_user_chars,
        max_assistant_chars=args.max_assistant_chars,
        max_digit_ratio=args.max_digit_ratio,
        allow_http=args.allow_http,
        reject_role_markers=args.reject_role_markers,
    )
    openhermes_val = openhermes_filtered[:openhermes_val_n]
    openhermes_train_pool = openhermes_filtered[openhermes_val_n:]
    openhermes_train = sample_up_to(openhermes_train_pool, args.openhermes_train_pairs, rng)

    slimorca_val_n = min(args.slimorca_val_pairs, len(slimorca_all))
    slimorca_filtered = filter_pairs(
        slimorca_all,
        max_user_chars=args.max_user_chars,
        max_assistant_chars=args.max_assistant_chars,
        max_digit_ratio=args.max_digit_ratio,
        allow_http=args.allow_http,
        reject_role_markers=args.reject_role_markers,
    )
    slimorca_val = slimorca_filtered[:slimorca_val_n]
    slimorca_train_pool = slimorca_filtered[slimorca_val_n:]
    slimorca_train = sample_up_to(slimorca_train_pool, args.slimorca_train_pairs, rng)

    logic_val_n = min(args.logic_val_pairs, len(logic_all))
    logic_filtered = filter_pairs(
        logic_all,
        max_user_chars=args.max_user_chars,
        max_assistant_chars=args.max_assistant_chars,
        max_digit_ratio=args.max_digit_ratio,
        allow_http=args.allow_http,
        reject_role_markers=args.reject_role_markers,
    )
    logic_val = logic_filtered[:logic_val_n]
    logic_train_pool = logic_filtered[logic_val_n:]
    logic_train = sample_up_to(logic_train_pool, args.logic_train_pairs, rng)

    train_pairs = (
        list(oasst_train) + list(openhermes_train) + list(slimorca_train) + list(logic_train)
    )
    val_pairs = list(oasst_val) + list(openhermes_val) + list(slimorca_val) + list(logic_val)
    train_lines = [
        serialize_pair(user, assistant, output_format=args.output_format)
        for user, assistant in train_pairs
    ]
    val_lines = [
        serialize_pair(user, assistant, output_format=args.output_format)
        for user, assistant in val_pairs
    ]
    train_lines = [line for line in train_lines if line]
    val_lines = [line for line in val_lines if line]
    rng.shuffle(train_lines)
    rng.shuffle(val_lines)

    suffix = "txt" if args.output_format == "chat_text" else "jsonl"
    stem = "sft_mix" if args.output_format == "chat_text" else "sft_instruction"
    train_path = out_dir / f"train_{stem}.{suffix}"
    val_path = out_dir / f"val_{stem}.{suffix}"
    meta_path = out_dir / "meta.json"

    write_lines(train_path, train_lines)
    write_lines(val_path, val_lines)

    meta = {
        "seed": args.seed,
        "output_format": args.output_format,
        "filters": {
            "max_user_chars": args.max_user_chars,
            "max_assistant_chars": args.max_assistant_chars,
            "max_digit_ratio": args.max_digit_ratio,
            "allow_http": args.allow_http,
            "reject_role_markers": args.reject_role_markers,
            "oasst_best_only": args.oasst_best_only,
            "oasst_min_quality": args.oasst_min_quality,
            "oasst_min_helpfulness": args.oasst_min_helpfulness,
            "oasst_max_fails_task": args.oasst_max_fails_task,
            "oasst_max_spam": args.oasst_max_spam,
        },
        "counts": {
            "train_total": len(train_lines),
            "val_total": len(val_lines),
            "oasst_train_used": len(oasst_train),
            "oasst_val_used": len(oasst_val),
            "openhermes_train_used": len(openhermes_train),
            "openhermes_val_used": len(openhermes_val),
            "slimorca_train_used": len(slimorca_train),
            "slimorca_val_used": len(slimorca_val),
            "logic_train_used": len(logic_train),
            "logic_val_used": len(logic_val),
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
