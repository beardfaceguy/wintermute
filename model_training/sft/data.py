"""
SFT data loading: chat-format JSONL → validated examples → HF Datasets.

Canonical input is one JSON object per line, each with a "messages" list in
OpenAI chat format:

    {"messages": [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."}
    ]}

Loading, validation, and splitting are stdlib-only so they can be tested without
transformers/datasets. Only build_datasets() touches the `datasets` library, and
it fails with a clear message if it is absent.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

VALID_ROLES = {"system", "user", "assistant"}
Example = dict[str, Any]


def load_jsonl(path: str | Path) -> list[Example]:
    """Read a JSONL file into a list of example dicts. Blank lines are skipped."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"training data not found: {path}")
    examples: list[Example] = []
    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
    return examples


def validate_examples(examples: list[Example]) -> None:
    """Raise ValueError if any example isn't a well-formed chat record.

    A valid example has a non-empty `messages` list; each message has a role in
    VALID_ROLES and string content; and at least one message is from the
    assistant (otherwise there's no target to learn).
    """
    if not examples:
        raise ValueError("no training examples found")
    for i, ex in enumerate(examples):
        if not isinstance(ex, dict) or "messages" not in ex:
            raise ValueError(f"example {i}: missing 'messages' key")
        msgs = ex["messages"]
        if not isinstance(msgs, list) or not msgs:
            raise ValueError(f"example {i}: 'messages' must be a non-empty list")
        for j, m in enumerate(msgs):
            if not isinstance(m, dict):
                raise ValueError(f"example {i} message {j}: must be an object")
            role = m.get("role")
            if role not in VALID_ROLES:
                raise ValueError(
                    f"example {i} message {j}: role must be one of "
                    f"{sorted(VALID_ROLES)}, got {role!r}"
                )
            if not isinstance(m.get("content"), str):
                raise ValueError(f"example {i} message {j}: 'content' must be a string")
        if not any(m["role"] == "assistant" for m in msgs):
            raise ValueError(f"example {i}: must contain at least one assistant message")


def split_examples(
    examples: list[Example], eval_split: float, seed: int = 42
) -> tuple[list[Example], list[Example]]:
    """Deterministically split into (train, eval). eval_split is the eval fraction."""
    if not (0.0 <= eval_split < 1.0):
        raise ValueError(f"eval_split must be in [0, 1), got {eval_split}")
    if eval_split == 0.0:
        return list(examples), []
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n_eval = int(len(shuffled) * eval_split)
    return shuffled[n_eval:], shuffled[:n_eval]


def render_chat(example: Example, tokenizer, add_generation_prompt: bool = False) -> str:
    """Render an example's messages to a single string via the tokenizer's chat template."""
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def build_datasets(data_cfg, seed: int = 42):
    """Load, validate, and split into (train_ds, eval_ds|None) as HF Datasets.

    data_cfg is duck-typed (a DataConfig): needs .train_path, .eval_path,
    .eval_split. Returns a tuple; eval_ds is None when there's no eval data.
    """
    try:
        from datasets import Dataset
    except ImportError as e:
        raise ImportError("build_datasets requires the 'datasets' package: pip install datasets") from e

    train_examples = load_jsonl(data_cfg.train_path)
    validate_examples(train_examples)

    if getattr(data_cfg, "eval_path", None):
        eval_examples = load_jsonl(data_cfg.eval_path)
        validate_examples(eval_examples)
    else:
        train_examples, eval_examples = split_examples(
            train_examples, data_cfg.eval_split, seed
        )

    train_ds = Dataset.from_list(train_examples)
    eval_ds = Dataset.from_list(eval_examples) if eval_examples else None
    return train_ds, eval_ds
