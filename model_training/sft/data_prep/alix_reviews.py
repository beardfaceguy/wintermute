"""
Convert codeJung PR-review JSON files into chat-format JSONL for SFT.

Reads the PR records under a dataset directory (schema: a record per PR with a
`comments` list, each comment having `path`, `context`, `body`, `author`, …) and
emits one chat example per high-quality inline review comment:

    {"messages": [
        {"role": "system",    "content": "You are a senior code reviewer ..."},
        {"role": "user",      "content": "File: <path>\n<code context>\n..."},
        {"role": "assistant", "content": "<the human review comment>"}
    ]}

This is Format A (code+context → review comment). Curation mirrors codeJung's
Tier-0 rules (drop bots, generic/empty bodies) but lives here so wintermute owns
its own training-data prep — codeJung stays the upstream dataset source.

    python -m model_training.sft.data_prep.alix_reviews \
        --input training_material/codeJung_datasets/meetalix \
        --output model_training/sft/data/alix_reviews.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Bot authors whose comments aren't human review signal (mirrors codeJung).
BOT_AUTHORS = frozenset(
    {
        "cursor[bot]",
        "copilot-pull-request-reviewer[bot]",
        "copilot",
        "coderabbitai[bot]",
        "coderabbit",
        "dependabot[bot]",
        "github-actions[bot]",
        "renovate[bot]",
        "renovate",
        "vercel[bot]",
        "vercel",
        "linear[bot]",
        "linear",
        "codecov[bot]",
        "codecov",
    }
)

# Low-signal bodies that teach no reusable review pattern.
GENERIC_BODIES = frozenset(
    {
        "lgtm",
        "lgtm!",
        "+1",
        "approved",
        "ship it",
        "ship it!",
        "looks good",
        "looks good!",
        "looks good to me",
        "looks good to me!",
        "nit",
        "ok",
        "okay",
        "done",
        "fixed",
        "yes",
        "no",
        "thanks",
        "thanks!",
    }
)

MIN_BODY_LEN = 15  # shorter bodies rarely carry a substantive review point

SYSTEM_PROMPT = (
    "You are a senior code reviewer for the {repo} repository. Given a code "
    "snippet, provide a specific, actionable review comment in the style of the "
    "team's reviewers."
)

Comment = dict[str, Any]
Example = dict[str, Any]


def is_bot(author: str | None) -> bool:
    if not author:
        return False
    a = author.strip().lower()
    return a in BOT_AUTHORS or a.endswith("[bot]")


def is_generic_body(body: str | None) -> bool:
    if not body or not body.strip():
        return True
    return body.strip().lower() in GENERIC_BODIES


def is_reviewable_comment(comment: Comment) -> bool:
    """A comment is reviewable if it's a human, substantive comment on real code."""
    if is_bot(comment.get("author")):
        return False
    body = comment.get("body") or ""
    if is_generic_body(body) or len(body.strip()) < MIN_BODY_LEN:
        return False
    if not (comment.get("path") or "").strip():
        return False
    if not (comment.get("context") or "").strip():
        return False
    return True


def build_example(comment: Comment, repo: str) -> Example:
    """Build a Format-A chat example from a reviewable comment."""
    path = comment["path"]
    line = comment.get("line")
    context = comment["context"].rstrip()
    loc = f"File: {path}" + (f" (around line {line})" if line else "")
    user = f"{loc}\n\n```\n{context}\n```\n\nProvide a review comment for this code."
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(repo=repo)},
            {"role": "user", "content": user},
            {"role": "assistant", "content": comment["body"].strip()},
        ]
    }


def pr_to_examples(pr_record: dict[str, Any]) -> list[Example]:
    repo = pr_record.get("repo") or "the"
    return [
        build_example(c, repo)
        for c in pr_record.get("comments", [])
        if is_reviewable_comment(c)
    ]


def convert_dataset(input_dir: str | Path, output_path: str | Path) -> dict[str, int]:
    """Walk PR JSONs under input_dir, emit chat-JSONL, return stats.

    Files without a `comments` key (e.g. pr-list.json enumerations) are skipped.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prs_processed = 0
    comments_seen = 0
    examples_written = 0

    with open(output_path, "w") as out:
        for json_path in sorted(input_dir.rglob("*.json")):
            try:
                record = json.loads(json_path.read_text())
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or "comments" not in record:
                continue  # not a PR record (e.g. pr-list.json)
            prs_processed += 1
            comments_seen += len(record.get("comments", []))
            for ex in pr_to_examples(record):
                out.write(json.dumps(ex) + "\n")
                examples_written += 1

    return {
        "prs_processed": prs_processed,
        "comments_seen": comments_seen,
        "examples_written": examples_written,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert PR review JSONs to chat-format JSONL")
    parser.add_argument("--input", required=True, help="Dataset directory of PR JSON files")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args(argv)

    stats = convert_dataset(args.input, args.output)
    print(
        f"Wrote {stats['examples_written']} examples from {stats['prs_processed']} PRs "
        f"({stats['comments_seen']} comments seen) → {args.output}"
    )


if __name__ == "__main__":
    main()
