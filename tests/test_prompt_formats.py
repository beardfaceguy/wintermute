"""
Tests for model_training/titanProject/prompt_formats.py — prompt rendering and extraction.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_training" / "titanProject"))

from prompt_formats import (
    _extract_instruction_parts,
    default_prompts,
    default_stop_strings,
    extract_completion,
    infer_prompt_family,
    render_instruction_prompt,
)

# ── render_instruction_prompt ────────────────────────────────────────────────


def test_render_instruction_prompt_basic():
    result = render_instruction_prompt("Summarize this.")
    assert "### Instruction:\nSummarize this." in result
    assert result.endswith("### Response:\n")


def test_render_instruction_prompt_with_input():
    result = render_instruction_prompt("Translate", "Hello")
    assert "### Input:\nHello" in result
    assert "### Instruction:\nTranslate" in result


def test_render_instruction_prompt_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        render_instruction_prompt("")


def test_render_instruction_prompt_none_raises():
    with pytest.raises(ValueError, match="non-empty"):
        render_instruction_prompt(None)


# ── _extract_instruction_parts ───────────────────────────────────────────────


def test_extract_parts_no_marker():
    result = _extract_instruction_parts("plain text without markers")
    assert result == {"instruction": "", "input": ""}


def test_extract_parts_roundtrip():
    prompt = render_instruction_prompt("Do something", "with this input")
    parts = _extract_instruction_parts(prompt)
    assert parts["instruction"] == "Do something"
    assert parts["input"] == "with this input"


def test_extract_parts_no_input_roundtrip():
    prompt = render_instruction_prompt("Do something")
    parts = _extract_instruction_parts(prompt)
    assert parts["instruction"] == "Do something"
    assert parts["input"] == ""


def test_extract_parts_missing_response_marker_does_not_crash():
    """If ### Response: is missing, the function should not raise ValueError."""
    malformed = "### Instruction:\nDo something\n\nThis text has no response marker"
    try:
        _extract_instruction_parts(malformed)
    except ValueError:
        pytest.fail(
            "_extract_instruction_parts crashed on malformed prompt missing ### Response: marker"
        )


def test_extract_parts_missing_response_after_input_does_not_crash():
    """If ### Input: is present but ### Response: is missing, should not crash."""
    malformed = (
        "### Instruction:\nDo something\n\n### Input:\nSome input text\n\nNo response marker here"
    )
    try:
        _extract_instruction_parts(malformed)
    except ValueError:
        pytest.fail(
            "_extract_instruction_parts crashed on malformed prompt with input but no ### Response:"
        )


# ── extract_completion ───────────────────────────────────────────────────────


def test_extract_completion_chat_strips_prefix():
    prompt = "User: Hello Assistant:"
    decoded = "User: Hello Assistant: Hi there!"
    result = extract_completion(decoded, prompt=prompt, prompt_family="chat")
    assert "Hi there!" in result


def test_extract_completion_instruction_basic():
    prompt = render_instruction_prompt("Say hello")
    decoded = prompt + "Hello!"
    result = extract_completion(decoded, prompt=prompt, prompt_family="instruction")
    assert result == "Hello!"


def test_extract_completion_unsupported_family():
    with pytest.raises(ValueError, match="Unsupported"):
        extract_completion("text", prompt="p", prompt_family="unknown")


# ── infer_prompt_family ──────────────────────────────────────────────────────


def test_infer_prompt_family_jsonl():
    cfg = {"data": {"train_path": "data/train.jsonl"}}
    assert infer_prompt_family(cfg) == "instruction"


def test_infer_prompt_family_chat():
    cfg = {"data": {"train_path": "data/train.txt"}}
    assert infer_prompt_family(cfg) == "chat"


def test_infer_prompt_family_instruction_keyword():
    cfg = {"data": {"train_path": "data/instruction_data.txt"}}
    assert infer_prompt_family(cfg) == "instruction"


# ── default_prompts / default_stop_strings ───────────────────────────────────


def test_default_prompts_chat():
    prompts = default_prompts("chat")
    assert len(prompts) > 0
    assert all("User:" in p for p in prompts)


def test_default_prompts_instruction():
    prompts = default_prompts("instruction")
    assert len(prompts) > 0
    assert all("### Instruction:" in p for p in prompts)


def test_default_prompts_unsupported():
    with pytest.raises(ValueError):
        default_prompts("bogus")


def test_default_stop_strings_chat():
    stops = default_stop_strings("chat")
    assert "\nUser:" in stops


def test_default_stop_strings_instruction():
    stops = default_stop_strings("instruction")
    assert "\n### Instruction:" in stops
