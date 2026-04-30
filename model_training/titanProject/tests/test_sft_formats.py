"""Tests for SFT data format parsing in finetune_sft.py.

Covers all four supported input formats:
  1. HF messages JSONL  ({"messages": [...]})
  2. ShareGPT JSONL     ({"conversations": [...]})
  3. Alpaca JSONL        ({"instruction": ..., "response"/\"output": ...})
  4. Chat text           (User: ... Assistant: ...)
"""

import json
import textwrap
from pathlib import Path

import pytest

from finetune_sft import (
    MaskedSFTDataset,
    _format_messages_as_chat,
    _split_sft_sample,
)


# ---------------------------------------------------------------------------
# _split_sft_sample — HF messages format
# ---------------------------------------------------------------------------

class TestHFMessagesFormat:
    def test_single_turn(self):
        line = json.dumps({
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        })
        prompt, response = _split_sft_sample(line)
        assert "User: What is 2+2?" in prompt
        assert prompt.endswith("Assistant:")
        assert response.strip() == "4"

    def test_multi_turn(self):
        line = json.dumps({
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        })
        prompt, response = _split_sft_sample(line)
        assert "User: Hello" in prompt
        assert "Assistant: Hi there!" in prompt
        assert "User: What is 2+2?" in prompt
        assert prompt.endswith("Assistant:")
        assert response.strip() == "4"

    def test_with_system_message(self):
        line = json.dumps({
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        })
        prompt, response = _split_sft_sample(line)
        assert "System: You are a helpful assistant." in prompt
        assert "User: Hello" in prompt
        assert response.strip() == "Hi!"

    def test_no_assistant_raises(self):
        line = json.dumps({
            "messages": [{"role": "user", "content": "Hello"}]
        })
        with pytest.raises(ValueError, match="no assistant turn"):
            _split_sft_sample(line)

    def test_empty_messages_raises(self):
        line = json.dumps({"messages": []})
        with pytest.raises(ValueError, match="no usable"):
            _split_sft_sample(line)

    def test_empty_assistant_content_raises(self):
        line = json.dumps({
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": ""},
            ]
        })
        with pytest.raises(ValueError, match="empty content"):
            _split_sft_sample(line)


# ---------------------------------------------------------------------------
# _split_sft_sample — ShareGPT format
# ---------------------------------------------------------------------------

class TestShareGPTFormat:
    def test_basic_sharegpt(self):
        line = json.dumps({
            "conversations": [
                {"from": "human", "value": "What color is the sky?"},
                {"from": "gpt", "value": "Blue."},
            ]
        })
        prompt, response = _split_sft_sample(line)
        assert "User: What color is the sky?" in prompt
        assert prompt.endswith("Assistant:")
        assert response.strip() == "Blue."

    def test_multi_turn_sharegpt(self):
        line = json.dumps({
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi!"},
                {"from": "human", "value": "Tell me a joke"},
                {"from": "gpt", "value": "Why did the chicken cross the road?"},
            ]
        })
        prompt, response = _split_sft_sample(line)
        assert "User: Hello" in prompt
        assert "Assistant: Hi!" in prompt
        assert "User: Tell me a joke" in prompt
        assert response.strip() == "Why did the chicken cross the road?"

    def test_conversation_key_variant(self):
        """Some datasets use 'conversation' (singular) instead of 'conversations'."""
        line = json.dumps({
            "conversation": [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello!"},
            ]
        })
        prompt, response = _split_sft_sample(line)
        assert "User: Hi" in prompt
        assert response.strip() == "Hello!"

    def test_role_variants(self):
        """ShareGPT datasets use varying role names."""
        line = json.dumps({
            "conversations": [
                {"from": "prompter", "value": "Question"},
                {"from": "chatbot", "value": "Answer"},
            ]
        })
        prompt, response = _split_sft_sample(line)
        assert "User: Question" in prompt
        assert response.strip() == "Answer"


# ---------------------------------------------------------------------------
# _split_sft_sample — Alpaca / instruction format
# ---------------------------------------------------------------------------

class TestAlpacaFormat:
    def test_instruction_response(self):
        line = json.dumps({
            "instruction": "Summarize this text.",
            "input": "",
            "response": "Here is the summary.",
        })
        prompt, response = _split_sft_sample(line)
        assert "### Instruction:" in prompt
        assert "Summarize this text." in prompt
        assert "### Response:" in prompt
        assert response == "Here is the summary."

    def test_instruction_with_input(self):
        line = json.dumps({
            "instruction": "Translate to French.",
            "input": "Hello world",
            "response": "Bonjour le monde",
        })
        prompt, response = _split_sft_sample(line)
        assert "### Input:" in prompt
        assert "Hello world" in prompt
        assert response == "Bonjour le monde"

    def test_output_alias(self):
        """'output' should work as alias for 'response' (standard Alpaca key)."""
        line = json.dumps({
            "instruction": "What is 1+1?",
            "input": "",
            "output": "2",
        })
        prompt, response = _split_sft_sample(line)
        assert "### Instruction:" in prompt
        assert response == "2"

    def test_response_takes_precedence_over_output(self):
        """If both 'response' and 'output' exist, 'response' wins."""
        line = json.dumps({
            "instruction": "Test",
            "input": "",
            "response": "from_response",
            "output": "from_output",
        })
        _, response = _split_sft_sample(line)
        assert response == "from_response"

    def test_missing_instruction_raises(self):
        line = json.dumps({"instruction": "", "response": "answer"})
        with pytest.raises(ValueError, match="instruction and response"):
            _split_sft_sample(line)

    def test_missing_response_raises(self):
        line = json.dumps({"instruction": "test", "input": ""})
        with pytest.raises(ValueError, match="instruction and response"):
            _split_sft_sample(line)


# ---------------------------------------------------------------------------
# _split_sft_sample — Chat text format
# ---------------------------------------------------------------------------

class TestChatTextFormat:
    def test_basic_chat(self):
        prompt, response = _split_sft_sample(
            "User: Hello there Assistant: Hi!"
        )
        assert prompt == "User: Hello there Assistant:"
        assert response == " Hi!"

    def test_missing_user_raises(self):
        with pytest.raises(ValueError, match="User: and Assistant:"):
            _split_sft_sample("Just some random text")

    def test_missing_assistant_raises(self):
        with pytest.raises(ValueError, match="User: and Assistant:"):
            _split_sft_sample("User: Hello")

    def test_empty_user_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _split_sft_sample("User: Assistant: answer")

    def test_empty_assistant_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _split_sft_sample("User: question Assistant: ")


# ---------------------------------------------------------------------------
# _split_sft_sample — edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _split_sft_sample("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _split_sft_sample("   \n  ")

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Invalid JSONL"):
            _split_sft_sample("{bad json")

    def test_json_array_raises(self):
        with pytest.raises(ValueError):
            _split_sft_sample("[1, 2, 3]")


# ---------------------------------------------------------------------------
# _format_messages_as_chat — unit tests
# ---------------------------------------------------------------------------

class TestFormatMessagesAsChat:
    def test_skips_non_dict_entries(self):
        prompt, response = _format_messages_as_chat([
            "not a dict",
            {"role": "user", "content": "Hi"},
            42,
            {"role": "assistant", "content": "Hello!"},
        ])
        assert "User: Hi" in prompt
        assert response.strip() == "Hello!"

    def test_content_key_variants(self):
        """Should accept 'content', 'value', and 'text' keys."""
        prompt, response = _format_messages_as_chat([
            {"role": "user", "text": "From text key"},
            {"role": "assistant", "value": "From value key"},
        ])
        assert "From text key" in prompt
        assert response.strip() == "From value key"


# ---------------------------------------------------------------------------
# MaskedSFTDataset — integration with all formats mixed
# ---------------------------------------------------------------------------

class TestMaskedSFTDatasetFormats:
    def _write_mixed_data(self, tmp_path: Path) -> Path:
        lines = [
            "User: Chat format question Assistant: Chat format answer",
            json.dumps({
                "messages": [
                    {"role": "user", "content": "HF messages question"},
                    {"role": "assistant", "content": "HF messages answer"},
                ]
            }),
            json.dumps({
                "conversations": [
                    {"from": "human", "value": "ShareGPT question"},
                    {"from": "gpt", "value": "ShareGPT answer"},
                ]
            }),
            json.dumps({
                "instruction": "Alpaca instruction",
                "input": "",
                "response": "Alpaca response",
            }),
            json.dumps({
                "instruction": "Alpaca output",
                "input": "",
                "output": "Output alias response",
            }),
        ]
        path = tmp_path / "mixed_sft.txt"
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_mixed_formats_load(self, tmp_path, dummy_tokenizer):
        path = self._write_mixed_data(tmp_path)
        ds = MaskedSFTDataset(
            str(path),
            dummy_tokenizer,
            seq_len=512,
            log_fn=lambda m: None,
            progress_label="test",
        )
        assert len(ds) == 5

    def test_all_samples_have_labels(self, tmp_path, dummy_tokenizer):
        path = self._write_mixed_data(tmp_path)
        ds = MaskedSFTDataset(
            str(path),
            dummy_tokenizer,
            seq_len=512,
            log_fn=lambda m: None,
            progress_label="test",
        )
        for i in range(len(ds)):
            x, y = ds[i]
            assert len(x) == len(y)
            has_real_label = any(label != -100 for label in y)
            assert has_real_label, f"Sample {i} has no trainable labels"
