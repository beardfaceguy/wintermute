"""Extended tests for finetune_sft.py — covers functions NOT in test_sft_formats.py.

Focuses on:
  - _tokenize_with_chat_template()
  - _forward_logits()
  - build_sft_dataloader()
"""

import json
import types
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader

from finetune_sft import (
    _forward_logits,
    _tokenize_with_chat_template,
    build_sft_dataloader,
    MaskedSFTDataset,
)


# ---------------------------------------------------------------------------
# Mock HF tokenizer for chat template tests
# ---------------------------------------------------------------------------

class FakeChatTokenizer:
    """Minimal tokenizer stub that supports apply_chat_template for testing."""

    def __init__(self):
        self.eos_token = "</s>"
        self.eos_token_id = 2
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.name_or_path = "fake-chat-model"

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return [ord(c) % 200 + 10 for c in text]

    def decode(self, ids, **kwargs) -> str:
        return "".join(chr(max(i, 32)) for i in ids)

    def apply_chat_template(
        self, messages, *, tokenize=True, add_generation_prompt=False
    ) -> List[int]:
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|{role}|>{content}<|end|>")
        if add_generation_prompt:
            parts.append("<|assistant|>")
        text = "".join(parts)
        if tokenize:
            return self.encode(text)
        return text


# ===========================================================================
# _tokenize_with_chat_template
# ===========================================================================

class TestTokenizeWithChatTemplate:
    def test_basic_chat_sample(self):
        tok = FakeChatTokenizer()
        sample = "User: Hi Assistant: Hello! How can I help you today? I am happy to assist with any questions you might have."
        result = _tokenize_with_chat_template(tok, sample, seq_len=512)

        assert result is not None
        x_tokens, labels = result
        assert len(x_tokens) == len(labels)
        assert len(x_tokens) > 0

    def test_strips_user_prefix(self):
        tok = FakeChatTokenizer()
        calls = []
        orig_apply = tok.apply_chat_template

        def tracking_apply(messages, **kwargs):
            calls.append(messages)
            return orig_apply(messages, **kwargs)

        tok.apply_chat_template = tracking_apply
        sample = "User: What is Python? Assistant: A programming language"
        _tokenize_with_chat_template(tok, sample, seq_len=512)

        full_call = calls[0]
        user_msg = next(m for m in full_call if m["role"] == "user")
        assert not user_msg["content"].startswith("User:")

    def test_strips_trailing_assistant_marker(self):
        tok = FakeChatTokenizer()
        calls = []
        orig_apply = tok.apply_chat_template

        def tracking_apply(messages, **kwargs):
            calls.append(messages)
            return orig_apply(messages, **kwargs)

        tok.apply_chat_template = tracking_apply
        sample = "User: Hello there Assistant: Hi!"
        _tokenize_with_chat_template(tok, sample, seq_len=512)

        full_call = calls[0]
        user_msg = next(m for m in full_call if m["role"] == "user")
        assert not user_msg["content"].endswith("Assistant:")

    def test_builds_user_and_assistant_messages(self):
        tok = FakeChatTokenizer()
        calls = []
        orig_apply = tok.apply_chat_template

        def tracking_apply(messages, **kwargs):
            calls.append(messages)
            return orig_apply(messages, **kwargs)

        tok.apply_chat_template = tracking_apply
        sample = "User: Tell me a joke Assistant: Why did the chicken cross the road?"
        _tokenize_with_chat_template(tok, sample, seq_len=512)

        full_msg_call = calls[0]
        roles = [m["role"] for m in full_msg_call]
        assert "user" in roles
        assert "assistant" in roles

    def test_returns_none_for_empty_sample(self):
        tok = FakeChatTokenizer()
        result = _tokenize_with_chat_template(tok, "", seq_len=512)
        assert result is None

    def test_returns_none_for_whitespace_only(self):
        tok = FakeChatTokenizer()
        result = _tokenize_with_chat_template(tok, "   \n  ", seq_len=512)
        assert result is None

    def test_returns_none_for_invalid_format(self):
        tok = FakeChatTokenizer()
        result = _tokenize_with_chat_template(tok, "just some random text", seq_len=512)
        assert result is None

    def test_prompt_masked_in_labels(self):
        tok = FakeChatTokenizer()
        sample = "User: Hello Assistant: World, this is a long enough response to ensure there are real labels after masking the prompt portion."
        result = _tokenize_with_chat_template(tok, sample, seq_len=512)

        assert result is not None
        _, labels = result
        masked_count = sum(1 for l in labels if l == -100)
        real_count = sum(1 for l in labels if l != -100)
        assert masked_count > 0, "Prompt portion should be masked"
        assert real_count > 0, "Answer portion should have real labels"

    def test_truncation_at_seq_len(self):
        tok = FakeChatTokenizer()
        long_answer = "x" * 5000
        sample = f"User: Hello Assistant: {long_answer}"
        result = _tokenize_with_chat_template(tok, sample, seq_len=64)

        assert result is not None
        x_tokens, labels = result
        assert len(x_tokens) <= 64

    def test_jsonl_hf_messages_format(self):
        tok = FakeChatTokenizer()
        sample = json.dumps({
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello! How can I help you today? I am an AI assistant ready to answer your questions."},
            ]
        })
        result = _tokenize_with_chat_template(tok, sample, seq_len=512)
        assert result is not None
        x_tokens, labels = result
        assert len(x_tokens) > 0

    def test_jsonl_alpaca_format(self):
        tok = FakeChatTokenizer()
        sample = json.dumps({
            "instruction": "Summarize",
            "input": "",
            "response": "Summary here",
        })
        result = _tokenize_with_chat_template(tok, sample, seq_len=512)
        assert result is not None


# ===========================================================================
# _forward_logits
# ===========================================================================

class TestForwardLogits:
    def test_titan_mode_returns_raw_tensor(self):
        logits = torch.randn(2, 10, 100)
        model = MagicMock(return_value=logits)

        result = _forward_logits(model, torch.zeros(2, 10, dtype=torch.long), hf_mode=False)

        assert torch.equal(result, logits)
        model.assert_called_once()

    def test_hf_mode_extracts_logits_attribute(self):
        logits = torch.randn(2, 10, 100)
        output = MagicMock()
        output.logits = logits
        model = MagicMock(return_value=output)

        result = _forward_logits(model, torch.zeros(2, 10, dtype=torch.long), hf_mode=True)

        assert torch.equal(result, logits)
        model.assert_called_once()
        _, kwargs = model.call_args
        assert "input_ids" in kwargs
        assert torch.equal(kwargs["input_ids"], torch.zeros(2, 10, dtype=torch.long))

    def test_hf_mode_passes_input_ids_kwarg(self):
        output = MagicMock()
        output.logits = torch.randn(1, 5, 50)
        model = MagicMock(return_value=output)
        x = torch.tensor([[1, 2, 3, 4, 5]])

        _forward_logits(model, x, hf_mode=True)

        _, kwargs = model.call_args
        assert "input_ids" in kwargs

    def test_titan_mode_passes_positional_arg(self):
        model = MagicMock(return_value=torch.randn(1, 5, 50))
        x = torch.tensor([[1, 2, 3, 4, 5]])

        _forward_logits(model, x, hf_mode=False)

        args, kwargs = model.call_args
        assert len(args) == 1
        assert "input_ids" not in kwargs

    def test_output_shape_matches_input_batch(self):
        batch_size, seq_len, vocab_size = 4, 16, 256
        logits = torch.randn(batch_size, seq_len, vocab_size)
        model = MagicMock(return_value=logits)
        x = torch.zeros(batch_size, seq_len, dtype=torch.long)

        result = _forward_logits(model, x, hf_mode=False)

        assert result.shape == (batch_size, seq_len, vocab_size)


# ===========================================================================
# build_sft_dataloader
# ===========================================================================

class TestBuildSFTDataloader:
    @pytest.fixture
    def sft_data_file(self, tmp_path):
        lines = [
            "User: Hi Assistant: Hello! How can I help you today? I am ready to assist with any questions you have.",
            "User: Hello Assistant: Hi there! Welcome. I am an AI assistant and I would be happy to help you out.",
            "User: Tell me about Python Assistant: Python is a high-level programming language known for its simplicity and readability. It is widely used.",
            "User: What color is the sky? Assistant: The sky appears blue during the day due to Rayleigh scattering of sunlight by the atmosphere.",
            "User: Name a fruit Assistant: Apple is a popular fruit that comes in many varieties including Fuji, Gala, and Granny Smith.",
        ]
        path = tmp_path / "train_sft.txt"
        path.write_text("\n".join(lines) + "\n")
        return str(path)

    def test_returns_dataloader_and_sampler(self, sft_data_file, dummy_tokenizer):
        loader, sampler = build_sft_dataloader(
            sft_data_file, dummy_tokenizer,
            seq_len=128, batch_size=2, shuffle=False,
            log_fn=lambda m: None, progress_label="test",
        )

        assert isinstance(loader, DataLoader)
        assert sampler is None  # single-GPU

    def test_batch_size_respected(self, sft_data_file, dummy_tokenizer):
        loader, _ = build_sft_dataloader(
            sft_data_file, dummy_tokenizer,
            seq_len=128, batch_size=2, shuffle=False,
            log_fn=lambda m: None, progress_label="test",
        )

        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape[0] == 2

    def test_batch_tensors_are_padded_to_same_length(self, sft_data_file, dummy_tokenizer):
        loader, _ = build_sft_dataloader(
            sft_data_file, dummy_tokenizer,
            seq_len=128, batch_size=3, shuffle=False,
            log_fn=lambda m: None, progress_label="test",
        )

        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape[1] == batch_y.shape[1]

    def test_labels_have_masked_and_real(self, sft_data_file, dummy_tokenizer):
        loader, _ = build_sft_dataloader(
            sft_data_file, dummy_tokenizer,
            seq_len=128, batch_size=1, shuffle=False,
            log_fn=lambda m: None, progress_label="test",
        )

        _, batch_y = next(iter(loader))
        labels = batch_y[0].tolist()
        assert any(l == -100 for l in labels), "Expected some masked labels"
        assert any(l != -100 for l in labels), "Expected some real labels"

    def test_missing_data_file_raises(self, dummy_tokenizer):
        with pytest.raises((FileNotFoundError, OSError)):
            build_sft_dataloader(
                "/nonexistent/path/data.txt", dummy_tokenizer,
                seq_len=128, batch_size=1, shuffle=False,
                log_fn=lambda m: None, progress_label="test",
            )

    def test_empty_data_file_raises(self, tmp_path, dummy_tokenizer):
        empty_path = tmp_path / "empty.txt"
        empty_path.write_text("")

        with pytest.raises(ValueError, match="No usable SFT samples"):
            build_sft_dataloader(
                str(empty_path), dummy_tokenizer,
                seq_len=128, batch_size=1, shuffle=False,
                log_fn=lambda m: None, progress_label="test",
            )

    def test_ddp_sampler_created_for_multi_gpu(self, sft_data_file, dummy_tokenizer):
        loader, sampler = build_sft_dataloader(
            sft_data_file, dummy_tokenizer,
            seq_len=128, batch_size=1, shuffle=True,
            log_fn=lambda m: None, progress_label="test",
            rank=0, world_size=2,
        )

        assert sampler is not None

    def test_with_chat_template_tokenizer(self, sft_data_file):
        tok = FakeChatTokenizer()

        dummy_encode = lambda text: [ord(c) % 200 for c in text]
        dummy_tok = MagicMock()
        dummy_tok.side_effect = dummy_encode
        dummy_tok.encode = dummy_encode
        dummy_tok.eos_id = 2
        dummy_tok.pad_id = 0

        loader, _ = build_sft_dataloader(
            sft_data_file, dummy_tok,
            seq_len=128, batch_size=2, shuffle=False,
            log_fn=lambda m: None, progress_label="test",
            chat_template_tokenizer=tok,
        )

        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape[0] == 2
        assert batch_x.shape[1] == batch_y.shape[1]

    def test_all_invalid_samples_raises(self, tmp_path, dummy_tokenizer):
        bad_path = tmp_path / "bad.txt"
        bad_path.write_text("no markers here\njust random text\nnothing useful\n")

        with pytest.raises(ValueError, match="No usable SFT samples"):
            build_sft_dataloader(
                str(bad_path), dummy_tokenizer,
                seq_len=128, batch_size=1, shuffle=False,
                log_fn=lambda m: None, progress_label="test",
            )

    def test_mixed_valid_invalid_keeps_valid(self, tmp_path, dummy_tokenizer):
        path = tmp_path / "mixed.txt"
        path.write_text(
            "garbage line\n"
            "User: valid question Assistant: valid answer\n"
            "more garbage\n"
        )

        loader, _ = build_sft_dataloader(
            str(path), dummy_tokenizer,
            seq_len=128, batch_size=1, shuffle=False,
            log_fn=lambda m: None, progress_label="test",
        )

        assert len(loader.dataset) == 1
