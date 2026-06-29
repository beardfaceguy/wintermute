"""Tests for data.py — chat-JSONL loading, validation, splitting, rendering."""

import json

import pytest
from model_training.sft.data import (
    load_jsonl,
    render_chat,
    split_examples,
    validate_examples,
)


def _ex(*roles_contents):
    """Build an example dict from (role, content) pairs."""
    return {"messages": [{"role": r, "content": c} for r, c in roles_contents]}


def _good():
    return _ex(("system", "be helpful"), ("user", "hi"), ("assistant", "hello"))


# ── load_jsonl ──────────────────────────────────────────────────────────────


class TestLoadJsonl:
    def test_reads_valid_lines(self, tmp_path):
        p = tmp_path / "train.jsonl"
        p.write_text("\n".join(json.dumps(_good()) for _ in range(3)) + "\n")
        examples = load_jsonl(p)
        assert len(examples) == 3
        assert examples[0]["messages"][0]["role"] == "system"

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "train.jsonl"
        p.write_text(json.dumps(_good()) + "\n\n  \n" + json.dumps(_good()) + "\n")
        assert len(load_jsonl(p)) == 2

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_jsonl(tmp_path / "nope.jsonl")

    def test_malformed_line_raises_with_lineno(self, tmp_path):
        p = tmp_path / "train.jsonl"
        p.write_text(json.dumps(_good()) + "\n{not json}\n")
        with pytest.raises(ValueError, match=":2:"):
            load_jsonl(p)


# ── validate_examples ─────────────────────────────────────────────────────────


class TestValidateExamples:
    def test_good_passes(self):
        validate_examples([_good(), _good()])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="no training examples"):
            validate_examples([])

    def test_missing_messages_key_raises(self):
        with pytest.raises(ValueError, match="messages"):
            validate_examples([{"text": "oops"}])

    def test_empty_messages_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            validate_examples([{"messages": []}])

    def test_bad_role_raises(self):
        with pytest.raises(ValueError, match="role"):
            validate_examples([_ex(("wizard", "hi"), ("assistant", "yo"))])

    def test_non_string_content_raises(self):
        bad = {"messages": [{"role": "user", "content": 123}, {"role": "assistant", "content": "x"}]}
        with pytest.raises(ValueError, match="content"):
            validate_examples([bad])

    def test_no_assistant_message_raises(self):
        with pytest.raises(ValueError, match="assistant"):
            validate_examples([_ex(("system", "x"), ("user", "y"))])


# ── split_examples ────────────────────────────────────────────────────────────


class TestSplitExamples:
    def test_zero_split_returns_all_train(self):
        examples = [_good() for _ in range(10)]
        train, eval_ = split_examples(examples, 0.0)
        assert len(train) == 10
        assert eval_ == []

    def test_fractional_split_counts(self):
        examples = [{"messages": [{"role": "assistant", "content": str(i)}]} for i in range(10)]
        train, eval_ = split_examples(examples, 0.2, seed=1)
        assert len(train) == 8
        assert len(eval_) == 2

    def test_split_is_deterministic(self):
        examples = [{"messages": [{"role": "assistant", "content": str(i)}]} for i in range(20)]
        a = split_examples(examples, 0.25, seed=7)
        b = split_examples(examples, 0.25, seed=7)
        assert a == b

    def test_split_is_disjoint_and_complete(self):
        examples = [{"messages": [{"role": "assistant", "content": str(i)}]} for i in range(20)]
        train, eval_ = split_examples(examples, 0.3, seed=3)
        contents = sorted(int(m["messages"][0]["content"]) for m in train + eval_)
        assert contents == list(range(20))  # no loss, no dupes

    def test_out_of_range_split_raises(self):
        with pytest.raises(ValueError, match="eval_split"):
            split_examples([_good()], 1.0)


# ── render_chat ───────────────────────────────────────────────────────────────


class TestRenderChat:
    def test_renders_roles_and_content(self, fake_tokenizer):
        text = render_chat(_good(), fake_tokenizer)
        assert "<|system|>be helpful" in text
        assert "<|user|>hi" in text
        assert "<|assistant|>hello" in text

    def test_generation_prompt_appended(self, fake_tokenizer):
        text = render_chat(
            _ex(("user", "hi")), fake_tokenizer, add_generation_prompt=True
        )
        assert text.endswith("<|assistant|>")


# ── build_datasets (needs the `datasets` lib — skipped if absent) ──────────────


class TestBuildDatasets:
    def test_build_with_eval_split(self, tmp_path):
        pytest.importorskip("datasets")
        from types import SimpleNamespace

        from model_training.sft.data import build_datasets

        p = tmp_path / "train.jsonl"
        p.write_text("\n".join(json.dumps(_good()) for _ in range(10)) + "\n")
        cfg = SimpleNamespace(train_path=str(p), eval_path=None, eval_split=0.2)
        train_ds, eval_ds = build_datasets(cfg, seed=1)
        assert len(train_ds) == 8
        assert len(eval_ds) == 2
        assert "messages" in train_ds.column_names

    def test_build_no_eval(self, tmp_path):
        pytest.importorskip("datasets")
        from types import SimpleNamespace

        from model_training.sft.data import build_datasets

        p = tmp_path / "train.jsonl"
        p.write_text("\n".join(json.dumps(_good()) for _ in range(5)) + "\n")
        cfg = SimpleNamespace(train_path=str(p), eval_path=None, eval_split=0.0)
        train_ds, eval_ds = build_datasets(cfg)
        assert len(train_ds) == 5
        assert eval_ds is None
