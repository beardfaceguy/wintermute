"""Tests for train.py — wiring only (mock-and-defer).

The real fine-tune needs trl/peft/transformers, which aren't installed on this
interpreter. These tests verify the orchestration LOGIC:
  - cfg → TRL SFTConfig kwargs
  - cfg → PEFT LoraConfig kwargs (or None when full fine-tune)
  - dtype string → torch dtype
  - train() calls the trainer + saves + returns the checkpoint dir

The heavy model/trainer construction is isolated behind two module-level seam
functions (`_load_model_and_tokenizer`, `_make_trainer`) that tests monkeypatch.
A real end-to-end tiny-model smoke is deferred to a pinned venv (Vikunja #911,
phase C) and lives behind the `slow`/`gpu` markers there.
"""

import json

import pytest
from model_training.sft import train as train_mod
from model_training.sft.config import SFTConfig


def _cfg(tmp_path, **over):
    d = {
        "model": {"base_model": "Qwen/Qwen2.5-0.5B-Instruct"},
        "data": {"train_path": str(tmp_path / "train.jsonl")},
        "training": {"output_dir": str(tmp_path / "out")},
    }
    for k, v in over.items():
        d.setdefault(k, {}).update(v)
    return SFTConfig.from_dict(d)


# ── build_sft_kwargs ──────────────────────────────────────────────────────────


class TestBuildSftKwargs:
    def test_maps_training_fields(self, tmp_path):
        cfg = _cfg(
            tmp_path,
            training={
                "epochs": 3,
                "learning_rate": 1e-4,
                "per_device_batch_size": 2,
                "grad_accum_steps": 4,
                "warmup_ratio": 0.1,
                "weight_decay": 0.01,
                "logging_steps": 5,
                "save_steps": 50,
                "seed": 7,
            },
        )
        kw = train_mod.build_sft_kwargs(cfg)
        assert kw["num_train_epochs"] == 3
        assert kw["learning_rate"] == 1e-4
        assert kw["per_device_train_batch_size"] == 2
        assert kw["gradient_accumulation_steps"] == 4
        assert kw["warmup_ratio"] == 0.1
        assert kw["weight_decay"] == 0.01
        assert kw["logging_steps"] == 5
        assert kw["save_steps"] == 50
        assert kw["seed"] == 7
        assert kw["output_dir"] == str(tmp_path / "out")

    def test_bf16_flag_for_bfloat16(self, tmp_path):
        kw = train_mod.build_sft_kwargs(_cfg(tmp_path, model={"dtype": "bfloat16"}))
        assert kw["bf16"] is True
        assert kw["fp16"] is False

    def test_fp16_flag_for_float16(self, tmp_path):
        kw = train_mod.build_sft_kwargs(_cfg(tmp_path, model={"dtype": "float16"}))
        assert kw["fp16"] is True
        assert kw["bf16"] is False

    def test_no_half_flags_for_float32(self, tmp_path):
        kw = train_mod.build_sft_kwargs(_cfg(tmp_path, model={"dtype": "float32"}))
        assert kw["bf16"] is False
        assert kw["fp16"] is False


# ── build_lora_kwargs ─────────────────────────────────────────────────────────


class TestBuildLoraKwargs:
    def test_enabled_returns_kwargs(self, tmp_path):
        cfg = _cfg(tmp_path, lora={"enabled": True, "r": 32, "alpha": 64, "dropout": 0.1})
        kw = train_mod.build_lora_kwargs(cfg)
        assert kw["r"] == 32
        assert kw["lora_alpha"] == 64
        assert kw["lora_dropout"] == 0.1
        assert kw["task_type"] == "CAUSAL_LM"
        assert kw["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]

    def test_disabled_returns_none(self, tmp_path):
        cfg = _cfg(tmp_path, lora={"enabled": False})
        assert train_mod.build_lora_kwargs(cfg) is None


# ── resolve_dtype ─────────────────────────────────────────────────────────────


class TestResolveDtype:
    def test_known_dtypes(self):
        import torch

        assert train_mod.resolve_dtype("bfloat16") is torch.bfloat16
        assert train_mod.resolve_dtype("float16") is torch.float16
        assert train_mod.resolve_dtype("float32") is torch.float32

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="dtype"):
            train_mod.resolve_dtype("int4")


# ── train() orchestration (seams monkeypatched) ───────────────────────────────


class _FakeTrainer:
    def __init__(self):
        self.trained = False
        self.saved_to = None

    def train(self):
        self.trained = True

    def save_model(self, output_dir):
        self.saved_to = output_dir


class _FakeTokenizer:
    def __init__(self):
        self.saved_to = None

    def save_pretrained(self, output_dir):
        self.saved_to = output_dir


class TestTrainOrchestration:
    def _write_data(self, tmp_path, n=4):
        p = tmp_path / "train.jsonl"
        ex = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
        p.write_text("\n".join(json.dumps(ex) for _ in range(n)) + "\n")
        return p

    def test_train_runs_saves_and_returns_dir(self, tmp_path, monkeypatch):
        self._write_data(tmp_path)
        cfg = _cfg(tmp_path)

        captured = {}
        fake_trainer = _FakeTrainer()
        fake_tok = _FakeTokenizer()

        monkeypatch.setattr(
            train_mod, "_load_model_and_tokenizer", lambda c: ("FAKE_MODEL", fake_tok)
        )

        def fake_make_trainer(model, tokenizer, train_ds, eval_ds, sft_kwargs, lora_kwargs):
            captured.update(
                model=model,
                tokenizer=tokenizer,
                train_ds=train_ds,
                eval_ds=eval_ds,
                sft_kwargs=sft_kwargs,
                lora_kwargs=lora_kwargs,
            )
            return fake_trainer

        monkeypatch.setattr(train_mod, "_make_trainer", fake_make_trainer)

        out = train_mod.train(cfg)

        assert fake_trainer.trained is True
        assert fake_trainer.saved_to == str(tmp_path / "out")
        assert fake_tok.saved_to == str(tmp_path / "out")
        assert out == str(tmp_path / "out")
        assert captured["model"] == "FAKE_MODEL"
        assert len(captured["train_ds"]) == 4
        assert captured["eval_ds"] is None  # eval_split defaults to 0
        assert captured["lora_kwargs"]["r"] == 16  # LoRA on by default

    def test_train_passes_eval_dataset_when_split_set(self, tmp_path, monkeypatch):
        self._write_data(tmp_path, n=10)
        cfg = _cfg(tmp_path, data={"eval_split": 0.2})

        captured = {}
        monkeypatch.setattr(
            train_mod, "_load_model_and_tokenizer", lambda c: ("M", _FakeTokenizer())
        )
        monkeypatch.setattr(
            train_mod,
            "_make_trainer",
            lambda *a: captured.update(eval_ds=a[3]) or _FakeTrainer(),
        )

        train_mod.train(cfg)
        assert captured["eval_ds"] is not None
        assert len(captured["eval_ds"]) == 2

    def test_train_validates_config_first(self, tmp_path, monkeypatch):
        # bad dtype should raise during train() before touching the heavy seams
        cfg = _cfg(tmp_path, model={"dtype": "int4"})
        monkeypatch.setattr(
            train_mod, "_load_model_and_tokenizer", lambda c: pytest.fail("should not load")
        )
        with pytest.raises(ValueError, match="dtype"):
            train_mod.train(cfg)
