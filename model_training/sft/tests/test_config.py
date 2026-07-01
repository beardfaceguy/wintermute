"""Tests for config.py — SFT configuration dataclasses, YAML loading, validation."""

import pytest
import yaml
from model_training.sft.config import (
    DataConfig,
    LoraConfig,
    ModelConfig,
    SFTConfig,
    TrainingConfig,
)


def _good_dict():
    """Minimal valid config: only the two required sections."""
    return {
        "model": {"base_model": "Qwen/Qwen2.5-0.5B-Instruct"},
        "data": {"train_path": "data/train.jsonl"},
    }


class TestFromDict:
    def test_builds_nested_configs(self):
        cfg = SFTConfig.from_dict(_good_dict())
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.data, DataConfig)
        assert isinstance(cfg.lora, LoraConfig)
        assert isinstance(cfg.training, TrainingConfig)

    def test_required_fields_set(self):
        cfg = SFTConfig.from_dict(_good_dict())
        assert cfg.model.base_model == "Qwen/Qwen2.5-0.5B-Instruct"
        assert cfg.data.train_path == "data/train.jsonl"

    def test_defaults_applied(self):
        cfg = SFTConfig.from_dict(_good_dict())
        assert cfg.lora.enabled is True
        assert cfg.lora.r == 16
        assert cfg.model.dtype == "bfloat16"
        assert cfg.training.epochs == 1.0

    def test_overrides_applied(self):
        d = _good_dict()
        d["lora"] = {"r": 64, "alpha": 128}
        d["training"] = {"learning_rate": 1e-4}
        cfg = SFTConfig.from_dict(d)
        assert cfg.lora.r == 64
        assert cfg.lora.alpha == 128
        assert cfg.training.learning_rate == 1e-4

    def test_missing_model_section_raises(self):
        d = _good_dict()
        del d["model"]
        with pytest.raises(ValueError, match="model"):
            SFTConfig.from_dict(d)

    def test_missing_data_section_raises(self):
        d = _good_dict()
        del d["data"]
        with pytest.raises(ValueError, match="data"):
            SFTConfig.from_dict(d)

    def test_unknown_top_level_key_raises(self):
        d = _good_dict()
        d["trainingg"] = {}  # typo
        with pytest.raises(ValueError, match="unknown"):
            SFTConfig.from_dict(d)

    def test_unknown_nested_key_raises(self):
        d = _good_dict()
        d["lora"] = {"rank": 64}  # should be 'r'
        with pytest.raises(ValueError, match="unknown"):
            SFTConfig.from_dict(d)


class TestFromYaml:
    def test_round_trip(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(_good_dict()))
        cfg = SFTConfig.from_yaml(p)
        assert cfg.model.base_model == "Qwen/Qwen2.5-0.5B-Instruct"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SFTConfig.from_yaml(tmp_path / "nope.yaml")


class TestValidation:
    def test_good_config_passes(self):
        SFTConfig.from_dict(_good_dict()).validate()

    def test_empty_base_model_raises(self):
        d = _good_dict()
        d["model"]["base_model"] = ""
        with pytest.raises(ValueError, match="base_model"):
            SFTConfig.from_dict(d).validate()

    def test_bad_dtype_raises(self):
        d = _good_dict()
        d["model"]["dtype"] = "int4"
        with pytest.raises(ValueError, match="dtype"):
            SFTConfig.from_dict(d).validate()

    def test_lora_r_must_be_positive_when_enabled(self):
        d = _good_dict()
        d["lora"] = {"enabled": True, "r": 0}
        with pytest.raises(ValueError, match="lora.r"):
            SFTConfig.from_dict(d).validate()

    def test_lora_r_not_checked_when_disabled(self):
        d = _good_dict()
        d["lora"] = {"enabled": False, "r": 0}
        SFTConfig.from_dict(d).validate()  # full fine-tune, r irrelevant

    def test_eval_split_out_of_range_raises(self):
        d = _good_dict()
        d["data"]["eval_split"] = 1.0
        with pytest.raises(ValueError, match="eval_split"):
            SFTConfig.from_dict(d).validate()

    def test_max_seq_len_must_be_positive(self):
        d = _good_dict()
        d["data"]["max_seq_len"] = 0
        with pytest.raises(ValueError, match="max_seq_len"):
            SFTConfig.from_dict(d).validate()

    def test_epochs_must_be_positive(self):
        d = _good_dict()
        d["training"] = {"epochs": 0}
        with pytest.raises(ValueError, match="epochs"):
            SFTConfig.from_dict(d).validate()

    def test_learning_rate_must_be_positive(self):
        d = _good_dict()
        d["training"] = {"learning_rate": 0}
        with pytest.raises(ValueError, match="learning_rate"):
            SFTConfig.from_dict(d).validate()

    def test_negative_max_steps_raises(self):
        d = _good_dict()
        d["training"] = {"max_steps": -1}
        with pytest.raises(ValueError, match="max_steps"):
            SFTConfig.from_dict(d).validate()

    def test_warmup_ratio_out_of_range_raises(self):
        d = _good_dict()
        d["training"] = {"warmup_ratio": 1.5}
        with pytest.raises(ValueError, match="warmup_ratio"):
            SFTConfig.from_dict(d).validate()
