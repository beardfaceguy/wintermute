"""
SFT pipeline configuration.

Plain dataclasses + YAML loading + validation. Deliberately free of
torch/trl/transformers imports so configs can be loaded and tested without the
training stack installed.

A config has two required sections (`model`, `data`) and two optional ones
(`lora`, `training`) that fall back to sensible defaults:

    model:
      base_model: Qwen/Qwen3-8B
    data:
      train_path: data/train.jsonl
    lora:        # optional — LoRA enabled by default
      r: 32
    training:    # optional
      learning_rate: 2.0e-4
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

VALID_DTYPES = {"bfloat16", "float16", "float32"}


@dataclass
class ModelConfig:
    base_model: str
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    attn_implementation: str | None = None


@dataclass
class LoraConfig:
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class DataConfig:
    train_path: str
    eval_path: str | None = None
    eval_split: float = 0.0  # fraction carved from train when eval_path is unset
    max_seq_len: int = 2048


@dataclass
class TrainingConfig:
    output_dir: str = "outputs"
    epochs: float = 1.0
    max_steps: int = 0  # >0 caps total optimizer steps (overrides epochs); 0 = use epochs
    learning_rate: float = 2e-4
    per_device_batch_size: int = 1
    grad_accum_steps: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 10
    save_steps: int = 200
    seed: int = 42


def _build(dc_type: type, data: Any, name: str):
    """Construct a dataclass from a mapping, rejecting unknown keys (typo guard)."""
    if not isinstance(data, dict):
        raise ValueError(f"'{name}' section must be a mapping, got {type(data).__name__}")
    valid = {f.name for f in fields(dc_type)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"unknown key(s) in '{name}' section: {sorted(unknown)}")
    return dc_type(**data)


@dataclass
class SFTConfig:
    model: ModelConfig
    data: DataConfig
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SFTConfig:
        valid = {f.name for f in fields(cls)}
        unknown = set(d) - valid
        if unknown:
            raise ValueError(f"unknown top-level config key(s): {sorted(unknown)}")
        if "model" not in d:
            raise ValueError("config missing required section 'model'")
        if "data" not in d:
            raise ValueError("config missing required section 'data'")
        return cls(
            model=_build(ModelConfig, d["model"], "model"),
            data=_build(DataConfig, d["data"], "data"),
            lora=_build(LoraConfig, d.get("lora", {}), "lora"),
            training=_build(TrainingConfig, d.get("training", {}), "training"),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> SFTConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def validate(self) -> None:
        """Raise ValueError on any out-of-range or missing field."""
        model, data, lora, training = self.model, self.data, self.lora, self.training

        if not model.base_model:
            raise ValueError("model.base_model must be non-empty")
        if model.dtype not in VALID_DTYPES:
            raise ValueError(
                f"model.dtype must be one of {sorted(VALID_DTYPES)}, got '{model.dtype}'"
            )

        if not data.train_path:
            raise ValueError("data.train_path must be non-empty")
        if not (0.0 <= data.eval_split < 1.0):
            raise ValueError(f"data.eval_split must be in [0, 1), got {data.eval_split}")
        if data.max_seq_len <= 0:
            raise ValueError(f"data.max_seq_len must be > 0, got {data.max_seq_len}")

        if lora.enabled:
            if lora.r <= 0:
                raise ValueError(f"lora.r must be > 0 when LoRA is enabled, got {lora.r}")
            if lora.alpha <= 0:
                raise ValueError(f"lora.alpha must be > 0 when LoRA is enabled, got {lora.alpha}")
            if not (0.0 <= lora.dropout < 1.0):
                raise ValueError(f"lora.dropout must be in [0, 1), got {lora.dropout}")

        if training.epochs <= 0:
            raise ValueError(f"training.epochs must be > 0, got {training.epochs}")
        if training.max_steps < 0:
            raise ValueError(f"training.max_steps must be >= 0, got {training.max_steps}")
        if training.learning_rate <= 0:
            raise ValueError(
                f"training.learning_rate must be > 0, got {training.learning_rate}"
            )
        if training.per_device_batch_size < 1:
            raise ValueError("training.per_device_batch_size must be >= 1")
        if training.grad_accum_steps < 1:
            raise ValueError("training.grad_accum_steps must be >= 1")
        if not (0.0 <= training.warmup_ratio <= 1.0):
            raise ValueError(
                f"training.warmup_ratio must be in [0, 1], got {training.warmup_ratio}"
            )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
