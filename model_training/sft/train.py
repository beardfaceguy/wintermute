"""
TRL SFTTrainer entrypoint for the host-agnostic SFT pipeline.

Run locally or as a SageMaker training-job entry point:

    python -m model_training.sft.train --config configs/qwen3_8b_lora.yaml

Design for testability: the heavy model/trainer construction is isolated behind
two module-level seam functions (`_load_model_and_tokenizer`, `_make_trainer`)
that do their trl/peft/transformers imports lazily. The pure mapping helpers
(`build_sft_kwargs`, `build_lora_kwargs`, `resolve_dtype`) carry the logic and
are unit-tested without the training stack installed.
"""

from __future__ import annotations

import argparse
from typing import Any

# When run as a bare script (e.g. a SageMaker training-job entry point) rather
# than as `python -m model_training.sft.train`, the package root isn't on
# sys.path and the package imports below fail. Bootstrap the repo root first.
if __package__ in (None, ""):
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from model_training.sft.config import SFTConfig  # noqa: E402

# torch is light enough (and installed) to import at module load for dtype mapping.
import torch  # noqa: E402

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPES:
        raise ValueError(f"unknown dtype '{name}'; expected one of {sorted(_DTYPES)}")
    return _DTYPES[name]


def build_sft_kwargs(cfg: SFTConfig) -> dict[str, Any]:
    """Map our TrainingConfig → kwargs for TRL's SFTConfig (a TrainingArguments subclass)."""
    t = cfg.training
    return {
        "output_dir": t.output_dir,
        "num_train_epochs": t.epochs,
        # -1 disables the step cap in TRL/transformers (train by epochs instead)
        "max_steps": t.max_steps if t.max_steps > 0 else -1,
        "learning_rate": t.learning_rate,
        "per_device_train_batch_size": t.per_device_batch_size,
        "gradient_accumulation_steps": t.grad_accum_steps,
        "warmup_ratio": t.warmup_ratio,
        "weight_decay": t.weight_decay,
        "logging_steps": t.logging_steps,
        "save_steps": t.save_steps,
        "seed": t.seed,
        # TRL 1.x renamed max_seq_length → max_length (verified against trl 1.7.0).
        "max_length": cfg.data.max_seq_len,
        "bf16": cfg.model.dtype == "bfloat16",
        "fp16": cfg.model.dtype == "float16",
        "report_to": "none",
    }


def build_lora_kwargs(cfg: SFTConfig) -> dict[str, Any] | None:
    """Map our LoraConfig → kwargs for peft's LoraConfig, or None for full fine-tune."""
    lora = cfg.lora
    if not lora.enabled:
        return None
    return {
        "r": lora.r,
        "lora_alpha": lora.alpha,
        "lora_dropout": lora.dropout,
        "target_modules": lora.target_modules,
        "task_type": "CAUSAL_LM",
        "bias": "none",
    }


# ── heavy seams (lazy imports; monkeypatched in tests) ─────────────────────────


def _load_model_and_tokenizer(cfg: SFTConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.base_model, trust_remote_code=cfg.model.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": resolve_dtype(cfg.model.dtype),
        "trust_remote_code": cfg.model.trust_remote_code,
    }
    if cfg.model.attn_implementation:
        model_kwargs["attn_implementation"] = cfg.model.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(cfg.model.base_model, **model_kwargs)
    return model, tokenizer


def _make_trainer(model, tokenizer, train_ds, eval_ds, sft_kwargs, lora_kwargs):
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    peft_config = None
    if lora_kwargs is not None:
        from peft import LoraConfig

        peft_config = LoraConfig(**lora_kwargs)

    args = TRLSFTConfig(**sft_kwargs)
    return SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )


# ── orchestration ─────────────────────────────────────────────────────────────


def train(cfg: SFTConfig) -> str:
    """Run the SFT job end to end. Returns the output checkpoint directory."""
    cfg.validate()

    from model_training.sft.data import build_datasets

    train_ds, eval_ds = build_datasets(cfg.data, seed=cfg.training.seed)
    model, tokenizer = _load_model_and_tokenizer(cfg)
    trainer = _make_trainer(
        model,
        tokenizer,
        train_ds,
        eval_ds,
        build_sft_kwargs(cfg),
        build_lora_kwargs(cfg),
    )
    trainer.train()
    trainer.save_model(cfg.training.output_dir)
    tokenizer.save_pretrained(cfg.training.output_dir)
    return cfg.training.output_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run SFT from a YAML config")
    parser.add_argument("--config", required=True, help="Path to the SFT YAML config")
    args = parser.parse_args(argv)

    cfg = SFTConfig.from_yaml(args.config)
    out = train(cfg)
    print(f"Training complete. Checkpoint saved to: {out}")


if __name__ == "__main__":
    main()
