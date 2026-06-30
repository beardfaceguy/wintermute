"""
Merge a LoRA adapter into its base model to produce a standalone HF model dir.

The SFT pipeline emits a LoRA adapter (adapter_config.json + adapter weights),
which most serving backends can't consume directly. merge_adapter() folds the
adapter into the base weights via peft `merge_and_unload`, yielding a plain
HF model directory that SageMaker LMI / vLLM / Ollama can all serve uniformly.

The peft/transformers import is lazy (behind _load_and_merge) so the pure logic
is testable without the ML stack.

    python -m model_training.sft.export --adapter outputs/smoke --out merged/smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_base_model(adapter_dir: str | Path) -> str:
    """Return base_model_name_or_path from an adapter's adapter_config.json."""
    cfg_path = Path(adapter_dir) / "adapter_config.json"
    if not cfg_path.exists():
        raise ValueError(f"adapter_config.json not found in {adapter_dir}")
    cfg = json.loads(cfg_path.read_text())
    base = cfg.get("base_model_name_or_path")
    if not base:
        raise ValueError(f"base_model_name_or_path missing from {cfg_path}")
    return base


def _load_and_merge(base_model: str, adapter_dir: str):
    """Heavy seam: load base + adapter, merge, return (model, tokenizer).

    NOTE: loads in the checkpoint's default precision (float32 for most bases).
    Fine for small models; for large bases (e.g. 8B) pass/load in bf16 to avoid
    OOM during the merge. Left as a follow-up since the validated path is 0.5B.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(base_model)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    return merged, tokenizer


def merge_adapter(
    adapter_dir: str | Path, output_dir: str | Path, base_model: str | None = None
) -> str:
    """Merge a LoRA adapter into its base model and save to output_dir.

    base_model defaults to the id recorded in the adapter's config. Returns the
    output directory path.
    """
    adapter_dir = Path(adapter_dir)
    if not (adapter_dir / "adapter_config.json").exists():
        raise ValueError(f"adapter_config.json not found in {adapter_dir}")
    if not str(output_dir).strip():
        raise ValueError("output_dir must be a non-empty path")

    base = base_model or read_base_model(adapter_dir)
    output_dir = str(output_dir)

    model, tokenizer = _load_and_merge(base, str(adapter_dir))
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model")
    parser.add_argument("--adapter", required=True, help="Path to the LoRA adapter directory")
    parser.add_argument("--out", required=True, help="Output directory for the merged model")
    parser.add_argument("--base-model", help="Override base model id (default: from adapter config)")
    args = parser.parse_args(argv)

    out = merge_adapter(args.adapter, args.out, base_model=args.base_model)
    print(f"Merged model saved to: {out}")


if __name__ == "__main__":
    main()
