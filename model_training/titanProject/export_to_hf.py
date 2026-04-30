"""
Export a Titans checkpoint to a Hugging Face-style folder.

Notes:
- This writes a HF-like layout (config.json, pytorch_model.bin, tokenizer file),
  but Titans is not a standard transformers architecture. Loading will require
  a custom HF model class (e.g., modeling_titans.py) that can ingest the saved
  state_dict and config fields.
- No external HF dependencies are required for this export.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
import yaml

from model import ModelConfig, build_model
from train_utils import resolve_path


def load_config(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Export Titans checkpoint to HF-style directory.")
    parser.add_argument("--config", required=True, help="YAML config path used for the run")
    parser.add_argument("--ckpt", required=True, help="Checkpoint (.pt) path with model state")
    parser.add_argument("--out", required=True, help="Output directory for HF-style artifacts")
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Override tokenizer path (if not using config data.tokenizer_path)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(resolve_path(args.config))
    mcfg = ModelConfig(**cfg["model"])

    # Build and load model
    model = build_model(mcfg)
    ckpt = torch.load(resolve_path(args.ckpt), map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)

    # Save weights in HF-style filename
    torch.save(model.state_dict(), out_dir / "pytorch_model.bin")

    # Write a minimal config.json with model_type tag for custom loader
    hf_config = {
        "model_type": "titans",
        "architectures": ["TitansForCausalLM"],
        "auto_map": {
            "AutoConfig": "modeling_titans.TitansConfig",
            "AutoModelForCausalLM": "modeling_titans.TitansForCausalLM",
        },
        "vocab_size": mcfg.vocab_size,
        "variant": mcfg.variant,
        "dim": mcfg.dim,
        "depth": mcfg.depth,
        "heads": mcfg.heads,
        "ff_mult": mcfg.ff_mult,
        "segment_len": mcfg.segment_len,
        "num_persist_mem_tokens": mcfg.num_persist_mem_tokens,
        "num_longterm_mem_tokens": mcfg.num_longterm_mem_tokens,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)

    # Copy tokenizer file if available
    tok_path = args.tokenizer_path or cfg["data"].get("tokenizer_path")
    if tok_path:
        tok_src = resolve_path(tok_path)
        if tok_src.exists():
            shutil.copy(tok_src, out_dir / tok_src.name)
        else:
            print(f"Warning: tokenizer file not found at {tok_src}, skipping copy.")
    else:
        print("Warning: tokenizer path not provided and not found in config; skipping copy.")

    # Write a short README to clarify loading expectations
    readme = """\
This folder is a Hugging Face-style export of a titans-pytorch model.

Files:
- config.json: minimal config with model_type='titans' and architecture fields.
- pytorch_model.bin: state_dict saved from titans-pytorch.
- <tokenizer_file>: SentencePiece model copied if available.

Loading:
- This is NOT a standard transformers architecture. To load via transformers,
  you need a custom modeling_titans.py that subclasses PreTrainedModel and
  PretrainedConfig to wrap titans-pytorch's MemoryAsContextTransformer and
  load this state_dict.
- Alternatively, load with titans-pytorch directly and ignore HF helpers.
"""
    with open(out_dir / "README_HF.txt", "w") as f:
        f.write(readme)

    # Copy modeling_titans.py alongside for trust_remote_code usage
    modeling_src = Path(__file__).resolve().parent / "modeling_titans.py"
    if modeling_src.exists():
        shutil.copy(modeling_src, out_dir / "modeling_titans.py")
    else:
        print(f"Warning: modeling_titans.py not found at {modeling_src}; trust_remote_code loads may fail.")

    print(f"Export complete to {out_dir}")


if __name__ == "__main__":
    main()
