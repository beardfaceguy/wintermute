"""
Quick text generation script for Titans checkpoints.

Usage:
  python generate.py --config configs/config_combo_all.yaml --ckpt ckpt_step_4000.pt --prompt "Once upon a time" --top-k 20 --temperature 0.8
  # greedy (no top-k) example: --top-k 0 --temperature 1.0
"""

import argparse
from pathlib import Path
from typing import List, Optional

import torch
import yaml

from model import ModelConfig, build_model, load_model_source
from prompt_formats import default_stop_strings, infer_prompt_family
from train import get_tokenizer


def resolve_path(path_str: str) -> Path:
    if path_str.startswith("hf://"):
        return path_str  # type: ignore[return-value]
    p = Path(path_str)
    if p.is_absolute():
        return p
    # cwd
    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return cwd_candidate
    # relative to script
    script_dir = Path(__file__).resolve().parent
    script_candidate = script_dir / p
    if script_candidate.exists():
        return script_candidate
    # repo root (wintermute) is two levels up from titanProject/
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / p


def load_config(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_tokenizer(path):
    return get_tokenizer(str(path))


@torch.no_grad()
def generate(
    model,
    tokenizer,
    device,
    prompt: str,
    max_new_tokens: int,
    top_k: int = 20,
    temperature: float = 0.8,
    stop_strings: Optional[List[str]] = None,
):
    ids = tokenizer.encode(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, seq)
    temp = max(temperature, 1e-5)
    eos_id = getattr(tokenizer, "eos_id", -1)
    for _ in range(max_new_tokens):
        logits = model(x)  # (1, seq, vocab)
        logits = logits[:, -1, :] / temp
        if top_k and top_k > 0:
            vals, idx = torch.topk(logits, top_k, dim=-1)
            mask = torch.full_like(logits, -float("inf"))
            mask.scatter_(1, idx, vals)
            logits = mask
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (1,1)
        x = torch.cat([x, next_id], dim=1)
        if isinstance(eos_id, int) and eos_id >= 0 and next_id.item() == eos_id:
            break
        if stop_strings:
            decoded = tokenizer.decode(x.squeeze(0).tolist())
            completion = decoded[len(prompt) :] if decoded.startswith(prompt) else decoded
            if any(stop in completion for stop in stop_strings):
                break
    return tokenizer.decode(x.squeeze(0).tolist())


def main():
    parser = argparse.ArgumentParser(description="Generate text from a Titans checkpoint.")
    parser.add_argument("--config", type=str, default="configs/config_combo_all.yaml", help="YAML config path")
    parser.add_argument("--ckpt", type=str, default="ckpt_step_4000.pt", help="Checkpoint path")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt text")
    parser.add_argument("--max-new", type=int, default=64, help="Max new tokens to generate")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument(
        "--prompt-family",
        type=str,
        default="none",
        choices=["none", "auto", "chat", "instruction"],
        help="Optional prompt family for stop handling; 'auto' infers from config data paths",
    )
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    mcfg = ModelConfig(**cfg["model"])

    # device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    tokenizer = load_tokenizer(resolve_path(cfg["data"]["tokenizer_path"]))
    model = build_model(mcfg).to(device)

    ckpt_path = resolve_path(args.ckpt)
    load_model_source(model, ckpt_path, map_location=device, strict=True)
    model.eval()
    if args.prompt_family == "none":
        stop_strings = None
    else:
        prompt_family = infer_prompt_family(cfg) if args.prompt_family == "auto" else args.prompt_family
        stop_strings = default_stop_strings(prompt_family)

    out = generate(
        model,
        tokenizer,
        device=device,
        prompt=args.prompt,
        max_new_tokens=args.max_new,
        top_k=args.top_k,
        temperature=args.temperature,
        stop_strings=stop_strings,
    )
    print(out)


if __name__ == "__main__":
    main()

