"""
Simple interactive chat-style REPL for Titans checkpoints.

This is a convenience interface for quick qualitative checks, not a production chat server.
"""

import argparse
from pathlib import Path

import torch
from generate import generate, load_config, load_tokenizer, resolve_path
from model import ModelConfig, build_model, load_model_source
from prompt_formats import default_stop_strings, extract_completion


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def build_prompt(
    history: list[tuple[str, str]],
    user_text: str,
    tokenizer,
    user_prefix: str,
    assistant_prefix: str,
    max_prompt_tokens: int,
) -> str:
    # Keep recent turns that fit context budget.
    trimmed = list(history)
    while True:
        lines: list[str] = []
        for user_msg, assistant_msg in trimmed:
            lines.append(f"{user_prefix} {user_msg}")
            lines.append(f"{assistant_prefix} {assistant_msg}")
        lines.append(f"{user_prefix} {user_text}")
        lines.append(f"{assistant_prefix}")
        prompt = "\n".join(lines)
        token_count = len(tokenizer.encode(prompt))
        if token_count <= max_prompt_tokens or not trimmed:
            return prompt
        trimmed.pop(0)


def postprocess_completion(raw_completion: str, user_prefix: str) -> str:
    return extract_completion(
        raw_completion,
        prompt="",
        prompt_family="chat",
        user_prefix=user_prefix,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive chat REPL for Titans checkpoint.")
    parser.add_argument(
        "--config", type=str, default="configs/config_baseline_nomem.yaml", help="YAML config path"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="ckpt_step_4000.pt",
        help="Checkpoint path or Hugging Face ref like hf://gpt2",
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"]
    )
    parser.add_argument("--max-new", type=int, default=80, help="Max new tokens per assistant turn")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=None,
        help="Prompt token budget; default uses train.seq_len",
    )
    parser.add_argument("--user-prefix", type=str, default="User:", help="User line prefix")
    parser.add_argument(
        "--assistant-prefix", type=str, default="Assistant:", help="Assistant line prefix"
    )
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    mcfg = ModelConfig(**cfg["model"])
    tokenizer = load_tokenizer(resolve_path(cfg["data"]["tokenizer_path"]))
    device = pick_device(args.device)

    model = build_model(mcfg).to(device)
    ckpt_path = resolve_path(args.ckpt)
    load_model_source(model, ckpt_path, map_location=device, strict=True)
    model.eval()

    max_prompt_tokens = args.max_prompt_tokens or int(cfg["train"]["seq_len"])
    history: list[tuple[str, str]] = []
    stop_strings = default_stop_strings("chat")

    print("Titans chat REPL ready.")
    print("Commands: /reset, /exit")
    print(f"device={device} | ckpt={Path(ckpt_path).name} | max_prompt_tokens={max_prompt_tokens}")

    while True:
        try:
            user_text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0

        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            print("Exiting.")
            return 0
        if user_text == "/reset":
            history.clear()
            print("History cleared.")
            continue

        prompt = build_prompt(
            history=history,
            user_text=user_text,
            tokenizer=tokenizer,
            user_prefix=args.user_prefix,
            assistant_prefix=args.assistant_prefix,
            max_prompt_tokens=max_prompt_tokens,
        )

        out = generate(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_new_tokens=args.max_new,
            top_k=args.top_k,
            temperature=args.temperature,
            stop_strings=stop_strings,
        )

        raw_completion = out[len(prompt) :] if out.startswith(prompt) else out
        assistant_text = postprocess_completion(raw_completion, args.user_prefix)
        if not assistant_text:
            assistant_text = "(empty completion)"
        print(f"Assistant> {assistant_text}")

        history.append((user_text, assistant_text))


if __name__ == "__main__":
    raise SystemExit(main())
