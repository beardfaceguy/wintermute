"""
Inference smoke test for Titans checkpoints.

Purpose:
- verify checkpoint + tokenizer load correctly
- generate responses for a small fixed prompt suite
- provide a simple pass/fail signal for CI/manual checks
"""

import argparse
import json
import time
from pathlib import Path
from typing import List

import torch

from generate import generate, load_config, load_tokenizer, resolve_path
from model import ModelConfig, build_model


DEFAULT_PROMPTS: List[str] = [
    "User: Hello there. Can you introduce yourself in one sentence? Assistant:",
    "User: Tell me a short story about a blue cat and a red kite. Assistant:",
    "User: What is 2 plus 2? Assistant:",
]


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a quick inference smoke suite.")
    parser.add_argument("--config", type=str, default="configs/config_baseline_nomem.yaml", help="YAML config path")
    parser.add_argument("--ckpt", type=str, default="ckpt_step_4000.pt", help="Checkpoint path")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--max-new", type=int, default=80, help="Max new tokens per prompt")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument(
        "--min-completion-chars",
        type=int,
        default=20,
        help="Minimum completion length to count as a pass",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    args = parser.parse_args()

    t0 = time.time()
    device = pick_device(args.device)

    cfg = load_config(resolve_path(args.config))
    mcfg = ModelConfig(**cfg["model"])
    tokenizer = load_tokenizer(resolve_path(cfg["data"]["tokenizer_path"]))

    model = build_model(mcfg).to(device)
    ckpt_path = resolve_path(args.ckpt)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    results = []
    all_passed = True
    for idx, prompt in enumerate(DEFAULT_PROMPTS, start=1):
        gen_start = time.time()
        out = generate(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_new_tokens=args.max_new,
            top_k=args.top_k,
            temperature=args.temperature,
        )
        latency = time.time() - gen_start
        completion = out[len(prompt) :].strip() if out.startswith(prompt) else out.strip()
        passed = len(completion) >= args.min_completion_chars
        all_passed = all_passed and passed
        results.append(
            {
                "prompt_id": idx,
                "prompt": prompt,
                "completion": completion,
                "completion_chars": len(completion),
                "latency_s": round(latency, 3),
                "passed": passed,
            }
        )

    summary = {
        "ok": all_passed,
        "device": str(device),
        "config": str(resolve_path(args.config)),
        "ckpt": str(ckpt_path),
        "total_runtime_s": round(time.time() - t0, 3),
        "results": results,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("== Inference Smoke Test ==")
        print(f"ok: {summary['ok']}")
        print(f"device: {summary['device']}")
        print(f"ckpt: {summary['ckpt']}")
        print(f"runtime_s: {summary['total_runtime_s']}")
        print()
        for row in results:
            print(f"[prompt_{row['prompt_id']}] passed={row['passed']} chars={row['completion_chars']} latency_s={row['latency_s']}")
            print(f"prompt: {row['prompt']}")
            print(f"completion: {row['completion']}")
            print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
