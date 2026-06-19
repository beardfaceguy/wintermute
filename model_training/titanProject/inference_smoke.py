"""
Inference smoke test for Titans checkpoints.

Purpose:
- verify checkpoint + tokenizer load correctly
- generate responses for a small fixed prompt suite
- provide a simple pass/fail signal for CI/manual checks
"""

import argparse
import json
import re
import time

import torch
from generate import generate, load_config, load_tokenizer, resolve_path
from model import ModelConfig, build_model, load_model_source
from prompt_formats import (
    default_prompts,
    default_stop_strings,
    extract_completion,
    infer_prompt_family,
)


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


def _has_format_leakage(completion: str) -> bool:
    return any(
        marker in completion
        for marker in ("### Instruction:", "### Input:", "### Response:", "User:", "Assistant:")
    )


def _story_task_ok(completion: str) -> bool:
    text = completion.strip()
    lowered = f" {text.lower()} "
    if "cat" not in lowered or "kite" not in lowered:
        return False
    if re.match(r"^\s*(blue\s+cats?|cats?|the\s+cat|a\s+cat)\s+(are|is)\b", text.lower()):
        return False
    has_action = bool(
        re.search(
            r"\b(chased|chase|raced|race|leaped|leap|jumped|jump|watched|watch|followed|follow|"
            r"caught|catch|curled|curl|drifted|drift|flew|fly|sailed|sail|landed|land|"
            r"tugged|tug|pounced|pounce|batted|bat|guarded|guard|sat|soared|soar|"
            r"danced|dance|carried|carry)\b",
            text.lower(),
        )
    )
    if not has_action:
        return False
    sentence_count = len(re.findall(r"[.!?]", text))
    has_connector = any(
        token in lowered
        for token in (" when ", " while ", " after ", " then ", " until ", " before ")
    )
    return sentence_count >= 2 or has_connector


def _instruction_task_ok(prompt_id: int, completion: str) -> bool:
    text = completion.strip()
    lowered = text.lower()
    if prompt_id == 1:
        return 5 <= len(text) <= 120 and "\n" not in text
    if prompt_id == 2:
        return _story_task_ok(completion)
    if prompt_id == 3:
        first_line = text.splitlines()[0].strip()
        normalized = re.sub(r"\s+", " ", first_line.lower())
        if normalized in {"4", "four", "2 plus 2 is 4.", "2 plus 2 is 4", "2+2=4", "2 + 2 = 4"}:
            return True
        if normalized.startswith("2 plus 2 is 4"):
            return True
        if normalized.startswith("2 + 2 = 4"):
            return True
        return False
    return len(text) >= 3


def evaluate_completion(
    *,
    prompt_family: str,
    prompt_id: int,
    completion: str,
    min_completion_chars: int,
) -> dict[str, bool]:
    format_ok = not _has_format_leakage(completion)
    if prompt_family == "instruction":
        length_ok = len(completion.strip()) >= 3
        task_ok = _instruction_task_ok(prompt_id, completion)
    else:
        length_ok = len(completion.strip()) >= min_completion_chars
        task_ok = length_ok
    passed = format_ok and task_ok and length_ok
    return {
        "format_ok": format_ok,
        "length_ok": length_ok,
        "task_ok": task_ok,
        "passed": passed,
    }


def overall_smoke_ok(prompt_family: str, results: list[dict[str, object]]) -> bool:
    if not results:
        return False
    if prompt_family != "instruction":
        return all(bool(row.get("passed")) for row in results)

    if not all(bool(row.get("format_ok")) for row in results):
        return False
    arithmetic_row = next((row for row in results if row.get("prompt_id") == 3), None)
    arithmetic_ok = bool(arithmetic_row and arithmetic_row.get("task_ok"))
    task_passes = sum(1 for row in results if row.get("task_ok"))
    return arithmetic_ok and task_passes >= 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a quick inference smoke suite.")
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
    parser.add_argument("--max-new", type=int, default=80, help="Max new tokens per prompt")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument(
        "--min-completion-chars",
        type=int,
        default=20,
        help="Legacy minimum completion length; instruction-mode smoke now uses prompt-specific checks",
    )
    parser.add_argument(
        "--prompt-family",
        type=str,
        default="auto",
        choices=["auto", "chat", "instruction"],
        help="Prompt family to evaluate. 'auto' infers from config data paths.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    args = parser.parse_args()

    t0 = time.time()
    device = pick_device(args.device)

    cfg = load_config(resolve_path(args.config))
    mcfg = ModelConfig(**cfg["model"])
    tokenizer = load_tokenizer(resolve_path(cfg["data"]["tokenizer_path"]))
    prompt_family = infer_prompt_family(cfg) if args.prompt_family == "auto" else args.prompt_family
    prompts = default_prompts(prompt_family)
    stop_strings = default_stop_strings(prompt_family)

    model = build_model(mcfg).to(device)
    ckpt_path = resolve_path(args.ckpt)
    load_model_source(model, ckpt_path, map_location=device, strict=True)
    model.eval()

    results = []
    for idx, prompt in enumerate(prompts, start=1):
        gen_start = time.time()
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
        latency = time.time() - gen_start
        completion = extract_completion(out, prompt=prompt, prompt_family=prompt_family)
        checks = evaluate_completion(
            prompt_family=prompt_family,
            prompt_id=idx,
            completion=completion,
            min_completion_chars=args.min_completion_chars,
        )
        results.append(
            {
                "prompt_id": idx,
                "prompt": prompt,
                "completion": completion,
                "completion_chars": len(completion),
                "latency_s": round(latency, 3),
                **checks,
            }
        )

    all_passed = overall_smoke_ok(prompt_family, results)
    summary = {
        "ok": all_passed,
        "device": str(device),
        "config": str(resolve_path(args.config)),
        "ckpt": str(ckpt_path),
        "prompt_family": prompt_family,
        "scoring": {
            "format_required_all": True,
            "instruction_rule": "arithmetic prompt must pass and at least 2/3 task checks must pass",
            "legacy_min_completion_chars": args.min_completion_chars,
        },
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
        print(f"prompt_family: {summary['prompt_family']}")
        print()
        for row in results:
            print(
                f"[prompt_{row['prompt_id']}] passed={row['passed']} "
                f"format_ok={row['format_ok']} length_ok={row['length_ok']} task_ok={row['task_ok']} "
                f"chars={row['completion_chars']} latency_s={row['latency_s']}"
            )
            print(f"prompt: {row['prompt']}")
            print(f"completion: {row['completion']}")
            print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
