"""
Evaluate a checkpoint on the synthetic memory test.

For each line in the dataset, we run the model and extract the top token
for the answer; compare exact match vs. the provided answers file.
"""

import argparse
from pathlib import Path
import re

import torch
import yaml
import sentencepiece as spm

from model import ModelConfig, build_model
from train_utils import resolve_path


def load_tokenizer(path: Path):
    sp = spm.SentencePieceProcessor()
    if not sp.load(str(path)):
        raise RuntimeError(f"Failed to load tokenizer at {path}")
    return sp


def normalize_token(tok: str) -> str:
    # strip sentencepiece underline and punctuation, lowercase
    tok = tok.replace("▁", "")
    tok = re.sub(r"[^\w]+", "", tok)
    return tok.lower()


@torch.no_grad()
def generate_answer(model, tokenizer, device, ids, max_new_tokens: int = 3):
    """
    Assumes prompt ends with 'Answer:'; decode up to max_new_tokens and normalize.
    """
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    gen_ids = []
    for _ in range(max_new_tokens):
        logits = model(x)
        logits = logits[:, -1, :]
        next_id = logits.argmax(dim=-1)  # greedy
        gen_ids.append(next_id.item())
        x = torch.cat([x, next_id.unsqueeze(0)], dim=1)
    decoded = tokenizer.decode(gen_ids)
    return [normalize_token(t) for t in decoded.strip().split() if t.strip()]


@torch.no_grad()
def evaluate(model, tokenizer, device, texts, answers, max_len: int = 512, max_answer_tokens: int = 3):
    correct = 0
    total = 0
    for text, gold in zip(texts, answers):
        ids = tokenizer.encode(text)
        # leave room for answer tokens
        if len(ids) > max_len - max_answer_tokens:
            ids = ids[: max_len - max_answer_tokens]
        preds = generate_answer(model, tokenizer, device, ids, max_new_tokens=max_answer_tokens)
        gold_norm = normalize_token(gold)
        if any(p == gold_norm for p in preds):
            correct += 1
        total += 1
    return correct, total


def main():
    parser = argparse.ArgumentParser(description="Evaluate Titans checkpoint on memory test.")
    parser.add_argument("--config", type=str, default="configs/config_combo_all.yaml", help="YAML config path")
    parser.add_argument("--ckpt", type=str, default="ckpt_step_4000.pt", help="Checkpoint path")
    parser.add_argument("--data", type=str, default="memory_test.txt", help="Memory test text file")
    parser.add_argument("--answers", type=str, default="memory_test_answers.txt", help="Answers file")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--max-answer-tokens", type=int, default=3, help="Greedy answer decode length")
    parser.add_argument("--strict-load", action="store_true", help="Enforce strict checkpoint loading (default: tolerant)")
    args = parser.parse_args()

    cfg = yaml.safe_load(resolve_path(args.config).open("r"))
    mcfg = ModelConfig(**cfg["model"])

    # device selection
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
    state = torch.load(ckpt_path, map_location=device)
    if args.strict_load:
        model.load_state_dict(state["model"])
    else:
        # Drop incompatible memory keys when evaluating a no-memory baseline
        filtered = {
            k: v
            for k, v in state["model"].items()
            if "longterm_mems" not in k and "persistent_memory" not in k
        }
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if missing or unexpected:
            print(f"Loaded with missing={missing}, unexpected={unexpected}")
    model.eval()

    texts = resolve_path(args.data).read_text(encoding="utf-8").strip().splitlines()
    answers = resolve_path(args.answers).read_text(encoding="utf-8").strip().splitlines()
    correct, total = evaluate(
        model,
        tokenizer,
        device,
        texts,
        answers,
        max_len=cfg["train"]["seq_len"],
        max_answer_tokens=args.max_answer_tokens,
    )
    acc = correct / max(total, 1)
    print(f"Accuracy: {correct}/{total} = {acc*100:.2f}%")


if __name__ == "__main__":
    main()

