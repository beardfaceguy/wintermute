"""
Quick finetune on the synthetic memory QA set (memory_test.txt + answers).

We append the gold answer to the prompt (which ends with "Answer:") and run LM training
to teach the model to emit the answer token(s).
"""

import argparse
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F
import yaml
from model import ModelConfig, build_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from train_utils import resolve_path


def load_config(path: Path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_tokenizer(path: Path):
    sp = spm.SentencePieceProcessor()
    if not sp.load(str(path)):
        raise RuntimeError(f"Failed to load tokenizer at {path}")
    return sp


class QADataset(Dataset):
    def __init__(self, texts, answers, tokenizer, seq_len: int):
        self.samples = []
        for t, a in zip(texts, answers, strict=False):
            full = f"{t} {a}"
            ids = tokenizer.encode(full)
            if len(ids) < 2:
                continue
            if len(ids) > seq_len:
                ids = ids[:seq_len]
            input_ids = ids[:-1]
            target_ids = ids[1:]
            self.samples.append(
                (
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(target_ids, dtype=torch.long),
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def main():
    parser = argparse.ArgumentParser(description="Finetune Titans on synthetic memory QA.")
    parser.add_argument("--config", type=str, default="configs/config_combo_all.yaml")
    parser.add_argument("--ckpt", type=str, default="ckpt_step_4000.pt")
    parser.add_argument("--data", type=str, default="memory_test.txt")
    parser.add_argument("--answers", type=str, default="memory_test_answers.txt")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"]
    )
    parser.add_argument("--save", type=str, default="ckpt_finetune_qa.pt")
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
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])

    texts = resolve_path(args.data).read_text(encoding="utf-8").strip().splitlines()
    answers = resolve_path(args.answers).read_text(encoding="utf-8").strip().splitlines()

    ds = QADataset(texts, answers, tokenizer, seq_len=cfg["train"]["seq_len"])
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=cfg["train"]["weight_decay"])

    model.train()
    step = 0
    for epoch in range(1000):  # will break by steps
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"[ft] step {step} loss {loss.item():.4f}")
            if step >= args.steps:
                break
        if step >= args.steps:
            break

    torch.save({"model": model.state_dict()}, resolve_path(args.save))
    print(f"Saved finetuned checkpoint to {args.save}")


if __name__ == "__main__":
    main()
