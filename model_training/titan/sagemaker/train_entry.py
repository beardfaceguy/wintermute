"""
SageMaker training entry point for the Titan pretraining PoC.

Runs inside a SageMaker PyTorch DLC (script mode). Reads a raw text corpus +
the 8k BPE tokenizer from the 'training' input channel, tokenizes in-container,
trains a MAC-Titan with titans-trainer, and writes the checkpoint + tokenizer to
SM_MODEL_DIR (auto-uploaded to S3 by SageMaker).

Hyperparameters arrive as --flag value argv (SageMaker converts the
hyperparameters dict). Channel/dir locations come from SM_* env vars.
"""
import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer
from titans_trainer import TitansConfig, TitansModel, TitansTrainer


class LMWindows(Dataset):
    def __init__(self, ids, seq, max_windows=0):
        self.ids, self.seq = ids, seq
        n = (len(ids) - 1) // seq
        self.n = min(n, max_windows) if max_windows else n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = i * self.seq
        w = torch.from_numpy(self.ids[s:s + self.seq + 1].astype(np.int64))
        return {"input_ids": w[:-1], "labels": w[1:]}


def tokenize_corpus(train_dir, tok, seq_len):
    eos = tok.token_to_id("</s>")
    corpus = os.path.join(train_dir, "corpus.txt")
    print(f"[titan] tokenizing {corpus}", flush=True)
    buf = []
    with open(corpus) as f:
        for line in f:
            line = line.strip()
            if line:
                buf.extend(tok.encode(line).ids)
                buf.append(eos)
    ids = np.array(buf, dtype=np.uint16)
    print(f"[titan] {len(ids)} tokens", flush=True)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--vocab-size", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-windows", type=int, default=0, help="cap dataset (0=all); dry-run bound")
    ap.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAINING", "."))
    ap.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "./out"))
    args = ap.parse_args()

    print(f"[titan] torch {torch.__version__} cuda={torch.cuda.is_available()} "
          f"gpus={torch.cuda.device_count()}", flush=True)

    tok = ByteLevelBPETokenizer(
        os.path.join(args.train, "vocab.json"), os.path.join(args.train, "merges.txt")
    )
    ids = tokenize_corpus(args.train, tok, args.seq_len)

    ds = LMWindows(ids, args.seq_len, args.max_windows)
    n_val = max(4, int(len(ds) * 0.05))
    val_ds = LMWindows(ids[-(n_val * args.seq_len + 1):], args.seq_len, 0)
    print(f"[titan] train windows={len(ds)} val windows={len(val_ds)}", flush=True)

    cfg = TitansConfig(
        vocab_size=args.vocab_size, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, max_seq_len=args.seq_len, causal=True, chunk_size=128,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, warmup_steps=10,
        use_amp=True, log_interval=5, val_every_steps=0, save_every_steps=0,
        output_dir=args.model_dir,
    )
    model = TitansModel.from_config(cfg)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[titan] model params: {nparams / 1e6:.1f}M", flush=True)

    t0 = time.time()
    TitansTrainer(model, ds, val_ds, cfg).train()

    os.makedirs(args.model_dir, exist_ok=True)
    ckpt = os.path.join(args.model_dir, "final.pt")
    model.save_pretrained(ckpt)
    tok.save_model(args.model_dir)  # tokenizer travels with the checkpoint
    print(f"[titan] SAGEMAKER_RUN_DONE params={nparams/1e6:.1f}M "
          f"time={time.time()-t0:.0f}s ckpt={ckpt}", flush=True)


if __name__ == "__main__":
    main()
