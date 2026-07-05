"""
Pretrain a small MAC-Titan on the tokenized TinyStories corpus.

Tokenizes the corpus with the 8k BPE (cached to $TITAN_WORK), builds contiguous
causal-LM windows, and trains with titans-trainer's TitansTrainer. --smoke does a
tiny fast wiring run first.
"""
import os
import argparse
import time
import numpy as np
import torch
from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer
from titans_trainer import TitansConfig, TitansModel, TitansTrainer

WORK = os.environ.get("TITAN_WORK", "titan_work")
HERE = os.path.dirname(os.path.abspath(__file__))
SEQ = 512
VOCAB = 8000


def tokenizer_dir():
    """Prefer a freshly-trained tokenizer in WORK, else the repo-shipped one."""
    w = os.path.join(WORK, f"tok_{VOCAB}")
    return w if os.path.exists(os.path.join(w, "vocab.json")) else os.path.join(HERE, "tokenizer")


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


def load_tokens(tok):
    cache = os.path.join(WORK, f"tokens_{VOCAB}.npy")
    if os.path.exists(cache):
        print(f"loaded cached tokens: {os.path.getsize(cache)/1e6:.0f} MB", flush=True)
        return np.load(cache)
    eos = tok.token_to_id("</s>")
    corpus = os.path.join(WORK, "tinystories_corpus.txt")
    print("tokenizing corpus...", flush=True)
    t0, buf = time.time(), []
    with open(corpus) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            buf.extend(tok.encode(line).ids)
            buf.append(eos)
            if i % 50000 == 0:
                print(f"  {i} stories, {len(buf)/1e6:.1f}M tokens, {time.time()-t0:.0f}s", flush=True)
    ids = np.array(buf, dtype=np.uint16)
    np.save(cache, ids)
    print(f"tokenized: {len(ids)/1e6:.1f}M tokens", flush=True)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny fast wiring run")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(WORK, "titan_run"))
    args = ap.parse_args()

    td = tokenizer_dir()
    tok = ByteLevelBPETokenizer(os.path.join(td, "vocab.json"), os.path.join(td, "merges.txt"))
    ids = load_tokens(tok)

    max_w = 4000 if args.smoke else 0
    n_val = max(8, int(((len(ids) - 1) // SEQ) * 0.02))
    val_ids = ids[-(n_val * SEQ + 1):]
    train_ds = LMWindows(ids if args.smoke else ids[:-(n_val * SEQ + 1)], SEQ, max_w)
    val_ds = LMWindows(val_ids, SEQ, 0)
    print(f"train windows: {len(train_ds)}  val windows: {len(val_ds)}  seq={SEQ}", flush=True)

    cfg = TitansConfig(
        vocab_size=VOCAB, d_model=args.d_model, n_layers=args.n_layers, n_heads=8,
        max_seq_len=SEQ, causal=True, chunk_size=128,
        batch_size=16, epochs=args.epochs, lr=3e-4,
        warmup_steps=50 if args.smoke else 300, weight_decay=0.01, use_amp=True,
        log_interval=20, val_every_steps=0 if args.smoke else 500,
        save_every_steps=0 if args.smoke else 1000, output_dir=args.out,
    )
    model = TitansModel.from_config(cfg)
    print(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    TitansTrainer(model, train_ds, val_ds, cfg).train()

    ckpt = os.path.join(args.out, "final.pt")
    model.save_pretrained(ckpt)
    tokdir = os.path.join(args.out, "tokenizer")
    os.makedirs(tokdir, exist_ok=True)
    tok.save_model(tokdir)
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nRUN_DONE time={time.time()-t0:.0f}s peakVRAM={peak:.2f}GiB ckpt={ckpt} tok={tokdir}", flush=True)


if __name__ == "__main__":
    main()
