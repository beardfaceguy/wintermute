"""
Stage-A training: recall-aware, long-context Titan (local, gaming-pc).

Blends TinyStories (flat LM windows, for fluency) with synthetic recall docs
(each padded to a whole seq_len window so a target fact + its late query land in
the SAME window). Trains a small MAC-Titan at long context (seq >> the block's
~512 attention window) so the test-time neural memory is the only way to carry
a far fact to the query. recall_probe.py then measures far-recall.

--no-recall trains the matched CONTROL (TinyStories only) for attribution.
"""
import argparse
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from tokenizers import ByteLevelBPETokenizer
from titans_trainer import TitansConfig, TitansModel, TitansTrainer

import recall_data  # same dir

HERE = os.path.dirname(os.path.abspath(__file__))
PAD = 0  # padding_idx / <pad>


class FlatWindows(Dataset):
    """Contiguous LM windows over a flat token stream (TinyStories)."""
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


class RecallWindows(Dataset):
    """One recall doc per window, padded to seq_len. COMPLETION-ONLY loss:
    supervise ONLY the answer code tokens; facts/template/padding are all -100.
    Full-sequence loss failed — the model learned the template and emitted a
    generic code; grading only the answer forces it to actually retrieve."""
    def __init__(self, tok, n_examples, facts, target_frac, seq, seed=0, letters=False):
        self.items = []
        rng = random.Random(seed)
        tries = 0
        while len(self.items) < n_examples and tries < n_examples * 4:
            tries += 1
            prefix, answer = recall_data.make_example_split(facts, target_frac, rng, letters=letters)
            pids = tok.encode(prefix).ids
            fids = tok.encode(prefix + answer).ids
            if len(fids) > seq:
                continue
            cp = 0  # common token prefix (robust to BPE boundary merges) → answer span [cp, len(fids))
            while cp < len(pids) and cp < len(fids) and pids[cp] == fids[cp]:
                cp += 1
            arr = np.array(fids + [PAD] * ((seq + 1) - len(fids)), dtype=np.int64)
            self.items.append((arr[:seq + 1], cp, len(fids)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        arr, a0, a1 = self.items[i]
        w = torch.from_numpy(arr)
        inp, lab = w[:-1].clone(), w[1:].clone()
        mask = torch.full_like(lab, -100)
        lo, hi = max(0, a0 - 1), max(0, a1 - 1)  # label idx i predicts token i+1; answer = tokens [a0,a1)
        mask[lo:hi] = lab[lo:hi]
        return {"input_ids": inp, "labels": mask}


def load_tinystories_ids(tok, n_stories):
    from datasets import load_dataset
    eos = tok.token_to_id("</s>") or 0
    print(f"[stageA] streaming {n_stories} TinyStories...", flush=True)
    buf = []
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    for i, ex in enumerate(ds):
        if i >= n_stories:
            break
        buf.extend(tok.encode(" ".join(ex["text"].split())).ids)
        buf.append(eos)
    print(f"[stageA] tinystories tokens: {len(buf)/1e6:.1f}M", flush=True)
    return np.array(buf, dtype=np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="stage_a_run")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--vocab-size", type=int, default=8000)
    ap.add_argument("--tinystories-n", type=int, default=30000)
    ap.add_argument("--recall-n", type=int, default=4000, help="recall windows (0 with --no-recall)")
    ap.add_argument("--recall-facts", type=int, default=90, help="facts/recall-doc (size to ~fill seq_len)")
    ap.add_argument("--no-recall", action="store_true", help="CONTROL: TinyStories only")
    ap.add_argument("--value-mode", choices=["code", "letter"], default="code", help="letter=single-token values (easier copy)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--tokenizer-dir", default=os.path.join(HERE, "tokenizer"))
    args = ap.parse_args()

    tok = ByteLevelBPETokenizer(os.path.join(args.tokenizer_dir, "vocab.json"),
                                os.path.join(args.tokenizer_dir, "merges.txt"))

    parts, ts_ids = [], None
    if args.tinystories_n > 0:
        ts_ids = load_tinystories_ids(tok, args.tinystories_n)
        parts.append(FlatWindows(ts_ids, args.seq_len))
    if not args.no_recall and args.recall_n > 0:
        parts.append(RecallWindows(tok, args.recall_n, args.recall_facts, 1.0, args.seq_len, letters=(args.value_mode == "letter")))
    if not parts:
        raise SystemExit("nothing to train on (need tinystories-n>0 or recall)")
    print("[stageA] windows: " + ", ".join(f"{type(p).__name__}={len(p)}" for p in parts), flush=True)
    train_ds = ConcatDataset(parts)

    # small held-out val: a few recall windows (or tinystories for the control)
    if not args.no_recall and args.recall_n > 0:
        val_ds = RecallWindows(tok, 64, args.recall_facts, 1.0, args.seq_len, seed=999, letters=(args.value_mode == "letter"))
    else:
        val_ds = FlatWindows(ts_ids[-(65 * args.seq_len + 1):], args.seq_len)

    cfg = TitansConfig(
        vocab_size=args.vocab_size, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, max_seq_len=args.seq_len, causal=True, chunk_size=128,
        batch_size=args.batch_size, epochs=args.epochs, lr=3e-4, warmup_steps=100,
        weight_decay=0.01, use_amp=True, log_interval=20,
        val_every_steps=500, save_every_steps=1000, output_dir=args.out,
    )
    model = TitansModel.from_config(cfg)
    print(f"[stageA] params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"| seq={args.seq_len} batch={args.batch_size} recall={'OFF' if args.no_recall else 'ON'}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    TitansTrainer(model, train_ds, val_ds, cfg).train()
    ckpt = os.path.join(args.out, "final.pt")
    model.save_pretrained(ckpt)
    tok.save_model(args.out)
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n[stageA] STAGE_A_DONE time={time.time()-t0:.0f}s peakVRAM={peak:.2f}GiB ckpt={ckpt}", flush=True)


if __name__ == "__main__":
    main()
