"""
Long-context TEXT recall on the lucidrains MAC (Vikunja #952, option 4) — the
product-relevant test: a fact buried in a long document, retrieved via the neural
memory (not attention).

Needle-in-haystack per example:
    <filler prose> ... "the secret <NAME> value is <VALUE> ." ... <filler> ...
    "the secret <NAME> value is" -> predict <VALUE>
with TinyStories filler padding the fact far beyond the sliding attention window,
so recall MUST route through the test-time neural memory.

Design choices (informed by the titans-trainer-era lessons):
- COMPLETION-ONLY loss on the single answer token — full-seq loss would train the
  model to language-model the filler and drown the 1-token retrieval signal.
- Single-BPE-token VALUES -> answer is exactly the last position => clean batching,
  loss, and argmax accuracy (no span bookkeeping).
- DISJOINT train/eval name+value pools => measures generalization, not memorization.
- Depth sweep at eval: fact placed at varying depths; far buckets (fact near the
  start, ~seq tokens before the query) are beyond the window => memory-only.

Uses the repo 8k BPE tokenizer. VRAM probe (#952) showed the small model runs
seq 2048 @ batch 8 in ~7.8GiB, so this fits an 8GB card at seq<=1024.
"""
import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import ByteLevelBPETokenizer
from titans_pytorch import MemoryAsContextTransformer, MemoryMLP

HERE = os.path.dirname(os.path.abspath(__file__))

NAMES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Iris",
         "Jack", "Karen", "Leo", "Mia", "Noah", "Olivia", "Paul", "Quinn", "Rose",
         "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier", "Yara", "Zack", "Aaron",
         "Bella", "Chloe", "Dan", "Ella", "Finn", "Gina", "Hugo", "Ivy", "Jake"]
WORDS = ["red", "blue", "green", "gold", "black", "white", "pink", "gray", "brown",
         "purple", "cat", "dog", "bird", "fish", "lion", "bear", "wolf", "fox",
         "frog", "duck", "apple", "bread", "water", "stone", "cloud", "river",
         "forest", "ocean", "desert", "happy", "angry", "quiet", "brave", "gentle",
         "silver", "orange", "yellow", "green", "north", "south", "east", "west"]


def single_token_values(tok):
    """Keep only words that encode (with a leading space) to exactly one token."""
    out = []
    for w in dict.fromkeys(WORDS):  # dedupe, keep order
        ids = tok.encode(" " + w).ids
        if len(ids) == 1:
            out.append((w, ids[0]))
    return out


def load_filler(tok, n_stories):
    from datasets import load_dataset
    print(f"[textrecall] streaming {n_stories} TinyStories for filler...", flush=True)
    buf = []
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    for i, ex in enumerate(ds):
        if i >= n_stories:
            break
        buf.extend(tok.encode(" ".join(ex["text"].split())).ids)
    print(f"[textrecall] filler pool: {len(buf)/1e6:.1f}M tokens", flush=True)
    return np.array(buf, dtype=np.int64)


class NeedleDocs(Dataset):
    """One buried-fact document per item. Answer = single value token at the last
    position. depth=None -> random depth per item (training); else fixed (eval)."""
    def __init__(self, n, tok, filler, names, values, seq, seed=0, depth=None):
        self.n, self.tok, self.filler = n, tok, filler
        self.names, self.values, self.seq = names, values, seq
        self.seed, self.depth = seed, depth

    def __len__(self):
        return self.n

    def _build(self, rng):
        name = rng.choice(self.names)
        word, ans_tok = rng.choice(self.values)
        fact = self.tok.encode(f" the secret {name} value is {word} .").ids
        query = self.tok.encode(f" the secret {name} value is").ids
        tail = query + [ans_tok]
        budget = self.seq - len(tail)                 # tokens for [filler + fact]
        fill_n = budget - len(fact)                   # pure filler tokens
        depth = rng.random() if self.depth is None else self.depth
        n_before = int(depth * fill_n)
        off = rng.randint(0, len(self.filler) - fill_n - 1)
        fill = self.filler[off:off + fill_n]
        seq = np.concatenate([fill[:n_before], np.array(fact, dtype=np.int64),
                              fill[n_before:], np.array(tail, dtype=np.int64)])
        return seq  # length == self.seq, answer token at index seq-1

    def __getitem__(self, i):
        rng = random.Random((self.seed * 1_000_003) ^ i)
        return torch.from_numpy(self._build(rng))


@torch.no_grad()
def eval_depths(model, tok, filler, names, values, seq, n=256, seed=999,
                depths=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9)):
    model.eval()
    dev = next(model.parameters()).device
    mb = 16  # mini-batch: a single [n, seq, vocab] forward would OOM at vocab 8k
    out = {}
    for d in depths:
        ds = NeedleDocs(n, tok, filler, names, values, seq, seed=seed, depth=d)
        correct = 0
        for s in range(0, n, mb):
            x = torch.stack([ds[i] for i in range(s, min(s + mb, n))]).to(dev)
            pred = model(x, return_loss=False)[:, seq - 2].argmax(-1)  # predicts token at seq-1
            correct += (pred == x[:, seq - 1]).sum().item()
        out[d] = correct / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--segment-len", type=int, default=128, help="sliding attn window")
    ap.add_argument("--nmem-seg", type=int, default=16)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--neural-mem-layers", type=int, nargs="*", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--tinystories-n", type=int, default=20000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--early-stop-far", type=float, default=0.9,
                    help="stop once far-recall (depth<=0.25 mean) exceeds this")
    ap.add_argument("--tokenizer-dir", default=os.path.join(HERE, "tokenizer"))
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    tok = ByteLevelBPETokenizer(os.path.join(args.tokenizer_dir, "vocab.json"),
                                os.path.join(args.tokenizer_dir, "merges.txt"))
    vocab = tok.get_vocab_size()
    vals = single_token_values(tok)
    # disjoint pools
    n_tr = int(len(NAMES) * 0.75)
    names_tr, names_ev = NAMES[:n_tr], NAMES[n_tr:]
    nv = int(len(vals) * 0.6)
    vals_tr, vals_ev = vals[:nv], vals[nv:]
    print(f"[textrecall] vocab={vocab} names(tr/ev)={len(names_tr)}/{len(names_ev)} "
          f"values(tr/ev)={len(vals_tr)}/{len(vals_ev)} single-token", flush=True)
    assert len(names_ev) and len(vals_ev), "need non-empty disjoint eval pools"

    filler = load_filler(tok, args.tinystories_n)

    mem_layers = tuple(args.neural_mem_layers) if args.neural_mem_layers else (args.layers,)
    model = MemoryAsContextTransformer(
        num_tokens=vocab, dim=args.dim, depth=args.layers,
        segment_len=args.segment_len, heads=args.heads, dim_head=args.dim // args.heads,
        num_persist_mem_tokens=4, num_longterm_mem_tokens=4,
        neural_memory_layers=mem_layers, neural_memory_segment_len=args.nmem_seg,
        sliding_window_attn=True, use_flex_attn=False,
        neural_memory_model=MemoryMLP(dim=64, depth=2),
        neural_memory_kwargs=dict(dim_head=64, heads=4, use_accelerated_scan=False),
    ).to(dev)
    print(f"[textrecall] MAC params={sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"seq={args.seq} window={args.segment_len} mem_layers={mem_layers} "
          f"steps={args.steps} (far buckets depth<=0.25 = memory-forced)", flush=True)

    train_ds = NeedleDocs(args.steps * args.batch_size, tok, filler, names_tr, vals_tr,
                          args.seq, seed=0, depth=None)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, args.warmup))
        * max(0.1, 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))))

    def fmt(d):
        return " ".join(f"{k}:{v:.2f}" for k, v in d.items())

    pre = eval_depths(model, tok, filler, names_ev, vals_ev, args.seq)
    print(f"[textrecall] PRE-TRAIN depth-acc {fmt(pre)}", flush=True)

    model.train()
    step = 0
    S = args.seq
    for x in loader:
        if step >= args.steps:
            break
        x = x.to(dev)
        logits = model(x, return_loss=False)                      # [B, S, V]
        loss = F.cross_entropy(logits[:, S - 2], x[:, S - 1])     # completion-only (answer token)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0:
            acc = eval_depths(model, tok, filler, names_ev, vals_ev, args.seq, n=128)
            model.train()
            far = (acc[0.0] + acc[0.1] + acc[0.25]) / 3
            print(f"[textrecall] step {step} loss {loss.item():.3f} far {far:.3f} | {fmt(acc)}", flush=True)
            if args.early_stop_far and far > args.early_stop_far:
                print(f"[textrecall] EARLY_STOP step {step} far={far:.3f}", flush=True)
                break
        step += 1

    acc = eval_depths(model, tok, filler, names_ev, vals_ev, args.seq, n=512)
    far = (acc[0.0] + acc[0.1] + acc[0.25]) / 3
    near = acc[0.9]
    print(f"\n[textrecall] RESULT far-recall(depth<=0.25)={far:.3f}  near(depth0.9)={near:.3f}  "
          f"| full {fmt(acc)}", flush=True)
    print("[textrecall] " + ("TEXT_RECALL_WORKS" if far > 0.8 else
                              "TEXT_RECALL_PARTIAL" if far > 0.3 else "TEXT_RECALL_FAILS"), flush=True)
    print("[textrecall] TEXT_RECALL_DONE", flush=True)


if __name__ == "__main__":
    main()
