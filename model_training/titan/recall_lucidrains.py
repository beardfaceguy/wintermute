"""
Stage-A associative-recall go/no-go on the TRUSTED backend (lucidrains/titans-pytorch).

This is the real Stage-A question, not just copy/induction: can the MAC neural
memory learn key->value BINDING and retrieve the value for a queried key? This is
exactly what pafos-ai/titans-trainer failed at (comment #329: it emitted the
marginal value distribution, never the queried binding). induction_lucidrains.py
already showed lucidrains forms induction; this raises the bar to associative recall.

Task (format/tokenizer-free, MQAR-style single query):
    k1 v1 k2 v2 ... kN vN  SEP  kq  -> predict vq   (kq is one of the earlier keys)
Keys sampled without replacement from a key range, values from a disjoint value
range, fresh every example -> can't memorize, must bind-and-retrieve. Metric:
argmax accuracy at the query position (predicting vq). Chance = 1/|value range|.

--sliding forces the pairs out of the local attention window so retrieval MUST go
through the neural memory (the true long-term-memory test); default full attention
first establishes that binding forms at all.
"""
import argparse
import math
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from titans_pytorch import MemoryAsContextTransformer, MemoryMLP

SEP = 5
KEY_LO, KEY_HI = 10, 130       # 120 possible keys
VAL_LO, VAL_HI = 130, 256      # 126 possible values


def make_seq(n_pairs, rng):
    keys = rng.sample(range(KEY_LO, KEY_HI), n_pairs)
    vals = [rng.randrange(VAL_LO, VAL_HI) for _ in range(n_pairs)]
    qi = rng.randrange(n_pairs)
    seq = []
    for k, v in zip(keys, vals):
        seq += [k, v]
    seq += [SEP, keys[qi], vals[qi]]     # query key then its value (answer)
    return seq


class RecallSeqs(Dataset):
    def __init__(self, n, n_pairs, seed=0):
        self.n, self.n_pairs, self.seed = n, n_pairs, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = random.Random((self.seed * 1_000_003) ^ i)
        return torch.tensor(make_seq(self.n_pairs, rng), dtype=torch.long)


@torch.no_grad()
def evaluate(model, n_pairs, vocab, n=512, seed=999):
    model.eval()
    device = next(model.parameters()).device
    rng = random.Random(seed)
    seqs = [make_seq(n_pairs, rng) for _ in range(n)]
    x = torch.tensor(seqs, dtype=torch.long, device=device)   # [n, 2N+3]
    logits = model(x, return_loss=False)                      # [n, L, vocab]
    ans = 2 * n_pairs + 1                                     # index of kq; predicts vq at ans+1
    pred = logits[:, ans].argmax(-1)
    tgt = x[:, ans + 1]
    acc = (pred == tgt).float().mean().item()
    ce = F.cross_entropy(logits[:, ans], tgt).item()
    return acc, ce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--n-pairs", type=int, default=24)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--segment-len", type=int, default=64, help=">=seq => full attn")
    ap.add_argument("--sliding", action="store_true", help="force recall through neural memory")
    ap.add_argument("--nmem-seg", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    V, N = args.vocab, args.n_pairs
    seq = 2 * N + 3
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = MemoryAsContextTransformer(
        num_tokens=V, dim=args.dim, depth=args.depth,
        segment_len=args.segment_len, heads=args.heads, dim_head=args.dim // args.heads,
        num_persist_mem_tokens=4, num_longterm_mem_tokens=4,
        neural_memory_layers=(args.depth,), neural_memory_segment_len=args.nmem_seg,
        sliding_window_attn=args.sliding, use_flex_attn=False,
        neural_memory_model=MemoryMLP(dim=64, depth=2),
        neural_memory_kwargs=dict(dim_head=64, heads=4, use_accelerated_scan=False),
    ).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 200) * max(0.1, 0.5 * (1 + math.cos(math.pi * s / args.steps))))
    chance = 1.0 / (VAL_HI - VAL_LO)
    print(f"[recall] MAC params={sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"pairs={N} seq={seq} segment_len={args.segment_len} sliding={args.sliding} "
          f"nmem_seg={args.nmem_seg} steps={args.steps} chance={chance:.4f}", flush=True)
    a0, c0 = evaluate(model, N, V)
    print(f"[recall] PRE-TRAIN query-acc={a0:.3f} query-ce={c0:.3f}", flush=True)

    ds = RecallSeqs(args.steps * args.batch_size, N, seed=0)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
    model.train()
    step = 0
    for x in loader:
        if step >= args.steps:
            break
        x = x.to(dev)
        loss = model(x, return_loss=True)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 200 == 0:
            acc, _ = evaluate(model, N, V, n=256)
            model.train()
            print(f"[recall] step {step} loss {loss.item():.3f} query-acc {acc:.3f}", flush=True)
        step += 1

    acc, ce = evaluate(model, N, V)
    print(f"\n[recall] RESULT query-acc={acc:.3f}  query-ce={ce:.3f}  (chance={chance:.4f})", flush=True)
    print("[recall] " + ("RECALL_WORKS" if acc > 0.8 else
                          "RECALL_PARTIAL" if acc > 5 * chance else "RECALL_FAILS"), flush=True)
    print("[recall] RECALL_DONE", flush=True)


if __name__ == "__main__":
    main()
