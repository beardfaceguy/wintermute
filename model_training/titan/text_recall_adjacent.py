"""
Salvage test for #952 option 4: MQAR-in-text — is the text-recall failure caused
by the answer not being ADJACENT to a unique matching token?

The failed text task keyed the answer off " ...value is <VALUE>", where "is" is
ambiguous (everywhere in filler) and the distinctive anchor (name) sat several
tokens before the answer. MQAR worked because the key was a UNIQUE token with the
value IMMEDIATELY after it ([kq][vq]) -> trivial single-token induction.

This isolates that one variable: keep real TinyStories filler, but structure the
doc as MQAR — a unique key token directly followed by the value:
    <filler> <KEY> <VALUE> <filler> <KEY> -> predict <VALUE>
KEY is guaranteed absent from the filler slice, so induction on KEY is unambiguous.

  learns (in-window seq128, then far seq512) => text recall works with a distinctive
     adjacent cue; the earlier failure was cue structure, not the arch.
  fails even here                            => natural-text filler itself is the wall.
"""
import argparse
import math
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import ByteLevelBPETokenizer
import os

from text_recall_lucidrains import single_token_values, load_filler
from titans_pytorch import MemoryAsContextTransformer, MemoryMLP

HERE = os.path.dirname(os.path.abspath(__file__))


class AdjDocs(Dataset):
    """[filler] KEY VALUE [filler] KEY -> VALUE. KEY unique (absent from filler)."""
    def __init__(self, n, filler, keys, values, seq, seed=0, depth=None):
        self.n, self.filler, self.keys, self.values = n, filler, keys, values
        self.seq, self.seed, self.depth = seq, seed, depth

    def __len__(self):
        return self.n

    def _build(self, rng):
        ktok = rng.choice(self.keys)
        vtok = rng.choice(self.values)
        fact = [ktok, vtok]
        tail = [ktok, vtok]                      # query KEY, then answer VALUE (last position)
        budget = self.seq - len(tail)
        fill_n = budget - len(fact)
        for _ in range(12):                      # ensure KEY is unique -> filler must not contain it
            off = rng.randint(0, len(self.filler) - fill_n - 1)
            fill = self.filler[off:off + fill_n]
            if ktok not in fill:
                break
        depth = rng.random() if self.depth is None else self.depth
        nb = int(depth * fill_n)
        seq = np.concatenate([fill[:nb], np.array(fact), fill[nb:], np.array(tail)])
        return seq.astype(np.int64)

    def __getitem__(self, i):
        rng = random.Random((self.seed * 1_000_003) ^ i)
        return torch.from_numpy(self._build(rng))


@torch.no_grad()
def eval_depths(model, filler, keys, values, seq, n=256, seed=999,
                depths=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9)):
    model.eval()
    dev = next(model.parameters()).device
    mb, out = 16, {}
    for d in depths:
        ds = AdjDocs(n, filler, keys, values, seq, seed=seed, depth=d)
        correct = 0
        for s in range(0, n, mb):
            x = torch.stack([ds[i] for i in range(s, min(s + mb, n))]).to(dev)
            pred = model(x, return_loss=False)[:, seq - 2].argmax(-1)
            correct += (pred == x[:, seq - 1]).sum().item()
        out[d] = correct / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--segment-len", type=int, default=128)
    ap.add_argument("--nmem-seg", type=int, default=16)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--neural-mem-layers", type=int, nargs="*", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--tinystories-n", type=int, default=15000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--early-stop-far", type=float, default=0.8)
    ap.add_argument("--tokenizer-dir", default=os.path.join(HERE, "tokenizer"))
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    tok = ByteLevelBPETokenizer(os.path.join(args.tokenizer_dir, "vocab.json"),
                                os.path.join(args.tokenizer_dir, "merges.txt"))
    vocab = tok.get_vocab_size()
    words = [t for _, t in single_token_values(tok)]     # token ids only
    keys, values = words[:500], words[500:1000]
    print(f"[adj] vocab={vocab} keys={len(keys)} values={len(values)} (in-distribution eval, "
          f"fresh combos) seq={args.seq} window={args.segment_len}", flush=True)
    filler = load_filler(tok, args.tinystories_n)

    model = MemoryAsContextTransformer(
        num_tokens=vocab, dim=args.dim, depth=args.layers, segment_len=args.segment_len,
        heads=args.heads, dim_head=args.dim // args.heads,
        num_persist_mem_tokens=4, num_longterm_mem_tokens=4,
        neural_memory_layers=tuple(args.neural_mem_layers) if args.neural_mem_layers else (args.layers,),
        neural_memory_segment_len=args.nmem_seg,
        sliding_window_attn=True, use_flex_attn=False,
        neural_memory_model=MemoryMLP(dim=64, depth=2),
        neural_memory_kwargs=dict(dim_head=64, heads=4, use_accelerated_scan=False),
    ).to(dev)
    print(f"[adj] MAC params={sum(p.numel() for p in model.parameters())/1e6:.1f}M steps={args.steps}", flush=True)

    ds = AdjDocs(args.steps * args.batch_size, filler, keys, values, args.seq, seed=0)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, args.warmup))
        * max(0.1, 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))))

    def fmt(d):
        return " ".join(f"{k}:{v:.2f}" for k, v in d.items())

    print(f"[adj] PRE-TRAIN {fmt(eval_depths(model, filler, keys, values, args.seq))}", flush=True)
    model.train()
    step, S = 0, args.seq
    for x in loader:
        if step >= args.steps:
            break
        x = x.to(dev)
        logits = model(x, return_loss=False)
        loss = torch.nn.functional.cross_entropy(logits[:, S - 2], x[:, S - 1])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0:
            acc = eval_depths(model, filler, keys, values, args.seq, n=128)
            model.train()
            far = (acc[0.0] + acc[0.1] + acc[0.25]) / 3
            print(f"[adj] step {step} loss {loss.item():.3f} far {far:.3f} near {acc[0.9]:.3f} | {fmt(acc)}", flush=True)
            if args.early_stop_far and far > args.early_stop_far:
                print(f"[adj] EARLY_STOP step {step} far={far:.3f}", flush=True)
                break
        step += 1

    acc = eval_depths(model, filler, keys, values, args.seq, n=512)
    far = (acc[0.0] + acc[0.1] + acc[0.25]) / 3
    print(f"\n[adj] RESULT far(depth<=0.25)={far:.3f} near(0.9)={acc[0.9]:.3f} | {fmt(acc)}", flush=True)
    print("[adj] " + ("ADJ_WORKS" if far > 0.8 else "ADJ_NEAR_ONLY" if acc[0.9] > 0.8 else "ADJ_FAILS"), flush=True)
    print("[adj] ADJ_DONE", flush=True)


if __name__ == "__main__":
    main()
