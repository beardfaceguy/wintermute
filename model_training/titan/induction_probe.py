"""
Canonical induction sanity check for the MAC-Titan, run through the REAL
`TitansTrainer` (NOT a hand-rolled AdamW loop).

Why this exists: Stage-A found the Titan never learns key->value binding, and a
prior induction side-test (`induction_test.py`) used a bare manual loop whose
numbers were discarded as harness-limited (train loss flat at ln(vocab)). The
robust Stage-A results all came through `TitansTrainer` (grad-clip / cosine LR /
warmup / AMP). This rewrite settles the crux gate with that same trainer:

  Does the MAC-Titan form an induction / associative-copy circuit AT ALL?

Task (format/tokenizer-free): a random block of L tokens repeated twice ->
sequence of length 2L. An induction head predicts every 2nd-copy token
perfectly, because each token appeared exactly L positions earlier. Each block
is freshly random, so the model cannot memorize -- it must learn the copy rule
(measured on a disjoint eval set => generalization).

Metric: per-position CE loss / argmax accuracy on 2nd-copy positions vs 1st.
  2nd-copy loss -> ~0 / acc -> ~1.0   => induction FORMS
     (=> our recall failure is recipe/scale: longer training / LR sweep / control)
  2nd-copy loss stays ~ln(vocab)      => induction does NOT form in this arch/impl
     (=> pivot the "agent that remembers me" goal to the external-memory track, #182)

titans-trainer gotcha honored: TitansModel.forward does NOT shift labels, so the
Dataset pre-shifts (input=tokens[:-1], labels=tokens[1:]) with contiguous slices.
"""
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from titans_trainer import TitansConfig, TitansModel, TitansTrainer


class InductionData(Dataset):
    """Map-style dataset of repeated-random-block sequences, pre-shifted for the
    TitansModel forward. __getitem__(i) is deterministic (seeded by index) so the
    set is fixed, reproducible, and safe to shuffle. Token ids are drawn from
    [10, vocab) (0 = pad reserved), matching the original manual test."""

    def __init__(self, n, vocab, block_len, seed=0, mask_first=False):
        self.n, self.vocab, self.L = n, vocab, block_len
        self.seed, self.mask_first = seed, mask_first

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = random.Random((self.seed * 1_000_003) ^ i)
        blk = [rng.randrange(10, self.vocab) for _ in range(self.L)]
        seq = torch.tensor(blk + blk, dtype=torch.long)  # [2L]
        inp = seq[:-1].contiguous()                      # position j predicts token j+1
        lab = seq[1:].clone().contiguous()
        if self.mask_first:
            lab[: self.L - 1] = -100                      # supervise ONLY 2nd-copy targets
        return {"input_ids": inp, "labels": lab}


@torch.no_grad()
def evaluate(model, vocab, block_len, n=256, seed=999):
    """Per-position CE / accuracy on a fresh (disjoint) batch of induction seqs."""
    model.eval()
    device = next(model.parameters()).device  # eval both pre-train (CPU) and post (CUDA)
    L = block_len
    rng = random.Random(seed)
    seqs = []
    for _ in range(n):
        blk = [rng.randrange(10, vocab) for _ in range(L)]
        seqs.append(blk + blk)
    x = torch.tensor(seqs, dtype=torch.long, device=device)      # [n, 2L]
    logits = model(x[:, :-1].contiguous())["logits"]             # [n, 2L-1, V]; pos j predicts x[:,j+1]
    tgt = x[:, 1:]
    ce = F.cross_entropy(logits.reshape(-1, vocab), tgt.reshape(-1),
                         reduction="none").reshape(n, 2 * L - 1)
    first = ce[:, : L - 1].mean().item()     # predicting 1st-copy tokens x[1..L-1] (random) -> high
    second = ce[:, L - 1:].mean().item()     # predicting 2nd-copy tokens x[L..2L-1] -> induction target
    acc2 = (logits[:, L - 1:].argmax(-1) == x[:, L:]).float().mean().item()
    return first, second, acc2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--block-len", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--train-n", type=int, default=320000, help="distinct seqs/epoch")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--mask-first", action="store_true",
                    help="supervise only 2nd-copy targets (cleaner induction signal)")
    ap.add_argument("--out", default="induction_tt")
    args = ap.parse_args()

    L, V = args.block_len, args.vocab
    seq = 2 * L

    train_ds = InductionData(args.train_n, V, L, seed=0, mask_first=args.mask_first)
    val_ds = InductionData(2048, V, L, seed=7, mask_first=args.mask_first)

    cfg = TitansConfig(
        vocab_size=V, d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        max_seq_len=seq, causal=True, chunk_size=args.chunk_size,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
        warmup_steps=args.warmup_steps, weight_decay=0.01, use_amp=True,
        log_interval=50, val_every_steps=500, save_every_steps=0, output_dir=args.out,
    )
    model = TitansModel.from_config(cfg)
    steps = int((args.train_n // args.batch_size) * args.epochs)
    print(f"[induction] params={sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"vocab={V} L={L} seq={seq} ~{steps} opt-steps mask_first={args.mask_first} "
          f"(via TitansTrainer)", flush=True)

    pre = evaluate(model, V, L)
    print(f"[induction] PRE-TRAIN 1st-copy={pre[0]:.3f} 2nd-copy={pre[1]:.3f} "
          f"acc2={pre[2]:.3f}  (ln(vocab)={np.log(V):.3f})", flush=True)

    TitansTrainer(model, train_ds, val_ds, cfg).train()

    first, second, acc2 = evaluate(model, V, L)
    print(f"\n[induction] RESULT 1st-copy loss={first:.3f}  2nd-copy loss={second:.3f}  "
          f"2nd-copy acc={acc2:.3f}", flush=True)
    print("[induction] " + ("INDUCTION_WORKS" if acc2 > 0.8 else "INDUCTION_FAILS"), flush=True)
    print("[induction] INDUCTION_DONE", flush=True)


if __name__ == "__main__":
    main()
