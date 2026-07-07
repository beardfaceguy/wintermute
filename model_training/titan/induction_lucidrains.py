"""
Induction gate against the TRUSTED reference impl: lucidrains/titans-pytorch.

Same repeated-block induction task + per-position eval as induction_probe.py,
but the model is lucidrains' MemoryAsContextTransformer (MAC). Purpose: decide
whether the induction failure is specific to pafos-ai/titans-trainer's bespoke
MAC block, or intrinsic to the MAC design.

Regime is deliberately hard for the memory: segment_len=32 with sliding-window
attention, while the repeat is at distance L=64 -> local attention CANNOT bridge
the copy; only the neural long-term memory can. So 2nd-copy acc -> ~1.0 means the
Titan neural memory genuinely performs associative recall.

  lucidrains MAC LEARNS  => titans-trainer impl was the blocker; switch backend.
  lucidrains MAC FAILS   => induction/recall is intrinsically hard for MAC here;
                            external-memory track (#182) is the right pivot.

lucidrains forward(x, return_loss=True) computes the autoregressive loss
internally (it shifts), so the dataset yields the FULL 2L sequence (no pre-shift).
"""
import argparse
import math
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from titans_pytorch import MemoryAsContextTransformer, MemoryMLP


class BlockSeqs(Dataset):
    """Full repeated-random-block sequences [2L] (lucidrains shifts internally)."""
    def __init__(self, n, vocab, block_len, seed=0):
        self.n, self.vocab, self.L, self.seed = n, vocab, block_len, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = random.Random((self.seed * 1_000_003) ^ i)
        blk = [rng.randrange(10, self.vocab) for _ in range(self.L)]
        return torch.tensor(blk + blk, dtype=torch.long)


@torch.no_grad()
def evaluate(model, vocab, L, n=256, seed=999):
    model.eval()
    device = next(model.parameters()).device
    rng = random.Random(seed)
    seqs = [[*(b := [rng.randrange(10, vocab) for _ in range(L)]), *b] for _ in range(n)]
    x = torch.tensor(seqs, dtype=torch.long, device=device)      # [n, 2L]
    logits = model(x, return_loss=False)                          # [n, 2L, vocab]
    logits = logits[:, :-1]                                       # pos j predicts x[:,j+1]
    tgt = x[:, 1:]
    ce = F.cross_entropy(logits.reshape(-1, vocab), tgt.reshape(-1),
                         reduction="none").reshape(n, 2 * L - 1)
    first = ce[:, : L - 1].mean().item()
    second = ce[:, L - 1:].mean().item()
    acc2 = (logits[:, L - 1:].argmax(-1) == x[:, L:]).float().mean().item()
    return first, second, acc2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--block-len", type=int, default=64)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--segment-len", type=int, default=128, help="local attn window (>=seq => full attn)")
    ap.add_argument("--sliding", action="store_true", help="sliding-window attn (forces cross-window via memory)")
    ap.add_argument("--nmem-seg", type=int, default=16, help="neural memory segment len (coarser=cheaper)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    L, V = args.block_len, args.vocab
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = MemoryAsContextTransformer(
        num_tokens=V, dim=args.dim, depth=args.depth,
        segment_len=args.segment_len, heads=args.heads, dim_head=args.dim // args.heads,
        num_persist_mem_tokens=4, num_longterm_mem_tokens=4,
        neural_memory_layers=(args.depth,),
        neural_memory_segment_len=args.nmem_seg,
        sliding_window_attn=args.sliding,
        use_flex_attn=False,  # avoid fragile flex-attn compile path
        neural_memory_model=MemoryMLP(dim=64, depth=2),
        neural_memory_kwargs=dict(dim_head=64, heads=4, use_accelerated_scan=False),  # match ref; avoid CUDA-kernel dep
    ).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 200) * max(0.1, 0.5 * (1 + math.cos(math.pi * s / args.steps))))
    print(f"[lucid] MAC params={sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"vocab={V} L={L} seq={2*L} segment_len={args.segment_len} sliding={args.sliding} "
          f"neural_mem_layers=({args.depth},) nmem_seg={args.nmem_seg} steps={args.steps}", flush=True)
    pre = evaluate(model, V, L)
    print(f"[lucid] PRE-TRAIN 1st={pre[0]:.3f} 2nd={pre[1]:.3f} acc2={pre[2]:.3f} "
          f"(ln(V)={math.log(V):.3f})", flush=True)

    ds = BlockSeqs(args.steps * args.batch_size, V, L, seed=0)
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
            print(f"[lucid] step {step} loss {loss.item():.3f}", flush=True)
        step += 1

    first, second, acc2 = evaluate(model, V, L)
    print(f"\n[lucid] RESULT 1st-copy loss={first:.3f}  2nd-copy loss={second:.3f}  "
          f"2nd-copy acc={acc2:.3f}", flush=True)
    print("[lucid] " + ("LUCID_LEARNS_INDUCTION" if acc2 > 0.8 else "LUCID_FAILS"), flush=True)
    print("[lucid] LUCID_DONE", flush=True)


if __name__ == "__main__":
    main()
