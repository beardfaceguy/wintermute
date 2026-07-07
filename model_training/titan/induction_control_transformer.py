"""
Plain-transformer CONTROL for the induction gate.

Trains a vanilla causal Transformer (no Titan memory) on the SAME repeated-block
induction task + the SAME InductionData/eval as induction_probe.py. Purpose: tell
apart two hypotheses for the Titan's flat-at-uniform failure —

  control LEARNS (2nd-copy acc -> ~1.0)  => task/data are trivially learnable;
      the Titan's failure is IMPLEMENTATION-SPECIFIC to titans-trainer's MAC block.
  control ALSO FAILS                     => something about the task/eval harness is
      off (should not happen — this is textbook induction).

A hand-rolled AdamW loop is unquestionably correct for a standard transformer
(the earlier manual-loop doubt was about faithfully training the Titan's custom
test-time memory, which this control does not have).
"""
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from induction_probe import InductionData, evaluate  # same dir


class TinyGPT(nn.Module):
    def __init__(self, vocab, d_model, n_layers, n_heads, max_seq):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True, activation="gelu", norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)
        self.max_seq = max_seq

    def forward(self, x, labels=None):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.tok(x) + self.pos(pos)[None]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), 1)
        h = self.norm(self.enc(h, mask=mask, is_causal=True))
        logits = self.head(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   labels.reshape(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--block-len", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mask-first", action="store_true")
    args = ap.parse_args()

    L, V = args.block_len, args.vocab
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = InductionData(args.steps * args.batch_size, V, L, seed=0, mask_first=args.mask_first)
    model = TinyGPT(V, args.d_model, args.n_layers, args.n_heads, 2 * L).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 200) * max(0.1, 0.5 * (1 + math.cos(math.pi * s / args.steps))))
    print(f"[control] TinyGPT params={sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"vocab={V} L={L} seq={2*L} steps={args.steps} mask_first={args.mask_first}", flush=True)
    print(f"[control] PRE-TRAIN 1st={evaluate(model, V, L)[0]:.3f} "
          f"2nd={evaluate(model, V, L)[1]:.3f} acc2={evaluate(model, V, L)[2]:.3f} "
          f"(ln(V)={math.log(V):.3f})", flush=True)

    model.train()
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                         drop_last=True, pin_memory=True)
    step = 0
    for b in loader:
        if step >= args.steps:
            break
        x, y = b["input_ids"].to(dev), b["labels"].to(dev)
        loss = model(x, labels=y)["loss"]
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 200 == 0:
            print(f"[control] step {step} loss {loss.item():.3f}", flush=True)
        step += 1

    first, second, acc2 = evaluate(model, V, L)
    print(f"\n[control] RESULT 1st-copy loss={first:.3f}  2nd-copy loss={second:.3f}  "
          f"2nd-copy acc={acc2:.3f}", flush=True)
    print("[control] " + ("CONTROL_LEARNS_INDUCTION" if acc2 > 0.8 else "CONTROL_ALSO_FAILS"), flush=True)
    print("[control] CONTROL_DONE", flush=True)


if __name__ == "__main__":
    main()
