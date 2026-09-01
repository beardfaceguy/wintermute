"""
Pretrain a small MAC-Titan on TinyStories using the ADOPTED backend
lucidrains/titans-pytorch (MemoryAsContextTransformer).

This replaces pafos-ai/titans-trainer's bespoke MAC block, whose implementation
never formed induction/recall (Vikunja #938 #340/#341; task #951). Same 8k BPE
tokenizer + TinyStories corpus as pretrain.py, but the model and loop are
lucidrains-native:
  - lucidrains ships no trainer, so this is a clean AdamW + cosine-warmup + grad-clip
    loop (fp32; NOT AMP -- the neural memory's inner functorch grads are unstable
    under autocast, and the validated recall runs were fp32).
  - forward(x, return_loss=True) shifts labels internally, so windows are FULL
    length seq_len+1 (do NOT pre-shift, unlike the titans-trainer path).

Defaults are cloud-oriented (task #952); --smoke is the local 8GB validation.
"""
import argparse
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import ByteLevelBPETokenizer
from titans_pytorch import MemoryAsContextTransformer, MemoryMLP

HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB = 8000


class LMWindows(Dataset):
    """Full-length (seq+1) contiguous windows; lucidrains shifts labels internally."""
    def __init__(self, ids, seq, max_windows=0):
        self.ids, self.seq = ids, seq
        n = (len(ids) - 1) // seq
        self.n = min(n, max_windows) if max_windows else n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = i * self.seq
        return torch.from_numpy(self.ids[s:s + self.seq + 1].astype(np.int64))


def load_tinystories_ids(tok, n_stories):
    from datasets import load_dataset
    eos = tok.token_to_id("</s>") or 0
    print(f"[lucid-pretrain] streaming {n_stories} TinyStories...", flush=True)
    buf = []
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    for i, ex in enumerate(ds):
        if i >= n_stories:
            break
        buf.extend(tok.encode(" ".join(ex["text"].split())).ids)
        buf.append(eos)
    print(f"[lucid-pretrain] tokens: {len(buf)/1e6:.1f}M", flush=True)
    return np.array(buf, dtype=np.uint16)


def cycle(loader):
    while True:
        for b in loader:
            yield b


@torch.no_grad()
def evaluate(model, loader, device, max_batches=20):
    model.eval()
    tot, n = 0.0, 0
    for i, x in enumerate(loader):
        if i >= max_batches:
            break
        tot += model(x.to(device), return_loss=True).item()
        n += 1
    return tot / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="lucid_pretrain")
    ap.add_argument("--smoke", action="store_true", help="tiny local wiring run (fits 8GB)")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--segment-len", type=int, default=128, help="local attn window")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--neural-mem-layers", type=int, nargs="*", default=[2, 4])
    ap.add_argument("--nmem-seg", type=int, default=16, help="neural-memory segment len")
    ap.add_argument("--tinystories-n", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--tokenizer-dir", default=os.path.join(HERE, "tokenizer"))
    args = ap.parse_args()

    if args.smoke:
        (args.seq_len, args.segment_len, args.dim, args.depth, args.heads,
         args.neural_mem_layers, args.tinystories_n, args.steps, args.warmup) = (
            128, 64, 256, 4, 4, [3], 3000, 400, 40)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = ByteLevelBPETokenizer(os.path.join(args.tokenizer_dir, "vocab.json"),
                                os.path.join(args.tokenizer_dir, "merges.txt"))
    ids = load_tinystories_ids(tok, args.tinystories_n)
    n_val = max(4, int(((len(ids) - 1) // args.seq_len) * 0.02))
    split = n_val * args.seq_len + 1
    train_ds = LMWindows(ids[:-split], args.seq_len)
    val_ds = LMWindows(ids[-split:], args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    print(f"[lucid-pretrain] train windows={len(train_ds)} val windows={len(val_ds)} "
          f"seq={args.seq_len}", flush=True)

    dim_head = args.dim // args.heads
    model = MemoryAsContextTransformer(
        num_tokens=VOCAB, dim=args.dim, depth=args.depth,
        segment_len=args.segment_len, heads=args.heads, dim_head=dim_head,
        num_persist_mem_tokens=4, num_longterm_mem_tokens=4,
        neural_memory_layers=tuple(args.neural_mem_layers),
        neural_memory_segment_len=args.nmem_seg,
        sliding_window_attn=True, use_flex_attn=False,
        neural_memory_model=MemoryMLP(dim=64, depth=2),
        neural_memory_kwargs=dict(dim_head=64, heads=4, use_accelerated_scan=False),
    ).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[lucid-pretrain] MAC params={n_params/1e6:.1f}M dim={args.dim} depth={args.depth} "
          f"heads={args.heads} seg={args.segment_len} mem_layers={tuple(args.neural_mem_layers)} "
          f"batch={args.batch_size} steps={args.steps}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, args.warmup))
        * max(0.1, 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))))

    os.makedirs(args.out, exist_ok=True)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model.train()
    it = cycle(train_loader)
    for step in range(args.steps):
        x = next(it).to(dev)
        loss = model(x, return_loss=True)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            print(f"[lucid-pretrain] step {step} loss {loss.item():.3f} "
                  f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
        if step > 0 and step % 500 == 0:
            print(f"[lucid-pretrain]   val loss {evaluate(model, val_loader, dev):.3f}", flush=True)
            model.train()

    val = evaluate(model, val_loader, dev)
    ckpt = os.path.join(args.out, "final.pt")
    torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt)  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
    tok.save_model(args.out)

    # sample to prove the generation path works end-to-end
    model.eval()
    prompt = torch.tensor(tok.encode("Once upon a time").ids, dtype=torch.long, device=dev)[None]
    with torch.no_grad():
        gen = model.sample(prompt, 60, show_progress=False)
    text = tok.decode(gen[0].tolist())
    peak = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0
    print(f"\n[lucid-pretrain] SAMPLE: {text!r}", flush=True)
    print(f"[lucid-pretrain] LUCID_PRETRAIN_DONE val={val:.3f} time={time.time()-t0:.0f}s "
          f"peakVRAM={peak:.2f}GiB ckpt={ckpt}", flush=True)


if __name__ == "__main__":
    main()
