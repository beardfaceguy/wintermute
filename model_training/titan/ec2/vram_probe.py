"""
VRAM / sequence-length probe for the lucidrains MAC neural memory (#952).

Runs ONE (seq, batch) cell: builds the proven small MAC (d256/L2/1-mem) with
sliding-window attention, does a few forward+backward training steps on random
tokens at the given seq length, and reports peak VRAM or OOM. run_sweep.sh calls
this once per grid cell (fresh process => clean CUDA state after any OOM).

Answers: how long a sequence fits on this GPU with the current (no-accelerated-scan)
neural memory, and therefore whether long-context text recall needs the CUDA
kernel / gradient checkpointing before it can run at a useful seq length.
"""
import argparse

import torch
from titans_pytorch import MemoryAsContextTransformer, MemoryMLP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--segment-len", type=int, default=128, help="sliding attn window")
    ap.add_argument("--nmem-seg", type=int, default=16, help="neural-memory segment len")
    ap.add_argument("--steps", type=int, default=3)
    a = ap.parse_args()
    tag = f"seq={a.seq} batch={a.batch} seg={a.segment_len} nmem_seg={a.nmem_seg}"

    if not torch.cuda.is_available():
        print(f"PROBE {tag} RESULT=NO_CUDA", flush=True)
        return
    dev = "cuda"
    torch.cuda.reset_peak_memory_stats()
    try:
        model = MemoryAsContextTransformer(
            num_tokens=256, dim=a.dim, depth=a.depth, segment_len=a.segment_len,
            heads=a.heads, dim_head=a.dim // a.heads,
            num_persist_mem_tokens=4, num_longterm_mem_tokens=4,
            neural_memory_layers=(a.depth,), neural_memory_segment_len=a.nmem_seg,
            sliding_window_attn=True, use_flex_attn=False,
            neural_memory_model=MemoryMLP(dim=64, depth=2),
            neural_memory_kwargs=dict(dim_head=64, heads=4, use_accelerated_scan=False),
        ).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for _ in range(a.steps):
            x = torch.randint(10, 256, (a.batch, a.seq), device=dev)
            loss = model(x, return_loss=True)
            opt.zero_grad()
            loss.backward()
            opt.step()
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"PROBE {tag} RESULT=OK peak={peak:.2f}GiB", flush=True)
    except Exception as e:  # noqa: BLE001 - classify OOM vs real error
        if "out of memory" in str(e).lower():
            print(f"PROBE {tag} RESULT=OOM", flush=True)
        else:
            print(f"PROBE {tag} RESULT=ERR {type(e).__name__}: {str(e)[:200]}", flush=True)


if __name__ == "__main__":
    main()
