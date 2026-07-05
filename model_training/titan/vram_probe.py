"""
Find the largest MAC-Titan that trains within the local GPU's VRAM.

Runs a few real train steps per config (fwd + double-backward through the
test-time memory + optimizer step) at PoC-realistic settings and records peak
VRAM until it OOMs. Run with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
"""
import torch
from titans_trainer import TitansConfig, TitansModel

DEV = "cuda"
B, L, STEPS = 8, 512, 4
SWEEP = [
    (256, 4, 4), (384, 6, 6), (512, 6, 8), (512, 8, 8),
    (640, 8, 10), (768, 10, 12), (768, 12, 12), (1024, 12, 16),
]

print(f"probe: batch={B} seq={L} amp=True | {torch.cuda.get_device_name(0)}")
print(f"{'d_model':>7} {'layers':>6} {'params(M)':>10} {'peakVRAM(GiB)':>13}  result")
for d_model, n_layers, n_heads in SWEEP:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        cfg = TitansConfig(vocab_size=8000, d_model=d_model, n_layers=n_layers,
                           n_heads=n_heads, max_seq_len=L, causal=True, chunk_size=128)
        model = TitansModel.from_config(cfg).to(DEV)
        params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        model.train()
        for _ in range(STEPS):
            x = torch.randint(0, 8000, (B, L), device=DEV)
            with torch.amp.autocast("cuda"):
                loss = model(x, labels=x)["loss"]
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"{d_model:>7} {n_layers:>6} {params/1e6:>10.1f} {peak:>13.2f}  OK")
        del model, opt, scaler, loss
    except RuntimeError as e:
        tag = "OOM" if "out of memory" in str(e).lower() else f"ERR: {str(e)[:40]}"
        print(f"{d_model:>7} {n_layers:>6} {'-':>10} {'-':>13}  {tag}")
        torch.cuda.empty_cache()
print("PROBE_DONE")
