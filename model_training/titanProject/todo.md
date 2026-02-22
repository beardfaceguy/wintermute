# Titans Small-Model Prototype TODO

- [in_progress] Set up small Titans MAC config (dim 256–384, depth 4–6, heads 4–6, mem tokens 2–4 / 8–16, seq_len 256–512) and scaffold in `titans-pytorch`. **Tokenizer: bpe_50k_bf (vocab 50k).**
- [pending] Prepare tokenizer (using bpe_50k_bf) and PyTorch dataloader with proper shuffling for a small corpus.
- [pending] Implement train loop (AdamW, warmup+cosine, grad clip, checkpoints, eval perplexities).
- [pending] Run a smoke train (few k steps) to verify stability; tune batch/LR as needed.
- [pending] Design a simple test to validate Titans test-time memory vs. a baseline on a short-context task.

