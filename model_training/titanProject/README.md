# Titans Small-Model Prototype

Goal: stand up a tiny Titans-based language model (MAC variant) to verify test-time memory on a modest Mac (16GB unified). All project files live in this directory; the `titans-pytorch` repo remains at workspace root for the library code.

## Layout
- `todo.md` — task tracker.
- `config_small.yaml` — baseline hyperparameters for the small model.
- `model.py` — model factory: tokenizer hook + Titans MAC + LM head.
- `data.py` — dataset/dataloader scaffolding (tokenize → chunk → shift targets).
- `train.py` — training loop skeleton (AdamW, warmup+cosine, checkpoints, eval).

## Quick start
- Train (example):
  ```bash
  source .venv/bin/activate
  python train.py --device mps --config configs/config_combo_all.yaml --max-steps 4000 --log-every 100 --no-amp
  ```
- Generate (example):
  ```bash
  source .venv/bin/activate
  python generate.py --device mps --config configs/config_combo_all.yaml --ckpt ckpt_step_4000.pt --prompt "Once upon a time" --max-new 50 --top-k 20 --temperature 0.8
  ```

## Notes
- Keep model tiny (≈50–120M params): dim 256–384, depth 4–6, heads 4–6, FFN 4×, mem tokens small.
- Seq len 256–512 to start; batch size small (8–16 sequences) to fit 16GB with optimizer states.
- Tokenizer: reuse an existing BPE (16–32k). Wire vocab size into `config_small.yaml`.

## Build-from-scratch checklist (Raschka) — status
- [x] BPE tokenizer with stable vocab; consistent train/eval paths
- [x] Sliding-window sampling, fixed train/val split, token caps for smoke
- [x] GPT decoder stack with causal mask, MH attention, GELU MLPs, residuals, layer norm
- [x] Positional encodings for target seq_len; LM head tied? (Titans ties embeddings internally)
- [x] AdamW, warmup + cosine with LR floor; grad clip; AMP off on MPS for stability
- [x] Regular checkpoints with config and tokenizer reference
- [x] Train/val loss + perplexity logged each eval
- [x] Basic generation sanity checks (greedy); decode span toggles for QA eval
- [x] Top-k/temperature decoding option exposed in generate script
- [x] Supervised finetune aligned with tokenizer/context; QA prompts “Answer:”
- [ ] Optional PEFT/LoRA (not needed for current tiny model)
- [ ] HF/transformers export script (not yet needed)


