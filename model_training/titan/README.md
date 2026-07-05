# Titan Memory Model — local pretraining PoC (Track 2)

A **proof-of-work** pipeline that pretrains a small MAC-Titan (test-time
neural-memory) language model from scratch on a single 8 GB GPU. It exists to
de-risk the Titan track (Vikunja project #177) **before** committing GPU/SageMaker
budget to a paper-scale run: it proves the full creation pipeline runs end-to-end
on local hardware — data → BPE → tokenize → pretrain (with the surprise-driven
test-time memory update) → checkpoint → generate.

This is **not** a capable model. The goal is a working pipeline plus enough
coherence to eyeball ("does it produce real English?"). Paper-scale Titans
(≥170M params) still need a real GPU / SageMaker — see the VRAM ceiling below.

## Built on

[`pafos-ai/titans-trainer`](https://github.com/pafos-ai/titans-trainer) (MAC-only
Titan, HuggingFace-style trainer), installed from our fork's
`fix/save-pretrained-full-config` branch, which fixes the `save_pretrained` /
`from_pretrained` full-architecture round-trip (upstream PR:
`pafos-ai/titans-trainer#1`). `titanProject/` in this repo is a separate, earlier
hand-rolled reference and is unrelated to this pipeline.

## PoC result (2026-07-04)

- **Model:** 46.6M-param MAC-Titan — `d_model=512, n_layers=8, n_heads=8,
  causal, chunk_size=128, seq_len=512`, vocab 8k.
- **Corpus:** TinyStories (`roneneldan/TinyStories`), 300k-story subset ≈ 64M tokens.
- **Hardware:** 1× RTX 2080 SUPER (8 GB, Turing), ~15k tok/s, 6.26 GB peak VRAM.
- **Training:** 2 epochs / 15,318 steps / ~2.3 h. Loss: train 8.96 → **val 1.92**
  (epoch 1 val 2.15 → epoch 2 val 1.92), next-token accuracy 0.55.
- **Coherence** (final model, top-k sampling):
  > *"Once upon a time, there was a little boy named Timmy. Timmy loved to play
  > outside and explore. One day, he found a bottle on the ground. He picked it
  > up and showed it to his mom. 'Mommy, look at this bottle!' he said..."*

  Fluent grammar, dialogue, and story arcs, with the expected tiny-model artifacts
  (occasional repetition / mild logical drift).

## Vocab-size finding

BPE compression on TinyStories plateaus almost immediately — chars/token barely
moves from 8k → 48k vocab (4.09 → 4.12), while the embedding table grows 6×. At
48k vocab the embeddings alone would consume the entire ~50M budget. **8k is the
sweet spot** for this corpus; bigger vocab is wasted capacity.

## VRAM ceiling (RTX 2080 SUPER, 8 GB; batch 8 / seq 512 / AMP / causal)

| params | peak VRAM | fits? |
|-------:|----------:|:------|
| ~50M (d512/L8) | 4.9 GB | ✅ |
| ~87M (d640/L8) | 6.0 GB | ✅ (ceiling) |
| ~130M (d768/L10) | — | ❌ OOM |

Paper-scale (≥170M) needs a bigger card → the Stage-1b GPU/SageMaker gate.

## Pipeline (run in order)

```bash
# 0. one-time: create venv + install titans-trainer (fork) + torch(CUDA)
bash install.sh

# work dir for corpus/tokenizer/checkpoints (default ./titan_work; override with $TITAN_WORK)
export TITAN_WORK=/path/to/scratch

# 1. download TinyStories subset + train the 8k BPE (also sweeps vocab sizes)
python prepare_data.py

# 2. (optional) find the largest model that trains in your GPU's VRAM
python vram_probe.py

# 3. pretrain (add --smoke for a fast wiring test first)
python pretrain.py --smoke        # tiny/fast sanity run
python pretrain.py --epochs 2     # full PoC run

# 4. talk to it
python generate.py $TITAN_WORK/titan_run/final.pt
```

`tokenizer/` holds the trained 8k BPE (`vocab.json`, `merges.txt`) used for the
PoC — kept in-repo because the model is unusable without its exact tokenizer.

## Artifacts NOT in git (see .gitignore)

The corpus, token cache, venv, and model checkpoints are regenerable or too large
to commit. The trained `final.pt` (~186 MB) lives on `gaming-pc-linux`
(`~/titan_poc/titan_run/`); push to S3 if it needs to be durable.

## Environment notes

- Python 3.11 venv (`titans-trainer` wheels don't build on the 3.14 system interp).
- `titan-venv` resolved `torch 2.12.1+cu130` — CUDA works on the Turing card.
- This box time-shares the GPU with a vLLM benchmark server; stop vLLM before training.
