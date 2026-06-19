# Titans Small-Model Prototype TODO

## Completed local model work
- [done] Set up small Titans MAC config scaffold with `bpe_50k_bf` tokenizer.
- [done] Prepare tokenizer and dataloader pipeline for TinyStories sampled data.
- [done] Implement train loop (AdamW, warmup+cosine, grad clip, checkpoints, eval perplexity).
- [done] Run smoke and extended training runs (see `run_log.md`).
- [done] Implement and run memory-vs-baseline evaluation flow.

## Active AWS hosting continuation
- [done] Audit live AWS state (`aws_titan_next_steps.py audit`) with working profile/credentials.
- [done] Ensure SG exists (`alix-pc-llm-model-training` -> `sg-05ca8b4be3b26ef52`).
- [done] Launch/start training runner (`i-050bce8db858dfa89` now running in `us-east-1f`).
- [done] Bootstrap instance (`/mnt/data` mounted, SSM online, code restored to `/mnt/data/code/wintermute`, Python deps installed on instance).
- [done] Regenerate tokenizer and TinyStories sampled data on EC2, and sync to S3:
  - `s3://alix-ai-ml-staging-data/titan/code/wintermute/model_training/LLM/tokenizers/bpe_50k_bf.model`
  - `s3://alix-ai-ml-staging-data/titan/data/tinystories_sampled/{train_sample.txt,val_sample.txt}`
- [done] Run baseline training on EC2 and sync checkpoint to `s3://alix-ai-ml-staging-data/titan/checkpoints/` (`ckpt_step_4000.pt`).
- [done] Validate spot-first additional runner flow (`i-0b788d7634d3a40c4`, spot request `sir-zxezf49j`, SSM online).
- [done] Implement and smoke-test periodic checkpoint auto-sync hooks in `train.py` (`--save-every`, `--checkpoint-dir`, `--s3-checkpoint-uri`).
- [done] Add and validate inference smoke harness (`inference_smoke.py`) for checkpoint usability checks.
- [done] Add and validate interactive qualitative interface (`chat_repl.py`) for manual user-style testing.
- [done] Add and validate HTTP endpoint interface (`chat_http.py`) for programmatic chat testing (`/health`, `/chat`, `/reset`).
- [done] Publish HTTP endpoint on EC2 for browser testing with restricted SG ingress on port `8000`.
- [done] Run first OASST1 + Dolly SFT pilot (600 steps) and switch live HTTP endpoint to the resulting checkpoint (`ckpt_sft_step_600.pt`).
- [done] Configure nginx path routing so the service can be exposed at `/titan` on domain-hosted HTTP.

## Current blockers
- [none] No critical blockers for baseline AWS run completion.
