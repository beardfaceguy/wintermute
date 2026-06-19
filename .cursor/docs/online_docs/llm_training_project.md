## LLM Training Side-Quest – Project Plan

### Objectives
- Understand the end-to-end workflow for training a compact, general-purpose LLM.
- Stand up an experimental pipeline that can be iterated locally or on rented GPUs.
- Capture lessons learned to feed back into Wintermute’s governance memory.

### Scope & Assumptions
- Target model size: 110M–350M parameters for on-device experiments (fits 16 GB unified memory); scale to 1B+ only when cloud GPUs are available.
- Focus on supervised pretraining + light instruction tuning; RLHF out of scope for first pass.
- Primary tooling: Apple’s MLX framework for local experimentation; optionally mirror configs in PyTorch/Hugging Face when scaling to cloud.
- Compute assumed: Apple M3 Pro (16 GB unified) for prototyping; schedule cloud (A100/H100) sessions for large-batch or long trainings.

### High-Level Phases
1. **Project Setup**
   - Confirm hardware access (local + cloud) and storage needs.
   - Define success metrics (perplexity targets, evaluation tasks).
2. **Data Strategy**
   - Identify base corpora (OpenWebText2, SlimPajama, code, Wintermute docs).
   - Establish filtering, dedup, and chunking process.
   - Train or adopt tokenizer; document vocabulary decisions.
3. **Model Configuration**
   - Select architecture template compatible with MLX (e.g., GPT-NeoX-style blocks) and adapt configs.
   - Decide precision (float16/bfloat16) and optimizer (AdamW with weight decay) supported by MLX.
4. **Training Pipeline**
   - Build MLX training scripts (Dataloader, optimizer step, checkpointing).
   - Integrate logging (MLX metrics, optional Weights & Biases via Python hooks).
   - Establish checkpoint cadence and validation loop.
5. **Evaluation & Alignment**
   - Set up perplexity measurement on held-out set.
   - Integrate basic instruction tuning dataset (e.g., Alpaca-style).
   - Draft plan for human eval or automated benchmarks (MT-Bench-lite).
6. **Deployment Experiment**
   - Export MLX checkpoints to Hugging Face/PyTorch format if needed.
   - Run smoke tests via simple CLI or TalkingHead integration (Metal inference or converted model).
7. **Governance & Documentation**
   - Log decisions in `Wintermute_dev_log.md` as DECs or CPs if needed.
   - Record infra steps, costs, and issues in this project doc.

### Risk & Mitigation Notes
- **Compute Limits:** Constrain local runs to ≤350M params, use sequence length ≤1K, gradient accumulation, and MLX optimizations; schedule cloud bursts for larger models or longer horizons.
- **Data Quality:** Run toxicity checks and enforce dedup using MinHash.
- **Training Instability:** Use gradient clipping, LR warmup, and monitor loss.
- **Cost Overruns:** Track GPU hours; prototype configs with tiny model before scaling.

### Working TODO (single list)
- [ ] Confirm available GPU resources (local M3 Pro + optional cloud) and budget.
- [x] Install and validate MLX environment (Python version, Metal support).
- [x] Choose base dataset mix and document licensing constraints (SlimPajama 100 MB + pentest/cybersec + Canstralian).
- [x] Decide tokenizer approach and build tokenization script (current best: `bpe_50k_bf` at `model_training/LLM/tokenizers/bpe_50k_bf.model`; prior baseline `bpe_32k` retained for comparison).
- [ ] For any future tabular prep, prefer Polars for performance; keep pandas only as a dependency for datasets.
- [ ] Draft model config (hidden size, layers, heads) and memory estimates.
- [x] Scaffold MLX training script (data pipeline, optimizer loop, checkpointing; model-only checkpoints via `model.save_weights`; optimizer state not saved yet).
- [ ] Define validation/eval datasets and baseline metrics (start with 95/5 line-split on combined_corpus; may move to a held-out val file later for better control).
- [ ] Plan lightweight instruction tuning phase (dataset + procedure).
- [ ] Outline deployment experiment (vLLM/TGI) and testing checklist.
- [ ] Capture governance updates (DEC/CP) once scope solidifies.
- [ ] Review lessons learned; feed cross-cutting insights into `memory.md`.

### Notes on checkpoints
- Model checkpointing works via `model.save_weights` / `load_weights`. Optimizer state is not saved yet. Future task: implement flat save/load of optimizer state if full resume is needed.
- Checkpoints: `checkpoints/step_50.npz`, `step_100.npz`, `step_150.npz`, `step_200.npz` from a 200-step run (seq_len=128, batch=4). Best-checkpoint support added: `--best-ckpt` saves the lowest-val model; used in recent runs (see session log).

### Next Conversation Steps
1. Align on compute availability and acceptable model size.
2. Prioritize dataset acquisition tasks.
3. Walk through each TODO item with concrete actions and owners.

### Future Integration (HRM sidecar idea)
- Idea: keep first LLM simple (GPT-style on `bpe_32k`), and later add Hierarchical Reasoning Model (HRM) as an external reasoning service for structured tasks (grids/puzzles/pathfinding).
- Deployment split: run HRM on the Windows gaming PC with 8 GB NVIDIA GPU (PyTorch + CUDA/FlashAttention); keep the LLM on the Mac (MLX/Metal).
- Interaction model: expose HRM via a small API/tool; LLM delegates structured problems (JSON schema) to HRM, consumes returned solutions.
- Scope notes: HRM is PyTorch/CUDA; adapting to Metal isn’t worth it for first pass. Best treated as a sidecar, not fused into the LLM architecture.
- Tasks HRM suits: small structured datasets (ARC, Sudoku, mazes) where HRM is sample-efficient and strong.
- Next steps (later): set up HRM env on GPU PC, run provided training/eval scripts, wrap with an API for LLM calls; compare against fine-tuning GPT-style model on same tasks.
- Reference docs:
  - MLX official docs: https://ml-explore.github.io/mlx/build/html/index.html
  - Hugging Face MLX hub guide: https://huggingface.co/docs/hub/en/mlx

### Session Log (2025-12-09)
- Environment: working in `model_training` venv on M3 Pro; installed `datasets`, `huggingface_hub` (pinned to 0.36.0 for transformers compat), added `zstandard` for zstd shards, and `beautifulsoup4` for HTML cleanup.
- Data pulls:
  - SlimPajama: streamed 100 MB slice to `data/slimpajama_sample_100mb.jsonl` (streaming requires zstd).
  - Pentesting corpora: combined accessible HF sets (`preemware/pentesting-eval`, `0dAI/PentestingCommandLogic`, `boapro/PentestingCommandLogic`, `resk-fr/pentesting-for-agents`) into `data/pentesting_corpus_10k_each.jsonl` (20,295 rows; gated `Canstralian` HF dataset skipped).
  - Cybersec README corpus from awesome-genai-cyberhub into `data/cybersec_corpus.txt` (light HTML residue acceptable for tokenizer input).
  - Canstralian local drop (vulnerability CSVs + README) flattened to sentences in `data/canstalian_corpus.txt` and appended.
- Combined corpus: `data/combined_corpus.txt` now contains SlimPajama slice + pentest corpora + cybersec README + Canstralian vulnerabilities (approx 112 MB total). Ratio currently ~88/12 (general/domain); user accepted as-is.
- Noted: `slimpajama_sample_500mb.jsonl` is empty placeholder; unused.

### Next Steps (when back)
- Train tokenizers on `data/combined_corpus.txt` (SentencePiece recommended):
  - BPE and Unigram at vocab sizes 8k/16k/32k; try with/without `byte_fallback=true`.
  - Report mean/median tokens per line and spot-check domain terms (CVE/SQLi/RCE/commands).
- If desired later, regenerate a larger SlimPajama slice and rebuild the combined corpus, but current ratio is accepted.
- Keep a reproducible log of corpus sizes/paths before training; freeze `combined_corpus.txt` for this run.

### Session Log (latest)
- Implemented dataloader shuffle buffer (`--shuffle-buffer`); lines buffered, shuffled, then packed/batched. Val loader remains unshuffled and is rebuilt every eval to avoid exhaustion.
- Added best-checkpoint saving (`--best-ckpt`), triggered on lowest val loss; eval returns `None` if val data is empty to avoid bogus zeros.
- Sanity run (30 steps) with shuffle buffer 200k, lr 5e-5, warmup 20, cosine floor 0.1, seq_len 64, batch 4, dropout 0.1:
  - Val losses: 9.80 → 9.07 → 8.74; best ckpt saved at `checkpoints/best_sanity.npz`.
- Long run (1000 steps) with lr 5e-5, warmup 100, cosine floor 0.1, seq_len 64, batch 4, dropout 0.1, shuffle buffer 200k, eval every 20:
  - Val best ≈ 7.17 at step 1000 (median 7.30, mean 7.47); final train loss ≈ 7.01.
  - Best checkpoint: `checkpoints/best_longrun_1000.npz`; metrics: `logs/metrics_longrun_1000_cosine_floor0.1.csv`.
- Suggestion before longer runs: try a slight LR tweak (e.g., 4.5e-5 to 5e-5) or slightly longer warmup (120–150) to see if val dips below ~7.0; otherwise current setup appears stable for extended training.

### Session Log (new best config)
- Dataset: `slimpajama3b_train_split.txt` / `slimpajama3b_val_split.txt`; shuffle-buffer 200,000; eval-every 20, eval-batches 5.
- Tokenizer: `tokenizers/bpe_50k_bf.model` (byte_fallback, nfkc).
- Model/training: seq_len 64, batch 16, d_model 512, n_layers 6, n_heads 8, d_ff 2048, dropout 0.1, lr 4.5e-5, warmup 120, cosine schedule with lr_min_factor 0.1, steps 3500.
- Result: best val loss 1.7199 (ppl 5.58) at step 3480. Final val at step 3500: loss 1.7253 (ppl 5.61).
- Artifacts: metrics `logs/metrics_slim3b_bpe50kbf_lr4.5e-5_warm120_floor0.1_s3500_seq64_bs16.csv`; best checkpoint `checkpoints/best_slim3b_bpe50kbf_lr4.5e-5_warm120_floor0.1_s3500_seq64_bs16.npz`.

### Session Log (dead ends)
- Config: bpe_50k_bf, seq64, batch 16, d640/L6/H10/FF2560, dropout 0.1, lr 4.5e-5, warmup 120, cosine lr_min_factor 0.05, steps 2000. Outcome: val loss plateaued high at 2.288 (ppl 9.86) with late train spike; worse than the 1.72 baseline—do not pursue further without major LR/shape changes.

### Session Log (recent runs)
- Config: bpe_50k_bf, seq64, batch 16, d512/L6/H8/FF2048, dropout 0.1, lr 4.5e-5, warmup 120, cosine lr_min_factor 0.1, steps 4000. Outcome: best val loss 1.7530 (ppl 5.77) at step 3960; final val loss 1.7639 (ppl 5.84) at step 4000. Slightly worse than the 3500-step best (1.7199), so 3500-step checkpoint remains SOTA for this shape.
