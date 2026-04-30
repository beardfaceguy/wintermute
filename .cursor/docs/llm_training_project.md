## LLM Training Side-Quest – Project Plan

**Status: COMPLETED (2026-04-16)**

### Objectives
- Understand the end-to-end workflow for training a compact, general-purpose LLM.
- Stand up an experimental pipeline that can be iterated locally or on rented GPUs.
- Capture lessons learned to feed back into Wintermute's governance memory.

### Final Outcome

The side-quest completed the full loop: data preparation, tokenizer training, pretraining, SFT, and qualitative evaluation. The pipeline is proven end-to-end, but the model itself (GPT-small, ~117M params) is too small to produce practically useful outputs even after instruction tuning.

### What Was Built

#### Infrastructure
- **AWS pipeline**: EC2 on-demand launcher scripts, SSM-based remote execution with CloudWatch logging, S3 checkpoint sync, disk-backed token caching, self-stop on completion.
- **Instance profile**: `g5.xlarge` / `g6.2xlarge` on `experimental-admin` account (`491794274773`), IAM role with scoped S3 + SSM permissions.
- **Data volume**: 500 GB gp3 EBS mounted at `/mnt/data`, with auto-formatting and mounting in launch scripts.
- **Tooling**: `generate.py` (sampling), `finetune_sft.py` (SFT loop), `prepare_sft_mix.py` (multi-source data prep), `chat_http.py` (HTTP endpoint), `chat_repl.py` (interactive REPL), `export_to_hf.py` (HF export), vLLM serving validated.

#### Data
- **Pretraining corpus**: FineWeb-Edu (100BT sample) + The-Stack-Smol (31 languages), processed to ~32 GB train / ~1.6 GB val text.
- **Tokenizer**: SentencePiece BPE, 50k vocab, byte fallback, NFKC normalization (`bpe_50k_fw_stack.model`).
- **SFT data**: OASST1 + OpenHermes + SlimOrca + GSM8K mix, filtered with role-marker rejection.

#### Training Runs
- **Sanity overfit**: Tiny-shard (~200k tokens, 1000 steps) — loss collapsed to ~0.02, confirming pipeline correctness.
- **40k-step pretraining** (`gpt_small_pretrain_20260414162538`):
  - Config: d=768, L=12, heads=12, ff_mult=4, seq_len=1024, batch=2, grad_accum=32, lr=3e-4 cosine, warmup=3k.
  - Final metrics at step 40k: train loss **2.8660**, val loss **3.3644**, val perplexity **28.91**.
  - ~2.62B tokens seen. Checkpoints saved every 2k steps, synced to S3.
- **SFT pilot** (3000 steps from 40k pretrain checkpoint):
  - Config: lr=5e-5 cosine, warmup=200, seq_len=1024, batch=2, grad_accum=8.
  - Data: ~27k train examples, ~1.6k val examples (OASST1 + OpenHermes + SlimOrca + GSM8K).
  - Final eval loss: ~3.19.
  - Checkpoints at steps 1000, 2000, 3000; synced to `s3://alix-ai-ml-staging-data/titan/checkpoints/gpt_small_sft_20260419/`.

### Evaluation Results

#### Pretraining Quality
- Val perplexity reached **28.91** at 40k steps — well below the <50 SFT gate target.
- Three-phase learning curve observed: rapid descent (0–10k), steady improvement (10k–25k), plateau/refinement (25k–40k).
- The model produces coherent sentence fragments and topical continuations, but lacks factual accuracy and deep reasoning (expected for 117M params).

#### SFT Quality (vs. Pretrained Base)
- **Improved**: Format awareness (User/Assistant structure recognized), shorter/more focused replies, basic structured output attempts, reduced rambling.
- **Not improved**: Repetition loops, factual inaccuracy, shallow reasoning, math errors, hallucination. These are fundamental capacity limits of a 117M-param model.
- **Verdict**: The SFT workflow is functional and directionally correct. The base model is the bottleneck — at this scale, SFT cannot teach knowledge the model doesn't have.

### Key Lessons Learned

1. **Disk-backed tokenization is essential**: In-memory token accumulation destabilizes instances with large corpora. The `TokenCache` (np.memmap shards) + S3 pre-staging pattern eliminates this.
2. **SSM execution timeout is a hidden default**: `AWS-RunShellScript` has a 3600s default `executionTimeout` separate from `--timeout-seconds`. Must pass `executionTimeout` in parameters for any run >1h.
3. **HuggingFace temp files fill root disk**: `TMPDIR` and `XDG_CACHE_HOME` must be redirected to `/mnt/data` alongside `HF_HOME` to prevent download artifacts from filling the root filesystem.
4. **EBS auto-mount is unreliable**: Launch scripts should verify mount state and handle manual format/mount if the volume is attached but not mounted.
5. **Always use on-demand instances for training**: One-time spot requests only support terminate, not stop — AWS reclamation destroys all local state. Even with frequent checkpointing, the lost compute and manual relaunch overhead isn't worth the savings. The GPT-Medium spot failure (CLA-143) confirmed this. On-demand is ~2x the hourly rate but guarantees completion.
6. **117M params is too small for instruction following**: The model can learn format conventions from SFT but cannot learn new knowledge. Minimum viable assistant likely requires 1B+ params.
7. **Sanity overfit gate is invaluable**: The tiny-shard overfit check caught config regressions before expensive GPU runs multiple times.

### Artifacts (S3)

| Path | Description |
|------|-------------|
| `s3://alix-ai-ml-staging-data/titan/checkpoints/gpt_small_pretrain_20260414162538/` | 40k pretrain checkpoints (every 2k steps) |
| `s3://alix-ai-ml-staging-data/titan/checkpoints/gpt_small_sft_20260419/` | SFT checkpoints (steps 1000, 2000, 3000) |
| `s3://alix-ai-ml-staging-data/titan/data/processed/` | Processed train/val text (~34 GB) |
| `s3://alix-ai-ml-staging-data/titan/tokenizers/new_bpe_50k/` | BPE tokenizer (50k vocab) |
| `s3://alix-ai-ml-staging-data/titan/token_cache/` | Pre-built disk-backed token cache |

### GPT-Medium (407M) — PRETRAINING + SFT COMPLETE (2026-04-30)

**Status: PRETRAINING + SFT COMPLETE (CLA-143)**

Scale-up to 407M params (d=1024, L=24, heads=16, ff_mult=4) completed successfully.

**Pretraining:**
- **Completed run**: `gpt_medium_pretrain_20260419232521` on `i-03cd3114ec2d3299b` (g6.2xlarge on-demand)
- **Final metrics**: step 125,000/125,000, train loss 2.87, val loss 2.9641, **val ppl 19.38**
- **Total tokens**: 8.19 billion (~214 hours, ~$210)
- **Throughput**: ~10,627 tok/s
- **Last checkpoint**: step 124,000 (63 checkpoints in S3)
- **SFT gate**: **PASSED** — val ppl 19.38 is well below the <50 threshold
- **Sanity gate**: PASSED (loss 8.60 → 0.0006 on 50k-token shard)
- **Failed attempt**: Spot instance `i-0d08f0992a79333d7` was reclaimed by AWS; switched to on-demand + `save_every=2000`
- **Full run log**: `.cursor/docs/TRAINING_RUN_GPT_MEDIUM_20260419.md`

**SFT (General):**
- **Completed run**: `gpt_medium_sft_20260430052503` on `i-03cd3114ec2d3299b` (g6.2xlarge on-demand)
- **Steps**: 5,000 / 5,000 (from pretrain step 124,000 checkpoint)
- **Final metrics**: train loss 1.38, eval loss 1.92, **eval ppl 6.79**
- **Data**: OASST1 (24K) + OpenHermes (20K) + SlimOrca (10K) + GSM8K (1K) = ~55K train samples
- **Duration**: ~55 minutes
- **Checkpoints**: 5 in S3 (steps 1K-5K), each ~4.88 GB
- **Weights-only copy**: `saved_models/gpt_medium_407m_sft_5000_weights.pt` (1.63 GB)
- **Conversation tests**: Passes factual Q&A, coding (correct Python), explanations. Weak on creative writing, math (expected at 407M).
- **Instance self-stopped**: cleanly after checkpoint sync

**Pipeline Unification (2026-04-29):**
- Merged `train.py` and `train_multi_gpu.py` into a single unified `train.py` (single-GPU, multi-GPU DDP, CPU/MPS)
- Extracted shared utilities into `train_utils.py` (DDP helpers, LR schedules, checkpointing, S3 sync)
- Updated `finetune_sft.py` with DDP support using shared `train_utils`
- `train_multi_gpu.py` deleted

**SFT Format Support (2026-04-30):**
- `finetune_sft.py` now auto-detects 4 formats per line: HF messages JSONL, ShareGPT JSONL, Alpaca JSONL, chat text
- Compatible with most datasets on the Hugging Face Hub without conversion
- 29 new tests (test_sft_formats.py), full suite now 142 tests

**Next**: Domain-specific SFT forks from the general SFT checkpoint, 7B model scale-up
- **Future training runs**: Use `torchrun train.py` on multi-GPU instance (g5.12xlarge 4x A10G) — 3x faster, lower total cost

### Pipeline Fixes (2026-04-28)

Two bugs were fixed after the GPT-Medium run revealed them:

1. **Final checkpoint save**: `train.py` now saves a final checkpoint + eval when training ends between `save_every` boundaries. (`finetune_sft.py` already had this.)
2. **Instance self-stop**: Runner scripts now auto-tag instances with `Purpose=titan-training` before calling `stop-instances`, and log explicit failures with remediation steps. The GPT-Medium instance ran idle because the tag didn't match the IAM policy condition.

### Multi-GPU DDP: VALIDATED (2026-04-22)

Multi-GPU distributed training is production-ready. Validated on a g5.12xlarge (4x A10G) with real NCCL:

- **Script**: `model_training/titanProject/train.py` (unified single/multi-GPU)
- **Launch**: `torchrun --nproc_per_node=N train.py --config <config.yaml>` (falls back to single-GPU with plain `python3`)
- **Throughput**: ~32,000 tok/s on 4x A10G (vs ~8,700 tok/s single-GPU g6.2xlarge = 3.7x speedup)
- **Weight sync**: Perfect (max_diff=0.00e+00 across all 4 ranks)
- **Test suite**: 142 tests, including 6 Gloo-based multi-process DDP tests, 29 SFT format tests, and a 20-step NCCL sanity check
- **Features**: Auto-scaled grad accumulation, `no_sync()` optimization, rank-0-only I/O, barrier coordination, checkpoint portability (DDP → bare model)
- **Cost implication**: 4x A10G at $5.67/hr vs single L4 at $0.98/hr — but 3.7x faster means similar $/token. For 1B+ models that need multi-GPU anyway, this is the path forward.

Other potential future work:
- **Branch A (GPT-2 bootstrap)**: Validates workflow against a known-good pretrained base. Partially complete; see Linear project `Titan Branch A`.
- **Branch B (controlled scratch)**: Full-control lineage with operator-chosen data. Not yet started; see Linear project `Titan Branch B`.
- **NoProp experiments**: Parallel-layer training experiments, scoped but not executed. See CLA-134.

### Linear Tracking
- Program issue: `CLA-33` (Titan GPT-small pretraining stabilization) — **Done**
- Project: `Titan GPT-small Pretraining Stabilization`
- Related projects: `Titan Branch A`, `Titan Branch B`
