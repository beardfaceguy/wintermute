# GPT-Medium (407M) Training Run — 2026-04-19

## Completed Run (on-demand, attempt 2) — DONE 2026-04-28

| Field | Value |
|-------|-------|
| **Run ID** | `gpt_medium_pretrain_20260419232521` |
| **Instance** | `i-03cd3114ec2d3299b` (g6.2xlarge **on-demand**, us-east-1d) |
| **Instance Name** | `titan-medium-pretrain` |
| **Model** | GPT (x-transformers Decoder), 407M params |
| **Config** | `config_gpt_medium.yaml` (d=1024, L=24, heads=16, ff_mult=4) |
| **Steps completed** | 125,000 / 125,000 |
| **Target tokens** | ~8.2B (Chinchilla 20x for 407M) |
| **Tokens processed** | 8,192,000,000 (8.19B) |
| **Effective batch** | 65,536 tokens/step (batch=2, seq=1024, grad_accum=32) |
| **Throughput** | ~10,627 tok/s (measured at completion) |
| **Total training time** | ~214 hours (~8.9 days) |
| **Actual cost** | ~$210 (g6.2xlarge on-demand at ~$0.98/hr) |
| **Auto-stop** | FAILED — instance tag mismatch (see Post-Mortem below) |
| **Final train loss** | 2.87 |
| **Final val loss** | 2.9641 (at step 124,000 eval) |
| **Final val perplexity** | **19.38** |
| **Last checkpoint** | `ckpt_step_124000.pt` (63 checkpoints total in S3) |
| **SSM Command ID** | `2a68015a-c47e-40dd-836f-e95be8674122` |
| **Linear Issue** | CLA-143 |
| **save_every** | 2000 steps (changed from 5000 after spot failure) |
| **eval_every** | 2000 steps |
| **SFT gate** | **PASSED** — val ppl 19.38 is well below the <50 threshold |

## Failed Run (spot, attempt 1)

| Field | Value |
|-------|-------|
| **Run ID** | `gpt_medium_pretrain_20260419115339` |
| **Instance** | `i-0d08f0992a79333d7` (g6.2xlarge **spot**, us-east-1d) — **terminated by AWS** |
| **Outcome** | Spot instance reclaimed before first checkpoint at step 5000. No checkpoints saved. |
| **Lesson** | One-time spot requests are terminated (not stopped) on reclamation. For multi-day runs, use on-demand or reduce save_every so checkpoints sync before reclamation. |

## Hyperparameters

```yaml
lr: 0.0003
lr_min: 0.00003
weight_decay: 0.1
warmup_steps: 3000
cosine_decay: true
grad_clip: 1.0
betas: [0.9, 0.98]
eps: 1.0e-8
```

## Data

| Dataset | Path | Token Cap |
|---------|------|-----------|
| Train | `s3://alix-ai-ml-staging-data/titan/data/processed/train.txt` (31 GB) | 2B |
| Val | `s3://alix-ai-ml-staging-data/titan/data/processed/val.txt` (1.6 GB) | 5M |
| Tokenizer | `s3://alix-ai-ml-staging-data/titan/tokenizers/new_bpe_50k/bpe_50k_fw_stack.model` | — |

## Training Progress (final)

| Step | Train Loss | Val Loss | Val PPL | LR | Tokens Seen | Progress |
|------|-----------|----------|---------|-----|-------------|----------|
| 100 | 8.9113 | — | — | 0.000010 | 6.5M | 0.08% |
| 2,000 | 5.0657 | 5.0356 | 153.79 | 0.000200 | 131M | 1.60% |
| 2,500 | 4.7012 | — | — | 0.000250 | 164M | 2.00% |
| 20,000 | — | 3.4644 | 31.96 | — | 1.3B | 16.0% |
| 40,000 | — | 3.2223 | 25.09 | — | 2.6B | 32.0% |
| 60,000 | — | 3.1154 | 22.54 | — | 3.9B | 48.0% |
| 80,000 | — | 3.0450 | 21.00 | — | 5.2B | 64.0% |
| 100,000 | — | 2.9892 | 19.87 | — | 6.6B | 80.0% |
| 120,000 | 2.9360 | 2.9644 | 19.38 | 0.000031 | 7.9B | 95.9% |
| 122,000 | — | 2.9620 | 19.34 | 0.000030 | 8.0B | 97.5% |
| 124,000 | 2.9180 | 2.9641 | 19.38 | 0.000030 | 8.1B | 99.1% |
| 125,000 | 2.8708 | — | — | 0.000030 | 8.19B | 99.9% |

GPU: 100% utilization, 10.7 GB / 23 GB VRAM.

## Post-Mortem: Pipeline Issues (fixed 2026-04-28)

Two bugs were identified and fixed after the run completed:

### 1. No final checkpoint saved

`train.py` only saved checkpoints when `global_step % save_every == 0`. With `save_every=2000` and `max_steps=125,000`, the last save was at step 124,000. The final 1,000 steps were never persisted — the training completed but the weights were lost.

**Fix**: `train.py` now runs a final eval and saves a final checkpoint whenever training ends off a `save_every` boundary. (`finetune_sft.py` already had this logic.) Note: `train_multi_gpu.py` was later merged into `train.py` (2026-04-29).

### 2. Instance self-stop failed silently

The IAM policy for self-stop requires tag `Purpose=titan-training`, but the instance had `Purpose=titan-medium-pretrain`. The `ec2:StopInstances` call was silently denied by IAM, leaving the instance running idle.

**Fix**: The runner scripts now auto-tag the instance with `Purpose=titan-training` before calling `stop-instances`, and log explicit FAILED messages with remediation steps if the stop fails.

## S3 Artifacts

| Path | Content |
|------|---------|
| `s3://alix-ai-ml-staging-data/titan/checkpoints/gpt_medium_pretrain_20260419232521/` | Checkpoints (every 2k steps) |
| `s3://alix-ai-ml-staging-data/titan/checkpoints/gpt_medium_pretrain_20260419232521/run_artifacts/` | train.log, run_status.json (on completion) |

## Instance Logs

| Path on Instance | Content |
|------------------|---------|
| `/mnt/data/ssm_runs/gpt_medium_pretrain_20260419232521/train.log` | Training output |
| `/mnt/data/ssm_runs/gpt_medium_pretrain_20260419232521/run_status.json` | Run status (written on completion) |
| `/mnt/data/ssm_runs/gpt_medium_pretrain_20260419232521/launcher.log` | Detached launcher output |

## Sanity Gate Results (PASSED — 2026-04-19)

Run ID: `gpt_medium_pretrain_20260419095516` (on terminated spot instance)

| Step | Loss | LR |
|------|------|----|
| 100 | 8.5969 | 0.000010 |
| 200 | 5.8603 | 0.000020 |
| 300 | 3.4386 | 0.000030 |
| 400 | 1.5191 | 0.000040 |
| 500 | 0.2134 | 0.000050 |
| 600 | 0.0131 | 0.000060 |
| 700 | 0.0032 | 0.000070 |
| 800 | 0.0011 | 0.000080 |
| 900 | 0.0023 | 0.000090 |
| 1000 | 0.0006 | 0.000100 |

Exit code: 0. Loss collapsed from 8.60 to 0.0006 on a 50k-token shard.

## Monitoring Commands

```bash
# Tail CloudWatch logs live
AWS_PROFILE=experimental-admin aws logs tail /aws/ssm/titan-llm-training --follow --region us-east-1

# Quick loss check (via SSM — only works while instance is running)
AWS_PROFILE=experimental-admin aws ssm send-command \
  --region us-east-1 --document-name "AWS-RunShellScript" \
  --instance-ids i-03cd3114ec2d3299b \
  --parameters '{"commands":["grep \"\\[train\\]\" /mnt/data/ssm_runs/gpt_medium_pretrain_20260419232521/train.log | tail -10"],"executionTimeout":["60"]}' \
  --query "Command.CommandId" --output text --no-cli-pager

# Check instance state
AWS_PROFILE=experimental-admin aws ec2 describe-instances --instance-ids i-03cd3114ec2d3299b \
  --query 'Reservations[0].Instances[0].State.Name' --output text --region us-east-1

# Check S3 checkpoints
AWS_PROFILE=experimental-admin aws s3 ls s3://alix-ai-ml-staging-data/titan/checkpoints/gpt_medium_pretrain_20260419232521/ --region us-east-1
```

## Multi-GPU DDP Validation (PASSED — 2026-04-22)

### Summary

Multi-GPU distributed training validated end-to-end on a 4x A10G instance. This unlocks future training runs at 3-4x throughput. (Originally validated with `train_multi_gpu.py`, which was later merged into unified `train.py` on 2026-04-29.)

### Test Infrastructure

| Field | Value |
|-------|-------|
| **Instance type** | g5.12xlarge (4x NVIDIA A10G, 24 GB each) |
| **Instance ID** | `i-0fbf856cf80d48969` (us-east-1d, on-demand) |
| **Cost** | ~$5.67/hr, ~15 min total runtime, ~$1.50 |
| **SSM Command** | `c5e31553-9376-45a1-8226-57e70c339ac8` |
| **CloudWatch** | `/aws/ssm/titan-llm-training` |
| **Instance status** | Stopped (auto-shutdown via EXIT trap) |

### Local Test Suite (pytest)

- **111 passed**, 2 expected failures (`hf_gpt2` variant tests — `transformers` not installed)
- Tests cover: data pipeline, model construction, training utilities, checkpoint round-trip, and **6 Gloo-based multi-process DDP tests** (gradient sync, all-reduce, rank-guarded I/O, barriers, full DDP training loop, checkpoint portability)
- Runtime: 27 seconds on 4x A10G

### NCCL 4-GPU Sanity Check

| Metric | Value |
|--------|-------|
| **World size** | 4 (NCCL backend) |
| **Model** | 406.7M params (same GPT-Medium config) |
| **Grad accumulation** | 8 (auto-scaled from 32 for 4 GPUs) |
| **Effective batch** | 65,536 tokens/step |
| **Steps** | 20 |
| **Throughput** | ~32,000 tok/s |
| **Loss** | 10.9799 → 10.9363 (decreasing) |
| **Weight sync** | max_diff=0.00e+00 across all ranks (perfect) |
| **Duration** | 41.1 seconds |

### Key Files

| File | Purpose |
|------|---------|
| `model_training/titanProject/train.py` | Unified training script (single-GPU, multi-GPU DDP via torchrun, CPU/MPS) |
| `model_training/titanProject/tests/test_multi_gpu.py` | 30+ DDP unit tests including Gloo-based multi-process tests |
| `model_training/titanProject/tests/test_data.py` | 28 data pipeline tests |
| `model_training/titanProject/tests/test_model.py` | 11 model construction tests |
| `model_training/titanProject/tests/test_train_utils.py` | 16 training utility tests |
| `model_training/titanProject/tests/conftest.py` | Shared fixtures (tiny configs, dummy tokenizer, token cache) |

### DDP Features Validated

- `setup_distributed()` with torchrun env detection and single-GPU fallback
- `DistributedDataParallel` wrapping with `device_ids`
- `DistributedSampler` with epoch-based shuffling
- `model.no_sync()` for gradient accumulation optimization
- `reduce_scalar()` for cross-rank metric aggregation
- Automatic `grad_accum_steps` scaling (÷ world_size)
- Rank-0-only logging, checkpointing, and S3 sync
- Barrier coordination for token cache build
- Checkpoint portability: DDP `model.module.state_dict()` loads into bare model

## Next Steps

1. ~~Verify val perplexity < 50 (gate for SFT)~~ — **PASSED** (ppl 19.38)
2. Run SFT using step 124,000 checkpoint: `INSTANCE_ID=i-03cd3114ec2d3299b BASE_CKPT_S3_URI=s3://alix-ai-ml-staging-data/titan/checkpoints/gpt_medium_pretrain_20260419232521/ckpt_step_124000.pt bash scripts/aws_commands/gpt_medium_sft_cloudwatch.sh`
3. Export to HF format
4. Serve via vLLM, test in talkingHead
5. Evaluate conversational quality — does 407M meet the bar?

## Next Training Run: Use Multi-GPU DDP

**Decision (2026-04-26)**: The next training run (whether a re-run, SFT, or 1B+ scale-up) must use multi-GPU DDP to reduce wall-clock time.

**Rationale**: The current single-GPU run (g6.2xlarge) achieves ~10,600 tok/s and takes ~9 days for 125k steps. The validated DDP path on g5.12xlarge (4x A10G) achieved ~32,000 tok/s — a 3x speedup that would cut a comparable run to ~3 days.

**Plan**:
- Launch on g5.12xlarge (4x A10G, $5.67/hr on-demand) or larger if scaling to 1B+
- Use `torchrun --nproc_per_node=4 train.py --config <config.yaml>`
- Grad accumulation auto-scales (32 → 8 per GPU), effective batch stays at 65,536 tokens/step
- Same checkpointing, S3 sync, and auto-stop behavior as current run
- Expected throughput: ~32,000 tok/s (validated)
- Expected cost for 125k steps at 407M: ~$170 (3 days × $5.67/hr) vs ~$213 for current single-GPU run — faster AND cheaper

**Cost comparison for future runs**:

| Config | Instance | GPUs | tok/s | 125k steps | Cost |
|--------|----------|------|-------|------------|------|
| Current run | g6.2xlarge | 1x L4 | 10,600 | ~9 days | ~$213 |
| **Next run** | **g5.12xlarge** | **4x A10G** | **32,000** | **~3 days** | **~$170** |
| 1B+ (future) | p4d.24xlarge | 8x A100 | est. ~100k+ | est. ~1 day | ~$787 |
