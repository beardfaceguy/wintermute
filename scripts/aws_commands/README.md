# Titan Training Launch Scripts

SSM-based launch scripts for GPT model training on AWS EC2 GPU instances.
Each script bootstraps a fresh instance, installs dependencies, downloads data,
and launches training in a detached process that survives the SSM command timeout.

## Production Scripts

| Script | Model | Mode | GPUs |
|--------|-------|------|------|
| `gpt_medium_pretrain_long_cloudwatch.sh` | GPT-Medium (407M) | Pretrain | 1 |
| `gpt_medium_pretrain_multigpu_cloudwatch.sh` | GPT-Medium (407M) | Pretrain (DDP) | N (auto-detected) |
| `gpt_medium_sft_cloudwatch.sh` | GPT-Medium (407M) | SFT | 1 |
| `gpt_small_pretrain_long_cloudwatch.sh` | GPT-Small | Pretrain | 1 |

## Prerequisites

### IAM

The instance role (`alix-llm-training-role`) needs two inline policies:

1. **CloudWatch Logs** — `iam/ssm_cloudwatch_logs_inline_policy.json`
2. **Self-Stop** — `iam/ssm_long_run_self_stop_inline_policy.json`
   - Requires the instance tag `Purpose=titan-training`
   - Scripts auto-apply this tag before self-stop, but pre-tagging is safer

### Instance Tag

```bash
aws ec2 create-tags --resources <instance-id> --tags Key=Purpose,Value=titan-training
```

### Code Bundle

Upload the code bundle to S3 before launching. Exclude large local artifacts:

```bash
cd ~/work/wintermute
tar -czf /tmp/titanProject_bundle.tar.gz \
  --exclude='*.pt' --exclude='*.pth' \
  --exclude='.titan_token_cache' \
  --exclude='__pycache__' \
  --exclude='logs' \
  model_training/titanProject/
aws s3 cp /tmp/titanProject_bundle.tar.gz \
  s3://alix-ai-ml-staging-data/titan/code_bundles/titanProject_bundle.tar.gz
```

### CloudWatch Log Group

```
/aws/ssm/titan-llm-training
```

## Quick Start

```bash
# Single-GPU pretrain
INSTANCE_ID=i-xxx bash scripts/aws_commands/gpt_medium_pretrain_long_cloudwatch.sh

# Multi-GPU pretrain (auto-detects GPU count)
INSTANCE_ID=i-xxx bash scripts/aws_commands/gpt_medium_pretrain_multigpu_cloudwatch.sh

# Resume from checkpoint
INSTANCE_ID=i-xxx \
  RESUME_CKPT_S3_URI=s3://alix-ai-ml-staging-data/titan/checkpoints/.../ckpt_step_124000.pt \
  MAX_STEPS=125000 \
  bash scripts/aws_commands/gpt_medium_pretrain_long_cloudwatch.sh

# SFT (requires pretrain checkpoint)
INSTANCE_ID=i-xxx \
  BASE_CKPT_S3_URI=s3://alix-ai-ml-staging-data/titan/checkpoints/.../ckpt_step_125000.pt \
  bash scripts/aws_commands/gpt_medium_sft_cloudwatch.sh

# Monitor via CloudWatch
AWS_PROFILE=experimental-admin aws logs tail /aws/ssm/titan-llm-training --follow --region us-east-1

# Check detached run status
RUN_ID=gpt_medium_pretrain_20260419... \
  INSTANCE_ID=i-xxx \
  bash scripts/aws_commands/check_detached_titan_status.sh
```

## Environment Variables

All scripts accept these common variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `INSTANCE_ID` | (required) | EC2 instance ID |
| `REGION` | `us-east-1` | AWS region |
| `AWS_PROFILE` | `experimental-admin` | Local AWS profile for `ssm send-command` |
| `MAX_STEPS` | (from YAML config) | Override training steps |
| `SAVE_EVERY` | (from YAML config) | Checkpoint save interval |
| `LOG_EVERY` | `100` | Log metrics interval |
| `RESUME_CKPT_S3_URI` | (unset) | S3 URI of checkpoint to resume from |
| `DETACH_TRAINING` | `1` | Detach training from SSM session |
| `STOP_INSTANCE_ON_EXIT` | `1` | Self-stop instance after training |
| `TRAIN_MAX_TOKENS_OVERRIDE` | (unset) | Cap training tokens |
| `VAL_MAX_TOKENS_OVERRIDE` | (unset) | Cap validation tokens |

Multi-GPU script also accepts:

| Variable | Default | Description |
|----------|---------|-------------|
| `NPROC_PER_NODE` | `auto` | Number of GPUs (auto-detects all available) |

## Architecture & Robustness Features

### Data Root Detection

Scripts auto-detect the correct data root on the instance:

1. **LVM ephemeral** (`/opt/dlami/nvme`) — AWS DL AMI instances (g5.*)
2. **Mounted EBS** (`/mnt/data`) — instances with separate EBS volumes (g6.*)
3. **LVM activation** — if LVM volumes exist but aren't mounted, activates and mounts them
4. **Root fallback** (`/mnt/data` on root) — last resort

All paths (checkpoints, datasets, token cache, run artifacts) are relative to the
detected `DATA_ROOT`. This means the same script works on any supported instance type.

### Token Cache Portability

The token cache key uses the **filename** (not full path), so caches built on one
data root path work on another. The source fingerprint (content hash) ensures
correctness. Set `TITAN_TOKEN_CACHE_TRUST_EXISTING=1` (default in scripts) to
skip the content hash check when reusing a pre-built cache from S3.

### Self-Stop with Auto-Tagging

On training completion (success or failure), scripts:

1. Sync final checkpoints and logs to S3
2. Auto-apply the `Purpose=titan-training` tag (required by IAM policy)
3. Attempt `ec2:StopInstances`
4. Log explicit diagnostics if self-stop fails

### Final Checkpoint Guarantee

`train.py` saves a final checkpoint + eval when training
ends at a step that isn't on a `save_every` boundary (e.g. 125,000 steps with
`save_every=2000`). This prevents losing the last segment of training.

### Detached Execution

Training runs detached from SSM via `nohup`. The SSM command exits after
bootstrap, but training continues. This avoids SSM timeout killing long runs.

## Known Pitfalls

1. **Code bundle too large** — If the tarball includes local checkpoints or
   token cache files, it can be multi-GB. Always use the `--exclude` flags above.

2. **g5 instance capacity** — `g5.12xlarge` often has `InsufficientInstanceCapacity`
   in popular AZs. Try `g5.24xlarge` (same 4x A10G GPUs) in alternative AZs.

3. **SSM timeout** — `executionTimeout` defaults to 43200s (12h). For very long
   bootstrap phases (large token cache builds), this may need increasing.

4. **Token cache rebuild** — First run on new data tokenizes from scratch (~80 min
   for 2B tokens). Upload the cache to S3 after the first run so subsequent runs
   skip this step.

## Legacy/Pilot Scripts

These scripts are from earlier experiments and may have hardcoded paths:

- `gpt_small_ssm_with_logs.sh` — Original foreground SSM runner
- `gpt_small_sft_pilot_cloudwatch.sh` — Small model SFT pilot
- `gpt_small_fresh_10k_with_logs.sh` — Quick 10k-step test
- `ssm_timeout_sleep_test.sh` / `ssm_timeout_wait_for_command.sh` — SSM timeout testing
- `vllm_serve.sh` — vLLM inference server
