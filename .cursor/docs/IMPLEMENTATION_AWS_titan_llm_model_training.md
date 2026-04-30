# Implementation Plan: AWS Titans LLM Model Training

> Execution tracking for this effort now lives in Linear.
> GPT-small companion docs have been archived to `.cursor/docs/archive/`. Use this file as the detailed historical archive and `.cursor/docs/llm_training_project.md` for the project summary.

## Overview
Set up and run Titans small-model training on an EC2 GPU instance (us-east-1), using **on-demand instances**. All resources are dedicated to avoid impacting existing systems: new IAM role/profile, new S3 bucket/prefix, existing training-dedicated SG, new EBS volume.

### Lessons Learned: Spot vs On-Demand

**Policy: Always use on-demand instances for training runs.**

During the GPT-Medium (407M) training (CLA-143), a spot instance (`i-0d08f0992a79333d7`) was reclaimed by AWS after several hours of training, before the first checkpoint could be saved. One-time spot requests result in instance *termination* (not stop) on reclamation, meaning all local state is lost. Even with frequent checkpointing, spot reclamation wastes hours of compute and requires manual relaunching.

The on-demand premium (~$0.98/hr vs ~$0.42/hr for g6.2xlarge) is worth the reliability for multi-hour and multi-day training runs. A full 407M pretrain costs ~$213 on-demand vs a theoretical ~$91 on spot — but losing a run to reclamation easily exceeds that difference.

## Workload context (from prior plan)
- Code: `model_training/titanProject/` (train.py, model.py, data.py, configs/*; scripts: generate.py, memory_test_eval.py, finetune_memory_qa.py, memory_test_gen.py).
- Baseline config in use: `configs/config_baseline_nomem.yaml` (MAC variant, dim 384, depth 5, heads 6, seq_len 512, batch 6, lr 5e-4, cosine, 4k steps, no memory tokens).
- Tokenizer/data: `model_training/LLM/tokenizers/bpe_50k_bf.model`; TinyStories sampled train/val at `model_training/LLM/data/tinystories_sampled/`.
- Checkpoints: only `ckpt_step_4000.pt` (memory-enabled; no clean baseline checkpoint yet).
- Local hardware used earlier: M3/MPS 16 GB unified memory (slow for longer runs).

## Approvals & Dependencies
- Admin approval needed: create IAM role/profile and new S3 bucket (Daniil; Tim already aware/approved).
- Network: use existing SG `alix-pc-llm-model-training` (SSH only to approved /32).
- Quotas: sufficient (G/VT on-demand 768 vCPUs; spot 64 vCPUs).
- Account: new `experimental` account created (ID `491794274773`, email `patrick.clawson+aws-experimental@meetalix.com`). Need SSO/role wiring in that account before creating bucket/role/profile there.

## Resource Plan (final targets)
- IAM role: `alix-llm-training-role` (trust: EC2)
- Instance profile: `alix-llm-training-profile`
- S3 bucket: `alix-ai-ml-staging-data` (new), prefix `titan/` with subprefixes `code/`, `data/`, `checkpoints/`, `logs/`
- Permissions: SSM (`AmazonSSMManagedInstanceCore`) + inline S3 RW scoped to `s3://alix-ai-ml-staging-data/titan/*`; CloudWatch logging omitted for now
- Tags: `Owner=patrick.clawson`, `Project=Titan-LLM`, `Env=staging`, `CostCenter=ai-ml-training`, `Purpose=titan-training`, `Name=titan-train-staging-g6-2xlarge`
- EC2: `g6.2xlarge` default for main pretraining, **on-demand only** (spot is not suitable for multi-day training runs — see lessons learned below)
- EC2 smoke / bring-up runner: `g6.xlarge`
- AMI: `ami-0ad8dd83d01a01d3a` (DL OSS GPU PyTorch 2.7, Ubuntu 22.04, 20260118)
- EBS: dedicated gp3 500 GB data volume, mount `/mnt/data`
- EBS throughput/Iops: gp3 default 3000 IOPS / 125 MB/s; bump to ~6000 IOPS / 250 MB/s if dataloader or checkpoint writes bottleneck.
- Security group: `alix-pc-llm-model-training` (`sg-0bec109715d614af7`), ingress 22 from `23.93.208.154/32`; widen or add 8888/6006 only if browser access without tunnels is required.
- Key pair: `alix-pc-llm-training-key` in `~/.ssh/alix-pc-llm-training-key.pem` (chmod 600); SSH config entry suggested for Host `alix-llm-ec2`.
- Ports: SSH 22 only (tunnel for TB/Jupyter)
- Data locality: keep S3/ECR in us-east-1 to avoid cross-region egress surprises.
- Cost guardrail: $50/mo alert; per-run soft cap 2h

### Multi-GPU DDP Validated (2026-04-22)

Multi-GPU training via PyTorch DDP is now production-ready. Key results from live validation on g5.12xlarge (4x A10G):

- **Script**: `train.py` (unified single/multi-GPU; `train_multi_gpu.py` was merged in and deleted on 2026-04-29). Launch with `torchrun --nproc_per_node=N train.py` for DDP or plain `python3 train.py` for single-GPU.
- **SFT Script**: `finetune_sft.py` also supports DDP via `torchrun`, using shared `train_utils.py`
- **Throughput**: ~32,000 tok/s (vs ~8,700 tok/s on single L4) — 3.7x speedup
- **Weight sync**: Perfect (max_diff=0.00e+00) across 4 ranks via NCCL
- **Test coverage**: 142 pytest tests, including 6 Gloo-based multi-process DDP tests and 29 SFT format tests
- **Instance for test**: `i-0fbf856cf80d48969` (g5.12xlarge, us-east-1d, on-demand, $5.67/hr) — auto-stopped after test

**Multi-GPU pricing (on-demand, us-east-1)**:
| Instance | GPUs | GPU Type | $/hr | Notes |
|----------|------|----------|------|-------|
| g5.12xlarge | 4 | A10G 24GB | ~$5.67 | Validated for DDP; best value for 4-GPU |
| p3.8xlarge | 4 | V100 16GB | ~$12.24 | Older; less VRAM |
| p4d.24xlarge | 8 | A100 40GB | ~$32.77 | Overkill for 407M; appropriate for 1B+ |

**Recommendation**: Use `g5.12xlarge` for multi-GPU runs up to ~1B params. For 1B+ models needing more VRAM per GPU, consider `p4d.24xlarge` or `p5.48xlarge` (H100).

## Instance choice & pricing notes (current recommendation)
- Recommended main runner: `g6.2xlarge` (L4 24 GB, 32 GiB RAM). Recommended smoke runner: `g6.xlarge` (L4 24 GB, 16 GiB RAM).
- On-demand pricing snapshot in `us-east-1`:
  - `g5.xlarge`: ~$1.006/hr
  - `g5.2xlarge`: ~$1.212/hr
  - `g6.xlarge`: ~$0.8048/hr
  - `g6.2xlarge`: ~$0.9776/hr
- Recent spot samples in `us-east-1`:
  - `g6.2xlarge`: 1d ~$0.4227/hr, 1c ~$0.4571/hr, 1b ~$0.5268/hr
  - `g5.2xlarge`: 1f ~$0.4656/hr, 1b ~$0.4649/hr, 1a ~$0.5065/hr
  - `g6.xlarge`: 1c ~$0.3486/hr, 1a ~$0.3991/hr, 1f ~$0.4797/hr
- Why `g6.2xlarge` replaced `g5.xlarge` in the recommendation:
  - It keeps a single `24 GiB` GPU while doubling host RAM from `16 GiB` to `32 GiB`.
  - It is cheaper on-demand than `g5.xlarge` in the current pricing snapshot.
  - The extra host RAM materially reduces fragility during dataset preload and tokenization, though it does not remove the underlying loader design problem.
- Not recommended as the default Titan pretraining runner:
  - `g4dn.xlarge`: cheapest, but still only `16 GiB` RAM and less GPU memory.
  - `g5.xlarge`: same host-RAM trap as the current failed path and worse on-demand price than `g6.xlarge`.
  - Multi-GPU / large-node options: outside the current `$50/mo` guarded burst model unless the project scope changes.
- Expected runtime/cost for a 4k-step-class small run on the recommended single-GPU hardware remains roughly ~0.4–1.4 hours; at current `g6.2xlarge` prices that is about ~$0.20–$0.75 on spot or ~$0.40–$1.40 on-demand depending on dataloader throughput and config changes.

## SSM Run Command: execution timeout (root cause and fix)

Long jobs launched with `AWS-RunShellScript` can hit **`ExecutionTimedOut` at ~1 hour** even when `aws ssm send-command` uses a large **`--timeout-seconds`**. AWS treats two limits separately: the document’s **`executionTimeout`** (how long the shell may run, default **3600** for `AWS-RunShellScript`) and the delivery/outer timeout. Passing only `{ "commands": [...] }` leaves the **default 3600 s** execution cap—this matched the HF GPT-2 smoke that failed after ~1h while mapping data.

**Fix (required for any long SSM shell):** include **`executionTimeout`** in the **`--parameters`** JSON next to **`commands`**, as string values in arrays (e.g. `"executionTimeout": ["43200"]`), set to at least the expected wall-clock runtime (max **172800** s per AWS). Also keep **`--timeout-seconds`** high (e.g. **43200** for 12h) via **`SSM_DELIVERY_TIMEOUT_SECONDS`** in repo scripts.

**Repo changes:** `scripts/aws_commands/gpt_small_pretrain_long_cloudwatch.sh`, `gpt_small_ssm_with_logs.sh`, and `gpt_small_fresh_10k_with_logs.sh` now emit **`executionTimeout`** via **`SSM_EXEC_TIMEOUT_SECONDS`** (default **43200**). Helpers and full write-up: **`.cursor/docs/ssm_timeout_fixes.md`**, **`scripts/aws_commands/ssm_timeout_sleep_test.sh`** (including **`SSM_LONG_VERIFY=1`** for a >1h sleep proof), **`scripts/aws_commands/ssm_timeout_wait_for_command.sh`**.

**Verified (2026-03-31):** **`SSM_LONG_VERIFY=1`** (4200 s sleep, **`executionTimeout=7200`**, **`--timeout-seconds 43200`**) completed with **`Status=Success`**, **`ResponseCode=0`**.

## Step-by-Step (after admin approval)
1) Create S3 bucket (if not already present)
```bash
AWS_PROFILE=225079546399_AdministratorAccess \
aws s3api create-bucket --bucket alix-ai-ml-staging-data --region us-east-1
```

2) Create IAM role and instance profile
```bash
# Trust policy
cat > /tmp/alix-llm-trust.json <<'EOF'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}
EOF

AWS_PROFILE=225079546399_AdministratorAccess aws iam create-role \
  --role-name alix-llm-training-role \
  --assume-role-policy-document file:///tmp/alix-llm-trust.json

# Inline S3 policy scoped to titan/*
cat > /tmp/alix-llm-s3.json <<'EOF'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["s3:GetObject","s3:PutObject","s3:DeleteObject","s3:ListBucket"],"Resource":["arn:aws:s3:::alix-ai-ml-staging-data","arn:aws:s3:::alix-ai-ml-staging-data/titan/*"]}]}
EOF

AWS_PROFILE=225079546399_AdministratorAccess aws iam put-role-policy \
  --role-name alix-llm-training-role \
  --policy-name alix-llm-training-s3 \
  --policy-document file:///tmp/alix-llm-s3.json

# Attach SSM
AWS_PROFILE=225079546399_AdministratorAccess aws iam attach-role-policy \
  --role-name alix-llm-training-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

# Create instance profile and add role
AWS_PROFILE=225079546399_AdministratorAccess aws iam create-instance-profile \
  --instance-profile-name alix-llm-training-profile
AWS_PROFILE=225079546399_AdministratorAccess aws iam add-role-to-instance-profile \
  --instance-profile-name alix-llm-training-profile \
  --role-name alix-llm-training-role
```

3) Launch EC2 (spot-first example; fill SG ID, key)
```bash
SG_ID=sg-0bec109715d614af7
KEY=alix-pc-llm-training-key
AZ=us-east-1d
AMI=ami-0ad8dd83d01a01d3a

AWS_PROFILE=225079546399_AdministratorAccess aws ec2 request-spot-instances --instance-count 1 --type one-time \
  --launch-specification "{
    \"ImageId\":\"$AMI\",
    \"InstanceType\":\"g6.2xlarge\",
    \"Placement\":{\"AvailabilityZone\":\"$AZ\"},
    \"IamInstanceProfile\":{\"Name\":\"alix-llm-training-profile\"},
    \"KeyName\":\"$KEY\",
    \"SecurityGroupIds\":[\"$SG_ID\"],
    \"BlockDeviceMappings\":[{\"DeviceName\":\"/dev/sdf\",\"Ebs\":{\"VolumeSize\":500,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}],
    \"TagSpecifications\":[{\"ResourceType\":\"instance\",\"Tags\":[
      {\"Key\":\"Owner\",\"Value\":\"patrick.clawson\"},
      {\"Key\":\"Project\",\"Value\":\"Titan-LLM\"},
      {\"Key\":\"Env\",\"Value\":\"staging\"},
      {\"Key\":\"CostCenter\",\"Value\":\"ai-ml-training\"},
      {\"Key\":\"Purpose\",\"Value\":\"titan-training\"},
      {\"Key\":\"Name\",\"Value\":\"titan-train-staging-g6-2xlarge\"}
    ]}]
  }"
```
Fallback: use `aws ec2 run-instances` on-demand with same parameters (and `--placement AvailabilityZone=$AZ`).

4) Instance setup (once running)
- SSH with key or SSM Session Manager.
- Format/mount the **non-root** EBS data volume:
```bash
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT
# Root filesystem disk (do NOT format this):
ROOT_DISK="/dev/$(lsblk -no PKNAME \"$(findmnt -n -o SOURCE /)\")"
# First non-root disk (the attached data volume)
DATA_DEV="$(lsblk -dpno NAME,TYPE | awk '$2==\"disk\"{print $1}' | grep -v \"^${ROOT_DISK}$\" | head -1)"
[ -n "$DATA_DEV" ] || { echo "No non-root data disk found; stop and verify EBS attachment."; exit 1; }

sudo mkfs -t xfs "$DATA_DEV"
sudo mkdir -p /mnt/data
UUID="$(sudo blkid -s UUID -o value "$DATA_DEV")"
echo "UUID=$UUID /mnt/data xfs defaults,nofail 0 2" | sudo tee -a /etc/fstab
sudo mount -a
```
- Code/data layout:
  - `/mnt/data/code/wintermute` (git clone or rsync)
  - `/mnt/data/data` (optional cache)
  - `/mnt/data/checkpoints`
- Sync artifacts:
```bash
aws s3 cp s3://alix-ai-ml-staging-data/titan/code/ /mnt/data/code/ --recursive
aws s3 cp s3://alix-ai-ml-staging-data/titan/data/ /mnt/data/data/ --recursive
```
- Dataloaders now accept `s3://` paths directly (streaming via boto3); configs can point to S3 objects without a manual pre-sync if desired.

5) Run training
```bash
cd /mnt/data/code/wintermute/model_training/titanProject
python train.py --config configs/config_combo_all.yaml --device cuda --log-every 100 --resume <ckpt_optional>
```
- Save checkpoints to `/mnt/data/checkpoints`, then sync:
```bash
aws s3 sync /mnt/data/checkpoints s3://alix-ai-ml-staging-data/titan/checkpoints/
```
- To minimize loss on spot interruption: save every 500–1000 steps and sync after each save.

6) Teardown / cost control
- Stop or terminate the instance after runs; do not leave on-demand running idle.
- Keep `$50` monthly alert in place; rerun plan if exceeded.

## Local validation (pre-approval)
- titans_pytorch import: ok (`python3 -c "import titans_pytorch"`).
- Artifacts present: tokenizer, train/val samples, checkpoint `ckpt_step_4000.pt`.
- SSH key present: `~/.ssh/alix-pc-llm-training-key.pem` (600 perms).
- CPU smoke run (max_tokens=20000, max_steps=5): succeeded, losses ~10.95–10.99; lr warmed from 0 to ~7e-6. Command:
  - `python3 model_training/titanProject/train.py --config model_training/titanProject/configs/config_combo_all.yaml --device cpu --max-steps 5 --max-tokens 20000 --log-every 1 --debug-every 1`
- Remote SSH note: ensure EC2 accepts `alix-pc-llm-training-key.pem` for user `ubuntu` (add to `~/.ssh/authorized_keys` on the instance or rely on EC2 keypair injection at launch).

## Open Items
- Optional: tune regeneration sample sizes for TinyStories subsets if we want smaller/faster iterations than current regenerated corpus.
- Cost-control follow-up: decide whether to keep or terminate the additional spot runner `i-0b788d7634d3a40c4` after validation.
- Once base ppl < 50: begin NoProp parallel-layer training experiments — see IMPLEMENTATION_NOPROP_EXPERIMENTS.md

## Long-term dataset redesign direction
- The AWS hardware recommendation changed because the current loader design, not only the instance class, is the main stability risk.
- `model_training/titanProject/data.py` still accumulates the tokenized corpus in Python memory before converting it to a tensor. With the current GPT-small pretrain config using `max_tokens: 1000000000`, this can overwhelm `16 GiB` hosts and destabilize the instance before training starts.
- Short-term operational stance:
  - Use `g6.2xlarge` for the extra host RAM.
  - Keep dataset preload to `/mnt/data`.
  - Use `TRAIN_MAX_TOKENS_OVERRIDE` / `VAL_MAX_TOKENS_OVERRIDE` for AWS bring-up runs when needed.
- Preferred long-term fix:
  - Pre-tokenize into disk-backed token artifacts.
  - Store them as shards or a memmap-style contiguous token file plus metadata.
  - Replace whole-corpus `tokens.extend(...)` accumulation with a dataset that slices windows lazily from disk-backed storage.
- Goal:
  - Make full pretraining runs depend on disk bandwidth and bounded window caching, not corpus-sized RAM.

### Handoff Priorities (timestamped, for next agent)

#### Local changes completed first (do not redo)
- [2026-03-28T21:40:00Z] Implemented gradient accumulation + token-budget instrumentation in `model_training/titanProject/train.py`:
  - Added `--grad-accum-steps` and optimizer stepping after accumulation.
  - Added `--target-tokens` and logging of `effective_tokens/step`, `tokens_seen`, `tok/s`, and target progress.
- [2026-03-28T21:41:00Z] Updated `model_training/titanProject/configs/config_gpt_small.yaml`:
  - `grad_accum_steps: 32`
  - `max_steps: 40000`
  - `target_tokens: 2300000000`
- [2026-03-28T21:50:00Z] Added required sanity-overfit config:
  - `model_training/titanProject/configs/config_gpt_small_sanity_overfit.yaml`
  - Tiny-shard (~200k tokens) overfit gate before any long run.
- [2026-03-28T21:30:00Z] Doc safety fixes already applied:
  - us-east-1 S3 bucket create example corrected (no `LocationConstraint`).
  - EBS mount instructions corrected to detect non-root disk via `lsblk` and mount by UUID.
  - vLLM/TGI section corrected to require HF export before serving (no raw `.pt` path).

#### AWS issues TODO (work these first when AWS work resumes)
- [2026-03-28T22:00:00Z] **P0** SSM long-run behavior (partially addressed).
  - **Resolved: `ExecutionTimedOut` ~1h** — caused by missing **`executionTimeout`** in `AWS-RunShellScript` parameters (default 3600 s). Training wrappers now set **`executionTimeout`** + high **`--timeout-seconds`**; see section **SSM Run Command: execution timeout** and `.cursor/docs/ssm_timeout_fixes.md`. Long sleep verification (**>1h**) succeeded with **`Success`**.
  - **Still watch:** intermittent **`Undeliverable`**, ghost **`InProgress`** when logs stop, S3 **`IncompleteReadError`**, or document/IPC-style failures—these are separate from the hourly execution cap. Follow-up if they recur:
    - Collect `/var/log/amazon/ssm/amazon-ssm-agent.log` and `/var/log/amazon/ssm/errors.log` around failures.
    - Correlate with uptime/reboots and concurrent SSM command volume (~5 concurrent commands per instance reported in the wild).
    - Optionally move very long training behind `nohup`/systemd + S3 log tail, or enforce watchdog + restart policy.
  - Historical command examples (pre-fix / other causes):
    - `a312856c-1f86-4397-9b74-057af22b05f9` failed with `IncompleteReadError` while streaming large S3 train file.
    - `4bd77a7b-568e-4521-aac0-52161a6d1d0f` and `bbe218e9-8887-44cc-905d-f573d11c76ca` failed with SSM document/IPC timeout style failures.
    - `493b6dca-dc54-4f9e-9eb4-c2a8df85d091` showed healthy tokenizer progress initially, then became stale while still reporting `InProgress`.
- [2026-03-28T22:05:00Z] **P0** Before any next long AWS run, execute the local sanity-overfit gate and confirm loss collapse.
  - Command:
    - `python model_training/titanProject/train.py --config model_training/titanProject/configs/config_gpt_small_sanity_overfit.yaml --device cuda --log-every 20 --data-log-every-lines 10000`
  - Pass criterion: train loss should drop sharply toward ~1 on this tiny shard.
- [2026-03-28T22:10:00Z] **P1** After local validation, perform one clean AWS rollout in one pass:
  - Sync updated `train.py` + `config_gpt_small.yaml` + new sanity config to S3 code path.
  - Relaunch with local dataset preload (already in `scripts/aws_commands/gpt_small_pretrain_long_cloudwatch.sh`).
  - Verify first `[data]` heartbeat and first `[train] step=... loss=...` line before leaving run unattended.
- [2026-03-28T22:12:00Z] **P1** Re-audit live AWS resources before restart (state drift expected).
  - Previous old pretrain instance `i-09fc4cdb410aac7ab` was terminated and its 300 GiB root volume deleted.
  - QA checkpoints copied locally to `backups/qa_checkpoints_20260328/`.
  - Reconfirm currently running/stopped instances, EBS volumes, and cost-bearing resources before new launch.

## Onboarding checklist (for new agent)
- Plan doc: `.cursor/docs/archive/PLAN_AWS_hosting.md` (archived; superseded by this file and `llm_training_project.md`).
- Titans code: `model_training/titanProject/` (`train.py`, `model.py`, `data.py`).
- Configs: `model_training/titanProject/configs/` (esp. `config_baseline_nomem.yaml`).
- Scripts: `generate.py`, `memory_test_eval.py`, `finetune_memory_qa.py`, `memory_test_gen.py`.
- Data/tokenizer: `model_training/LLM/tokenizers/bpe_50k_bf.model`; data at `model_training/LLM/data/tinystories_sampled/{train_sample.txt,val_sample.txt}`.
- Project notes: `model_training/titanProject/todo.md`, `run_log.md`.
- Checkpoints: `model_training/titanProject/ckpt_step_4000.pt` (memory-enabled; no clean baseline ckpt yet).
- Past terminal logs (if needed): `.cursor/projects/Users-beardface-lab-wintermute/terminals/`.

## Serving option (vLLM / TGI on EC2)
- Preferred self-hosted inference: run vLLM or Hugging Face TGI on the same EC2 GPU stack (DLAMI) after training.
- **Mandatory first step (before any vLLM/TGI serving): export to HF format**.
  - Raw Titans `.pt` checkpoints are not directly loadable by vLLM/TGI.
  - Export example:
    - `python model_training/titanProject/export_to_hf.py --ckpt /mnt/data/code/wintermute/model_training/titanProject/ckpt_step_4000.pt --out /mnt/data/hf_runs/gpt_small_nomem --tokenizer /mnt/data/code/wintermute/model_training/LLM/tokenizers/bpe_50k_bf.model`
- vLLM quick start (HF export only):
  - `pip install vllm`
  - `python -m vllm.entrypoints.api_server --model /mnt/data/hf_runs/gpt_small_nomem --host 0.0.0.0 --port 8000`
  - Open SG port 8000 only to trusted CIDR (or tunnel/SSM).
- TGI quick start (example):
  - `pip install text-generation` (or use TGI container)
  - `text-generation-launcher --model /mnt/data/hf_runs/gpt_small_nomem --port 8000 --num-shard 1`
  - Same SG guidance; front with nginx if exposing via `/titan/`.
- Keep HTTP surface minimal; prefer SSM/SSH or tunnel. Tag and stop/terminate instances when idle.

### Important: model format compatibility
- The Titans checkpoints are not in standard Hugging Face/transformers format. vLLM/TGI expect HF-compatible model artifacts.
- New helpers:
  - `model_training/titanProject/export_to_hf.py` exports a HF-style folder (`config.json`, `pytorch_model.bin`, tokenizer copy, auto_map entries).
  - `model_training/titanProject/modeling_titans.py` provides a minimal HF-compatible config/model wrapper (`TitansConfig`, `TitansForCausalLM`) that loads titans-pytorch checkpoints and exposes logits/loss for basic generation.
- Do not point vLLM/TGI directly at `.pt` checkpoints; always serve from HF-exported artifacts.

## Security hardening (latest)
- Enforced IMDSv2 on `i-050bce8db858dfa89` (instance metadata requires tokens).
- Removed SSH ingress from the training SG `sg-05ca8b4be3b26ef52` (SSH now closed; use SSM).
- SG ingress now limited to:
  - TCP 80: 23.93.208.154/32, 104.7.12.166/32
  - TCP 8000: 23.93.208.154/32, 104.7.12.166/32
- Pending Twingate: once connector SG/CIDR is known, restrict SG to Twingate and drop public ingress. Consider removing the public IP after that.

## Progress Log
- 2026-02-22: Env hook set (`AWS_PROFILE=experimental-admin`, `AWS_DEFAULT_REGION=us-east-1` via `.envrc`). New bucket created: `alix-ai-ml-staging-data`. IAM role `alix-llm-training-role` (EC2 trust) with inline S3 RW to `s3://alix-ai-ml-staging-data/titan/*` and `AmazonSSMManagedInstanceCore` attached. Instance profile `alix-llm-training-profile` created and linked to the role.
- 2026-02-22: Verified AWS CLI/profile on workstation (`491794274773_AdministratorAccess` works). No `g5.xlarge` instances currently running (describe-instances returned empty). Expected SG `alix-pc-llm-model-training` / `sg-0bec109715d614af7` is not present in this account/region; will need to create a new SG in us-east-1 (SSH 22 from approved /32, tunnel TB/Jupyter) before launch. Next session (Linux): create SG, then launch spot g5.xlarge with the existing IAM profile/bucket/tag settings.
- 2026-03-08: Added helper script `model_training/titanProject/aws_titan_next_steps.py` to make AWS continuation repeatable and idempotent.
  - `audit`: compares live AWS state to open items and reports done/pending for bucket, role/profile, SG, key pair, runners, spot requests, and checkpoint objects.
  - `ensure-sg`: creates `alix-pc-llm-model-training` in the selected VPC if missing, with SSH ingress on 22 for the approved CIDR.
  - `launch-spot`: launches `g5.xlarge` spot with AZ fallback order `us-east-1f, us-east-1d, us-east-1a, us-east-1c`.
  - `launch-ondemand`: launches on-demand `g5.xlarge` in a single-AZ fallback path.
  - Recommended operator flow: `audit` -> `ensure-sg` (if needed) -> `launch-spot` -> `audit` again.
- 2026-03-08: Hardened CLI detection in `aws_titan_next_steps.py` for WSL environments.
  - `run_aws()` now tries `aws`, `aws.exe`, and `/mnt/c/Program Files/Amazon/AWSCLIV2/aws.exe`.
  - Supports explicit override via `AWS_CLI_BIN` when AWS CLI is installed in a non-standard location.
- 2026-03-08: Executed AWS continuation commands and resolved local tooling blockers:
  - Installed AWS CLI v2 in WSL at `/home/zombi/.local/bin/aws` using zip installer (no apt package available in this environment).
  - Verified `audit` runs successfully with `AWS_CLI_BIN=/home/zombi/.local/bin/aws`.
  - Added explicit auth error surfacing in `audit` output (`AuthError: ... profile ... could not be found`) to avoid misleading all-false resource checks.
  - Current blocker: AWS profile `experimental-admin` is not configured in WSL and `/mnt/c/Users/zombi/.aws` does not exist, so no credentials/profile could be copied.
  - Next unblock step: configure SSO profile in WSL (`/home/zombi/.local/bin/aws configure sso --profile experimental-admin`), then re-run `audit`, `ensure-sg`, and `launch-spot`.
- 2026-03-08: Post-SSO execution completed and environment state advanced:
  - Confirmed identity for profile `experimental-admin`: account `491794274773`, role `AWSReservedSSO_AdministratorAccess`.
  - Live audit now resolves real resources:
    - S3 bucket exists: `alix-ai-ml-staging-data`
    - IAM role/profile exist and attached correctly (`alix-llm-training-role`, `alix-llm-training-profile`)
    - SG exists by name: `alix-pc-llm-model-training` with ID `sg-05ca8b4be3b26ef52`
    - Key pair exists: `alix-pc-llm-training-key`
    - Titan instance present: `i-050bce8db858dfa89` (`g5.xlarge`)
  - Started `i-050bce8db858dfa89`; state confirmed `running` and public IP `3.238.84.152`.
  - SSH path is currently blocked from local because private key file `/home/zombi/.ssh/alix-pc-llm-training-key.pem` is missing locally; switched to SSM control path.
  - SSM check confirms instance is online, `/mnt/data` is mounted, but `/mnt/data/code` is absent (no repo checkout on disk).
  - SSM dependency probe confirms training deps are missing on instance Python: `torch`, `sentencepiece`, `titans_pytorch` not installed.
  - Synced `model_training/titanProject` code to `s3://alix-ai-ml-staging-data/titan/code/wintermute/model_training/titanProject` (excluded `*.pt`).
  - Attempted tokenizer/data upload from local failed because local files are missing:
    - expected tokenizer `model_training/LLM/tokenizers/bpe_50k_bf.model` not present in workspace
    - expected sampled data files under `model_training/LLM/data/tinystories_sampled/` not present in workspace
  - New blocker to start training: need source of tokenizer + TinyStories sampled train/val, then complete EC2 bootstrap (pull code/data, install deps, run baseline).
- 2026-03-08: Additional execution completed after auth:
  - SSM confirms instance online and manageable (`PingStatus=Online`), so SSH key absence is not a hard blocker for automation.
  - Synced Titans code to S3 and restored on instance:
    - Local -> S3: `s3://alix-ai-ml-staging-data/titan/code/wintermute/model_training/titanProject`
    - S3 -> EC2 via SSM: `/mnt/data/code/wintermute/model_training/titanProject`
  - Installed Python deps on EC2 via SSM: `torch`, `sentencepiece`, `titans-pytorch` (plus transitive CUDA deps); follow-up check reports all required modules import successfully.
  - Verified current live audit state:
    - Titan instance is now `running` (`i-050bce8db858dfa89`, `g5.xlarge`, `us-east-1f`)
    - `titan/code/` prefix exists; `titan/data/` and `titan/checkpoints/` remain empty.
  - Asset search on EC2 for required files returned no matches:
    - `bpe_50k_bf.model`
    - `train_sample.txt`
    - `val_sample.txt`
  - Remaining blocker to start baseline run is now strictly data/tokenizer availability (restore original assets or regenerate replacements).
- 2026-03-09: Local Hugging Face CLI tooling installed in WSL for operator convenience.
  - Local `python3` initially had no `pip` module and no `ensurepip`.
  - Installed pip with `python3 /tmp/get-pip.py --user --break-system-packages`.
  - Installed HF CLI with `python3 -m pip install --user --upgrade --break-system-packages huggingface_hub`.
  - Verified local CLI at `/home/zombi/.local/bin/hf` (`hf --version` -> `1.6.0`).
- 2026-03-09: Attempted local HF auth using token file `.cursor/tmp/huggingface_token.txt`.
  - Token file is populated (non-empty), but `hf auth login --token ...` returns `Invalid user token`.
  - On this CLI version, identity check command is `hf auth whoami` (not `hf whoami`).
  - Current status: HF local CLI installed but not authenticated; requires refreshed/valid token.
- 2026-03-09: Refreshed HF token and completed authentication on both local WSL and EC2.
  - Local WSL:
    - `hf auth login` succeeded with token label `linux_wintermute`.
    - `hf auth whoami` returns user `beardface`.
  - EC2 instance (`i-050bce8db858dfa89`):
    - Logged in via SSM command using a short-lived encrypted token object in S3 (`titan/tmp/...`) that is removed at script exit.
    - SSM command id: `63ee789d-9f7a-4457-82fc-77667cc6854b`
    - Result: `Status=Success`, token valid (`read`), active token `linux_wintermute`, `whoami` user `beardface`.
  - This removes anonymous HF rate-limit risk for future model/dataset pulls on the training instance.
- 2026-03-09: Regenerated missing tokenizer/data assets directly on EC2 via SSM and synced to S3.
  - Installed `datasets` on instance and loaded `roneneldan/TinyStories`.
  - Generated sampled files:
    - `/mnt/data/code/wintermute/model_training/LLM/data/tinystories_sampled/train_sample.txt` (250k stories, ~217 MB)
    - `/mnt/data/code/wintermute/model_training/LLM/data/tinystories_sampled/val_sample.txt` (15k stories, ~13 MB)
  - Trained SentencePiece tokenizer (`vocab_size=50000`, BPE, `byte_fallback=true`, `nfkc`):
    - `/mnt/data/code/wintermute/model_training/LLM/tokenizers/bpe_50k_bf.model`
    - `/mnt/data/code/wintermute/model_training/LLM/tokenizers/bpe_50k_bf.vocab`
  - Synced assets to S3:
    - `s3://alix-ai-ml-staging-data/titan/code/wintermute/model_training/LLM/tokenizers/bpe_50k_bf.model`
    - `s3://alix-ai-ml-staging-data/titan/data/tinystories_sampled/{train_sample.txt,val_sample.txt}`
- 2026-03-09: Baseline EC2 training run completed successfully and checkpoint synced to S3.
  - SSM command: `839de07f-48ad-43d3-82db-6e5707716018`
  - Runtime: ~22m25s (step 0 -> 4000)
  - Final train loss at step 4000: `2.7927`
  - Eval at step 4000: loss `2.8044`, perplexity `16.52`
  - Checkpoint saved on instance: `/mnt/data/code/wintermute/model_training/titanProject/ckpt_step_4000.pt` (~678 MB)
  - Synced checkpoint to `s3://alix-ai-ml-staging-data/titan/checkpoints/ckpt_step_4000.pt`
- 2026-03-09: Completed spot-runner validation and periodic checkpoint sync automation.
  - Spot validation:
    - Launched spot runner via helper: instance `i-0b788d7634d3a40c4` (`us-east-1f`, `g5.xlarge`, spot fulfilled).
    - Spot request observed active/fulfilled: `sir-zxezf49j`.
    - SSM registration check confirms `PingStatus=Online` for the new spot instance.
  - Checkpoint sync automation:
    - Updated `model_training/titanProject/train.py` with CLI hooks:
      - `--save-every` (override checkpoint interval)
      - `--checkpoint-dir` (separate output path per run)
      - `--s3-checkpoint-uri` + `--aws-bin` (auto sync to S3 after each save)
      - Missing `aws` binary now logs a warning and skips sync instead of crashing training.
    - Moved eval/save hooks to step-level execution so long epochs do not delay checkpoint persistence.
    - Compatibility fix: removed `--no-cli-pager` from remote `aws s3 sync` invocation because instance AWS CLI rejected that flag.
  - Smoke verification on EC2 (`i-050bce8db858dfa89`):
    - Final successful SSM command: `bcab2033-f5b6-4eac-a8af-61e6ade3e38b` (runtime ~20s).
    - Config: `--max-steps 10 --save-every 5 --max-tokens 20000`.
    - Observed saves at steps `5` and `10` with successful sync after each save.
    - Verified objects under `s3://alix-ai-ml-staging-data/titan/checkpoints/sync_smoke_20260309034822/`:
      - `ckpt_step_5.pt`
      - `ckpt_step_10.pt`
- 2026-03-09: Added and validated inference smoke test for checkpoint usability.
  - New script: `model_training/titanProject/inference_smoke.py`
    - Loads config/tokenizer/checkpoint once and runs a fixed 3-prompt suite.
    - Emits pass/fail and structured JSON (`ok`, latencies, completion lengths).
  - EC2 validation command id: `35d9cc4e-d630-4f37-a082-ef00bb878065`
    - Result: `Status=Success`, `ok=true`, `device=cuda`, runtime ~12.7s.
    - Confirms end-to-end inference path (checkpoint load -> tokenization -> generation) is operational.
    - Output quality is coherent tiny-story continuation but not instruction-accurate chat behavior yet (expected for current baseline training objective).
- 2026-03-09: Added interactive qualitative test interface (`chat_repl.py`) and validated on EC2.
  - New script: `model_training/titanProject/chat_repl.py`
    - Interactive terminal loop with `/reset` and `/exit`.
    - Multi-turn history with context-budget trimming (`max_prompt_tokens`, default from `train.seq_len`).
    - Uses existing sampling knobs (`--top-k`, `--temperature`, `--max-new`) and checkpoint/config paths.
  - EC2 smoke command id: `a9d91143-ac27-4cb3-918c-cefacd1d78c3`
    - Result: `Status=Success`.
    - One-turn scripted interaction returns assistant text and exits cleanly.
  - Quick run command on instance:
    - `python3 chat_repl.py --config configs/config_baseline_nomem.yaml --ckpt ckpt_step_4000.pt --device cuda`
- 2026-03-09: Added lightweight HTTP chat endpoint and validated request/response flow on EC2.
  - New script: `model_training/titanProject/chat_http.py`
    - `GET /health` -> model/device/config health metadata.
    - `POST /chat` -> body fields: `session_id`, `message`, optional `reset`, `max_new`, `top_k`, `temperature`.
    - `POST /reset` -> clears in-memory session history for a `session_id`.
    - In-memory multi-turn sessions with context trimming against prompt token budget.
  - EC2 smoke command id: `899ef90e-2185-4c10-8944-34fbe88f02dd`
    - Result: `Status=Success`.
    - Verified `GET /health`, two sequential `POST /chat` calls on same session, and `POST /reset`.
  - Quick run command on instance:
    - `python3 chat_http.py --config configs/config_baseline_nomem.yaml --ckpt ckpt_step_4000.pt --device cuda --host 0.0.0.0 --port 8000`
- 2026-03-09: Published HTTP endpoint for direct browser testing.
  - Opened SG ingress rule on `sg-05ca8b4be3b26ef52` for TCP `8000` from `23.93.208.154/32` (same restricted CIDR pattern as SSH).
  - Updated `chat_http.py` to serve a lightweight browser UI at `GET /` in addition to JSON endpoints.
  - Started server on instance `i-050bce8db858dfa89` via SSM command `c41c87e0-30d7-4f7f-ac2a-e5c70e051aa4`.
  - Verified remote reachability from workstation:
    - `curl http://3.238.84.152:8000/` returns the chat UI HTML page.
    - `curl http://3.238.84.152:8000/health` returns `ok=true`.
  - Browser test URL (while server is running): `http://3.238.84.152:8000/`
- 2026-03-09: Added HF export shim and tightened security.
  - Added `export_to_hf.py` to produce a HF-style folder (config + state_dict + tokenizer copy) with auto_map entries.
  - Added `modeling_titans.py` HF wrapper (`TitansConfig`, `TitansForCausalLM`) to load Titans checkpoints in a HF-style flow; still depends on titans-pytorch and is a minimal implementation.
  - Hardened instance: enforced IMDSv2; removed SSH ingress (SSM-only); SG now only allows 80/8000 to the two /32s (work/home). Pending Twingate info to drop public ingress entirely.
- 2026-03-09: Commenced SFT pilot on OASST1 + Dolly instruction mix and deployed pilot checkpoint to HTTP endpoint.
  - Added new project scripts/config:
    - `model_training/titanProject/prepare_sft_mix.py` (builds one-line `User: ... Assistant: ...` corpus from `OpenAssistant/oasst1` + `databricks/dolly-15k`)
    - `model_training/titanProject/finetune_sft.py` (short supervised finetune loop with eval/save/s3-sync hooks)
    - `model_training/titanProject/configs/config_sft_pilot_oasst1_dolly.yaml`
  - SSM pilot run command: `72155daf-eb1f-4499-bbba-0e825d9eabaa` (status `Success`, runtime ~4m41s).
  - Data prep output:
    - OASST1 pairs extracted: train `23398`, val `1212`
    - Dolly pairs extracted: `14996`
    - Pilot corpus written: train `18000` lines, val `1600` lines at `model_training/LLM/data/sft_mix/`
  - Finetune summary (from `ckpt_step_4000.pt`):
    - 600 SFT steps on CUDA, checkpoints at steps `200`, `400`, `600`
    - Eval loss trend (40 val batches): `7.48` -> `7.08` -> `6.80` -> `6.64` -> `6.51` -> `6.41`
    - Final checkpoint: `/mnt/data/checkpoints/sft_pilot_20260309054346/ckpt_sft_step_600.pt`
    - Synced artifacts:
      - `s3://alix-ai-ml-staging-data/titan/checkpoints/sft_pilot_20260309054346/ckpt_sft_step_{200,400,600}.pt`
      - `.../inference_smoke.json`
      - `.../sample_chat.txt`
  - Live endpoint switched to the SFT checkpoint via SSM command `d6acfb2a-88d4-40f3-b8b6-ba337eb39913`.
    - `/health` now reports `ckpt_sft_step_600.pt`.
    - Browser URL remains: `http://3.238.84.152:8000/`
- 2026-03-09: Added domain-friendly path routing for `wint3rmute.com/titan` via nginx reverse proxy.
  - Opened SG ingress for TCP `80` on `sg-05ca8b4be3b26ef52` (restricted to `23.93.208.154/32`).
  - Installed/configured nginx on EC2 (`i-050bce8db858dfa89`) with:
    - `location = /titan` -> redirect to `/titan/`
    - `location /titan/` -> reverse proxy to `http://127.0.0.1:8000/`
  - Updated `chat_http.py` UI JS to call relative API paths based on current URL path (so `/titan/` correctly uses `/titan/health`, `/titan/chat`, `/titan/reset`).
  - Verified:
    - `http://3.238.84.152/titan/health` returns `ok=true`
    - `http://3.238.84.152/titan/chat` accepts POST and returns model reply
    - host header override test works: `curl --resolve wint3rmute.com:80:3.238.84.152 http://wint3rmute.com/titan/health`
  - DNS note:
    - Current public DNS lookup for `wint3rmute.com` resolves to `23.93.208.154` (operator network), not EC2 `3.238.84.152`.
    - To use `http://wint3rmute.com/titan` publicly, set/adjust DNS `A` record for `wint3rmute.com` to `3.238.84.152` (or CNAME via your preferred fronting layer).
- 2026-03-10: HF GPT-2 no-memory smoke via SSM (timed out).
  - Added `model_training/titanProject/scripts/run_gpt2_nomem_ssm.sh` (pins `numpy<2`, installs CPU-only `torch==2.2.2`, pulls `train_gpt2_nomem.py`, syncs outputs) and updated `train_gpt2_nomem.py` to drop empty lines before tokenization.
  - SSM command `18bd3237-5bab-442a-849d-a753ccc3efa8` ran ~1h and hit `ExecutionTimedOut` while mapping the dataset; no checkpoint or metrics produced. **Cause (later):** default `AWS-RunShellScript` **`executionTimeout`** (3600 s) without an override in `--parameters`; raising `--timeout-seconds` alone does not fix it. See **SSM Run Command: execution timeout** in this doc.
  - Log stored at `s3://alix-ai-ml-staging-data/titan/logs/hf_gpt2_nomem.log`.
  - Next options: install GPU `torch` and rerun on CUDA; or limit work (smaller sampled corpus and low `--max_steps`/`save_steps`/`eval_steps`) to finish quickly via SSM; or pre-shrink the dataset on S3 and rerun the same script; or pass **`executionTimeout`** in parameters for long shells.
- 2026-03-20: HF GPT-2 no-memory completion + vLLM smoke attempt.
  - Training rerun succeeded (SSM `e1f8911e-22aa-4747-ac8a-d63f12d22ad8`): 20k/2k TinyStories slice, max_steps=200, fp16 on GPU, filtered empty tokens. Artifacts in `s3://alix-ai-ml-staging-data/titan/hf_exports/gpt2_nomem_small/` (checkpoints 100/200, final model, samples, eval).
  - Eval + generations (SSM `0be40db5-1710-49a2-8fa9-91cf5c080b0a`): perplexity ~13.07, samples saved to `samples.txt`, metrics to `eval_metrics.json` in the same prefix.
  - Sanity load (SSM `3d217766-dfee-472a-8115-c93abfb1f1bc`): loaded saved model and generated 3 prompts successfully; output synced to `sanity.txt` in the same S3 prefix.
  - vLLM test: pulled `vllm/vllm-openai:latest` and attempted container on GPU. First run failed (port 8000 already in use); second run bound to localhost:8010 but `/v1/models` curled with connection reset; container was stopped and instance terminated to avoid costs. EC2 `i-050bce8db858dfa89` now `shutting-down`.
- 2026-03-20: Additional vLLM container attempts (all failed; instances terminated).
  - Launched fresh g5.xlarge; sync of HF export succeeded. vLLM tags tried: `latest` (hung on startup), `0.5.2-cuda12` (not found), `0.5.2-cpu` (not found), `0.4.0` (not found), `cpu` (not found). Subsequent GPU run of `latest` on port 8010 also hung; CPU tag pull not available. Instances `i-03d682920a52d247d` and `i-0df5277f9b1256a24` were terminated to avoid cost.
- 2026-03-21: vLLM CPU smoke + ECR push attempt.
  - Launched g5.xlarge `i-0785efbb8fb0873f2` (DLAMI `ami-0ad8dd83d01a01d3a`), synced `s3://alix-ai-ml-staging-data/titan/hf_exports/gpt2_nomem_small/` to `/mnt/data/hf_runs/gpt2_nomem_small`.
  - CPU run: `docker run -d --name vllm-test -p 127.0.0.1:8010:8000 -v /mnt/data/hf_runs/gpt2_nomem_small:/model vllm/vllm-openai-cpu:v0.17.1 --model /model --dtype float32 --port 8000`. Warmup ~34s; `/v1/models` succeeded after warmup (initial curls returned connection reset before warmup completed). Instance terminated after test to avoid cost.
  - GPU image push: created ECR repo `vllm-openai` in `us-east-1`; pulled `vllm/vllm-openai:latest` locally and began push to `491794274773.dkr.ecr.us-east-1.amazonaws.com/vllm-openai:latest`. Push is still in progress/slow (many layers reported “Unavailable” while uploading); will retry/resume to complete manifest.
- 2026-03-21: GPU serve from ECR (amd64) validated.
  - Added ECR pull/push permissions to `alix-llm-training-role` (GetAuthToken, BatchCheck/GetDownloadUrl/BatchGetImage, Initiate/Upload/CompleteLayerUpload, PutImage).
  - Launched g5.xlarge `i-03b5af3c45a1947bc` (AMI `ami-0ad8dd83d01a01d3a`), synced HF export to `/mnt/data/hf_runs/gpt2_nomem_small`.
  - Pulled `vllm/vllm-openai:latest` with `--platform linux/amd64`, pushed to ECR tag `vllm-openai:latest` (digest `sha256:c32358ebfc115d56ade2acfdbcd00df5b115417dbd6006547c88f07e2b39de06`).
  - Ran GPU server: `docker run -d --gpus all --name vllm-gpu -p 127.0.0.1:8010:8000 -v /mnt/data/hf_runs/gpt2_nomem_small:/model 491794274773.dkr.ecr.us-east-1.amazonaws.com/vllm-openai:latest --model /model --dtype float16 --port 8000`.
  - Warmup/compile finished (~80s total load/compile). `/v1/models` returned OK for `/model`. Instance terminated afterward to avoid cost.
- 2026-03-21: GPU serve re-verify with memory cap and completions.
  - Launched g5.xlarge `i-0287618bdb5235c3b`, synced HF export to `/mnt/data/hf_runs/gpt2_nomem_small`.
  - Ran: `docker run -d --gpus all --name vllm-gpu -p 127.0.0.1:8010:8000 -v /mnt/data/hf_runs/gpt2_nomem_small:/model 491794274773.dkr.ecr.us-east-1.amazonaws.com/vllm-openai:latest --model /model --dtype float16 --port 8000 --gpu-memory-utilization 0.85`.
  - Warmup ~90s; `/v1/models` OK; `/v1/completions` success (prompt “Hello from GPU test”, max_tokens=32). Instance terminated post-test to control cost.
- 2026-03-22: Dataset staging to S3 (for training prep).
  - HuggingFaceFW/fineweb-edu sample 100BT shards `000_00000`–`000_00009` synced to `s3://alix-ai-ml-staging-data/titan/data/fineweb-edu/sample-100BT/sample/100BT/` (local temp cleared).
  - bigcode/the-stack-smol (31 language JSON shards) synced to `s3://alix-ai-ml-staging-data/titan/data/stack-smol/` (local temp cleared).
  - Next use: point dataloaders to these prefixes or stage onto EBS before runs; instance role already has RW on `s3://alix-ai-ml-staging-data/titan/*`.
  - Bucket lifecycle set: objects under `titan/data/` transition to STANDARD_IA after 30 days (cost control).
- 2026-03-22: Checkpoints reset and preprocessing script added.
  - Removed prior checkpoints from `s3://alix-ai-ml-staging-data/titan/checkpoints/` to start fresh.
  - Added `model_training/titanProject/scripts/preprocess_corpus_and_tokenizer.py` to:
    - Sync fineweb parquet and stack-smol JSON from S3.
    - Convert to newline text, split train/val, train a fresh SentencePiece BPE (50k default).
    - Upload text and tokenizer back to S3.
  - Suggested run (on EC2):
    ```bash
    python3 model_training/titanProject/scripts/preprocess_corpus_and_tokenizer.py \
      --fineweb-s3 s3://alix-ai-ml-staging-data/titan/data/fineweb-edu/sample-100BT/sample/100BT/ \
      --stack-smol-s3 s3://alix-ai-ml-staging-data/titan/data/stack-smol/ \
      --text-out-s3 s3://alix-ai-ml-staging-data/titan/data/processed/ \
      --tokenizer-out-s3 s3://alix-ai-ml-staging-data/titan/tokenizers/new_bpe_50k/ \
      --vocab-size 50000 --spm-sample-lines 2000000 --val-ratio 0.05
    ```
    Requires: `pyarrow`, `pandas`, `sentencepiece`, `boto3`.
- 2026-03-22: Preprocessing executed; new tokenizer + text in S3; sanity train pending.
  - Processed corpus uploaded: `s3://alix-ai-ml-staging-data/titan/data/processed/{train.txt (~32.3GB), val.txt (~1.6GB), spm_sample.txt}`.
  - Trained tokenizer uploaded: `s3://alix-ai-ml-staging-data/titan/tokenizers/new_bpe_50k/{bpe_50k_fw_stack.model,vocab}`.
  - Config updated to use S3 paths and token caps: `config_baseline_nomem.yaml` now points to the above and caps `max_tokens=50,000,000`, `max_tokens_val=2,000,000`.
  - S3-ready dataloader is in place (`data.py`/`train.py` streaming). Code synced to S3 under `titan/code/wintermute/model_training/titanProject/`.
  - Sanity train attempt (200 steps, g5.xlarge) failed early due to mixing torch versions (torch 2.10 pulled via titans-pytorch deps overriding torch 2.2.2) and an old `train.py` lacking S3 path handling. Instance terminated after failure to avoid costs.
  - Next run instructions: (on fresh g5.xlarge)
    1) `aws s3 sync s3://alix-ai-ml-staging-data/titan/code/wintermute /home/ubuntu/wintermute --exclude '*/logs/LLM/.venv/*' --exclude '*/logs/LLM/.venv/**' --only-show-errors`
    2) Install deps without upgrading torch:  
       ```
       pip install --upgrade --no-cache-dir torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
       pip install --no-cache-dir sentencepiece boto3 pyarrow
       pip install --no-cache-dir titans-pytorch==0.5.3 --no-deps
       pip install --no-cache-dir einops==0.8.2 einx==0.4.2 hyper-connections==0.4.9 axial-positional-embedding==0.3.12 assoc-scan==0.0.4 ema-pytorch==0.7.9 tqdm fire loguru orjson
       ```
    3) Run:
       ```
       cd /home/ubuntu/wintermute
       python3 model_training/titanProject/train.py \
         --config model_training/titanProject/configs/config_baseline_nomem.yaml \
         --device cuda \
         --max-steps 200 \
         --log-every 50 \
         --save-every 100 \
         --checkpoint-dir /mnt/data/checkpoints/sanity_fw_stack \
         --s3-checkpoint-uri s3://alix-ai-ml-staging-data/titan/checkpoints/sanity_fw_stack/ \
         --aws-bin aws
       ```
    4) If any parquet read errors reoccur, they are now skipped with warnings; text/val already materialized, so reads will be from S3 text.
- 2026-03-22: Sanity train success and torch upgrade plan.
  - 200-step sanity run succeeded on g5.xlarge with local tokenizer and S3-streamed text; checkpoints synced to `s3://alix-ai-ml-staging-data/titan/checkpoints/sanity_fw_stack/{ckpt_step_100.pt,ckpt_step_200.pt}`. Loss at step 200: ~7.67. Instance terminated post-run.
  - `train.py` updated to use `torch.amp.autocast` (no `device_type` arg) and safer `GradScaler(enabled=...)`, keeping AMP on CUDA and avoiding prior TypeError.
  - Torch requirement: `titans-pytorch`/`accelerated-scan` expect `torch>=2.8/2.9`; move to torch 2.9.0 to clear resolver warnings.
  - Recommended install for next runs (CUDA 12.4 wheels):
    ```
    pip install --upgrade --no-cache-dir torch==2.9.0+cu124 torchvision==0.24.0+cu124 torchaudio==2.9.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
    pip install --no-cache-dir sentencepiece boto3 pyarrow
    pip install --no-cache-dir titans-pytorch==0.5.3 --no-deps
    pip install --no-cache-dir einops==0.8.2 einx==0.4.2 hyper-connections==0.4.9 axial-positional-embedding==0.3.12 assoc-scan==0.0.4 ema-pytorch==0.7.9 tqdm fire loguru orjson tensordict==0.11.0 x-transformers==2.17.7 rotary-embedding-torch==0.8.9 ninja --no-deps
    ```
  - If cu124 wheels are unavailable on the AMI mirror, fallback to `torch==2.8.0+cu124` with matching `torchvision/torchaudio` versions and rerun; keep `numpy==1.26.4` if base image preloads `numpy>=2`.
- 2026-03-23: Intermediate evals and scaling plan toward Wizard-Vicuna-13B class (ref: https://huggingface.co/QuixiAI/Wizard-Vicuna-13B-Uncensored).
  - Latest checkpoints: up to `ckpt_step_20000.pt` in `s3://alix-ai-ml-staging-data/titan/checkpoints/longrun_fw_stack/`; 20k eval: val_loss=5.3643, ppl≈213.6 on the capped val set (2M tokens). Earlier trend: at 17k evals ~5.42 (ppl ~226).
  - Current model is a ~0.4B MAC config (d=384, L=5, heads=6, seq=512, vocab=50k) — far below 13B-class capability; quality is appropriate for size/data/steps.
  - Bridge plan (staged):
    - 1B config (proposed): d≈1024, L≈24, heads≈16, ffn 4×, seq_len 2048, vocab 50k (reuse tokenizer). Batch target effective 1–2M tokens/step (micro-batch 8–16 × grad-acc 8–16 @ seq 2k). Train on billions of tokens (2–5B min), uncapped data. LR cosine, warmup ~2–3k, base LR ~3e-4, min LR ~3e-5.
    - 3B config (next): d≈1536, L≈32, heads≈24, seq_len 2048–4096, effective batch 2–4M tokens/step; needs multi-GPU (FSDP/ZeRO) or larger single GPUs than g5.xlarge. Base LR ~2e-4, similar schedule.
    - 13B target (Wizard-Vicuna class): adopt LLaMA-13B shape (d≈5120, L=40, heads=40, seq 4k+), train on tens/hundreds of billions of tokens; requires multi-GPU A100/H100/L4 cluster; follow with SFT/RLHF-style instruction tuning (Wizard/Vicuna data or similar).
  - Infra notes:
    - Remove train/val token caps when scaling; increase seq_len to 2k+; keep current tokenizer unless expanding domain.
    - Use larger instances (e.g., g5.12x / g6.xlarge or multi-GPU) for 1B+; set SSM timeouts generously (12h+); avoid per-run pip reinstall via prepared AMI/EBS if possible.
    - Save/eval cadence: for long runs use every 1k steps (later widen if stable); checkpoints sync to S3.
- 2026-03-24: Pause Titan-MAC training; shift to standard GPT-style bring-up.
  - Rationale: stabilize end-to-end training/serving on AWS with a standard HF-compatible causal LM before revisiting Titan memory variants.
  - Plan:
    - Add a GPT-small config (HF-compatible) for bring-up: e.g., d≈768, L≈12, heads≈12, ffn 4×, seq_len 2048, vocab 50k (reuse tokenizer). Target effective batch ~0.5–1M tokens/step via micro-batch + grad-acc.
    - Training settings: cosine LR, warmup ~2k steps, base LR ~3e-4, lr_min ~3e-5; AMP on; save/eval every 1k; log every 100. Start with a token cap (optional, e.g., 200M) for quick validation, then remove caps for longer runs.
    - Data: use existing processed train/val (fineweb+stack-smol) uncapped when stable; keep val slice consistent.
    - Infra: prefer a larger single GPU (g5.12x/g6.xlarge) or multi-GPU if we bump batch/seq; set long SSM timeout (12h+); avoid per-run pip installs if possible.
    - Serving: export checkpoints directly in HF format; serve with vLLM/TGI for smoke tests.
  - Next: implement GPT-small config, run bring-up, then lift caps and/or scale model size once pipeline is proven.
  - Future context window goal: plan a larger-context variant (target ~128k tokens) after hardware/resource analysis; will require longer seq_len configs, potentially larger instances or multi-GPU, and corresponding changes to serving stack.
- SSM completion note (for future runs):
  - Avoid calling `shutdown` inside the same SSM script that runs training; let training exit cleanly, then stop/terminate in a separate command. If shutdown must occur in-script, add a short sleep after training/sync so the agent can report `Success`.
  - Don’t background work inside the training script; ensure training and sync finish before exit.
  - Consider using `--output-s3-bucket-name/--output-s3-key-prefix` or CloudWatch logs to capture stdout/stderr, so logs are available even if the instance stops.
  - Leave the instance running after the command finishes so SSM can exit `InProgress` cleanly; only stop/terminate after SSM reports completion.
- 2026-03-22: Torch availability + rerun on 2.5.1 with extra deps (SUCCESS).
  - cu124/cu121 wheels for torch 2.8/2.9 were not available on the PyTorch index from the DLAMI; attempts to install `torch==2.9.0+cu124` and `torch==2.8.0+cu121` failed with “No matching distribution found”.
  - Settled on `torch==2.5.1+cu121` / `torchvision==0.20.1+cu121` / `torchaudio==2.5.1+cu121` plus the following extras required by titans/tensordict/einx: `pyvers`, `cloudpickle`, `frozendict` (installed with `--no-deps`), along with the pinned dependency set used earlier.
  - Re-ran 200-step sanity on g5.xlarge with the above stack; losses: step 50=9.97, 100=8.94, 150=7.92, 200=7.84. Checkpoints synced to `s3://alix-ai-ml-staging-data/titan/checkpoints/sanity_fw_stack/{ckpt_step_100.pt, ckpt_step_200.pt}`. Instance terminated post-run.
  - Current working install recipe (cu121):
    ```
    pip install --no-cache-dir numpy==1.26.4
    pip install --upgrade --no-cache-dir torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
    pip install --no-cache-dir sentencepiece boto3 pyarrow
    pip install --no-cache-dir titans-pytorch==0.5.3 --no-deps
    pip install --no-cache-dir einops==0.8.2 einx==0.4.2 hyper-connections==0.4.9 axial-positional-embedding==0.3.12 assoc-scan==0.0.4 ema-pytorch==0.7.9 tqdm fire loguru orjson tensordict==0.11.0 x-transformers==2.17.7 rotary-embedding-torch==0.8.9 ninja pyvers cloudpickle frozendict --no-deps
    ```
  - Next action: launch a longer training run from the same config/tokenizer using this torch stack; checkpoints will sync to `s3://alix-ai-ml-staging-data/titan/checkpoints/`.

- 2026-03-24: GPT-small bring-up run (SUCCESS; SSM stayed `InProgress` because the instance shut down immediately after training).
  - SSM command: `f0f4c128-7a7b-4013-a3a1-751c1893862c` on `i-095a84b978335b4f9` (TimeoutSeconds extended for long runs).
  - Config: `configs/config_gpt_small.yaml` (variant `gpt`, d=768, L=12, heads=12, ff_mult=4, max_seq_len=2048; train seq_len 1024, batch 2, warmup 2k, max_steps 8000, save/eval every 1000, token cap 200M for bring-up).
  - Stack: `torch==2.5.1+cu121` plus the extra deps above (tensordict/einx/pyvers/cloudpickle/frozendict, etc.).
  - Outcome: ran 0→8000 steps; checkpoints saved locally and synced to the S3 prefix passed via `--s3-checkpoint-uri` (look under `s3://alix-ai-ml-staging-data/titan/checkpoints/` for the run folder and `ckpt_step_8000.pt`). Instance shutdown prevented SSM from reporting `Success`; rely on S3 artifacts to confirm completion.
  - Monitoring note: the zsh status loop must not use the reserved `status` variable—use `job_status` instead (snippet below).
- 2026-03-24: SSM monitoring snippets (reusable).
  - Status loop (zsh-safe):
    ```bash
    while true; do
      job_status=$(AWS_PROFILE=experimental-admin \
        aws ssm list-command-invocations \
          --command-id <cmd-id> \
          --no-cli-pager \
        | jq -r '.CommandInvocations[0].Status // "None"')
      [[ "$job_status" != "InProgress" && "$job_status" != "None" ]] && { echo "Status: $job_status"; break; }
      sleep 10
    done
    ```
  - One-shot check:
    ```bash
    AWS_PROFILE=experimental-admin \
      aws ssm get-command-invocation \
        --command-id <cmd-id> \
        --instance-id <instance-id> \
        --no-cli-pager \
      | jq -r '.Status, .ResponseCode'
    ```
  - Reminder: avoid in-script `shutdown`; if unavoidable, add a short `sleep` after sync so SSM can report completion.

### OpenClaw compatibility notes
- OpenClaw has first-class vLLM support via OpenAI-compatible `/v1` endpoints (`docs/providers/vllm.md`). It can auto-discover models from `/v1/models` when `VLLM_API_KEY` is set (any value if no auth).
- Serving expectations: stable model id (e.g., `/model`), OpenAI surface at `http://127.0.0.1:8010/v1` (or tunneled), bearer auth optional; we can add a simple header or reverse-proxy if desired.
- Explicit provider config (if not using auto-discovery):
  ```json5
  {
    "models": {
      "providers": {
        "vllm": {
          "baseUrl": "http://127.0.0.1:8010/v1",
          "apiKey": "${VLLM_API_KEY}",
          "api": "openai-completions",
          "models": [
            { "id": "/model", "name": "Local vLLM", "contextWindow": 2048, "maxTokens": 1024 }
          ]
        }
      }
    }
  }
  ```
- Training impact: none. Keep HF-compatible exports (config/tokenizer/safetensors with `architectures`/`auto_map`) and a consistent chat template. Update `contextWindow`/`maxTokens` when we raise seq length. Auth/fronting to be decided (bearer or proxy), but not required for local testing.

### Recent ops (2026-03-26)
- Disk exhaustion fix:
  - Deleted older checkpoints (kept: `gpt_small_50k_resume_20260324234903`, `gpt_small_sft_filtered_20260326005546`, `gpt_small_sft_filtered_lowlr_20260326011212`, and new low-LR resume artifacts).
  - Expanded root EBS to 1TB and grew root FS (`/` now ~969G, ~840G free).
- SSM recovery:
  - Reinstalled/restarted SSM agent (snap) after freeing space; SSM now healthy.
  - DNS issues resolved by resetting `/etc/resolv.conf` (1.1.1.1, 8.8.8.8).
- Guardrails added to SFT command params:
  - `params_sft.json`: logs `df -h`, aborts if root free <100G, shows top checkpoint sizes, S3 sync enabled.
  - `params_sft_run.json` (low-LR resume): logs `df -h`, aborts if root free <50G, shows top checkpoint sizes (no S3 sync in this file).
- Low-LR resume (no S3 sync) from `gpt_small_sft_filtered_lowlr_20260326011212/ckpt_sft_step_900.pt`:
  - Run: 600 steps, lr=3e-5; checkpoints at steps 300 and 600 under `gpt_small_sft_filtered_lowlr_resume_20260326180710/`.
  - Eval ppl: ~139.7 @300; ~137.3 @600 (still incoherent outputs in smoke/chat).
- Eval on `ckpt_sft_step_600` confirmed health/chat works but generations remain poor.
- TODO: add SSH key for direct `ubuntu` SSH (currently relying on EC2 Instance Connect/SSM).

- 2026-03-26: Pretrain plan reset for GPT-small and disk guardrails
  - Reason for change: SFT runs from scratch GPT-small remain incoherent; eval perplexity ~137–152 at 600–1500 steps (low-LR resumes included), pointing to an undertrained base and mixed-format SFT data. More SFT at low LR will not fix this without strengthening the base.
  - New base pretraining run (config updated in `configs/config_gpt_small.yaml`):
    - Train: lr `3e-4`, lr_min `3e-5`, warmup `3k`, cosine decay, weight_decay `0.1`, max_steps `40k`, save/eval every `2k`, grad_clip `1.0`, betas `[0.9, 0.98]`, eps `1e-8`, `grad_accum_steps=32`.
    - Data caps lifted for full-stream: `max_tokens=1,000,000,000`, `max_tokens_val=5,000,000`; tokenizer/path unchanged.
    - Token math (batch=2, seq=1024, grad_accum=32): `65,536` effective tokens/optimizer-step. At `40k` steps this is ~`2.62B` tokens (crosses the ~2.3B minimum target for a ~117M model).
    - Gating: do not SFT until base val ppl is in a reasonable band (aim <~50, push toward ~30). Keep checkpoints at ~20k and ~40k as SFT starting points.
    -  Once base ppl < 50: begin NoProp parallel-layer training experiments. See IMPLEMENTATION_NOPROP_EXPERIMENTS.md
    - Sanity (required before long pretrain): run tiny-shard overfit (`~200k` tokens, `200–500` steps) and verify train loss can collapse toward ~`1` on same train/val shard. Use `configs/config_gpt_small_sanity_overfit.yaml`.
      - Example local command:
        - `python model_training/titanProject/train.py --config model_training/titanProject/configs/config_gpt_small_sanity_overfit.yaml --device cuda --log-every 20 --data-log-every-lines 10000`
  - Disk safety: checkpoint writers now skip saves when free space is below a threshold (default 20 GiB) to avoid filling instance disks; applies to `train.py` and `finetune_sft.py` (S3 sync is also skipped when a save is skipped).

### SSM execution timeout fix (2026-03-31)
- Documented root cause and fix under **SSM Run Command: execution timeout** (above); details in `.cursor/docs/ssm_timeout_fixes.md`.
- `gpt_small_*` SSM launch scripts now pass **`executionTimeout`** in `send-command` parameters; test harness `scripts/aws_commands/ssm_timeout_sleep_test.sh` with **`SSM_LONG_VERIFY=1`** (~70m) completed **`Success`**, confirming long shells no longer die at the default 3600 s cap when parameters are set correctly.

## Final Status (2026-04-16) — Side-Quest Complete

### 40k-step pretraining completed
- Run ID: `gpt_small_pretrain_20260414162538`
- Instance: `i-095a84b978335b4f9` (g5.xlarge, spot)
- Duration: ~40k optimizer steps over ~24h wall clock
- Final metrics: train loss **2.8660**, val loss **3.3644**, val perplexity **28.91**
- Tokens seen: ~2.62B (exceeded 2.3B target)
- Checkpoints: every 2k steps, all synced to `s3://alix-ai-ml-staging-data/titan/checkpoints/gpt_small_pretrain_20260414162538/`
- Perplexity gate (**<50**) passed — SFT was authorized

### SFT pilot completed
- Base checkpoint: `ckpt_step_40000.pt` from the 40k pretrain run
- Instance: `i-038216b02ecd17662` (g6.2xlarge, spot) — terminated after completion
- Duration: 3000 SFT steps
- Config: lr=5e-5 cosine, warmup=200, seq_len=1024, batch=2, grad_accum=8
- Data: ~27k train / ~1.6k val examples (OASST1 + OpenHermes + SlimOrca + GSM8K)
- Final eval loss: ~3.19
- Checkpoints: `s3://alix-ai-ml-staging-data/titan/checkpoints/gpt_small_sft_20260419/ckpt_sft_step_{1000,2000,3000}.pt`

### Qualitative evaluation
- SFT improved format awareness and reply structure vs. raw pretrained base
- Fundamental limits remain: repetition, factual errors, shallow reasoning, math failures
- Root cause: 117M params is insufficient for instruction-following; SFT cannot teach knowledge the model lacks
- Verdict: **pipeline validated, model size is the bottleneck**

### AWS resource cleanup
- SFT spot instance `i-038216b02ecd17662` terminated
- Three stopped instances remain (`i-095a84b978335b4f9`, `i-041856a8d4276ce06`, `i-079d25bc4d7483504`) — tracked in CLA-133 for termination and EBS audit
- S3 data lifecycle rule: `titan/data/` transitions to STANDARD_IA after 30 days

### Linear disposition
- CLA-33 (program tracker): Done
- CLA-36 (Phase 3 long-run + gating): Done
- CLA-38 (SFT perplexity gate): Done — gate met, SFT executed
- CLA-135 (stage SFT recipe): Done — recipe executed
- CLA-35 (AWS orchestration hardening): Done
- CLA-133 (terminate stale instances): remains open for cleanup
- CLA-134 (NoProp experiments): remains planned for future work
