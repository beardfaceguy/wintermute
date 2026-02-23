# Implementation Plan: AWS Titans LLM Model Training

## Overview
Set up and run Titans small-model training on an EC2 GPU instance (us-east-1), using spot-first with on-demand fallback. All resources are dedicated to avoid impacting existing systems: new IAM role/profile, new S3 bucket/prefix, existing training-dedicated SG, new EBS volume.

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
- Tags: `Owner=patrick.clawson`, `Project=Titan-LLM`, `Env=staging`, `CostCenter=ai-ml-training`, `Purpose=titan-training`, `Name=titan-train-staging-g5xlarge`
- EC2: `g5.xlarge`, spot-first (AZ order: 1f → 1d → 1a → 1c), on-demand fallback
- AMI: `ami-0ad8dd83d01a01d3a` (DL OSS GPU PyTorch 2.7, Ubuntu 22.04, 20260118)
- EBS: gp3 300 GB (can bump IOPS if needed), mount `/mnt/data`
- EBS throughput/Iops: gp3 default 3000 IOPS / 125 MB/s; bump to ~6000 IOPS / 250 MB/s if dataloader or checkpoint writes bottleneck.
- Security group: `alix-pc-llm-model-training` (`sg-0bec109715d614af7`), ingress 22 from `23.93.208.154/32`; widen or add 8888/6006 only if browser access without tunnels is required.
- Key pair: `alix-pc-llm-training-key` in `~/.ssh/alix-pc-llm-training-key.pem` (chmod 600); SSH config entry suggested for Host `alix-llm-ec2`.
- Ports: SSH 22 only (tunnel for TB/Jupyter)
- Data locality: keep S3/ECR in us-east-1 to avoid cross-region egress surprises.
- Cost guardrail: $50/mo alert; per-run soft cap 2h

## Instance choice & pricing notes (from prior plan)
- g5.xlarge (A10G 24 GB) target; spot price samples in us-east-1: 1f ~$0.4009/hr (cheapest), 1d ~$0.4089/hr, 1a ~$0.4134/hr, 1c ~$0.4200/hr, 1b ~$0.4545/hr. On-demand ~$1.006/hr. Set max spot price to on-demand.
- Alternatives: `g4dn.xlarge` is cheaper/slower (T4 16 GB); `g6.xlarge` (L4 24 GB) comparable vRAM, check pricing if capacity tight. P3/P5 overkill for this tiny run unless configs scale up significantly.
- Expected runtime/cost for current 4k-step small run: ~0.4–1.4 hours on g5.xlarge; ~$0.20–0.60 on spot, ~$0.40–1.40 on on-demand (depends on dataloader throughput and config changes).

## Step-by-Step (after admin approval)
1) Create S3 bucket (if not already present)
```bash
AWS_PROFILE=225079546399_AdministratorAccess \
aws s3api create-bucket --bucket alix-ai-ml-staging-data --region us-east-1 --create-bucket-configuration LocationConstraint=us-east-1
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
AZ=us-east-1f
AMI=ami-0ad8dd83d01a01d3a

AWS_PROFILE=225079546399_AdministratorAccess aws ec2 request-spot-instances --instance-count 1 --type one-time \
  --launch-specification "{
    \"ImageId\":\"$AMI\",
    \"InstanceType\":\"g5.xlarge\",
    \"Placement\":{\"AvailabilityZone\":\"$AZ\"},
    \"IamInstanceProfile\":{\"Name\":\"alix-llm-training-profile\"},
    \"KeyName\":\"$KEY\",
    \"SecurityGroupIds\":[\"$SG_ID\"],
    \"BlockDeviceMappings\":[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":300,\"VolumeType\":\"gp3\"}}],
    \"TagSpecifications\":[{\"ResourceType\":\"instance\",\"Tags\":[
      {\"Key\":\"Owner\",\"Value\":\"patrick.clawson\"},
      {\"Key\":\"Project\",\"Value\":\"Titan-LLM\"},
      {\"Key\":\"Env\",\"Value\":\"staging\"},
      {\"Key\":\"CostCenter\",\"Value\":\"ai-ml-training\"},
      {\"Key\":\"Purpose\",\"Value\":\"titan-training\"},
      {\"Key\":\"Name\",\"Value\":\"titan-train-staging-g5xlarge\"}
    ]}]
  }"
```
Fallback: use `aws ec2 run-instances` on-demand with same parameters (and `--placement AvailabilityZone=$AZ`).

4) Instance setup (once running)
- SSH with key or SSM Session Manager.
- Format/mount EBS:
```bash
sudo mkfs -t xfs /dev/sda1
sudo mkdir -p /mnt/data
echo "/dev/sda1 /mnt/data xfs defaults,nofail 0 2" | sudo tee -a /etc/fstab
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
- Launch training runner (spot-first g5.xlarge in us-east-1f, fallback 1d/1a/1c or on-demand).
- Bootstrap instance (format/mount gp3 300 GB to /mnt/data; sync code/data).
- Run baseline training and checkpoint to S3 (sync frequently to survive spot evictions).

## Onboarding checklist (for new agent)
- Plan doc: `.cursor/docs/PLAN_AWS_hosting.md`.
- Titans code: `model_training/titanProject/` (`train.py`, `model.py`, `data.py`).
- Configs: `model_training/titanProject/configs/` (esp. `config_baseline_nomem.yaml`).
- Scripts: `generate.py`, `memory_test_eval.py`, `finetune_memory_qa.py`, `memory_test_gen.py`.
- Data/tokenizer: `model_training/LLM/tokenizers/bpe_50k_bf.model`; data at `model_training/LLM/data/tinystories_sampled/{train_sample.txt,val_sample.txt}`.
- Project notes: `model_training/titanProject/todo.md`, `run_log.md`.
- Checkpoints: `model_training/titanProject/ckpt_step_4000.pt` (memory-enabled; no clean baseline ckpt yet).
- Past terminal logs (if needed): `.cursor/projects/Users-beardface-lab-wintermute/terminals/`.

## Progress Log
- 2026-02-22: Env hook set (`AWS_PROFILE=experimental-admin`, `AWS_DEFAULT_REGION=us-east-1` via `.envrc`). New bucket created: `alix-ai-ml-staging-data`. IAM role `alix-llm-training-role` (EC2 trust) with inline S3 RW to `s3://alix-ai-ml-staging-data/titan/*` and `AmazonSSMManagedInstanceCore` attached. Instance profile `alix-llm-training-profile` created and linked to the role.
- 2026-02-22: Verified AWS CLI/profile on workstation (`491794274773_AdministratorAccess` works). No `g5.xlarge` instances currently running (describe-instances returned empty). Expected SG `alix-pc-llm-model-training` / `sg-0bec109715d614af7` is not present in this account/region; will need to create a new SG in us-east-1 (SSH 22 from approved /32, tunnel TB/Jupyter) before launch. Next session (Linux): create SG, then launch spot g5.xlarge with the existing IAM profile/bucket/tag settings.
