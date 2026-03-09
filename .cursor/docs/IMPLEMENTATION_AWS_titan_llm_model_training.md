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
- Optional: tune regeneration sample sizes for TinyStories subsets if we want smaller/faster iterations than current regenerated corpus.
- Cost-control follow-up: decide whether to keep or terminate the additional spot runner `i-0b788d7634d3a40c4` after validation.

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
