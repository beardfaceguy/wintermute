## PLAN — AWS hosting for model training

### Quick project recap (Titans small prototype)
- Code: `model_training/titanProject/` (train.py, model.py, data.py, configs/*).
- Tokenizer/data: `bpe_50k_bf.model`; TinyStories sampled train/val at `model_training/LLM/data/tinystories_sampled/`.
- Config in use for baseline: `configs/config_baseline_nomem.yaml` (MAC variant, dim 384, depth 5, heads 6, seq_len 512, batch 6, lr 5e-4, cosine decay, 4k steps, no memory tokens).
- Checkpoints: only `ckpt_step_4000.pt` (memory-enabled model) exists; no clean baseline checkpoint yet.
- Training notes: added `--resume` to train.py (restores model/opt/scaler). Baseline run was mid-way; stopped. Run log in terminal files under `.cursor/projects/Users-beardface-lab-wintermute/terminals/`.
- Key scripts: train (`train.py`), generate (`generate.py`), memory eval (`memory_test_eval.py`), finetune QA (`finetune_memory_qa.py`).
- Local hardware: M3/MPS 16 GB unified memory (slow for long runs).
- AWS accounts: `experimental` created (ID `491794274773`, email `patrick.clawson+aws-experimental@meetalix.com`). Use this account for training once SSO/access is wired.

Context: local runs are on M3/MPS (16 GB unified). AWS profile `225079546399_AdministratorAccess` is available, region `us-east-1`.

### GPU options available (us-east-1)
- G5 (A10G 24 GB): g5.xlarge/2xlarge/4xlarge/8xlarge/12xlarge/16xlarge/24xlarge/48xlarge.
- G6 (NVIDIA L4 24 GB).
- G4dn (T4 16 GB).
- P3 (V100 16 GB).
- P5 (H100 80 GB) — expensive/overkill for current tiny model.

### Pricing (g5.xlarge, Linux, us-east-1)
- On-demand: ~$1.006/hr.
- Spot (recent samples):
  - us-east-1f: ~$0.4009/hr (lowest)
  - us-east-1d: ~$0.4089/hr
  - us-east-1a: ~$0.4134/hr
  - us-east-1c: ~$0.4200/hr
  - us-east-1b: ~$0.4545/hr

### Recommended target
- Use `g5.xlarge` (1× A10G 24 GB) in AZ `us-east-1f` for best spot price; fall back to 1d/1a/1c or on-demand if capacity is tight.
- If cost is critical, `g4dn.xlarge` is cheaper but slower and only 16 GB vRAM.
- If you need similar cost but newer GPU, compare `g6.xlarge` (L4 24 GB) prices.

### Next steps (CLI-ready)
1) Confirm/align with admins on region/AZ usage and quotas.
2) Launch spot in cheapest AZ (example for us-east-1f):
   - `aws ec2 request-spot-instances --instance-count 1 --type one-time --launch-specification '{"ImageId":"<AMI>","InstanceType":"g5.xlarge","Placement":{"AvailabilityZone":"us-east-1f"},"KeyName":"<key>","SecurityGroupIds":["<sg>"]}'`
3) Or launch on-demand:
   - `aws ec2 run-instances --image-id <AMI> --instance-type g5.xlarge --placement AvailabilityZone=us-east-1f --key-name <key> --security-group-ids <sg>`
4) Sync code/data or point to S3; run training.
5) Track spend; consider shutting down between runs.

### Latest AMI (PyTorch, Ubuntu 22.04, us-east-1)
- `ami-0ad8dd83d01a01d3a` — Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04) 20260118

### Security group for training
- Name: `alix-pc-llm-model-training` (SG ID: `sg-0bec109715d614af7`, VPC `vpc-08b77ec258b3875af`)
- Ingress: TCP 22 from `23.93.208.154/32` (“SSH from current IP”)
- Add ports (e.g., 8888/6006) or widen CIDR if your IP changes/you need broader access.

### Key pair
- Created key pair `alix-pc-llm-training-key` (us-east-1). Private key saved locally at `~/lab/wintermute/alix-pc-llm-training-key.pem` (chmod 600). Keep it safe; rotate if exposed.
- Moved to `~/.ssh/alix-pc-llm-training-key.pem` and added SSH config entry:
  - Host `alix-llm-ec2`
  - `User ubuntu`
  - `IdentityFile ~/.ssh/alix-pc-llm-training-key.pem`
  - Set `HostName` to the EC2 public IP when launched; `IdentitiesOnly yes`

### Notes
- Spot availability varies; keep a fallback AZ or on-demand.
- Data locality: keep S3/ECR in us-east-1 to avoid cross-region egress.
- Tag instances for cost tracking.

### Rough time/cost estimate (current small Titans run: 4k steps, seq_len 512, batch 6)
- Local M3/MPS: ~2–2.8 hours (based on ~2–2.5s/step).
- g5.xlarge (A10G): expect ~3–5× faster (range 2–6×):
  - Time: ~0.4–1.4 hours (most likely ~0.55–1.0h).
  - Cost spot (~$0.40/hr in us-east-1f): ~$0.20–$0.60/run.
  - Cost on-demand (~$1.01/hr): ~$0.40–$1.40/run.
Notes: actuals depend on dataloader throughput, host CPU contention, and larger model/batch configs; bigger configs will scale time/cost up proportionally.

### Latest run state (experimental account)
- Instance: on-demand `g5.xlarge` (i-050bce8db858dfa89), AZ `us-east-1f`, SG `sg-05ca8b4be3b26ef52`, key `alix-pc-llm-training-key`.
- Storage: using bind-mount `/opt/dlami/nvme` → `/mnt/data` (~217G free).
- Repo: cloned to `/mnt/data/code/wintermute`.
- Tokenizer: copied `bpe_50k_bf.model` to `/mnt/data/code/wintermute/model_training/LLM/tokenizers/`.
- Last error: running `python train.py --config configs/config_combo_all.yaml --device cuda --log-every 100` failed earlier due to missing tokenizer (fixed); pending rerun after restart.

### Still needed to finalize launch/runbook
- IAM role/instance profile for EC2 with S3 access (specify role name or policy).
- S3 bucket/prefix for code/data/checkpoints (or confirm git clone + download path).
- EBS size/mount point (recommendation below).
- Ports to open beyond SSH (e.g., 8888/6006) and any broader SSH CIDR.
- Tagging schema (Owner/Project/CostCenter/Env).
- Spot vs on-demand policy (spot-first fallback or on-demand only).

### EBS recommendation
- Volume: gp3 300 GB (default 3000 IOPS / 125 MB/s). Bump to 6000 IOPS / 250 MB/s if dataloader or checkpoint write throughput becomes a bottleneck.
- Mount: single volume mounted at `/mnt/data` (or `/data`); place code checkout, virtualenv, cache, and checkpoints there.
- Rationale: TinyStories + checkpoints are modest in size; 300 GB leaves headroom for multiple runs, logs, and future larger samples without paying for overprovisioned storage.

### Ports
- For now, open only SSH (22) from the approved CIDR. TensorBoard/Jupyter can be tunneled over SSH; no additional inbound ports required unless browser access without tunnels is desired later.

### Tagging (proposed)
- `Owner=patrick.clawson`
- `Project=Titan-LLM`
- `Env=staging`
- `CostCenter=ai-ml-training`
- `Purpose=titan-training`
- `Name=titan-train-<env>-<instance>` (e.g., `titan-train-staging-g5xlarge`)

### Spot policy (agreed)
- Strategy: spot-first; fallback to on-demand if spot unavailable after a few attempts.
- AZ order: `us-east-1f` first (cheapest recent), then `1d`, `1a`, `1c`.
- Instance type: `g5.xlarge` (default max price = on-demand).
- Interruption handling: checkpoint to S3 prefix frequently so runs can resume after spot interruption.

### Quotas (align with spot policy)
- Verify EC2 quotas in us-east-1 for:
  - Running On-Demand G/VT family instances (target: >= 1 g5.xlarge = 4 vCPUs; request 8–16 vCPUs buffer if low).
  - Running Spot G/VT family instances (target: >= 1 g5.xlarge = 4 vCPUs; request 8–16 vCPUs buffer if low).
- If either quota is below 4–8 vCPUs, submit a quota increase to cover at least one g5.xlarge for spot plus on-demand fallback.
- Current quotas (us-east-1): On-Demand G/VT vCPUs = **768** (`L-DB2E81BA`), Spot G/VT vCPUs = **64** (`L-3819A6DF`). Both comfortably cover a g5.xlarge (4 vCPUs); no increase needed for this plan.

### Cost guardrails (proposed)
- Expected per-run cost (current config): ~$0.20–$0.60 on spot, ~$0.40–$1.40 on on-demand for a 4k-step run.
- Soft cap per run: stop/fallback if runtime exceeds 2 hours (way above expected) to avoid runaway spend.
- Usage discipline: stop/terminate instances after jobs complete; do not leave on-demand instances idle.
- Billing alert: set a project tag-based monthly alarm at **$50**; if exceeded, pause and re-evaluate the plan.

### Proposed IAM role/profile (pending admin approval; not created yet)
- Role: `alix-llm-training-role`
- Trust: EC2 (`ec2.amazonaws.com`)
- Permissions:
  - Inline S3 policy: allow Get/Put/Delete/List on `arn:aws:s3:::<bucket>` and `arn:aws:s3:::<bucket>/<prefix>/*` (fill in bucket/prefix for code/data/checkpoints)
  - Managed: `AmazonSSMManagedInstanceCore` (SSM/Session Manager)
  - Managed: *(omit for now)* `CloudWatchAgentServerPolicy` (logs/metrics) — if we need persistent off-box logs later, enable this.
- Instance profile: `alix-llm-training-profile` (attach the role; pass `--iam-instance-profile Name=alix-llm-training-profile` at EC2 launch)
- Admin approvals needed: confirm S3 bucket/prefix scope, whether CloudWatch logging is desired, required tags (Owner/Project/CostCenter/Env)

### S3 naming observations & recommendation
- Existing buckets follow `alix-<env>-<purpose>` (qa1–qa6, staging) and `alix-ai-<env>-data` for AI datasets.
- To avoid impacting other teams, create a new bucket, e.g. `alix-ai-ml-staging-data` (or `alix-ai-ml-data` if you prefer non-env-specific) with a dedicated prefix `titan/` and sub-prefixes `code/`, `data/`, `checkpoints/`, `logs/`.
- Required access: read/write to `s3://alix-ai-ml-staging-data/titan/*` (or your final bucket name). Keep scope at the prefix level in the IAM inline policy.

### Onboarding checklist for a new assistant
To get full context, review these files/paths:
- Plan doc (this file): `.cursor/docs/PLAN_AWS_hosting.md`
- Titans project code: `model_training/titanProject/`
  - `train.py`, `model.py`, `data.py`
  - Configs: `model_training/titanProject/configs/` (esp. `config_baseline_nomem.yaml`)
  - Scripts: `generate.py`, `memory_test_eval.py`, `finetune_memory_qa.py`, `memory_test_gen.py`
- Data/tokenizer references:
  - Tokenizer: `model_training/LLM/tokenizers/bpe_50k_bf.model`
  - Data: `model_training/LLM/data/tinystories_sampled/train_sample.txt`, `val_sample.txt`
- Project notes/todo: `model_training/titanProject/todo.md`, `run_log.md`
- Checkpoints: only `model_training/titanProject/ckpt_step_4000.pt` (memory-enabled; no baseline ckpt)
- Terminal logs (if needed): `.cursor/projects/Users-beardface-lab-wintermute/terminals/` for past training output

Instruction: read the above files to fully restore context (code, configs, data paths, and plan). Use the plan doc for AWS setup; use the code/config files for model/training details.
