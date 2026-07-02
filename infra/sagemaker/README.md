# Hosting LLMs on SageMaker

Reusable runbook for deploying a Hugging Face LLM to a SageMaker real-time
endpoint in the `experimental` AWS account, using the LMI (Large Model
Inference) container.

First proven with `huihui-ai/Qwen3-8B-abliterated` (Vikunja #889). This guide
generalizes that work so any HF model can be hosted the same way.

> **The deploy tooling now lives in `serving/sagemaker.py`** (the
> `SageMakerServeBackend`), part of the host-agnostic serving layer. The old
> `infra/sagemaker/deploy_qwen3.py` script has been retired. This doc remains the
> operational runbook (account, quota, sizing, troubleshooting).

---

## TL;DR

```bash
# from the repo root
aws sso login --profile experimental

# deploy any HF model (or an s3:// artifact) — pass --model, no code edits
python3 -m serving.sagemaker --profile experimental --model huihui-ai/Qwen3-8B-abliterated

# delete when done (~$1.04/hr while running)
python3 -m serving.sagemaker --profile experimental --delete <endpoint-name>
```

Programmatic use:

```python
from serving.sagemaker import SageMakerServeBackend
handle = SageMakerServeBackend(profile="experimental").deploy("huihui-ai/Qwen3-8B-abliterated")
# ... handle.endpoint_name ...
SageMakerServeBackend(profile="experimental").delete(handle)
```

---

## Account / environment

| Item | Value |
|------|-------|
| AWS account | `491794274773` (SSO profile `experimental`) |
| Region | `us-east-1` |
| IAM execution role | `arn:aws:iam::491794274773:role/SageMakerExecutionRole` (auto-created by the script if missing) |
| Container | LMI V20 — `763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.36.0-lmi20.0.0-cu128-v1.0` (vLLM 0.15.1) |
| Default instance | `ml.g5.2xlarge` — 1× A10G, 24 GB VRAM, ~$1.04/hr |
| g5.2xlarge quota | **2** in the experimental account — see "Quota" below |

---

## Prerequisites

1. **AWS SSO login** (credentials expire — re-run when calls start failing):
   ```bash
   aws sso login --profile experimental
   ```

2. **Python deps** (pin sagemaker to v2 — see "SDK version" below):
   ```bash
   pip install 'sagemaker<3' boto3 'botocore[crt]'
   ```
   `botocore[crt]` is required for the SSO credential provider; without it you
   get `MissingDependencyException: Using the login credential provider
   requires an additional dependency`.

3. **Hugging Face token** (needed for gated models like Qwen3-abliterated).
   Accept the model terms on its HF page first, then:
   ```bash
   pip install 'click>=8.4.0'   # older click breaks the hf CLI
   hf auth login                # a read-only token is sufficient
   ```
   This saves the token to `~/.cache/huggingface/token`, where the deploy
   script picks it up automatically. You can also pass `--hf-token hf_xxx`
   or set `HF_TOKEN` in the environment.

---

## Deploying a different model

Pass `--model <hf-id-or-s3-uri>` and `--instance-type` on the CLI — no code
edits. To tweak inference settings, pass `env_extra` to `deploy()` (or extend
`build_serving_env` in `serving/sagemaker.py`). The defaults are:

```python
# serving/sagemaker.py — _BASE_ENV
TENSOR_PARALLEL_DEGREE = "max"          # shard across all GPUs on the instance
OPTION_DTYPE = "bf16"                    # fp16/bf16 — half precision
OPTION_MAX_MODEL_LEN = "8192"            # context window cap
OPTION_MAX_ROLLING_BATCH_SIZE = "8"      # concurrent requests
OPTION_GPU_MEMORY_UTILIZATION = "0.90"
```

`HF_MODEL_ID` is set from `--model` (HF ids only; S3 artifacts are mounted).
The HF token is injected automatically when found (`--hf-token`,
`~/.cache/huggingface/token`, or `HF_TOKEN`).

### Instance sizing

Rough rule: half-precision (bf16) weights need ~2 GB VRAM per 1 B params, plus
headroom for KV cache and activations.

| Model size | Min VRAM | Instance | GPUs | ~$/hr |
|-----------|----------|----------|------|-------|
| ≤8 B | 24 GB | `ml.g5.2xlarge` | 1× A10G | 1.04 |
| 13–34 B | 48 GB | `ml.g5.12xlarge` | 4× A10G | 5.67 |
| 70 B | 192 GB | `ml.g5.48xlarge` | 8× A10G | 16.29 |
| 70 B (faster) | 320 GB | `ml.p4d.24xlarge` | 8× A100 | 32.77 |

`TENSOR_PARALLEL_DEGREE=max` automatically shards across all GPUs on the
chosen instance, so multi-GPU instances need no extra config.

---

## Lifecycle

```bash
# deploy — model downloads from HF Hub at container start; allow 10–15 min
python3 -m serving.sagemaker --profile experimental --model <hf-id>

# invoke later via boto3 sagemaker-runtime.invoke_endpoint, or
# sagemaker.Predictor(endpoint_name).predict({...})

# delete — endpoints bill per hour whether or not they serve traffic
python3 -m serving.sagemaker --profile experimental --delete <endpoint-name>
```

**Always delete endpoints you are not actively using.** A forgotten
g5.2xlarge is ~$25/day.

---

## Quota

The experimental account is limited to **2** `ml.g5.2xlarge` instances. Each
InService endpoint consumes one. If you hit:

```
ResourceLimitExceeded: ... ml.g5.2xlarge ... current limit of 2
```

list and clean up stale endpoints first:

```bash
aws sagemaker list-endpoints --profile experimental --region us-east-1
aws sagemaker delete-endpoint --profile experimental --region us-east-1 \
    --endpoint-name <stale-endpoint>
```

Request a higher limit via Service Quotas if you need more concurrent
endpoints.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: No module named 'sagemaker.model'` | sagemaker v3 removed `sagemaker.model.Model` | `pip install 'sagemaker<3'` (we pin v2.257.3) |
| `MissingDependencyException` on the login credential provider | SSO needs CRT | `pip install 'botocore[crt]'` |
| `hf` CLI: `TypeError: type 'Choice' is not subscriptable` | click < 8.4 | `pip install 'click>=8.4.0'` |
| `huggingface-cli: command not found` / deprecation notice | old CLI renamed | use `hf auth login`, not `huggingface-cli login` |
| Model download fails / 401 at container start | gated repo, no token | accept terms on the HF page, `hf auth login`, redeploy |
| `ResourceLimitExceeded` | g5.2xlarge quota (2) reached | delete a stale endpoint (see Quota) |

---

## SDK version

We are pinned to **sagemaker v2** (`2.257.3`). v3 was still
`Development Status :: 3 - Alpha` on PyPI as of 2026-07-02 (latest `3.15.0`),
with broken LMI imports.

- Migration reference: [`SAGEMAKER_V2_TO_V3_MIGRATION.md`](./SAGEMAKER_V2_TO_V3_MIGRATION.md)
- Re-evaluate v3 when it reaches Beta/Stable — tracked in Vikunja #888 (due 2026-10-01, Q4 re-check).

---

## Files in this directory

| File | Purpose |
|------|---------|
| `README.md` | This runbook (account, quota, sizing, troubleshooting) |
| `SAGEMAKER_V2_TO_V3_MIGRATION.md` | SDK v2→v3 migration reference |

The deploy tooling moved to `serving/sagemaker.py` (retired: `deploy_qwen3.py`,
`test_deploy_qwen3.py`, `conftest.py`, `endpoint.json`).
