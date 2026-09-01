# Titan pretraining on SageMaker (Stage 1b)

Cloud path for the real Titan run, scaling the local PoC (`../`) past the 8 GB
laptop ceiling. **Validated end-to-end 2026-07-06** with a dry run.

## Headline finding: no BYOC needed for training

Titan trains on SageMaker via **plain script mode** — the stock PyTorch DLC
provides torch+CUDA, `requirements.txt` adds the `titans-trainer` fork, and
`train_entry.py` runs the loop. The custom MAC-Titan architecture (test-time
neural memory + second-order autograd) is just PyTorch code SageMaker runs; it
is **not** a training blocker. The dry run confirmed `titans-trainer` installs
into the DLC **without clobbering** its torch/CUDA.

(Custom infra — BYOC container or EC2 GPU — is only needed to *serve* a trained
Titan: managed LMI/TGI/vLLM won't serve this architecture. Training is fine on
managed SageMaker.)

## Dry-run proof (2026-07-06)

Job `titan-dryrun-2026-07-06-18-09-38-987`, `ml.g5.2xlarge` (A10G 24 GB),
256s billable (~$0.09):

```
[titan] torch 2.3.0 cuda=True gpus=1
[titan] 19670 tokens -> train windows=153
TitansModel: 7.4M params  Using GPU: NVIDIA A10G
Epoch 1 loss 9.03 -> 7.71 (19 steps)   SAGEMAKER_RUN_DONE -> /opt/ml/model/final.pt
Reporting training SUCCESS
```

## Files

- `train_entry.py` — SageMaker entry point. Reads `corpus.txt` + `vocab.json` +
  `merges.txt` from the `training` channel, tokenizes in-container, trains a
  MAC-Titan, writes `final.pt` + tokenizer to `SM_MODEL_DIR` (→ S3).
- `requirements.txt` — `titans-trainer` (fork) + `tokenizers`. **Do not pin
  torch** — the DLC owns it.
- `launch.py` — PyTorch estimator (script mode). `--dry-run` = tiny (256/4, 200
  windows); default = ~50M (512/8), tune toward ~170M for the real run.

## Prereqs (experimental account, us-east-1)

- Role: `arn:aws:iam::491794274773:role/SageMakerExecutionRole`
  (`AmazonSageMakerFullAccess`) — hence use the SageMaker **default bucket**
  `sagemaker-us-east-1-491794274773` (the role's S3 access is scoped to
  `sagemaker-*`).
- `sagemaker` SDK v2 + `boto3` (v3 SDK still Alpha — see #888).
- GPU quota (training jobs): A10G `ml.g5.2xlarge/4x/8x/16x` = 1, `ml.g5.12xlarge`
  (4× A10G) = 1. **No A100/H100** (`p4d`/`p5` = 0 → quota increase needed).

## Run

```bash
# stage input (corpus.txt + tokenizer) to S3, then:
python launch.py --dry-run --s3-input s3://sagemaker-us-east-1-491794274773/titan-poc/dry-run-input/
python launch.py           --s3-input s3://.../<real-corpus-prefix>/   # full run
```

## Scale ceiling on current quota

Single A10G (24 GB) ≈ 3× the local 8 GB → **~170–350M** Titan (covers the
paper's smallest model = the Stage-1b target). `g5.12xlarge` (4× A10G) scales
batch/throughput via `titans-trainer`'s DataParallel but **not** max model size
(DataParallel replicates, doesn't shard). The paper's 760M needs FSDP/sharding
(code work) or an A100 (quota increase).
