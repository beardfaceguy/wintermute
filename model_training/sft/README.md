# Host-agnostic SFT pipeline

Supervised fine-tuning for LLMs built on **TRL `SFTTrainer`** with **LoRA/PEFT**.
Model-agnostic: the base model, LoRA params, and hyperparameters all live in a
YAML config. Qwen3-8B is just the default.

Tracked in Vikunja project #176 (LLM SFT and Serving Pipeline), task #911.

## Layout

```
model_training/sft/
  config.py            # dataclass configs + YAML load + validation
  data.py              # chat-JSONL → validate → split → HF Datasets
  train.py             # TRL SFTTrainer entrypoint (local + SageMaker)
  backends/
    base.py            # TrainBackend interface: run(cfg) -> checkpoint_uri
    local.py           # run in-process (tiny-model smoke / local GPU)
    sagemaker.py       # submit a SageMaker Training Job (HF estimator)
  configs/
    qwen3_8b_lora.yaml # default 8B LoRA config (SageMaker paths)
    smoke_local.yaml   # tiny-model CPU smoke config
  tests/               # 63 tests, TDD-first
```

## Input format

Canonical training data is **chat-format JSONL** — one JSON object per line:

```json
{"messages": [
  {"role": "system", "content": "You are a helpful code reviewer."},
  {"role": "user", "content": "<diff + context>"},
  {"role": "assistant", "content": "<review comment>"}
]}
```

Each example must have a non-empty `messages` list and at least one assistant
message. Dataset-specific conversion (e.g. codeJung PR JSONs → this format) is a
**separate** step, not part of this pipeline.

## Usage

### Local (in-process)

```python
from model_training.sft.config import SFTConfig
from model_training.sft.backends.local import LocalBackend

cfg = SFTConfig.from_yaml("model_training/sft/configs/smoke_local.yaml")
ckpt = LocalBackend().run(cfg)   # path to the saved checkpoint
```

or as a script:

```bash
python -m model_training.sft.train --config model_training/sft/configs/smoke_local.yaml
```

### SageMaker Training Job

```python
from model_training.sft.config import SFTConfig
from model_training.sft.backends.sagemaker import SageMakerBackend

cfg = SFTConfig.from_yaml("model_training/sft/configs/qwen3_8b_lora.yaml")
model_data = SageMakerBackend(profile="experimental").run(
    cfg,
    source_dir=".",  # repo root — train.py is the entry point
    config_path="model_training/sft/configs/qwen3_8b_lora.yaml",
)
# model_data → s3://.../output/model.tar.gz
```

`cfg.data.train_path` is the S3 URI of the training JSONL; it's passed as the
SageMaker `train` channel (mounted at `/opt/ml/input/data/train`). The config's
`train_path` should point at that mount for the on-instance run.

## Config reference

| Section | Key | Default | Notes |
|---------|-----|---------|-------|
| model | `base_model` | — (required) | HF id or local path |
| model | `dtype` | `bfloat16` | `bfloat16` / `float16` / `float32` |
| data | `train_path` | — (required) | JSONL path or S3 URI |
| data | `eval_path` | `None` | explicit eval set; else split from train |
| data | `eval_split` | `0.0` | fraction carved from train for eval |
| data | `max_seq_len` | `2048` | truncation length |
| lora | `enabled` | `true` | `false` = full fine-tune |
| lora | `r` / `alpha` / `dropout` | `16` / `32` / `0.05` | |
| lora | `target_modules` | `q,k,v,o_proj` | |
| training | `epochs` | `1.0` | |
| training | `learning_rate` | `2e-4` | |
| training | `per_device_batch_size` | `1` | |
| training | `grad_accum_steps` | `8` | |
| training | `warmup_ratio` | `0.03` | |
| training | `save_steps` / `logging_steps` | `200` / `10` | |
| training | `seed` | `42` | |

## Dependencies

Exact pins are deferred to the phase-C venv (see below); these are the
known-compatible floors:

```
torch>=2.4
transformers>=4.49
trl>=0.12
peft>=0.13
datasets>=2.20
accelerate>=0.34
pyyaml>=6

# SageMaker backend — only to SUBMIT training jobs, not to train locally.
sagemaker<3   # v3 is Alpha with broken LMI imports (see infra/sagemaker/)
boto3
botocore[crt]
```

A pinned, pip-installable `requirements.txt` lands in phase C as its own
reviewed change (kept out of this commit so it doesn't trip the repo's
osv-scanner dependency gate on torch's unpatched CVEs).

## Testing

```bash
pytest model_training/sft/tests/
```

63 tests, all green. The suite is **wiring-only** — it verifies config
validation, data handling, the cfg→TRL/PEFT kwargs mapping, and backend
orchestration **without** importing trl/peft/transformers (mock-and-defer). The
heavy model/trainer construction sits behind seam functions
(`_load_model_and_tokenizer`, `_make_trainer`, `_make_estimator`,
`_resolve_role`) that tests monkeypatch.

## Status / deferred

- **Done**: config, data, train wiring, three training-compute backends — fully
  unit-tested.
- **Deferred to phase C** (Vikunja #911): a pinned venv (Python 3.11) with
  `trl`/`peft`/`transformers` installed to run a real tiny-model smoke fine-tune
  and validate the SageMaker framework versions / TRL kwarg names (note
  `max_seq_length` vs `max_length` in `train.build_sft_kwargs`). The system
  interpreter here is Python 3.14, where those wheels are not yet reliable.
  A pinned `requirements.txt` is added then (kept out of Phase 1 to avoid the
  osv-scanner gate on torch's unpatched CVEs).
- **Phase 2** (Vikunja #912): the `serving/` layer to deploy the resulting
  checkpoint (SageMaker LMI / vLLM / Ollama).
