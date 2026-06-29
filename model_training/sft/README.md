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

Pinned in `requirements.txt`, validated by a real Qwen2.5-0.5B smoke fine-tune
on Python 3.11 (torch 2.12.1, transformers 5.12.1, trl 1.7.0, peft 0.19.1,
datasets 5.0.0, accelerate 1.14.0). Set up the venv with:

```bash
uv venv --python 3.11 model_training/sft/.venv
uv pip install --python model_training/sft/.venv/bin/python \
    torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu   # or default index for CUDA
uv pip install --python model_training/sft/.venv/bin/python -r model_training/sft/requirements.txt
```

The SageMaker submit-side deps (`sagemaker<3`, `boto3`, `botocore[crt]`) live
with the SageMaker tooling under `infra/sagemaker/`.

## Testing

```bash
pytest model_training/sft/tests/
```

64 wiring tests (green on the system interpreter, no ML stack needed) plus one
opt-in end-to-end integration test. The wiring suite verifies config validation,
data handling, the cfg→TRL/PEFT kwargs mapping, and backend orchestration
**without** importing trl/peft/transformers — heavy construction sits behind seam
functions (`_load_model_and_tokenizer`, `_make_trainer`, `_make_estimator`,
`_resolve_role`) that tests monkeypatch.

The integration test (`test_smoke_integration.py`) runs a real Qwen2.5-0.5B LoRA
fine-tune. It's gated (skips unless `SFT_RUN_SMOKE=1` and trl/peft importable):

```bash
SFT_RUN_SMOKE=1 model_training/sft/.venv/bin/python -m pytest \
    model_training/sft/tests/test_smoke_integration.py -v
```

## Status

- **Done** (Vikunja #911): config, data, train, three training-compute backends,
  pinned + validated dependencies, and a real smoke fine-tune that writes a LoRA
  checkpoint. TRL kwarg confirmed as `max_length` (TRL 1.x).
- **Open**:
  - SageMaker framework versions in `backends/sagemaker.py` are not yet validated
    against a live training job — the pinned stack (transformers 5.x) predates
    any HuggingFace DLC, so the first cloud run needs a matching DLC or a custom
    `image_uri`.
  - **Phase 2** (Vikunja #912): the `serving/` layer to deploy the checkpoint
    (SageMaker LMI / vLLM / Ollama).
