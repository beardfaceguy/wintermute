# SFT Pipeline Guide

Reference for agents and contributors preparing domain-specific fine-tuning runs on Wintermute's training infrastructure.

---

## Overview

The SFT (Supervised Fine-Tuning) pipeline takes a pretrained base checkpoint and teaches it to follow instructions and hold conversations in a specific style or domain. It supports single-GPU, multi-GPU (DDP via `torchrun`), and CPU/MPS development modes.

**Key files:**

| File | Purpose |
|------|---------|
| `model_training/titanProject/finetune_sft.py` | SFT training loop (Titan + HF models) |
| `model_training/titanProject/hf_utils.py` | HF model loading, LoRA/QLoRA wrapping, checkpoint I/O |
| `model_training/titanProject/prepare_sft_mix.py` | Multi-source data preparation (built-in sources) |
| `model_training/titanProject/train_utils.py` | Shared utilities (checkpointing, LR schedules, DDP helpers) |
| `model_training/titanProject/generate.py` | Text generation / evaluation (Titan + HF models) |
| `model_training/titanProject/export_to_hf.py` | Export weights to HF-style directory |
| `model_training/titanProject/configs/config_sft_gpt_medium_instruction.yaml` | Reference SFT config (407M) |
| `model_training/titanProject/configs/config_sft_hf_qlora.yaml` | Template QLoRA config for HF models |
| `scripts/aws_commands/gpt_medium_sft_cloudwatch.sh` | AWS launch script for Titan SFT runs |
| `scripts/aws_commands/hf_sft_cloudwatch.sh` | AWS launch script for HF model SFT |

---

## Data Format

The SFT pipeline reads plain text files where **each line is one training sample**. Four formats are supported and can be **mixed freely in the same file** — the parser auto-detects the format of each line.

### Format 1: HF Messages JSONL (recommended)

The Hugging Face / OpenAI standard. One JSON object per line with a `messages` array of `role`/`content` dicts. This is the most widely used format on the Hugging Face Hub and is directly compatible with datasets downloaded via the `datasets` library.

```json
{"messages": [{"role": "user", "content": "What is SQL injection?"}, {"role": "assistant", "content": "SQL injection is an attack where malicious SQL statements are inserted into input fields..."}]}
```

Multi-turn conversations are supported — include all turns in the array:

```json
{"messages": [{"role": "system", "content": "You are a cybersecurity expert."}, {"role": "user", "content": "What is SQL injection?"}, {"role": "assistant", "content": "SQL injection is an attack..."}, {"role": "user", "content": "How do I prevent it?"}, {"role": "assistant", "content": "Use parameterized queries..."}]}
```

Rules:
- Each message must have `role` and `content` keys.
- Supported roles: `user`, `assistant`, `system`.
- The model trains to predict the **last assistant turn**. All prior turns become the prompt (masked during training).
- System messages are included in the prompt as `System: <content>`.

### Format 2: ShareGPT JSONL

The format used by many popular datasets like OpenHermes-2.5 and SlimOrca. One JSON object per line with a `conversations` array using `from`/`value` keys.

```json
{"conversations": [{"from": "human", "value": "What is SQL injection?"}, {"from": "gpt", "value": "SQL injection is an attack..."}]}
```

Rules:
- Supported role names: `human`/`user`/`prompter` (mapped to user), `gpt`/`assistant`/`bot`/`chatbot` (mapped to assistant).
- Also accepts `conversation` (singular) as the key name.
- Multi-turn supported, same masking behavior as HF messages.

### Format 3: Alpaca JSONL

One JSON object per line with `instruction`, optional `input`, and `response` (or `output`) fields.

```json
{"instruction": "Explain what SQL injection is.", "input": "", "response": "SQL injection is an attack..."}
{"instruction": "Explain what SQL injection is.", "input": "", "output": "SQL injection is an attack..."}
```

Rules:
- `instruction` is required and must be non-empty.
- Either `response` or `output` must be present and non-empty. If both exist, `response` takes precedence.
- `input` is optional (use empty string `""` if not needed).
- Automatically wrapped in the Alpaca instruction template.

### Format 4: Chat Text

Our original format. One plain text line per sample with `User:` and `Assistant:` markers.

```
User: What is the capital of France? Assistant: The capital of France is Paris.
```

Rules:
- The line must contain both `User:` and `Assistant:` markers exactly once.
- Everything up to and including `Assistant:` becomes the prompt (masked).
- Everything after `Assistant:` becomes the response (trained on).
- Newlines within text should be replaced with spaces.

### Which format to use?

**Use HF Messages format** for new datasets. It is the industry standard, supported by TRL/SFTTrainer, OpenAI fine-tuning, and the vast majority of datasets on the Hugging Face Hub. It also supports multi-turn conversations and system prompts natively.

If you are downloading an existing dataset from Hugging Face, you can likely use it directly without conversion — most datasets use either HF messages or ShareGPT format, both of which are supported.

### Data quality guidelines

| Guideline | Why |
|-----------|-----|
| Keep samples under ~400 tokens total (prompt + response) | Longer samples get truncated at `seq_len` (default 1024 tokens). Short, focused samples train more efficiently. |
| Avoid samples where the response is just "Yes" or "OK" | These waste training compute on trivial targets. |
| Remove duplicates | Duplicate samples cause overfitting on repeated patterns. |
| Remove samples with excessive special characters or formatting | HTML tags, markdown tables, etc. tokenize poorly with our BPE tokenizer. |
| Balance your dataset | If 90% of your data is one topic, the model will be biased toward that topic. |
| Include ~10-20% general conversation data | When doing domain-specific SFT from a general SFT checkpoint, mixing in general data prevents catastrophic forgetting. |

### Preparing custom datasets

For domain-specific fine-tuning (e.g., cybersecurity, medical, legal), you typically:

1. Gather raw data (Q&A pairs, documentation, CTF writeups, etc.)
2. Convert each sample to HF messages JSONL format
3. Write one sample per line to a `.txt` or `.jsonl` file
4. Split into train (90%) and val (10%) files
5. Upload to S3 or provide local paths

Example Python snippet for converting a custom dataset:

```python
import json
import random

samples = []
with open("raw_cyber_qa.jsonl") as f:
    for line in f:
        item = json.loads(line)
        q = item["question"].strip()
        a = item["answer"].strip()
        if q and a:
            sample = {"messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]}
            samples.append(json.dumps(sample))

random.shuffle(samples)
split = int(len(samples) * 0.9)

with open("train_cyber.jsonl", "w") as f:
    f.write("\n".join(samples[:split]) + "\n")

with open("val_cyber.jsonl", "w") as f:
    f.write("\n".join(samples[split:]) + "\n")
```

To use a dataset directly from Hugging Face Hub:

```python
import json
import random
from datasets import load_dataset

ds = load_dataset("your-dataset-name", split="train")
samples = []
for row in ds:
    line = json.dumps({"messages": row["messages"]})
    samples.append(line)

random.shuffle(samples)
split = int(len(samples) * 0.9)

with open("train.jsonl", "w") as f:
    f.write("\n".join(samples[:split]) + "\n")

with open("val.jsonl", "w") as f:
    f.write("\n".join(samples[split:]) + "\n")
```

### Using the built-in data preparation script

For general-purpose SFT using standard public datasets, `prepare_sft_mix.py` automates downloading, filtering, and formatting:

```bash
python3 model_training/titanProject/prepare_sft_mix.py \
  --output-dir /path/to/output \
  --oasst-train-pairs 24000 \
  --oasst-val-pairs 2000 \
  --openhermes-train-pairs 20000 \
  --openhermes-val-pairs 2000 \
  --slimorca-train-pairs 10000 \
  --slimorca-val-pairs 1000 \
  --logic-train-pairs 1000 \
  --logic-val-pairs 200 \
  --seed 42
```

Built-in sources: OASST1, OpenHermes-2.5, SlimOrca, GSM8K.

---

## Configuration

SFT configs are YAML files with three sections: `model`, `train`, and `data`.

### Reference config (407M model)

```yaml
model:
  variant: gpt
  vocab_size: 50000
  dim: 1024
  depth: 24
  heads: 16
  ff_mult: 4
  max_seq_len: 2048

train:
  seq_len: 1024
  batch_size: 2
  grad_accum_steps: 8
  lr: 0.00005
  lr_min: 0.000005
  weight_decay: 0.01
  warmup_steps: 300
  max_steps: 5000
  grad_clip: 1.0
  betas: [0.9, 0.98]
  eps: 1.0e-8
  cosine_decay: true
  save_every: 1000
  eval_every: 500

data:
  train_path: /path/to/train_sft.txt
  val_path: /path/to/val_sft.txt
  tokenizer_path: s3://alix-ai-ml-staging-data/titan/tokenizers/new_bpe_50k/bpe_50k_fw_stack.model
  shuffle_buffer: 100000
```

### Key parameters explained

| Parameter | Recommended Range | Notes |
|-----------|------------------|-------|
| `lr` | 1e-5 to 5e-5 | SFT uses a lower LR than pretraining to avoid catastrophic forgetting. Start with 5e-5. |
| `lr_min` | lr / 10 | Floor for cosine decay. |
| `warmup_steps` | 200-500 | Linear warmup before cosine decay kicks in. |
| `max_steps` | 2000-10000 | Depends on dataset size. Watch for eval loss plateauing. |
| `batch_size` | 2-4 | Per-GPU batch size. Limited by GPU memory. |
| `grad_accum_steps` | 4-16 | Effective batch = batch_size * grad_accum_steps * num_gpus. |
| `seq_len` | 512-1024 | Max tokens per sample. Samples exceeding this are truncated. |
| `save_every` | 500-2000 | Checkpoint interval. Each checkpoint is ~4.9GB (with optimizer state). |
| `eval_every` | 250-500 | Validation loss evaluation interval. |
| `weight_decay` | 0.01-0.1 | Regularization. 0.01 is conservative for SFT. |

### Model section

The `model` section must match the architecture of the base checkpoint you are fine-tuning. Do not change these values unless you are starting from a different model. Current available base models:

| Model | Parameters | Config Values |
|-------|-----------|---------------|
| GPT-Small | 117M | dim=768, depth=12, heads=12 |
| GPT-Medium | 407M | dim=1024, depth=24, heads=16 |

---

## Running SFT

### Local development (CPU/MPS)

```bash
cd model_training/titanProject

python3 finetune_sft.py \
  --config configs/config_sft_gpt_medium_instruction.yaml \
  --ckpt /path/to/base_checkpoint.pt \
  --device cpu \
  --steps 100 \
  --log-every 10 \
  --eval-every 50 \
  --save-every 50
```

### Single GPU

```bash
python3 finetune_sft.py \
  --config configs/config_sft_gpt_medium_instruction.yaml \
  --ckpt /path/to/base_checkpoint.pt \
  --device cuda \
  --steps 5000 \
  --checkpoint-dir /path/to/checkpoints \
  --s3-checkpoint-uri s3://bucket/prefix/
```

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=4 finetune_sft.py \
  --config configs/config_sft_gpt_medium_instruction.yaml \
  --ckpt /path/to/base_checkpoint.pt \
  --steps 5000 \
  --checkpoint-dir /path/to/checkpoints \
  --s3-checkpoint-uri s3://bucket/prefix/
```

When using `torchrun`, the script auto-detects DDP mode and:
- Distributes data across GPUs using `DistributedSampler`
- Wraps the model in `DistributedDataParallel`
- Auto-scales `grad_accum_steps` by dividing by `world_size` (disable with `--no-accum-scale`)
- Only saves checkpoints from rank 0
- Reduces metrics across all ranks for accurate logging

### AWS remote execution

See `scripts/aws_commands/gpt_medium_sft_cloudwatch.sh`. Key environment variables:

```bash
INSTANCE_ID=i-xxxxx \
BASE_CKPT_S3_URI=s3://alix-ai-ml-staging-data/titan/checkpoints/.../ckpt_step_124000.pt \
STEPS=5000 \
LR=5.0e-05 \
bash scripts/aws_commands/gpt_medium_sft_cloudwatch.sh
```

The launch script handles: code bundle deployment, dependency installation, tokenizer/checkpoint download, SFT data preparation, config generation, detached training execution, checkpoint sync to S3, and instance self-stop on completion.

### Full CLI reference

```
finetune_sft.py arguments:

  --config PATH          YAML config file (model arch + training params + data paths)
  --ckpt PATH            Titan checkpoint (local path, S3 URI, or hf://gpt2). Mutually exclusive with --hf-model.

  HuggingFace model arguments (optional):
  --hf-model MODEL_ID   HuggingFace model ID (e.g. meta-llama/Meta-Llama-3-8B)
  --qlora               QLoRA: 4-bit quantized base + LoRA (default when --hf-model is used)
  --lora                LoRA: fp16 base + LoRA adapters
  --no-lora             Full fine-tuning (requires multi-GPU or large GPU)
  --lora-rank INT       LoRA rank (default 16)
  --lora-alpha INT      LoRA alpha (default 32)
  --lora-dropout FLOAT  LoRA dropout (default 0.05)
  --lora-targets MODS   Comma-separated target modules (auto-detected if omitted)
  --chat-template       Use model's native chat template for tokenization

  General arguments:
  --device DEVICE        Device override: auto, cpu, mps, cuda (ignored under torchrun)
  --steps N              Total training steps
  --log-every N          Print training loss every N steps
  --eval-every N         Run validation every N steps
  --eval-batches N       Number of val batches per evaluation
  --save-every N         Save checkpoint every N steps
  --checkpoint-dir PATH  Directory for checkpoint files
  --s3-checkpoint-uri S3 Sync checkpoints to this S3 prefix after each save
  --lr FLOAT             Override learning rate from config
  --grad-accum-steps N   Override gradient accumulation steps
  --no-accum-scale       Disable automatic grad_accum scaling by world_size in DDP
  --min-free-gb FLOAT    Minimum free disk (GiB) required to write a checkpoint (default 20)
  --amp / --no-amp       Force enable/disable mixed precision (auto-detected by default)
  --aws-bin PATH         Path to aws CLI binary (default: aws)
```

---

## Evaluating Results

### Quick generation test

```bash
python3 generate.py \
  --config /path/to/config.yaml \
  --ckpt /path/to/ckpt_sft_step_5000.pt \
  --prompt "User: Your test question here. Assistant:" \
  --max-new 200 \
  --top-k 40 \
  --temperature 0.7 \
  --prompt-family chat
```

Always use `--prompt-family chat` for models trained on chat text format. This enables stop-string detection at `User:` and `Assistant:` boundaries.

### What to look for

| Metric | Healthy Range (407M) | Notes |
|--------|---------------------|-------|
| Training loss | 1.2 - 2.0 (final) | Should decrease over training |
| Eval loss | 1.8 - 2.2 (final) | Should decrease, then plateau |
| Eval perplexity | 6 - 9 (final) | Lower is better. `ppl = exp(eval_loss)` |
| Response coherence | Subjective | Does it answer the question asked? |
| Domain accuracy | Subjective | Are domain-specific facts correct? |

If eval loss starts increasing while train loss keeps dropping, the model is overfitting. Reduce `--steps` or add more training data.

### Exporting weights only

Full training checkpoints include optimizer state (~4.9GB for 407M). To save just the model weights (~1.6GB for 407M):

```python
import torch, os

ckpt = torch.load("ckpt_sft_step_5000.pt", map_location="cpu", weights_only=False)
weights_only = {"model": ckpt["model"], "step": ckpt["step"]}
torch.save(weights_only, "model_weights_only.pt")
```

### Exporting to Hugging Face format

```bash
python3 export_to_hf.py \
  --config configs/config_sft_gpt_medium_instruction.yaml \
  --ckpt ckpt_sft_step_5000.pt \
  --out ./hf_export/
```

---

## HuggingFace Model Fine-Tuning (7B+ Models)

The SFT pipeline supports fine-tuning any HuggingFace causal LM (Llama 3, Mistral, Qwen, etc.) with three modes: **QLoRA** (4-bit quantized base + LoRA), **LoRA** (fp16 base + LoRA), and **full fine-tuning** (all parameters, multi-GPU).

### Key files

| File | Purpose |
|------|---------|
| `model_training/titanProject/hf_utils.py` | HF model loading, LoRA wrapping, checkpoint save/load |
| `model_training/titanProject/configs/config_sft_hf_qlora.yaml` | Template QLoRA config |
| `scripts/aws_commands/hf_sft_cloudwatch.sh` | AWS launch script for HF SFT |

### Quick start

**QLoRA on a single L4 GPU (g6.2xlarge) — best for experimentation:**

```bash
python3 finetune_sft.py \
  --config configs/config_sft_hf_qlora.yaml \
  --hf-model meta-llama/Meta-Llama-3-8B \
  --qlora \
  --chat-template \
  --steps 3000 \
  --checkpoint-dir ./checkpoints_hf_sft
```

**LoRA on a single GPU (fp16, needs 20+ GB VRAM):**

```bash
python3 finetune_sft.py \
  --config configs/config_sft_hf_qlora.yaml \
  --hf-model meta-llama/Meta-Llama-3-8B \
  --lora \
  --chat-template \
  --steps 3000
```

**Full fine-tuning with multi-GPU DDP (g5.12xlarge with 4x A10G):**

```bash
torchrun --nproc_per_node=4 finetune_sft.py \
  --config configs/config_sft_hf_qlora.yaml \
  --hf-model meta-llama/Meta-Llama-3-8B \
  --no-lora \
  --chat-template \
  --steps 3000
```

### HF-specific CLI arguments

| Argument | Default | Notes |
|----------|---------|-------|
| `--hf-model MODEL_ID` | — | HuggingFace model ID. Mutually exclusive with `--ckpt`. |
| `--qlora` | (default when `--hf-model` used) | 4-bit quantized base + LoRA. Best for single GPU. |
| `--lora` | — | fp16 base + LoRA adapters. |
| `--no-lora` | — | Full fine-tuning. Requires multi-GPU or A100. |
| `--lora-rank INT` | 16 | LoRA rank. Higher = more capacity, more memory. |
| `--lora-alpha INT` | 32 | LoRA alpha (scaling). Typically 2x rank. |
| `--lora-dropout FLOAT` | 0.05 | Dropout on LoRA layers. |
| `--lora-targets MODULES` | auto | Comma-separated target module names. Auto-detects all linear layers if omitted. |
| `--chat-template` | off | Use the model's native chat template (recommended for instruct/chat models). |

### VRAM estimates

| Mode | GPU | Base Model VRAM | Trainable Params | Total VRAM |
|------|-----|----------------|------------------|------------|
| QLoRA (4-bit) | 1x L4 24GB | ~4-5 GB | ~20M (0.3%) | ~8-10 GB |
| LoRA (fp16) | 1x L4 24GB | ~14 GB | ~20M | ~18-20 GB |
| Full fine-tune (fp16) | 4x A10G 24GB | ~14 GB/GPU (sharded) | 7B (100%) | ~20 GB/GPU |

QLoRA on the g6.2xlarge is the sweet spot for experimentation. Full fine-tuning on g5.12xlarge for production-quality runs.

### Chat template tokenization

When `--chat-template` is enabled, the pipeline uses the model's native `tokenizer.apply_chat_template()` to format input. This ensures special tokens (`<|begin_of_text|>`, `[INST]`, `<|im_start|>`, etc.) are placed correctly per model architecture.

The tokenizer handles prompt/response masking automatically: user turns are masked (`-100`) so the model only trains on assistant output.

### Checkpoint format

- **QLoRA/LoRA**: Saves only the adapter weights (~50-200 MB per checkpoint) plus `training_state.pt` with the step counter.
- **Full fine-tuning**: Saves the complete model via `save_pretrained()`.
- Checkpoints are saved as directories: `checkpoints_sft/step_500/`, `checkpoints_sft/step_1000/`, etc.

### Inference with HF models

```bash
# Base HF model
python3 generate.py --hf-model meta-llama/Meta-Llama-3-8B --prompt "Hello" --max-new 200

# With LoRA adapter
python3 generate.py --hf-model meta-llama/Meta-Llama-3-8B --adapter checkpoints_sft/step_3000 --prompt "Hello"

# With chat template formatting
python3 generate.py --hf-model meta-llama/Meta-Llama-3-8B --adapter checkpoints_sft/step_3000 --chat-template --prompt "What is machine learning?"

# Merge adapter for deployment
python3 generate.py --hf-model meta-llama/Meta-Llama-3-8B --adapter checkpoints_sft/step_3000 --merge-adapter --prompt "Hello"
```

### AWS remote execution (HF models)

Use `scripts/aws_commands/hf_sft_cloudwatch.sh`:

```bash
INSTANCE_ID=i-xxxxx \
HF_MODEL_ID=meta-llama/Meta-Llama-3-8B \
FINETUNE_MODE=qlora \
SFT_TRAIN_PATH=s3://alix-ai-ml-staging-data/titan/data/sft/train.jsonl \
SFT_VAL_PATH=s3://alix-ai-ml-staging-data/titan/data/sft/val.jsonl \
STEPS=3000 \
bash scripts/aws_commands/hf_sft_cloudwatch.sh
```

Key environment variables:

| Variable | Default | Notes |
|----------|---------|-------|
| `HF_MODEL_ID` | — | Required. e.g. `meta-llama/Meta-Llama-3-8B` |
| `FINETUNE_MODE` | `qlora` | `qlora`, `lora`, or `full` |
| `SFT_TRAIN_PATH` | — | S3 URI or local path to training data |
| `SFT_VAL_PATH` | — | S3 URI or local path to validation data |
| `NUM_GPUS` | `1` | Number of GPUs (>1 for full fine-tuning) |
| `LORA_RANK` | `16` | LoRA rank |
| `USE_CHAT_TEMPLATE` | `1` | Enable native chat template |

### Recommended open-source 7B base models

| Model | HF ID | Notes |
|-------|-------|-------|
| Llama 3 8B | `meta-llama/Meta-Llama-3-8B` | Strong general performance; requires HF access approval |
| Mistral 7B v0.3 | `mistralai/Mistral-7B-v0.3` | Good balance of quality and speed |
| Qwen 2.5 7B | `Qwen/Qwen2.5-7B` | Strong multilingual and coding ability |

---

## Domain-Specific Fine-Tuning (Forking)

The recommended approach for creating specialized models is to **fork from a general SFT checkpoint**:

```
Pretrained Base (general knowledge)
  └── General SFT (conversational ability)
        ├── Fork A: Cybersecurity SFT
        ├── Fork B: Code Review SFT
        ├── Fork C: Medical SFT
        └── Fork D: Any domain...
```

### Steps to create a domain fork

1. **Prepare domain data** in chat text format (see "Data Format" above).
2. **Mix in 10-20% general SFT data** to prevent catastrophic forgetting. Example: if you have 10,000 cyber samples, add 2,000 general conversation samples.
3. **Create a config** — copy `config_sft_gpt_medium_instruction.yaml` and update `data.train_path` and `data.val_path` to your new data files.
4. **Set the base checkpoint** to the general SFT checkpoint (not the pretrain checkpoint).
5. **Run SFT** with 2,000-5,000 steps (adjust based on dataset size).
6. **Evaluate** with domain-specific prompts.

### Available base checkpoints

| Checkpoint | S3 Location | Use As |
|-----------|-------------|--------|
| GPT-Medium pretrain (step 124K) | `s3://alix-ai-ml-staging-data/titan/checkpoints/gpt_medium_pretrain_20260419232521/ckpt_step_124000.pt` | Base for full SFT (general + domain mixed) |
| GPT-Medium general SFT (step 5K) | `s3://alix-ai-ml-staging-data/titan/checkpoints/gpt_medium_sft_20260430052503/ckpt_sft_step_5000.pt` | Base for domain-specific fork |
| GPT-Medium general SFT weights-only | `model_training/titanProject/saved_models/gpt_medium_407m_sft_5000_weights.pt` | Local copy, weights only (1.6GB) |
| Mistral 7B v0.3 QLoRA adapter | `s3://alix-ai-ml-staging-data/titan/checkpoints/hf_sft_mistral7b_qlora/` | Proven end-to-end (2026-05-01). Merge with base for serving. |

### Tokenizer

All current models use the same tokenizer:

```
s3://alix-ai-ml-staging-data/titan/tokenizers/new_bpe_50k/bpe_50k_fw_stack.model
```

SentencePiece BPE, 50,000 token vocabulary. Do not change the tokenizer without retraining from scratch.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `No usable SFT samples found` | Data file is empty, wrong format, or all samples filtered out | Check that each line matches chat or JSONL format. Verify the file path. |
| `SFT sample must contain both User: and Assistant: markers` | Chat-format line is missing one of the markers | Ensure every line has both `User:` and `Assistant:` |
| Eval loss immediately very high (>5) | Wrong base checkpoint or mismatched model config | Verify the config `model` section matches the checkpoint architecture |
| Training loss not decreasing | LR too low or data quality issues | Try increasing LR to 1e-4. Check data for empty/garbage samples. |
| CUDA OOM | Batch size or seq_len too large for GPU memory | Reduce `batch_size` to 1 and/or `seq_len` to 512. Use `grad_accum_steps` to maintain effective batch size. |
| Checkpoint not saving | Insufficient disk space | Check `--min-free-gb` threshold (default 20GB). The script logs a warning when skipping saves. |
| Model outputs gibberish after SFT | Overfitting or too high LR | Reduce steps, lower LR, add more diverse training data. |
| Good train loss but bad responses | Prompt format mismatch at inference | Use the same prompt format at inference as was used in training data. For chat format: `User: <question> Assistant:` |
