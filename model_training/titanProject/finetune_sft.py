"""
SFT finetuning loop for Titans checkpoints and HuggingFace causal LMs.

Accepts four data formats (auto-detected per line, can be mixed):
  1. HF messages JSONL:  {"messages": [{"role": "user", "content": "..."},  ...]}
  2. ShareGPT JSONL:     {"conversations": [{"from": "human", "value": "..."}, ...]}
  3. Alpaca JSONL:       {"instruction": "...", "input": "", "response"/"output": "..."}
  4. Chat text:          User: <question> Assistant: <answer>

Supports single-GPU, multi-GPU (DDP), and CPU/MPS development:

    # Titan checkpoint (existing workflow)
    python finetune_sft.py --config configs/config_sft.yaml --ckpt ckpt_step_124000.pt

    # HuggingFace model with QLoRA (single GPU)
    python finetune_sft.py --config configs/config_sft_hf_qlora.yaml --hf-model meta-llama/Meta-Llama-3-8B --qlora

    # HuggingFace model with LoRA (single GPU, fp16)
    python finetune_sft.py --config configs/config_sft_hf_qlora.yaml --hf-model meta-llama/Meta-Llama-3-8B --lora

    # HuggingFace model full fine-tuning (multi-GPU)
    torchrun --nproc_per_node=4 finetune_sft.py --config configs/config_sft_hf.yaml --hf-model meta-llama/Meta-Llama-3-8B --no-lora

    # CPU / Apple Silicon dev
    python finetune_sft.py --config configs/config_sft.yaml --ckpt ckpt.pt --device mps
"""

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from model import ModelConfig, build_model, load_model_source
from train_utils import (
    cleanup_distributed,
    cosine_lr,
    get_tokenizer,
    has_min_free_space,
    is_main,
    load_config,
    pick_device,
    reduce_scalar,
    resolve_checkpoint_dir,
    resolve_path,
    save_checkpoint,
    setup_distributed,
    sync_checkpoints_to_s3,
)


def cycle_batches(
    loader: DataLoader,
    sampler: Optional[DistributedSampler] = None,
):
    """Infinite iterator over a dataloader, calling set_epoch on each restart."""
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def _render_instruction_prompt(instruction_text: str, input_text: str = "") -> str:
    instruction = str(instruction_text or "").strip()
    input_block = str(input_text or "").strip()
    if not instruction:
        raise ValueError("Instruction-format SFT sample must include non-empty instruction text")
    prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}"
    )
    if input_block:
        prompt += f"\n\n### Input:\n{input_block}"
    prompt += "\n\n### Response:\n"
    return prompt


def _format_messages_as_chat(messages: list) -> Tuple[str, str]:
    """Convert a list of role/content message dicts into a (prompt, response) pair.

    Supports both HF messages format (role/content) and ShareGPT format
    (from/value).  All turns up to the final assistant turn become the prompt;
    the final assistant content becomes the response.
    """
    normalized: list = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or msg.get("from") or "").strip().lower()
        content = str(
            msg.get("content") or msg.get("value") or msg.get("text") or ""
        ).strip()
        if role in ("user", "human", "prompter"):
            normalized.append(("user", content))
        elif role in ("assistant", "gpt", "bot", "chatbot"):
            normalized.append(("assistant", content))
        elif role == "system":
            normalized.append(("system", content))

    if not normalized:
        raise ValueError("Messages array contains no usable user/assistant turns")

    last_assistant_idx = None
    for i in range(len(normalized) - 1, -1, -1):
        if normalized[i][0] == "assistant":
            last_assistant_idx = i
            break
    if last_assistant_idx is None:
        raise ValueError("Messages array contains no assistant turn")

    prompt_parts: list = []
    for role, content in normalized[:last_assistant_idx]:
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
    prompt_parts.append("Assistant:")
    prompt_text = " ".join(prompt_parts)

    response_text = normalized[last_assistant_idx][1]
    if not response_text:
        raise ValueError("Final assistant turn has empty content")
    return prompt_text, f" {response_text}"


def _split_sft_sample(sample: str) -> Tuple[str, str]:
    """Parse one line of SFT data into (prompt, response).

    Supported formats (auto-detected):
      1. HF messages JSONL:  {"messages": [{"role": "user", "content": "..."},
                                           {"role": "assistant", "content": "..."}]}
      2. ShareGPT JSONL:    {"conversations": [{"from": "human", "value": "..."},
                                                {"from": "gpt", "value": "..."}]}
      3. Alpaca JSONL:      {"instruction": "...", "input": "", "response": "..."}
                            (also accepts "output" as alias for "response")
      4. Chat text:         User: <question> Assistant: <answer>
    """
    sample = sample.strip()
    if not sample:
        raise ValueError("SFT sample must be non-empty")
    if sample.startswith("{"):
        try:
            payload = json.loads(sample)
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSONL SFT sample") from e
        if not isinstance(payload, dict):
            raise ValueError("JSONL SFT sample must be an object")

        # --- HF messages format ---
        if "messages" in payload and isinstance(payload["messages"], list):
            return _format_messages_as_chat(payload["messages"])

        # --- ShareGPT format ---
        conv_key = None
        for key in ("conversations", "conversation"):
            if key in payload and isinstance(payload[key], list):
                conv_key = key
                break
        if conv_key is not None:
            return _format_messages_as_chat(payload[conv_key])

        # --- Alpaca / instruction format ("output" accepted as alias) ---
        instruction_text = str(payload.get("instruction", "")).strip()
        input_text = str(payload.get("input", "")).strip()
        response_text = str(
            payload.get("response") or payload.get("output") or ""
        ).strip()
        if not instruction_text or not response_text:
            raise ValueError(
                "Instruction-format SFT sample must include instruction and response/output"
            )
        return _render_instruction_prompt(instruction_text, input_text), response_text

    if "User:" not in sample or "Assistant:" not in sample:
        raise ValueError("SFT sample must contain both User: and Assistant: markers")
    _, rest = sample.split("User:", 1)
    user_part, assistant_part = rest.split("Assistant:", 1)
    user_text = user_part.strip()
    assistant_text = assistant_part.strip()
    if not user_text or not assistant_text:
        raise ValueError("SFT sample must include non-empty user and assistant text")
    return f"User: {user_text} Assistant:", f" {assistant_text}"


def _get_boundary_tokens(tokenizer: Callable[[str], List[int]]) -> List[int]:
    eos_id = getattr(tokenizer, "eos_id", -1)
    if isinstance(eos_id, int) and eos_id >= 0:
        return [eos_id]
    return tokenizer("\n")


def _forward_logits(model, x: torch.Tensor, hf_mode: bool) -> torch.Tensor:
    """Get logits from either Titan or HF models.

    HF CausalLM models return a CausalLMOutput object, not bare logits.
    """
    out = model(x) if not hf_mode else model(input_ids=x)
    if hasattr(out, "logits"):
        return out.logits
    return out


def _hf_tokenizer_to_adapter(hf_tokenizer):
    """Wrap a raw HuggingFace tokenizer into a TokenizerAdapter for the SFT data pipeline."""
    from train_utils import TokenizerAdapter, _hf_tokenizer_fingerprint

    if hf_tokenizer.pad_token_id is None and hf_tokenizer.eos_token_id is not None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    return TokenizerAdapter(
        encode_fn=lambda text: hf_tokenizer.encode(text, add_special_tokens=False),
        decode_fn=lambda ids: hf_tokenizer.decode(
            ids, clean_up_tokenization_spaces=False, skip_special_tokens=True,
        ),
        tokenizer_fingerprint=_hf_tokenizer_fingerprint(hf_tokenizer),
        tokenizer_source_path=hf_tokenizer.name_or_path,
        eos_id=hf_tokenizer.eos_token_id if hf_tokenizer.eos_token_id is not None else -1,
        pad_id=hf_tokenizer.pad_token_id if hf_tokenizer.pad_token_id is not None else -1,
    )


def _tokenize_with_chat_template(
    hf_tokenizer,
    sample: str,
    seq_len: int,
) -> Optional[Tuple[List[int], List[int]]]:
    """Tokenize a sample using the model's native chat template.

    Returns (input_ids, labels) with proper masking: the user/system prompt
    portion is masked to -100 so the model only trains on assistant output.
    Returns None if the sample can't be processed.
    """
    sample = sample.strip()
    if not sample:
        return None

    try:
        prompt_text, answer_text = _split_sft_sample(sample)
    except ValueError:
        return None

    user_content = prompt_text
    if user_content.startswith("User:"):
        user_content = user_content[len("User:"):].strip()
    if user_content.endswith("Assistant:"):
        user_content = user_content[: -len("Assistant:")].strip()

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer_text.strip()},
    ]

    full_ids = hf_tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )

    prompt_only_ids = hf_tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=True,
        add_generation_prompt=True,
    )
    prompt_len = len(prompt_only_ids)

    max_total = seq_len + 1
    if len(full_ids) > max_total:
        full_ids = full_ids[:max_total]
    if len(full_ids) < 2:
        return None

    x_tokens = full_ids[:-1]
    y_tokens = full_ids[1:]
    mask_len = max(prompt_len - 1, 0)
    labels = [-100] * mask_len + y_tokens[mask_len:]

    if all(l == -100 for l in labels):
        return None

    return x_tokens, labels


class MaskedSFTDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer: Callable[[str], List[int]],
        seq_len: int,
        *,
        log_fn: Callable[[str], None],
        progress_label: str,
        chat_template_tokenizer=None,
    ):
        self.seq_len = seq_len
        self.input_ids: List[List[int]] = []
        self.labels: List[List[int]] = []

        use_chat_tpl = chat_template_tokenizer is not None
        boundary_tokens = _get_boundary_tokens(tokenizer) if not use_chat_tpl else []
        total = 0
        kept = 0
        skipped_empty = 0
        skipped_no_answer_room = 0
        truncated = 0

        with open(Path(path), "r", encoding="utf-8") as f:
            for raw_line in f:
                total += 1
                sample = raw_line.strip()
                if not sample:
                    continue

                if use_chat_tpl:
                    result = _tokenize_with_chat_template(
                        chat_template_tokenizer, sample, seq_len,
                    )
                    if result is None:
                        skipped_empty += 1
                        continue
                    x_tokens, masked_labels = result
                    self.input_ids.append(x_tokens)
                    self.labels.append(masked_labels)
                    kept += 1
                    continue

                try:
                    prompt_text, answer_text = _split_sft_sample(sample)
                except ValueError:
                    skipped_empty += 1
                    continue

                prompt_tokens = tokenizer(prompt_text)
                answer_tokens = tokenizer(answer_text)
                if not answer_tokens:
                    skipped_empty += 1
                    continue

                max_total_tokens = seq_len + 1
                if len(prompt_tokens) >= max_total_tokens:
                    skipped_no_answer_room += 1
                    continue

                available_answer_tokens = max_total_tokens - len(prompt_tokens)
                full_answer_tokens = list(answer_tokens) + list(boundary_tokens)
                if len(full_answer_tokens) > available_answer_tokens:
                    truncated += 1
                    if boundary_tokens and available_answer_tokens >= len(boundary_tokens):
                        answer_body_budget = available_answer_tokens - len(boundary_tokens)
                        full_answer_tokens = answer_tokens[:answer_body_budget] + list(boundary_tokens)
                    else:
                        full_answer_tokens = full_answer_tokens[:available_answer_tokens]

                full_tokens = prompt_tokens + full_answer_tokens
                if len(full_tokens) < 2:
                    skipped_empty += 1
                    continue

                x_tokens = full_tokens[:-1]
                y_tokens = full_tokens[1:]
                prompt_label_count = max(len(prompt_tokens) - 1, 0)
                masked_labels = [-100] * prompt_label_count + y_tokens[prompt_label_count:]
                if all(label == -100 for label in masked_labels):
                    skipped_no_answer_room += 1
                    continue

                self.input_ids.append(x_tokens)
                self.labels.append(masked_labels)
                kept += 1

        if not self.input_ids:
            raise ValueError(f"No usable SFT samples found in {path}")

        mode_str = "chat-template" if use_chat_tpl else "plain"
        log_fn(
            f"[data] [{progress_label}] masked SFT dataset built samples={kept:,} mode={mode_str} "
            f"seq_len<={seq_len} skipped_empty={skipped_empty:,} "
            f"skipped_no_answer_room={skipped_no_answer_room:,} truncated={truncated:,} total={total:,}"
        )

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.input_ids[idx], self.labels[idx]


def _maybe_enable_hf_gradient_checkpointing(model, *, log_fn: Callable[[str], None]) -> None:
    """Enable HF gradient checkpointing to cut activation VRAM (needed for 7B full FT on 40GB)."""
    fn = getattr(model, "gradient_checkpointing_enable", None)
    if fn is None:
        log_fn("[warn] gradient checkpointing requested but model has no gradient_checkpointing_enable()")
        return
    try:
        fn(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        fn()
    log_fn("[init] gradient checkpointing enabled (HF)")


def build_sft_dataloader(
    path: str,
    tokenizer: Callable[[str], List[int]],
    *,
    seq_len: int,
    batch_size: int,
    shuffle: bool,
    log_fn: Callable[[str], None],
    progress_label: str,
    rank: int = 0,
    world_size: int = 1,
    chat_template_tokenizer=None,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    dataset = MaskedSFTDataset(
        path,
        tokenizer,
        seq_len,
        log_fn=log_fn,
        progress_label=progress_label,
        chat_template_tokenizer=chat_template_tokenizer,
    )
    pad_id = getattr(tokenizer, "pad_id", -1)
    if not isinstance(pad_id, int) or pad_id < 0:
        pad_id = getattr(tokenizer, "eos_id", -1)
    if not isinstance(pad_id, int) or pad_id < 0:
        pad_id = 0

    def collate_fn(batch: Sequence[Tuple[List[int], List[int]]]) -> Tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(x) for x, _ in batch)
        x_out = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        y_out = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for row_idx, (x_tokens, y_tokens) in enumerate(batch):
            x_out[row_idx, : len(x_tokens)] = torch.tensor(x_tokens, dtype=torch.long)
            y_out[row_idx, : len(y_tokens)] = torch.tensor(y_tokens, dtype=torch.long)
        return x_out, y_out

    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        loader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            num_workers=0, drop_last=True, collate_fn=collate_fn,
        )
    else:
        sampler = None
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=0, drop_last=shuffle, collate_fn=collate_fn,
        )
    return loader, sampler


def main() -> int:
    parser = argparse.ArgumentParser(description="SFT finetune loop (single & multi-GPU)")
    parser.add_argument("--config", type=str, default="configs/config_sft_pilot_oasst1_dolly.yaml")
    parser.add_argument(
        "--ckpt", type=str, default=None,
        help="Titan checkpoint path or hf://gpt2 ref (mutually exclusive with --hf-model)",
    )

    # --- HuggingFace model arguments ---
    parser.add_argument(
        "--hf-model", type=str, default=None,
        help="HuggingFace model ID (e.g. meta-llama/Meta-Llama-3-8B). Mutually exclusive with --ckpt.",
    )
    lora_group = parser.add_mutually_exclusive_group()
    lora_group.add_argument("--qlora", action="store_true",
                            help="QLoRA: 4-bit quantized base + LoRA adapters (best for single GPU)")
    lora_group.add_argument("--lora", action="store_true",
                            help="LoRA: fp16 base + LoRA adapters")
    lora_group.add_argument("--no-lora", action="store_true",
                            help="Full fine-tuning of HF model (requires multi-GPU or large GPU)")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank (default 16)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (default 32)")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout (default 0.05)")
    parser.add_argument("--lora-targets", type=str, default=None,
                        help="Comma-separated LoRA target modules (auto-detected if omitted)")
    parser.add_argument("--chat-template", action="store_true",
                        help="Use the HF model's native chat template for tokenization")

    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "mps", "cuda"],
                        help="Device override (ignored under torchrun, which forces CUDA)")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=40)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--s3-checkpoint-uri", type=str, default=None)
    parser.add_argument("--aws-bin", type=str, default="aws")
    parser.add_argument("--lr", type=float, default=None, help="Override LR from config")
    parser.add_argument(
        "--grad-accum-steps", type=int, default=None,
        help="Override gradient accumulation steps (auto-scaled by world_size unless --no-accum-scale)",
    )
    parser.add_argument("--no-accum-scale", action="store_true",
                        help="Disable automatic grad_accum_steps scaling by world_size")
    parser.add_argument(
        "--min-free-gb", type=float, default=20.0,
        help="Minimum free disk space (GiB) required to write a checkpoint",
    )
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true",
                           help="Force enable AMP")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false",
                           help="Force disable AMP")
    parser.set_defaults(amp=None)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="HF activation checkpointing (config train.gradient_checkpointing if omitted)",
    )
    args = parser.parse_args()

    hf_mode = args.hf_model is not None
    if hf_mode and args.ckpt is not None:
        parser.error("--hf-model and --ckpt are mutually exclusive")
    if not hf_mode and args.ckpt is None:
        args.ckpt = "ckpt_step_4000.pt"
    if not hf_mode and (args.qlora or args.lora):
        parser.error("--qlora and --lora require --hf-model")

    use_lora = False
    use_qlora = False
    if hf_mode:
        if args.no_lora:
            use_lora = False
        elif args.qlora:
            use_qlora = True
            use_lora = True
        elif args.lora:
            use_lora = True
        else:
            use_qlora = True
            use_lora = True

    if args.steps <= 0:
        raise ValueError("--steps must be > 0")

    # ------------------------------------------------------------------
    # Distributed / device setup
    # ------------------------------------------------------------------
    use_ddp = "RANK" in os.environ

    if use_ddp:
        rank, local_rank, world_size = setup_distributed()
        device = torch.device("cuda", local_rank)
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = pick_device(args.device)

    started = time.time()

    def log(msg: str) -> None:
        if not is_main(rank):
            return
        print(f"[{time.time() - started:7.1f}s] {msg}")

    # ------------------------------------------------------------------
    # Config, tokenizer, data
    # ------------------------------------------------------------------
    cfg = load_config(resolve_path(args.config))

    if args.amp is not None:
        amp_enabled = args.amp
    elif device.type == "cuda":
        amp_enabled = True
    else:
        amp_enabled = False
    amp_device = device.type if amp_enabled and device.type in ("cuda", "mps") else None
    # FP16 weights + GradScaler + gradient checkpointing can fail at scaler.unscale_ on CUDA.
    # Non-QLoRA HF on bf16-capable GPUs: load weights in bf16, autocast bf16, no GradScaler.
    use_bf16_hf_non_qlora = (
        hf_mode
        and not use_qlora
        and amp_enabled
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    hf_tokenizer_obj = None  # raw HF tokenizer for chat template path

    if hf_mode:
        from hf_utils import get_hf_tokenizer
        hf_tokenizer_obj = get_hf_tokenizer(args.hf_model)
        tokenizer = _hf_tokenizer_to_adapter(hf_tokenizer_obj)
        log(f"[init] HF tokenizer loaded from {args.hf_model}")
    else:
        tokenizer_path = resolve_path(cfg["data"]["tokenizer_path"])
        tokenizer = get_tokenizer(str(tokenizer_path))
        log(f"[init] tokenizer={tokenizer_path}")

    use_chat_tpl = args.chat_template or cfg.get("data", {}).get("use_chat_template", False)
    if use_chat_tpl and hf_tokenizer_obj is None:
        log("[warn] --chat-template requires an HF tokenizer; falling back to plain tokenization")
        use_chat_tpl = False

    train_path = resolve_path(cfg["data"]["train_path"])
    val_path = resolve_path(cfg["data"]["val_path"])

    log(f"[init] world_size={world_size} rank={rank} local_rank={local_rank} device={device}")

    train_loader, train_sampler = build_sft_dataloader(
        str(train_path), tokenizer,
        seq_len=cfg["train"]["seq_len"],
        batch_size=cfg["train"]["batch_size"],
        shuffle=True, log_fn=log if is_main(rank) else lambda m: None,
        progress_label="train_sft",
        rank=rank, world_size=world_size,
        chat_template_tokenizer=hf_tokenizer_obj if use_chat_tpl else None,
    )
    val_loader, _ = build_sft_dataloader(
        str(val_path), tokenizer,
        seq_len=cfg["train"]["seq_len"],
        batch_size=cfg["train"]["batch_size"],
        shuffle=False, log_fn=log if is_main(rank) else lambda m: None,
        progress_label="val_sft",
        rank=rank, world_size=world_size,
        chat_template_tokenizer=hf_tokenizer_obj if use_chat_tpl else None,
    )
    log(
        f"[init] train_samples={len(train_loader.dataset)} val_samples={len(val_loader.dataset)} "
        f"seq_len={cfg['train']['seq_len']} batch={cfg['train']['batch_size']}"
    )

    grad_ckpt: bool
    if args.gradient_checkpointing is not None:
        grad_ckpt = bool(args.gradient_checkpointing)
    else:
        grad_ckpt = bool(cfg["train"].get("gradient_checkpointing", False))

    # ------------------------------------------------------------------
    # Model + checkpoint
    # ------------------------------------------------------------------
    if hf_mode:
        from hf_utils import load_hf_model, apply_lora, save_hf_checkpoint

        hf_load_kwargs = {}
        quant_cfg = cfg.get("quantization", {})
        lora_cfg = cfg.get("lora", {})

        if use_qlora:
            hf_load_kwargs["load_in_4bit"] = True
            hf_load_kwargs["bnb_4bit_compute_dtype"] = quant_cfg.get("bnb_4bit_compute_dtype", "float16")
            hf_load_kwargs["bnb_4bit_quant_type"] = quant_cfg.get("bnb_4bit_quant_type", "nf4")
            hf_load_kwargs["device_map"] = {"": local_rank} if use_ddp else "auto"
        else:
            hf_load_kwargs["device_map"] = None if use_ddp else "auto"
            hf_load_kwargs["torch_dtype"] = (
                torch.bfloat16 if use_bf16_hf_non_qlora else torch.float16
            )

        model = load_hf_model(args.hf_model, **hf_load_kwargs)
        log(f"[init] loaded HF model: {args.hf_model} qlora={use_qlora}")

        if use_lora:
            lora_targets = None
            if args.lora_targets:
                lora_targets = [t.strip() for t in args.lora_targets.split(",")]
            elif lora_cfg.get("target_modules"):
                lora_targets = lora_cfg["target_modules"]

            model = apply_lora(
                model,
                rank=args.lora_rank if args.lora_rank != 16 else lora_cfg.get("rank", 16),
                alpha=args.lora_alpha if args.lora_alpha != 32 else lora_cfg.get("alpha", 32),
                dropout=args.lora_dropout if args.lora_dropout != 0.05 else lora_cfg.get("dropout", 0.05),
                target_modules=lora_targets,
                log_fn=log,
            )

        if grad_ckpt:
            _maybe_enable_hf_gradient_checkpointing(model, log_fn=log)

        if use_ddp and not use_qlora:
            model.to(device)
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        raw_model = model.module if (use_ddp and not use_qlora) else model
        ckpt_path = args.hf_model
    else:
        mcfg = ModelConfig(**cfg["model"])
        model = build_model(mcfg).to(device)
        ckpt_path = resolve_path(args.ckpt)
        load_model_source(model, ckpt_path, map_location=device, strict=True)
        log(f"[init] loaded base checkpoint: {ckpt_path}")

        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        raw_model = model.module if world_size > 1 else model

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    lr = args.lr if args.lr is not None else float(cfg["train"]["lr"])
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if use_lora:
        log(f"[init] optimizing {len(trainable_params)} LoRA parameter groups")
    opt = AdamW(
        trainable_params,
        lr=lr,
        weight_decay=cfg["train"]["weight_decay"],
        betas=tuple(cfg["train"]["betas"]),
        eps=cfg["train"]["eps"],
    )
    if amp_device is None:
        autocast_dtype: torch.dtype | None = None
        use_amp_scaler = False
    elif use_bf16_hf_non_qlora:
        autocast_dtype = torch.bfloat16
        use_amp_scaler = False
    else:
        autocast_dtype = torch.float16
        use_amp_scaler = amp_device in ("cuda", "mps")
    scaler = GradScaler(enabled=use_amp_scaler)
    if is_main(rank) and hf_mode and not use_qlora and amp_enabled:
        if use_bf16_hf_non_qlora:
            log("[init] CUDA AMP: bfloat16 weights + autocast (GradScaler off)")
        else:
            log("[init] CUDA/MPS AMP: float16 autocast + GradScaler")

    # ------------------------------------------------------------------
    # Training params
    # ------------------------------------------------------------------
    grad_accum_steps = args.grad_accum_steps or cfg["train"].get("grad_accum_steps", 1)
    if grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be > 0")

    if world_size > 1 and not args.no_accum_scale:
        original_accum = grad_accum_steps
        grad_accum_steps = max(1, grad_accum_steps // world_size)
        log(f"[init] auto-scaled grad_accum_steps: {original_accum} -> {grad_accum_steps} (world_size={world_size})")

    use_cosine = cfg["train"].get("cosine_decay", False)
    warmup = cfg["train"].get("warmup_steps", 0)
    lr_min = cfg["train"].get("lr_min", 0.0)

    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_dir, Path(__file__).parent / "checkpoints_sft")

    log(
        f"[init] device={device} amp={amp_enabled} "
        f"amp_dtype={autocast_dtype} grad_scaler={use_amp_scaler} ckpt={ckpt_path}"
    )
    log(f"[init] steps={args.steps} grad_accum={grad_accum_steps} world_size={world_size} lr={lr}")
    log(f"[init] checkpoint_dir={checkpoint_dir}")
    if args.s3_checkpoint_uri:
        log(f"[init] periodic checkpoint sync enabled -> {args.s3_checkpoint_uri}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    model.train()
    train_iter = cycle_batches(train_loader, train_sampler)
    global_step = 0
    accum_in_step = 0
    accum_loss_sum = 0.0
    opt.zero_grad(set_to_none=True)

    while global_step < args.steps:
        x, y = next(train_iter)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        if use_cosine:
            current_lr = cosine_lr(global_step, warmup, args.steps, lr, lr_min)
            for pg in opt.param_groups:
                pg["lr"] = current_lr

        is_last_accum = (accum_in_step + 1) == grad_accum_steps
        ddp_wrapped = use_ddp and not (hf_mode and use_qlora)
        sync_context = nullcontext() if (not ddp_wrapped or is_last_accum) else model.no_sync()

        with sync_context:
            if amp_device:
                with autocast(device_type=amp_device, dtype=autocast_dtype):
                    logits = _forward_logits(model, x, hf_mode)
                    raw_loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100,
                    )
                    loss = raw_loss / grad_accum_steps
                if use_amp_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                logits = _forward_logits(model, x, hf_mode)
                raw_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100,
                )
                loss = raw_loss / grad_accum_steps
                loss.backward()

        accum_in_step += 1
        accum_loss_sum += raw_loss.item()
        if accum_in_step < grad_accum_steps:
            continue

        # --- Optimizer step ---
        if use_amp_scaler:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable_params, cfg["train"]["grad_clip"])
        if use_amp_scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)

        global_step += 1
        step_loss = accum_loss_sum / max(accum_in_step, 1)
        accum_in_step = 0
        accum_loss_sum = 0.0

        if global_step % args.log_every == 0 or global_step == 1:
            avg_loss = reduce_scalar(step_loss, world_size)
            log(f"[train] step {global_step}/{args.steps} loss {avg_loss:.4f}")

        if global_step % args.eval_every == 0:
            model.eval()
            total_val = 0.0
            count = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
                    if amp_device:
                        with autocast(device_type=amp_device, dtype=autocast_dtype):
                            vlogits = _forward_logits(model, bx, hf_mode)
                            vloss = F.cross_entropy(
                                vlogits.view(-1, vlogits.size(-1)), by.view(-1), ignore_index=-100,
                            )
                    else:
                        vlogits = _forward_logits(model, bx, hf_mode)
                        vloss = F.cross_entropy(
                            vlogits.view(-1, vlogits.size(-1)), by.view(-1), ignore_index=-100,
                        )
                    total_val += vloss.item()
                    count += 1
                    if count >= args.eval_batches:
                        break

            avg_val = reduce_scalar(total_val / max(count, 1), world_size)
            if is_main(rank):
                ppl = math.exp(avg_val)
                log(f"[eval] step {global_step} loss {avg_val:.4f} ppl {ppl:.2f} batches={count}")
            model.train()

        if (global_step % args.save_every == 0 or global_step == args.steps) and is_main(rank):
            if has_min_free_space(checkpoint_dir, args.min_free_gb, log):
                if hf_mode:
                    ckpt_out = checkpoint_dir / f"step_{global_step}"
                    save_hf_checkpoint(
                        raw_model, ckpt_out,
                        tokenizer=hf_tokenizer_obj,
                        step=global_step,
                        is_lora=use_lora,
                        log_fn=log,
                    )
                else:
                    ckpt_out = checkpoint_dir / f"ckpt_sft_step_{global_step}.pt"
                    save_checkpoint(ckpt_out, raw_model.state_dict(), opt.state_dict(), global_step,
                                    extra={"source_ckpt": str(ckpt_path)})
                    log(f"[save] {ckpt_out}")
                if args.s3_checkpoint_uri:
                    sync_checkpoints_to_s3(checkpoint_dir, args.s3_checkpoint_uri, args.aws_bin, log,
                                           glob_pattern="step_*" if hf_mode else "ckpt_sft_step_*.pt")

    log("[done] SFT finished")
    cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
