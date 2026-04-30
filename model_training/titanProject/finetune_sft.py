"""
SFT finetuning loop for Titans checkpoints.

Accepts four data formats (auto-detected per line, can be mixed):
  1. HF messages JSONL:  {"messages": [{"role": "user", "content": "..."},  ...]}
  2. ShareGPT JSONL:     {"conversations": [{"from": "human", "value": "..."}, ...]}
  3. Alpaca JSONL:       {"instruction": "...", "input": "", "response"/"output": "..."}
  4. Chat text:          User: <question> Assistant: <answer>

Supports single-GPU, multi-GPU (DDP), and CPU/MPS development:

    # Single GPU
    python finetune_sft.py --config configs/config_sft.yaml --ckpt ckpt_step_124000.pt

    # Multi-GPU via torchrun
    torchrun --nproc_per_node=4 finetune_sft.py --config configs/config_sft.yaml --ckpt ckpt_step_124000.pt

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


class MaskedSFTDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer: Callable[[str], List[int]],
        seq_len: int,
        *,
        log_fn: Callable[[str], None],
        progress_label: str,
    ):
        self.seq_len = seq_len
        self.input_ids: List[List[int]] = []
        self.labels: List[List[int]] = []

        boundary_tokens = _get_boundary_tokens(tokenizer)
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

        log_fn(
            f"[data] [{progress_label}] masked SFT dataset built samples={kept:,} "
            f"seq_len<={seq_len} skipped_empty={skipped_empty:,} "
            f"skipped_no_answer_room={skipped_no_answer_room:,} truncated={truncated:,} total={total:,}"
        )

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.input_ids[idx], self.labels[idx]


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
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    dataset = MaskedSFTDataset(
        path,
        tokenizer,
        seq_len,
        log_fn=log_fn,
        progress_label=progress_label,
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
        "--ckpt", type=str, default="ckpt_step_4000.pt",
        help="Checkpoint path or Hugging Face ref like hf://gpt2",
    )
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
    args = parser.parse_args()

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
    mcfg = ModelConfig(**cfg["model"])

    if args.amp is not None:
        amp_enabled = args.amp
    elif device.type == "cuda":
        amp_enabled = True
    else:
        amp_enabled = False
    amp_device = device.type if amp_enabled and device.type in ("cuda", "mps") else None

    tokenizer_path = resolve_path(cfg["data"]["tokenizer_path"])
    tokenizer = get_tokenizer(str(tokenizer_path))
    train_path = resolve_path(cfg["data"]["train_path"])
    val_path = resolve_path(cfg["data"]["val_path"])

    log(f"[init] world_size={world_size} rank={rank} local_rank={local_rank} device={device}")
    log(f"[init] tokenizer={tokenizer_path}")

    train_loader, train_sampler = build_sft_dataloader(
        str(train_path), tokenizer,
        seq_len=cfg["train"]["seq_len"],
        batch_size=cfg["train"]["batch_size"],
        shuffle=True, log_fn=log if is_main(rank) else lambda m: None,
        progress_label="train_sft",
        rank=rank, world_size=world_size,
    )
    val_loader, _ = build_sft_dataloader(
        str(val_path), tokenizer,
        seq_len=cfg["train"]["seq_len"],
        batch_size=cfg["train"]["batch_size"],
        shuffle=False, log_fn=log if is_main(rank) else lambda m: None,
        progress_label="val_sft",
        rank=rank, world_size=world_size,
    )
    log(
        f"[init] train_samples={len(train_loader.dataset)} val_samples={len(val_loader.dataset)} "
        f"seq_len={cfg['train']['seq_len']} batch={cfg['train']['batch_size']}"
    )

    # ------------------------------------------------------------------
    # Model + checkpoint
    # ------------------------------------------------------------------
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
    opt = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=cfg["train"]["weight_decay"],
        betas=tuple(cfg["train"]["betas"]),
        eps=cfg["train"]["eps"],
    )
    scaler = GradScaler(enabled=amp_device in ("cuda", "mps"))

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

    log(f"[init] device={device} amp={amp_enabled} ckpt={ckpt_path}")
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
        sync_context = nullcontext() if (world_size <= 1 or is_last_accum) else model.no_sync()

        with sync_context:
            if amp_device:
                with autocast(device_type=amp_device, dtype=torch.float16):
                    logits = model(x)
                    raw_loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100,
                    )
                    loss = raw_loss / grad_accum_steps
                scaler.scale(loss).backward()
            else:
                logits = model(x)
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
        if amp_device:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        if amp_device:
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
                        with autocast(device_type=amp_device, dtype=torch.float16):
                            vlogits = model(bx)
                            vloss = F.cross_entropy(
                                vlogits.view(-1, vlogits.size(-1)), by.view(-1), ignore_index=-100,
                            )
                    else:
                        vlogits = model(bx)
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
                ckpt_out = checkpoint_dir / f"ckpt_sft_step_{global_step}.pt"
                save_checkpoint(ckpt_out, raw_model.state_dict(), opt.state_dict(), global_step,
                                extra={"source_ckpt": str(ckpt_path)})
                log(f"[save] {ckpt_out}")
                if args.s3_checkpoint_uri:
                    sync_checkpoints_to_s3(checkpoint_dir, args.s3_checkpoint_uri, args.aws_bin, log,
                                           glob_pattern="ckpt_sft_step_*.pt")

    log("[done] SFT finished")
    cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
