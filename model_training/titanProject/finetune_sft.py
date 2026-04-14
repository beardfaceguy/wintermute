"""
SFT pilot finetuning loop for Titans checkpoint on instruction/chat text.
"""

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from model import ModelConfig, build_model, load_model_source
from train import get_tokenizer, has_min_free_space, load_config, resolve_path


def cycle_batches(loader: Iterable[Tuple[torch.Tensor, torch.Tensor]]):
    while True:
        for batch in loader:
            yield batch


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


def _split_sft_sample(sample: str) -> Tuple[str, str]:
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
        instruction_text = str(payload.get("instruction", "")).strip()
        input_text = str(payload.get("input", "")).strip()
        response_text = str(payload.get("response", "")).strip()
        if not instruction_text or not response_text:
            raise ValueError("Instruction-format SFT sample must include instruction and response")
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
) -> DataLoader:
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

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=shuffle,
        collate_fn=collate_fn,
    )


def sync_checkpoints_to_s3(checkpoint_dir: Path, s3_uri: str, aws_bin: str, log_fn) -> None:
    cmd = [
        aws_bin,
        "s3",
        "sync",
        str(checkpoint_dir),
        s3_uri,
        "--exclude",
        "*",
        "--include",
        "ckpt_sft_step_*.pt",
        "--only-show-errors",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        log_fn(f"[sync] warning: aws binary not found: {aws_bin}")
        return
    if proc.returncode != 0:
        log_fn(f"[sync] warning: command failed (exit={proc.returncode})")
        if proc.stderr.strip():
            log_fn(f"[sync] stderr: {proc.stderr.strip()}")
        elif proc.stdout.strip():
            log_fn(f"[sync] stdout: {proc.stdout.strip()}")
        return
    log_fn(f"[sync] checkpoints synced to {s3_uri}")


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description="SFT pilot finetune loop")
    parser.add_argument("--config", type=str, default="configs/config_sft_pilot_oasst1_dolly.yaml")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="ckpt_step_4000.pt",
        help="Checkpoint path or Hugging Face ref like hf://gpt2",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
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
        "--min-free-gb",
        type=float,
        default=20.0,
        help="Minimum free disk space (GiB) required to write a checkpoint; below this, saves are skipped",
    )
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be > 0")

    started = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - started:7.1f}s] {msg}")

    cfg = load_config(resolve_path(args.config))
    mcfg = ModelConfig(**cfg["model"])
    device = pick_device(args.device)
    amp_enabled = device.type == "cuda"
    amp_device = "cuda" if amp_enabled else None

    tokenizer_path = resolve_path(cfg["data"]["tokenizer_path"])
    tokenizer = get_tokenizer(str(tokenizer_path))
    train_path = resolve_path(cfg["data"]["train_path"])
    val_path = resolve_path(cfg["data"]["val_path"])

    train_loader = build_sft_dataloader(
        str(train_path),
        tokenizer,
        seq_len=cfg["train"]["seq_len"],
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        log_fn=log,
        progress_label="train_sft",
    )
    val_loader = build_sft_dataloader(
        str(val_path),
        tokenizer,
        seq_len=cfg["train"]["seq_len"],
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        log_fn=log,
        progress_label="val_sft",
    )
    log(
        f"[init] train_samples={len(train_loader.dataset)} val_samples={len(val_loader.dataset)} "
        f"seq_len={cfg['train']['seq_len']} batch={cfg['train']['batch_size']}"
    )

    model = build_model(mcfg).to(device)
    ckpt_path = resolve_path(args.ckpt)
    load_model_source(model, ckpt_path, map_location=device, strict=True)

    lr = args.lr if args.lr is not None else float(cfg["train"]["lr"])
    opt = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=cfg["train"]["weight_decay"],
        betas=tuple(cfg["train"]["betas"]),
        eps=cfg["train"]["eps"],
    )
    scaler = GradScaler(amp_device) if amp_device else None

    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir).expanduser()
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = (Path.cwd() / checkpoint_dir).resolve()
    else:
        checkpoint_dir = Path(__file__).parent / "checkpoints_sft"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log(f"[init] device={device} amp={amp_enabled} ckpt={ckpt_path}")
    log(f"[init] checkpoint_dir={checkpoint_dir}")
    if args.s3_checkpoint_uri:
        log(f"[init] periodic checkpoint sync enabled -> {args.s3_checkpoint_uri}")

    model.train()
    train_iter = cycle_batches(train_loader)

    for step in range(1, args.steps + 1):
        x, y = next(train_iter)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)

        if amp_device:
            with autocast(device_type=amp_device, dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()

        if step % args.log_every == 0 or step == 1:
            log(f"[train] step {step}/{args.steps} loss {loss.item():.4f}")

        if step % args.eval_every == 0:
            model.eval()
            total = 0.0
            count = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    vlogits = model(bx)
                    vloss = F.cross_entropy(vlogits.view(-1, vlogits.size(-1)), by.view(-1), ignore_index=-100)
                    total += vloss.item()
                    count += 1
                    if count >= args.eval_batches:
                        break
            avg = total / max(count, 1)
            ppl = math.exp(avg)
            log(f"[eval] step {step} loss {avg:.4f} ppl {ppl:.2f} batches={count}")
            model.train()

        if step % args.save_every == 0 or step == args.steps:
            if not has_min_free_space(checkpoint_dir, args.min_free_gb, log):
                continue
            ckpt_out = checkpoint_dir / f"ckpt_sft_step_{step}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "step": step,
                    "source_ckpt": str(ckpt_path),
                },
                ckpt_out,
            )
            log(f"[save] {ckpt_out}")
            if args.s3_checkpoint_uri:
                sync_checkpoints_to_s3(checkpoint_dir, args.s3_checkpoint_uri, args.aws_bin, log)

    log("[done] SFT pilot finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
