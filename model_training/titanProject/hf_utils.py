"""
Hugging Face model utilities for fine-tuning open-source LLMs (Llama, Mistral, Qwen, etc.).

Provides model loading (with optional 4-bit quantization), LoRA/QLoRA wrapping,
checkpoint saving/loading, and adapter merging for deployment.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)


def load_hf_model(
    model_id: str,
    *,
    load_in_4bit: bool = False,
    bnb_4bit_compute_dtype: str = "float16",
    bnb_4bit_quant_type: str = "nf4",
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Optional[str] = "auto",
    attn_implementation: Optional[str] = None,
    trust_remote_code: bool = False,
):
    """Load a HuggingFace causal LM, optionally quantized to 4-bit for QLoRA.

    Returns a model whose ``forward(input_ids)`` produces logits compatible
    with the existing SFT training loop.
    """
    try:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as e:
        raise ImportError(
            "transformers is required for HF model loading. "
            "Install with: pip install transformers accelerate"
        ) from e

    kwargs: Dict = {
        "trust_remote_code": trust_remote_code,
    }

    if torch_dtype is None:
        torch_dtype = torch.float16
    kwargs["torch_dtype"] = torch_dtype

    if device_map is not None:
        kwargs["device_map"] = device_map

    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    if load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "bitsandbytes is required for 4-bit quantization. "
                "Install with: pip install bitsandbytes"
            ) from e
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        compute_dtype = dtype_map.get(bnb_4bit_compute_dtype, torch.float16)
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return model


def apply_lora(
    model,
    *,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    task_type: str = "CAUSAL_LM",
    log_fn=None,
):
    """Wrap a HuggingFace model with LoRA adapters via PEFT.

    If *target_modules* is ``None``, targets all linear layers (the PEFT
    default when ``target_modules="all-linear"``).
    """
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    except ImportError as e:
        raise ImportError(
            "peft is required for LoRA fine-tuning. "
            "Install with: pip install peft"
        ) from e

    task_map = {"CAUSAL_LM": TaskType.CAUSAL_LM}
    peft_task = task_map.get(task_type, TaskType.CAUSAL_LM)

    if hasattr(model, "is_loaded_in_4bit") and model.is_loaded_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules or "all-linear",
        task_type=peft_task,
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    if log_fn:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        pct = 100.0 * trainable / total if total > 0 else 0
        log_fn(
            f"[lora] trainable={trainable:,} / {total:,} ({pct:.2f}%) "
            f"rank={rank} alpha={alpha} dropout={dropout}"
        )
    return model


def save_hf_checkpoint(
    model,
    path: Union[str, Path],
    *,
    tokenizer=None,
    step: int = 0,
    is_lora: bool = False,
    log_fn=None,
):
    """Save an HF model checkpoint (full or LoRA adapter only)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if is_lora:
        model.save_pretrained(path)
        if log_fn:
            log_fn(f"[save] LoRA adapter saved to {path}")
    else:
        unwrapped = model.module if hasattr(model, "module") else model
        unwrapped.save_pretrained(path)
        if log_fn:
            log_fn(f"[save] full model saved to {path}")

    if tokenizer is not None:
        tokenizer.save_pretrained(path)

    step_file = path / "training_state.pt"
    torch.save({"step": step}, step_file)


def load_hf_checkpoint(
    model_id: str,
    *,
    adapter_path: Optional[str] = None,
    merge: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Optional[str] = "auto",
    trust_remote_code: bool = False,
):
    """Load a HuggingFace model, optionally with a LoRA adapter.

    When *merge* is True the adapter weights are folded into the base model
    for deployment (no PEFT dependency needed at inference time).
    """
    model = load_hf_model(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )

    if adapter_path is not None:
        try:
            from peft import PeftModel
        except ImportError as e:
            raise ImportError("peft is required to load LoRA adapters") from e
        model = PeftModel.from_pretrained(model, adapter_path)
        if merge:
            model = model.merge_and_unload()
    return model


def get_hf_tokenizer(model_id: str, *, trust_remote_code: bool = False):
    """Load a HuggingFace tokenizer with sensible defaults for SFT."""
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError("transformers is required for HF tokenizer loading") from e

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not getattr(tokenizer, "chat_template", None):
        _CHATML_SPECIAL = ["<|im_start|>", "<|im_end|>"]
        for tok in _CHATML_SPECIAL:
            if tok not in tokenizer.get_vocab():
                tokenizer.add_special_tokens(
                    {"additional_special_tokens": [tok]}
                )
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}<|im_start|>system\n"
            "{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'user' %}<|im_start|>user\n"
            "{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n"
            "{{ message['content'] }}<|im_end|>\n"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )
    return tokenizer


class HFModelWrapper:
    """Thin wrapper so HF models expose the same ``forward(input_ids) -> logits``
    interface used by the Titan training loop.

    The wrapper also holds references needed for chat-template tokenization
    and checkpoint saving.
    """

    def __init__(self, model, tokenizer, *, is_lora: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.is_lora = is_lora

    @property
    def hf_model(self):
        return self.model

    @property
    def hf_tokenizer(self):
        return self.tokenizer
