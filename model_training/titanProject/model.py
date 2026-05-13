"""
Titans MAC small-model factory.

Assumes `titans-pytorch` is installed (or the repo is on PYTHONPATH).
This is a minimal scaffold to instantiate a tiny MAC transformer with an LM head.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

try:
    from titans_pytorch import MemoryAsContextTransformer
except ImportError as e:
    raise ImportError(
        "titans-pytorch is required. Install with `pip install titans-pytorch` "
        "or ensure the repo is on PYTHONPATH."
    ) from e

try:
    from x_transformers import Decoder
except ImportError as e:
    raise ImportError(
        "x-transformers is required for the GPT variant. Install with `pip install x-transformers`."
    ) from e


@dataclass
class ModelConfig:
    vocab_size: int
    variant: str = "mac"
    dim: int = 320
    depth: int = 5
    heads: int = 6
    ff_mult: int = 4
    segment_len: int = 256
    num_persist_mem_tokens: int = 4
    num_longterm_mem_tokens: int = 16
    max_seq_len: int = 2048  # used for GPT variant positional embeddings
    hf_model_name: Optional[str] = None

    # --- MAC neural memory options (passed via neural_memory_kwargs) ---
    store_with_lookahead_value: bool = False
    neural_memory_add_value_residual: bool = False
    neural_mem_gate_attn_output: bool = False
    neural_mem_weight_residual: bool = False
    sliding_window_attn: bool = False
    num_residual_streams: int = 4


class TitansLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.variant.lower() != "mac":
            raise ValueError(f"Unsupported variant {cfg.variant}; only 'mac' is implemented here.")

        neural_memory_kwargs = {}
        if cfg.store_with_lookahead_value:
            neural_memory_kwargs["store_with_lookahead_value"] = True

        self.model = MemoryAsContextTransformer(
            num_tokens=cfg.vocab_size,
            dim=cfg.dim,
            depth=cfg.depth,
            heads=cfg.heads,
            ff_mult=cfg.ff_mult,
            segment_len=cfg.segment_len,
            num_persist_mem_tokens=cfg.num_persist_mem_tokens,
            num_longterm_mem_tokens=cfg.num_longterm_mem_tokens,
            neural_memory_add_value_residual=cfg.neural_memory_add_value_residual,
            neural_mem_gate_attn_output=cfg.neural_mem_gate_attn_output,
            neural_mem_weight_residual=cfg.neural_mem_weight_residual,
            sliding_window_attn=cfg.sliding_window_attn,
            num_residual_streams=cfg.num_residual_streams,
            neural_memory_kwargs=neural_memory_kwargs,
        )

    def forward(self, x, return_loss: bool = False):
        return self.model(x, return_loss=return_loss)


class GPTLM(nn.Module):
    """
    Minimal GPT-style decoder-only LM using x-transformers. HF-compatible causal attention stack
    without Titan memory. Positional embeddings are learned; max_seq_len defines the table size.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.dim)
        self.decoder = Decoder(
            dim=cfg.dim,
            depth=cfg.depth,
            heads=cfg.heads,
            ff_mult=cfg.ff_mult,
            attn_flash=True,
            rotary_pos_emb=False,
        )
        self.to_logits = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    def forward(self, x, return_loss: bool = False):
        # x: (batch, seq)
        b, n = x.shape
        if n > self.pos_emb.num_embeddings:
            raise ValueError(f"Sequence length {n} exceeds max_seq_len {self.pos_emb.num_embeddings}")
        tok = self.token_emb(x)
        pos = self.pos_emb(torch.arange(n, device=x.device))[None, :, :]
        h = tok + pos
        h = self.decoder(h)
        logits = self.to_logits(h)
        return logits


class HFGPT2LM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        try:
            from transformers import GPT2Config, GPT2LMHeadModel
        except ImportError as e:
            raise ImportError(
                "transformers is required for the hf_gpt2 variant. "
                "Install it in the active Python environment first."
            ) from e

        hf_cfg = GPT2Config(
            vocab_size=cfg.vocab_size,
            n_positions=cfg.max_seq_len,
            n_ctx=cfg.max_seq_len,
            n_embd=cfg.dim,
            n_layer=cfg.depth,
            n_head=cfg.heads,
            n_inner=cfg.dim * cfg.ff_mult,
        )
        self.model = GPT2LMHeadModel(hf_cfg)
        self.default_hf_model_name = cfg.hf_model_name or "gpt2"

    def load_pretrained(self, hf_name: str) -> None:
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as e:
            raise ImportError(
                "transformers is required to load Hugging Face GPT-2 weights. "
                "Install it in the active Python environment first."
            ) from e

        hf_model = AutoModelForCausalLM.from_pretrained(hf_name)
        self.model.load_state_dict(hf_model.state_dict(), strict=True)

    def forward(self, x, return_loss: bool = False):
        return self.model(input_ids=x).logits


HF_SOURCE_PREFIX = "hf://"


def is_hf_source(source: Optional[Union[str, Path]]) -> bool:
    return isinstance(source, (str, Path)) and str(source).startswith(HF_SOURCE_PREFIX)


def normalize_hf_source(source: Union[str, Path]) -> str:
    source_str = str(source)
    if not source_str.startswith(HF_SOURCE_PREFIX):
        raise ValueError(f"Expected Hugging Face source with prefix {HF_SOURCE_PREFIX}, got {source_str}")
    return source_str[len(HF_SOURCE_PREFIX) :]


def _load_gpt2_weights_into_gptlm(model: GPTLM, hf_name: str) -> None:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        raise ImportError(
            "transformers is required to load Hugging Face GPT-2 weights. "
            "Install it in the active Python environment first."
        ) from e

    hf_model = AutoModelForCausalLM.from_pretrained(hf_name)
    hf_state = hf_model.state_dict()

    token_emb = hf_state["transformer.wte.weight"]
    pos_emb = hf_state["transformer.wpe.weight"]
    if model.token_emb.weight.shape != token_emb.shape:
        raise ValueError(
            f"GPT-2 token embedding shape mismatch: local={tuple(model.token_emb.weight.shape)} "
            f"hf={tuple(token_emb.shape)}"
        )
    if model.to_logits.weight.shape != token_emb.shape:
        raise ValueError(
            f"GPT-2 lm_head shape mismatch: local={tuple(model.to_logits.weight.shape)} "
            f"hf={tuple(token_emb.shape)}"
        )
    if model.pos_emb.weight.shape[1] != pos_emb.shape[1]:
        raise ValueError(
            f"GPT-2 positional embedding width mismatch: local={tuple(model.pos_emb.weight.shape)} "
            f"hf={tuple(pos_emb.shape)}"
        )
    if model.pos_emb.weight.shape[0] > pos_emb.shape[0]:
        raise ValueError(
            f"Local max_seq_len {model.pos_emb.weight.shape[0]} exceeds GPT-2 positional table "
            f"length {pos_emb.shape[0]}"
        )

    mapped_state = {
        "token_emb.weight": token_emb,
        "pos_emb.weight": pos_emb[: model.pos_emb.weight.shape[0]],
        "to_logits.weight": token_emb,
        "decoder.final_norm.gamma": hf_state["transformer.ln_f.weight"],
    }

    depth = len(model.decoder.layers) // 2
    for block_idx in range(depth):
        attn_layer_idx = block_idx * 2
        ff_layer_idx = attn_layer_idx + 1
        prefix = f"transformer.h.{block_idx}"

        c_attn_weight = hf_state[f"{prefix}.attn.c_attn.weight"]
        dim = model.token_emb.embedding_dim
        mapped_state[f"decoder.layers.{attn_layer_idx}.0.0.gamma"] = hf_state[f"{prefix}.ln_1.weight"]
        mapped_state[f"decoder.layers.{attn_layer_idx}.1.to_q.weight"] = c_attn_weight[:, :dim].T
        mapped_state[f"decoder.layers.{attn_layer_idx}.1.to_k.weight"] = c_attn_weight[:, dim : 2 * dim].T
        mapped_state[f"decoder.layers.{attn_layer_idx}.1.to_v.weight"] = c_attn_weight[:, 2 * dim :].T
        mapped_state[f"decoder.layers.{attn_layer_idx}.1.to_out.weight"] = hf_state[
            f"{prefix}.attn.c_proj.weight"
        ].T

        mapped_state[f"decoder.layers.{ff_layer_idx}.0.0.gamma"] = hf_state[f"{prefix}.ln_2.weight"]
        mapped_state[f"decoder.layers.{ff_layer_idx}.1.ff.0.0.weight"] = hf_state[f"{prefix}.mlp.c_fc.weight"].T
        mapped_state[f"decoder.layers.{ff_layer_idx}.1.ff.0.0.bias"] = hf_state[f"{prefix}.mlp.c_fc.bias"]
        mapped_state[f"decoder.layers.{ff_layer_idx}.1.ff.2.weight"] = hf_state[f"{prefix}.mlp.c_proj.weight"].T
        mapped_state[f"decoder.layers.{ff_layer_idx}.1.ff.2.bias"] = hf_state[f"{prefix}.mlp.c_proj.bias"]

    missing, unexpected = model.load_state_dict(mapped_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Failed GPT-2 load with missing={missing}, unexpected={unexpected}")


def load_model_source(
    model: nn.Module,
    source: Union[str, Path],
    *,
    map_location=None,
    strict: bool = True,
) -> str:
    if is_hf_source(source):
        hf_name = normalize_hf_source(source)
        if isinstance(model, HFGPT2LM):
            model.load_pretrained(hf_name)
            return f"hf://{hf_name}"
        if not isinstance(model, GPTLM):
            raise ValueError("Hugging Face GPT-2 bootstrap is only supported for GPT variants")
        _load_gpt2_weights_into_gptlm(model, hf_name)
        return f"hf://{hf_name}"

    state = torch.load(source, map_location=map_location)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict, strict=strict)
    return str(source)


def build_model(cfg: ModelConfig) -> TitansLM:
    if cfg.variant.lower() == "mac":
        return TitansLM(cfg)
    if cfg.variant.lower() == "gpt":
        return GPTLM(cfg)
    if cfg.variant.lower() == "hf_gpt2":
        return HFGPT2LM(cfg)
    raise ValueError(f"Unsupported variant {cfg.variant}; choose 'mac', 'gpt', or 'hf_gpt2'")

