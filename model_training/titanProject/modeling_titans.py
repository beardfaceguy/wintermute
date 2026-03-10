"""
Hugging Face compatibility shim for titans-pytorch checkpoints.

This provides:
- TitansConfig: minimal HF config wrapper
- TitansForCausalLM: HF-style causal LM wrapper over titans-pytorch MemoryAsContextTransformer

Notes:
- Requires titans-pytorch installed.
- Expects checkpoints exported via export_to_hf.py (state_dict + config.json).
- This is not an official transformers architecture; many optional features (attention mask, KV cache)
  are no-ops or ignored. It is sufficient for loading/saving and basic generation.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from model import ModelConfig, build_model


class TitansConfig(PretrainedConfig):
    model_type = "titans"

    def __init__(
        self,
        vocab_size: int = 50000,
        variant: str = "mac",
        dim: int = 384,
        depth: int = 5,
        heads: int = 6,
        ff_mult: int = 4,
        segment_len: int = 512,
        num_persist_mem_tokens: int = 0,
        num_longterm_mem_tokens: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.variant = variant
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.ff_mult = ff_mult
        self.segment_len = segment_len
        self.num_persist_mem_tokens = num_persist_mem_tokens
        self.num_longterm_mem_tokens = num_longterm_mem_tokens


class TitansForCausalLM(PreTrainedModel):
    config_class = TitansConfig
    base_model_prefix = "titans"

    def __init__(self, config: TitansConfig):
        super().__init__(config)
        mcfg = ModelConfig(
            vocab_size=config.vocab_size,
            variant=config.variant,
            dim=config.dim,
            depth=config.depth,
            heads=config.heads,
            ff_mult=config.ff_mult,
            segment_len=config.segment_len,
            num_persist_mem_tokens=config.num_persist_mem_tokens,
            num_longterm_mem_tokens=config.num_longterm_mem_tokens,
        )
        self.transformer = build_model(mcfg)

    def get_input_embeddings(self):
        # titans-pytorch does not expose embeddings directly; not supported here
        return None

    def set_input_embeddings(self, value):
        # Not supported
        pass

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithCrossAttentions:
        # titans-pytorch ignores attention_mask; segment handling is internal.
        out = self.transformer(input_ids, return_loss=False)
        logits = out if not isinstance(out, dict) else out.get("logits", out)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        **kwargs,
    ):
        # No KV cache support; generation will be slower but functional for small models.
        return {"input_ids": input_ids}

    @staticmethod
    def _reorder_cache(past, beam_idx):
        return past


__all__ = ["TitansConfig", "TitansForCausalLM"]
