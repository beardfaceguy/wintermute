"""
Titans MAC small-model factory.

Assumes `titans-pytorch` is installed (or the repo is on PYTHONPATH).
This is a minimal scaffold to instantiate a tiny MAC transformer with an LM head.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

try:
    from titans_pytorch import MemoryAsContextTransformer
except ImportError as e:
    raise ImportError(
        "titans-pytorch is required. Install with `pip install titans-pytorch` "
        "or ensure the repo is on PYTHONPATH."
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


class TitansLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.variant.lower() != "mac":
            raise ValueError(f"Unsupported variant {cfg.variant}; only 'mac' is implemented here.")
        self.model = MemoryAsContextTransformer(
            num_tokens=cfg.vocab_size,
            dim=cfg.dim,
            depth=cfg.depth,
            heads=cfg.heads,
            ff_mult=cfg.ff_mult,
            segment_len=cfg.segment_len,
            num_persist_mem_tokens=cfg.num_persist_mem_tokens,
            num_longterm_mem_tokens=cfg.num_longterm_mem_tokens,
        )
        # LM head is built into MemoryAsContextTransformer; if not, uncomment below:
        # self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    def forward(self, x, return_loss: bool = False):
        # titans-pytorch uses return_loss flag to compute CE internally
        return self.model(x, return_loss=return_loss)


def build_model(cfg: ModelConfig) -> TitansLM:
    return TitansLM(cfg)

