"""LocalBackend — run the SFT job in-process on the local machine.

Thin wrapper over train.train(). Suitable for tiny-model smoke runs and for
local GPUs large enough to hold the model; real 8B training targets SageMaker.
"""

from __future__ import annotations

from model_training.sft.backends.base import TrainBackend
from model_training.sft.config import SFTConfig


class LocalBackend(TrainBackend):
    def run(self, cfg: SFTConfig) -> str:
        # Lazy import: pulls in torch/trl/transformers only when actually training.
        from model_training.sft.train import train

        return train(cfg)
