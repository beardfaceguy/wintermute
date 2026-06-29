"""TrainBackend — the interface every training-compute backend implements.

A backend takes a validated SFTConfig, runs the fine-tune wherever it runs
(in-process, SageMaker Training Job, …), and returns a URI/path to the
resulting checkpoint.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from model_training.sft.config import SFTConfig


class TrainBackend(ABC):
    @abstractmethod
    def run(self, cfg: SFTConfig) -> str:
        """Run the SFT job and return a path/URI to the resulting checkpoint."""
