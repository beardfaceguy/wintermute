"""Base class all benchmark adapters inherit from."""

from __future__ import annotations

from abc import ABC, abstractmethod

from eval.model import GenerateConfig, ModelBackend
from eval.results import BenchmarkResult

DEFAULT_CFG = GenerateConfig(max_tokens=512, temperature=0.0)


class BaseBenchmark(ABC):
    # Subclasses set these as class attributes
    name: str  # e.g. "mmlu"
    suite: str  # "intelligence" | "coding" | "agentic" | "memory" | "groundedness" | "compliance" | "personality"
    metric: str  # e.g. "accuracy"

    @abstractmethod
    def run(self, model: ModelBackend, cfg: GenerateConfig = DEFAULT_CFG) -> BenchmarkResult:
        """Run the benchmark against model and return a BenchmarkResult."""

    def _result(self, score: float, details: dict | None = None) -> BenchmarkResult:
        return BenchmarkResult(
            benchmark=self.name,
            suite=self.suite,
            metric=self.metric,
            score=score,
            details=details or {},
        )
