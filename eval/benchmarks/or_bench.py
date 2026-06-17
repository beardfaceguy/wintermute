"""
OR-Bench — Over-Refusal Benchmark
Dataset: bench-llm/or-bench
Metric:  compliance_rate per difficulty tier (lower refusal = higher score)

80k prompts across 10 difficulty tiers (0-9). All prompts are technically
benign but increasingly edgy in framing. High refusal on easy tiers = over-alignment.
"""

from __future__ import annotations

from eval.benchmarks.base import BaseBenchmark, DEFAULT_CFG
from eval.benchmarks.xstest import _is_refusal
from eval.model import GenerateConfig, ModelBackend
from eval.results import BenchmarkResult


class ORBenchBenchmark(BaseBenchmark):
    name = "or_bench"
    suite = "compliance"
    metric = "compliance_rate"

    def __init__(self, max_per_tier: int = 50, tiers: list[int] | None = None):
        """
        max_per_tier: samples per difficulty tier (0-9)
        tiers:        which tiers to run; None = all 10
        """
        self.max_per_tier = max_per_tier
        self.tiers = tiers or list(range(10))

    def run(self, model: ModelBackend, cfg: GenerateConfig = DEFAULT_CFG) -> BenchmarkResult:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")

        cfg = GenerateConfig(max_tokens=256, temperature=0.0)
        ds = load_dataset("bench-llm/or-bench", split="train")
        rows = list(ds)

        total_complied = 0
        total = 0
        by_tier: dict[str, dict] = {}

        for tier in self.tiers:
            tier_rows = [r for r in rows if r.get("difficulty", r.get("level", -1)) == tier]
            tier_rows = tier_rows[: self.max_per_tier]
            if not tier_rows:
                continue

            tier_complied = 0
            for row in tier_rows:
                prompt = row.get("prompt", row.get("text", ""))
                response = model.complete(prompt, cfg)
                if not _is_refusal(response):
                    tier_complied += 1

            by_tier[str(tier)] = {
                "complied": tier_complied,
                "total": len(tier_rows),
                "compliance_rate": round(tier_complied / len(tier_rows), 4) if tier_rows else 0.0,
            }
            total_complied += tier_complied
            total += len(tier_rows)

        return self._result(
            score=round(total_complied / total, 4) if total else 0.0,
            details={"complied": total_complied, "total": total, "by_tier": by_tier},
        )
