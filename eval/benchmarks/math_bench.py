"""
MATH — Hendrycks Math Benchmark
Dataset: hendrycks/MATH
Metric:  accuracy (exact answer match after normalization)

Challenging problems from AMC, AIME, and competition math.
Answers are often expressions, fractions, or LaTeX — we normalize
before comparing. Covers 7 subjects: algebra, counting, geometry,
intermediate algebra, number theory, prealgebra, precalculus.
"""

from __future__ import annotations

import re
from typing import Optional

from eval.benchmarks.base import BaseBenchmark, DEFAULT_CFG
from eval.model import GenerateConfig, ModelBackend
from eval.results import BenchmarkResult

SYSTEM_PROMPT = (
    "Solve the math problem. Show your work, then write your final answer "
    "inside \\boxed{} at the end. Example: \\boxed{42}"
)

BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


def _extract(text: str) -> Optional[str]:
    m = BOXED_RE.search(text)
    return m.group(1).strip() if m else None


def _normalize(val: str) -> str:
    """Light normalization: strip whitespace, LaTeX commas, dollar signs."""
    val = val.strip().replace("$", "").replace(",", "").replace(" ", "")
    val = val.replace("\\!", "").replace("\\,", "")
    try:
        return str(float(val))
    except ValueError:
        return val.lower()


class MATHBenchmark(BaseBenchmark):
    name = "math"
    suite = "intelligence"
    metric = "accuracy"

    def __init__(self, max_samples: int = 0, levels: list[str] | None = None):
        """
        max_samples: cap total problems (0 = all ~5000)
        levels:      difficulty filter e.g. ["Level 1", "Level 2"] (None = all)
        """
        self.max_samples = max_samples
        self.levels = levels

    def run(self, model: ModelBackend, cfg: GenerateConfig = DEFAULT_CFG) -> BenchmarkResult:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")

        cfg = GenerateConfig(max_tokens=1024, temperature=0.0, system_prompt=SYSTEM_PROMPT)
        ds = load_dataset("TIGER-Lab/MATH-plus", split="train")
        rows = list(ds)
        if self.levels:
            rows = [r for r in rows if r.get("level") in self.levels]
        if self.max_samples:
            rows = rows[: self.max_samples]

        correct = 0
        by_type: dict[str, dict] = {}

        for row in rows:
            response = model.complete(row["instruction"], cfg)
            predicted = _extract(response)
            expected = _extract(row["output"]) or ""

            ok = bool(predicted and expected and _normalize(predicted) == _normalize(expected))
            if ok:
                correct += 1

            t = row.get("type", "unknown")
            if t not in by_type:
                by_type[t] = {"correct": 0, "total": 0}
            by_type[t]["total"] += 1
            by_type[t]["correct"] += int(ok)

        total = len(rows)
        return self._result(
            score=round(correct / total, 4) if total else 0.0,
            details={"correct": correct, "total": total, "by_type": by_type},
        )
