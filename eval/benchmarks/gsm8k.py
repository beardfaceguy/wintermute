"""
GSM8K — Grade School Math
Dataset: openai/gsm8k
Metric:  accuracy (exact final number match)

Multi-step arithmetic word problems. The model must reason through
the problem and produce a numeric answer. We extract the final number
from the response and compare to ground truth.
"""

from __future__ import annotations

import re

from eval.benchmarks.base import DEFAULT_CFG, BaseBenchmark
from eval.model import GenerateConfig, ModelBackend
from eval.results import BenchmarkResult

SYSTEM_PROMPT = (
    "Solve the math problem step by step. "
    "At the end of your response, write your final numeric answer on its own line "
    "in the format: #### <number>"
)

NUMBER_RE = re.compile(r"####\s*([\d,\.\-]+)")
FALLBACK_RE = re.compile(r"([\d,]+\.?\d*)\s*$", re.MULTILINE)


def _extract_answer(text: str) -> str | None:
    """Pull the final numeric answer out of a model response."""
    m = NUMBER_RE.search(text)
    if m:
        return m.group(1).replace(",", "").strip()
    # Fallback: last number in the response
    m = FALLBACK_RE.search(text.strip())
    return m.group(1).replace(",", "").strip() if m else None


def _normalize(val: str) -> str:
    try:
        return str(float(val.replace(",", "")))
    except ValueError:
        return val.strip().lower()


class GSM8KBenchmark(BaseBenchmark):
    name = "gsm8k"
    suite = "intelligence"
    metric = "accuracy"

    def __init__(self, max_samples: int = 0):
        self.max_samples = max_samples

    def run(self, model: ModelBackend, cfg: GenerateConfig = DEFAULT_CFG) -> BenchmarkResult:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets") from None

        cfg = GenerateConfig(max_tokens=256, temperature=0.0, system_prompt=SYSTEM_PROMPT)
        ds = load_dataset("openai/gsm8k", "main", split="test")
        rows = list(ds)
        if self.max_samples:
            rows = rows[: self.max_samples]

        correct = 0
        for row in rows:
            response = model.complete(row["question"], cfg)
            predicted = _extract_answer(response)
            # Ground truth is after #### in the answer field
            gt_match = NUMBER_RE.search(row["answer"])
            expected = gt_match.group(1).replace(",", "").strip() if gt_match else None
            if predicted and expected and _normalize(predicted) == _normalize(expected):
                correct += 1

        total = len(rows)
        return self._result(
            score=round(correct / total, 4) if total else 0.0,
            details={"correct": correct, "total": total},
        )
