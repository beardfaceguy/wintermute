"""
HellaSwag — Commonsense NLI
Dataset: allenai/HellaSwag
Metric:  accuracy

Given a partial description of an activity, pick the most plausible
sentence continuation from 4 options. Designed to be easy for humans
(~95%) but hard for models without genuine commonsense understanding.
"""

from __future__ import annotations

import re

from eval.benchmarks.base import DEFAULT_CFG, BaseBenchmark
from eval.model import GenerateConfig, ModelBackend
from eval.results import BenchmarkResult

SYSTEM_PROMPT = (
    "Choose the most natural and logical continuation of the given text. "
    "Respond with only the number of your answer (0, 1, 2, or 3)."
)

ANSWER_RE = re.compile(r"\b([0-3])\b")


def _parse(text: str) -> int | None:
    text = text.strip()
    if text and text[0] in "0123":
        return int(text[0])
    m = ANSWER_RE.search(text)
    return int(m.group(1)) if m else None


def _clean(text: str) -> str:
    """Strip HellaSwag's [header] tokens."""
    return re.sub(r"\s+", " ", text.replace("[header]", "").replace("[step]", "")).strip()


class HellaSwagBenchmark(BaseBenchmark):
    name = "hellaswag"
    suite = "intelligence"
    metric = "accuracy"

    def __init__(self, max_samples: int = 0):
        self.max_samples = max_samples

    def run(self, model: ModelBackend, cfg: GenerateConfig = DEFAULT_CFG) -> BenchmarkResult:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets") from None

        cfg = GenerateConfig(max_tokens=8, temperature=0.0, system_prompt=SYSTEM_PROMPT)
        ds = load_dataset("allenai/HellaSwag", split="validation")
        rows = list(ds)
        if self.max_samples:
            rows = rows[: self.max_samples]

        correct = 0
        for row in rows:
            ctx = _clean(row["ctx"])
            endings = [_clean(e) for e in row["endings"]]
            options = "\n".join(f"{i}) {e}" for i, e in enumerate(endings))
            prompt = f"Text: {ctx}\n\nContinuations:\n{options}\n\nBest continuation (0-3):"

            response = model.complete(prompt, cfg)
            predicted = _parse(response)
            expected = int(row["label"])

            if predicted is not None and predicted == expected:
                correct += 1

        total = len(rows)
        return self._result(
            score=round(correct / total, 4) if total else 0.0,
            details={"correct": correct, "total": total},
        )
