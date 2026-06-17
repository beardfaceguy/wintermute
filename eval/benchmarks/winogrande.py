"""
WinoGrande — Large-Scale Winograd Schema Challenge
Dataset: allenai/winogrande
Metric:  accuracy

Fill-in-the-blank pronoun resolution requiring commonsense reasoning.
Uses adversarial filtering (AF) to remove biases, making it harder
than the original Winograd schema. Two options per question.
"""

from __future__ import annotations

import re
from typing import Optional

from eval.benchmarks.base import BaseBenchmark, DEFAULT_CFG
from eval.model import GenerateConfig, ModelBackend
from eval.results import BenchmarkResult

SYSTEM_PROMPT = (
    "Complete the sentence by choosing the correct option. "
    "Respond with only '1' or '2'."
)

ANSWER_RE = re.compile(r"\b([12])\b")


def _parse(text: str) -> Optional[str]:
    text = text.strip()
    if text and text[0] in "12":
        return text[0]
    m = ANSWER_RE.search(text)
    return m.group(1) if m else None


class WinoGrandeBenchmark(BaseBenchmark):
    name = "winogrande"
    suite = "intelligence"
    metric = "accuracy"

    def __init__(self, max_samples: int = 0):
        self.max_samples = max_samples

    def run(self, model: ModelBackend, cfg: GenerateConfig = DEFAULT_CFG) -> BenchmarkResult:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")

        cfg = GenerateConfig(max_tokens=8, temperature=0.0, system_prompt=SYSTEM_PROMPT)
        ds = load_dataset("allenai/winogrande", "winogrande_debiased", split="validation")
        rows = list(ds)
        if self.max_samples:
            rows = rows[: self.max_samples]

        correct = 0
        for row in rows:
            prompt = (
                f"Sentence: {row['sentence']}\n\n"
                f"Option 1: {row['option1']}\n"
                f"Option 2: {row['option2']}\n\n"
                f"Which option correctly fills the blank? (1 or 2):"
            )
            response = model.complete(prompt, cfg)
            predicted = _parse(response)
            if predicted and predicted == str(row["answer"]):
                correct += 1

        total = len(rows)
        return self._result(
            score=round(correct / total, 4) if total else 0.0,
            details={"correct": correct, "total": total},
        )
