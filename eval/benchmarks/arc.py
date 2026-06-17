"""
ARC Challenge — AI2 Reasoning Challenge (hard subset)
Dataset: allenai/ARC (ARC-Challenge split)
Metric:  accuracy

Science questions that require reasoning beyond simple retrieval.
The Challenge set specifically filters out questions that retrieval-based
and word co-occurrence models can answer correctly.
"""

from __future__ import annotations

import re

from eval.benchmarks.base import DEFAULT_CFG, BaseBenchmark
from eval.model import GenerateConfig, ModelBackend
from eval.results import BenchmarkResult

SYSTEM_PROMPT = (
    "Answer the following science question by choosing the correct option. "
    "Respond with only the letter of your answer (A, B, C, or D)."
)

ANSWER_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)


def _parse(text: str) -> str | None:
    text = text.strip()
    if text and text[0].upper() in "ABCD":
        return text[0].upper()
    m = ANSWER_RE.search(text)
    return m.group(1).upper() if m else None


class ARCChallengeBenchmark(BaseBenchmark):
    name = "arc_challenge"
    suite = "intelligence"
    metric = "accuracy"

    def __init__(self, max_samples: int = 0):
        self.max_samples = max_samples

    def run(self, model: ModelBackend, cfg: GenerateConfig = DEFAULT_CFG) -> BenchmarkResult:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets") from None

        cfg = GenerateConfig(max_tokens=16, temperature=0.0, system_prompt=SYSTEM_PROMPT)
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        rows = list(ds)
        if self.max_samples:
            rows = rows[: self.max_samples]

        correct = 0
        for row in rows:
            choices = row["choices"]
            labels = choices["label"]  # e.g. ["A","B","C","D"] or ["1","2","3","4"]
            texts = choices["text"]

            options = "\n".join(f"{lbl}) {t}" for lbl, t in zip(labels, texts, strict=False))
            prompt = f"{row['question']}\n\n{options}\n\nAnswer:"

            response = model.complete(prompt, cfg)
            predicted = _parse(response)

            # Normalize answerKey — some datasets use 1/2/3/4 instead of A/B/C/D
            answer_key = row["answerKey"].strip()
            if answer_key.isdigit():
                idx = int(answer_key) - 1
                answer_key = labels[idx] if idx < len(labels) else answer_key

            if predicted and predicted == answer_key.upper():
                correct += 1

        total = len(rows)
        return self._result(
            score=round(correct / total, 4) if total else 0.0,
            details={"correct": correct, "total": total},
        )
