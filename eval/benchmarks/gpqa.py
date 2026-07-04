"""
GPQA Diamond — Graduate-level Google-Proof Q&A
Dataset: google-deepmind/gpqa (gpqa_diamond split, 448 questions)
Metric:  accuracy (0-shot)

Questions are intentionally difficult for non-experts; even with Google
access, non-specialists score ~65%. Expert baseline ~74%.
"""

from __future__ import annotations

import hashlib
import re

from eval.benchmarks.base import DEFAULT_CFG, BaseBenchmark
from eval.model import GenerateConfig, ModelBackend
from eval.results import BenchmarkResult

SYSTEM_PROMPT = (
    "You are an expert scientist answering multiple choice questions. "
    "Your response must be exactly one character: A, B, C, or D. "
    "Do not write any explanation, punctuation, or other text."
)

PROMPT_TEMPLATE = """{question}

A) {A}
B) {B}
C) {C}
D) {D}

Answer with a single letter (A, B, C, or D):"""

# Patterns to extract a letter from verbose responses (e.g. "The answer is B")
ANSWER_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)
ANSWER_PHRASE_RE = re.compile(
    r"(?:answer\s*(?:is|:)\s*\*{0,2}([A-D])\b|\b([A-D])\s*(?:is\s+correct|\)|\.))", re.IGNORECASE
)
BOXED_RE = re.compile(r"\\boxed\{\s*([A-D])\s*\}", re.IGNORECASE)
CHOICES = ["A", "B", "C", "D"]


def _parse(text: str) -> str | None:
    text = text.strip()
    # Fast path: single letter response
    if len(text) <= 2 and text and text[0].upper() in CHOICES:
        return text[0].upper()
    # LaTeX-formatting models may box the letter: \boxed{C}
    m = BOXED_RE.search(text)
    if m:
        return m.group(1).upper()
    # Look for "answer is B", "answer: C", "B is correct", etc.
    m = ANSWER_PHRASE_RE.search(text)
    if m:
        letter = m.group(1) or m.group(2)
        return letter.upper() if letter else None
    # Last A-D letter in the text (e.g. model finishes with "...therefore B.")
    matches = ANSWER_RE.findall(text)
    if matches:
        return matches[-1].upper()
    return None


class GPQADiamondBenchmark(BaseBenchmark):
    name = "gpqa_diamond"
    suite = "intelligence"
    metric = "accuracy"

    def __init__(self, max_samples: int = 0):
        # max_samples=0 means all 448
        self.max_samples = max_samples

    def run(self, model: ModelBackend, cfg: GenerateConfig = DEFAULT_CFG) -> BenchmarkResult:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets") from None

        cfg = GenerateConfig(max_tokens=16, temperature=0.0, system_prompt=SYSTEM_PROMPT)
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        rows = list(ds)
        if self.max_samples:
            rows = rows[: self.max_samples]

        correct = 0
        for row in rows:
            # Raw data always puts the correct answer at index 0 — shuffle
            # deterministically by question hash so the correct answer lands at
            # a different position each time, making the benchmark meaningful.
            choices = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            seed = int(hashlib.md5(row["Question"].encode()).hexdigest()[:8], 16)
            order = sorted(range(4), key=lambda i: (seed + i * 2654435761) % (2**32))
            shuffled = [choices[i] for i in order]
            correct_idx = order.index(0)  # where the correct answer landed
            correct_letter = CHOICES[correct_idx]

            prompt = PROMPT_TEMPLATE.format(
                question=row["Question"],
                A=shuffled[0],
                B=shuffled[1],
                C=shuffled[2],
                D=shuffled[3],
            )
            raw = model.complete(prompt, cfg)
            if _parse(raw) == correct_letter:
                correct += 1

        total = len(rows)
        return self._result(
            score=round(correct / total, 4) if total else 0.0,
            details={"correct": correct, "total": total},
        )
