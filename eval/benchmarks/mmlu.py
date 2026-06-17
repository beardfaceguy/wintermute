"""
MMLU — Massive Multitask Language Understanding
Dataset: cais/mmlu on HuggingFace
Metric:  accuracy (macro average across all subjects)

Each question is 4-way multiple choice. We prompt with the question and
choices A-D and parse the first letter from the model's response.

Subjects can be filtered with the `subjects` parameter to run a fast
subset rather than all 57 subjects.
"""

from __future__ import annotations

import re
from typing import Optional

from eval.benchmarks.base import BaseBenchmark, DEFAULT_CFG
from eval.model import GenerateConfig, ModelBackend
from eval.results import BenchmarkResult

SYSTEM_PROMPT = (
    "You are a helpful assistant answering multiple choice questions. "
    "Respond with only the letter of the correct answer: A, B, C, or D."
)

PROMPT_TEMPLATE = """{question}

A) {A}
B) {B}
C) {C}
D) {D}

Answer:"""

ANSWER_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)

CHOICES = ["A", "B", "C", "D"]


def _parse_answer(text: str) -> Optional[str]:
    text = text.strip()
    # First char shortcut
    if text and text[0].upper() in CHOICES:
        return text[0].upper()
    m = ANSWER_RE.search(text)
    return m.group(1).upper() if m else None


class MMLUBenchmark(BaseBenchmark):
    name = "mmlu"
    suite = "intelligence"
    metric = "accuracy"

    def __init__(
        self,
        subjects: Optional[list[str]] = None,
        max_per_subject: int = 100,
        split: str = "test",
    ):
        """
        subjects:         list of MMLU subject names, or None for all 57
        max_per_subject:  cap per subject to keep runtime reasonable
        split:            "test" (default) or "validation"
        """
        self.subjects = subjects
        self.max_per_subject = max_per_subject
        self.split = split

    def run(self, model: ModelBackend, cfg: GenerateConfig = DEFAULT_CFG) -> BenchmarkResult:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")

        cfg = GenerateConfig(
            max_tokens=cfg.max_tokens,
            temperature=0.0,  # always deterministic for MC
            top_p=cfg.top_p,
            system_prompt=SYSTEM_PROMPT,
        )

        subjects = self.subjects or self._all_subjects()
        correct = 0
        total = 0
        per_subject: dict[str, dict] = {}

        for subject in subjects:
            ds = load_dataset("cais/mmlu", subject, split=self.split)
            rows = list(ds)
            if self.max_per_subject:
                rows = rows[: self.max_per_subject]

            subj_correct = 0
            for row in rows:
                prompt = PROMPT_TEMPLATE.format(
                    question=row["question"],
                    A=row["choices"][0],
                    B=row["choices"][1],
                    C=row["choices"][2],
                    D=row["choices"][3],
                )
                raw = model.complete(prompt, cfg)
                predicted = _parse_answer(raw)
                expected = CHOICES[row["answer"]]
                if predicted == expected:
                    subj_correct += 1

            per_subject[subject] = {
                "correct": subj_correct,
                "total": len(rows),
                "accuracy": subj_correct / len(rows) if rows else 0.0,
            }
            correct += subj_correct
            total += len(rows)

        accuracy = correct / total if total else 0.0
        return self._result(
            score=round(accuracy, 4),
            details={"correct": correct, "total": total, "per_subject": per_subject},
        )

    @staticmethod
    def _all_subjects() -> list[str]:
        return [
            "abstract_algebra", "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology", "college_chemistry",
            "college_computer_science", "college_mathematics", "college_medicine",
            "college_physics", "computer_security", "conceptual_physics",
            "econometrics", "electrical_engineering", "elementary_mathematics",
            "formal_logic", "global_facts", "high_school_biology",
            "high_school_chemistry", "high_school_computer_science",
            "high_school_european_history", "high_school_geography",
            "high_school_government_and_politics", "high_school_macroeconomics",
            "high_school_mathematics", "high_school_microeconomics",
            "high_school_physics", "high_school_psychology",
            "high_school_statistics", "high_school_us_history",
            "high_school_world_history", "human_aging", "human_sexuality",
            "international_law", "jurisprudence", "logical_fallacies",
            "machine_learning", "management", "marketing", "medical_genetics",
            "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
            "philosophy", "prehistory", "professional_accounting",
            "professional_law", "professional_medicine", "professional_psychology",
            "public_relations", "security_studies", "sociology", "us_foreign_policy",
            "virology", "world_religions",
        ]
