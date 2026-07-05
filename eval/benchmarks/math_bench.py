"""
MATH — Competition Math Benchmark
Dataset: TIGER-Lab/MATH-plus (train split, capped at max_samples)
Metric:  accuracy (exact answer match after normalization)

Challenging competition math problems. Solutions contain \\boxed{} answers
which we extract and normalize for comparison.
"""

from __future__ import annotations

from eval.benchmarks.base import DEFAULT_CFG, BaseBenchmark
from eval.model import GenerateConfig, ModelBackend
from eval.results import BenchmarkResult

SYSTEM_PROMPT = (
    "Solve the math problem. Show your work, then write your final answer "
    "inside \\boxed{} at the end. Example: \\boxed{42}"
)

def _extract(text: str) -> str | None:
    r"""Return the content of the first \boxed{...}, matching nested braces.

    A regex like \\boxed\{([^}]+)\} stops at the first '}', which truncates
    answers containing nested braces (e.g. \boxed{(-1,\sqrt{3},\sqrt{2})} would
    yield only '(-1,\sqrt{3'). We scan for the balanced closing brace instead.
    """
    key = "\\boxed{"
    start = text.find(key)
    if start == -1:
        return None
    depth = 1
    inner_start = start + len(key)
    for j in range(inner_start, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[inner_start:j].strip()
    return None  # unbalanced — no matching close brace


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

    def __init__(self, max_samples: int = 0):
        """
        max_samples: cap total problems (0 = all ~893k — use a cap in practice)
        """
        self.max_samples = max_samples

    def run(self, model: ModelBackend, cfg: GenerateConfig = DEFAULT_CFG) -> BenchmarkResult:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets") from None

        cfg = GenerateConfig(max_tokens=1024, temperature=0.0, system_prompt=SYSTEM_PROMPT)
        ds = load_dataset("TIGER-Lab/MATH-plus", split="train")
        rows = list(ds)
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
