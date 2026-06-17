"""
MBPP — Mostly Basic Python Problems
Dataset: google-research-datasets/mbpp (sanitized split)
Metric:  pass@1

Each problem has a text description and 3 test assertions.
"""

from __future__ import annotations

from eval.benchmarks.base import BaseBenchmark, DEFAULT_CFG
from eval.model import GenerateConfig, ModelBackend
from eval.results import BenchmarkResult
from eval.sandbox import extract_code_block, run_code

SYSTEM_PROMPT = (
    "Write a Python function to solve the problem. "
    "Return only the function code, no explanation or test cases."
)


class MBPPBenchmark(BaseBenchmark):
    name = "mbpp"
    suite = "coding"
    metric = "pass@1"

    def __init__(self, max_samples: int = 0, timeout: int = 10):
        self.max_samples = max_samples
        self.timeout = timeout

    def run(self, model: ModelBackend, cfg: GenerateConfig = DEFAULT_CFG) -> BenchmarkResult:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")

        cfg = GenerateConfig(max_tokens=512, temperature=0.0, system_prompt=SYSTEM_PROMPT)
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        rows = list(ds)
        if self.max_samples:
            rows = rows[: self.max_samples]

        passed = 0
        for row in rows:
            prompt = f"Problem: {row['text']}\n\nWrite a Python function to solve it."
            raw = model.complete(prompt, cfg)
            code = extract_code_block(raw)
            test_code = "\n".join(row["test_list"])
            result = run_code(code, test_code, timeout=self.timeout)
            passed += int(result.passed)

        total = len(rows)
        return self._result(
            score=round(passed / total, 4) if total else 0.0,
            details={"passed": passed, "total": total},
        )
