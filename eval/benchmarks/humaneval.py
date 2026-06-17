"""
HumanEval
Dataset: openai_humaneval
Metric:  pass@1 (fraction of problems where the first attempt passes all tests)

Each problem provides a function signature + docstring. The model must complete
the function body. We execute the completion against the bundled test suite.
"""

from __future__ import annotations

from eval.benchmarks.base import BaseBenchmark, DEFAULT_CFG
from eval.model import GenerateConfig, ModelBackend
from eval.results import BenchmarkResult
from eval.sandbox import extract_code_block, run_code

SYSTEM_PROMPT = (
    "Complete the following Python function. "
    "Return only the function implementation, no explanation."
)

PROMPT_TEMPLATE = """{prompt}
    # Your implementation here:
"""


class HumanEvalBenchmark(BaseBenchmark):
    name = "humaneval"
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
        ds = load_dataset("openai_humaneval", split="test")
        rows = list(ds)
        if self.max_samples:
            rows = rows[: self.max_samples]

        passed = 0
        details = []
        for row in rows:
            prompt = PROMPT_TEMPLATE.format(prompt=row["prompt"])
            raw = model.complete(prompt, cfg)
            code = extract_code_block(raw)

            # The model may repeat the signature — if not, prepend it
            if row["entry_point"] not in code:
                code = row["prompt"] + "\n" + code

            result = run_code(code, row["test"] + f"\ncheck({row['entry_point']})", timeout=self.timeout)
            passed += int(result.passed)
            details.append({
                "task_id": row["task_id"],
                "passed": result.passed,
                "timed_out": result.timed_out,
            })

        total = len(rows)
        return self._result(
            score=round(passed / total, 4) if total else 0.0,
            details={"passed": passed, "total": total, "per_task": details},
        )
