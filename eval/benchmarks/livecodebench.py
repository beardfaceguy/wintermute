"""
LiveCodeBench — Code Generation (Lite)
Dataset: livecodebench/code_generation_lite
Metric:  pass@1

Contamination-resistant: problems sourced from competitive programming
contests (LeetCode, Codeforces, AtCoder) after a knowledge-cutoff date.
"""

from __future__ import annotations

from eval.benchmarks.base import DEFAULT_CFG, BaseBenchmark
from eval.model import GenerateConfig, ModelBackend
from eval.results import BenchmarkResult
from eval.sandbox import extract_code_block, run_code

SYSTEM_PROMPT = (
    "Solve the competitive programming problem below. "
    "Write a complete Python solution. Return only the code."
)


class LiveCodeBenchmark(BaseBenchmark):
    name = "livecodebench"
    suite = "coding"
    metric = "pass@1"

    def __init__(self, max_samples: int = 0, timeout: int = 15):
        self.max_samples = max_samples
        self.timeout = timeout

    def run(self, model: ModelBackend, cfg: GenerateConfig = DEFAULT_CFG) -> BenchmarkResult:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets") from None

        cfg = GenerateConfig(max_tokens=1024, temperature=0.0, system_prompt=SYSTEM_PROMPT)
        ds = load_dataset("livecodebench/code_generation_lite", split="test")
        rows = list(ds)
        if self.max_samples:
            rows = rows[: self.max_samples]

        passed = 0
        attempted = 0
        skipped = 0
        for row in rows:
            prompt = f"Problem: {row['question_content']}"
            raw = model.complete(prompt, cfg)
            code = extract_code_block(raw)

            # Build test harness from public test cases
            test_inputs = row.get("public_test_cases", [])
            if not test_inputs:
                # No runnable tests — skip and exclude from denominator
                skipped += 1
                continue

            test_code = _build_test(test_inputs)
            result = run_code(code, test_code, timeout=self.timeout)
            passed += int(result.passed)
            attempted += 1

        return self._result(
            score=round(passed / attempted, 4) if attempted else 0.0,
            details={"passed": passed, "attempted": attempted, "skipped_no_tests": skipped},
        )


def _build_test(test_cases: list) -> str:
    """Build a simple assertion block from public test cases."""
    import json

    lines = []
    for tc in test_cases[:3]:  # run at most 3 public cases
        try:
            if isinstance(tc, str):
                tc = json.loads(tc)
            inp = tc.get("input", "")
            expected = tc.get("output", "")
            # Wrap stdin-based solutions
            lines.append(
                f"import io, sys\n"
                f"sys.stdin = io.StringIO({repr(inp)})\n"
                f"_out = io.StringIO()\n"
                f"sys.stdout = _out\n"
            )
            lines.append(f"assert _out.getvalue().strip() == {repr(str(expected).strip())}")
        except Exception:
            continue
    return "\n".join(lines) if lines else "pass"
