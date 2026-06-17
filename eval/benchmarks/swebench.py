"""
SWE-bench Verified
Dataset: princeton-nlp/SWE-bench_Verified
Metric:  resolve_rate (% of issues where the patch passes the test suite)

NOTE: Full execution requires Docker for sandboxed repo checkout + test running.
This adapter generates patches but requires the official SWE-bench harness to
score them. See: https://github.com/swe-bench/SWE-bench

Quick setup for full scoring:
  pip install swebench
  python -m swebench.harness.run_evaluation \
    --predictions_path eval/results/<run_id>_swebench_patches.jsonl \
    --run_id <run_id>

This adapter runs in two modes:
  - generate (default): writes patches to a JSONL file for offline scoring
  - score: reads a pre-scored results file if available
"""

from __future__ import annotations

import json

from eval.benchmarks.base import DEFAULT_CFG, BaseBenchmark
from eval.model import GenerateConfig, ModelBackend
from eval.results import RESULTS_DIR, BenchmarkResult
from eval.sandbox import extract_code_block

SYSTEM_PROMPT = """You are an expert software engineer. Given a GitHub issue and relevant code context,
produce a minimal unified diff patch that resolves the issue.
Return ONLY the patch in unified diff format (--- a/file, +++ b/file, @@ ... @@)."""

PROMPT_TEMPLATE = """Repository: {repo}
Issue: {problem_statement}

Relevant code:
{hints}

Produce a unified diff patch to fix this issue:"""


class SWEBenchVerifiedBenchmark(BaseBenchmark):
    name = "swebench_verified"
    suite = "agentic"
    metric = "resolve_rate"

    def __init__(self, max_samples: int = 0):
        self.max_samples = max_samples

    def run(self, model: ModelBackend, cfg: GenerateConfig = DEFAULT_CFG) -> BenchmarkResult:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets") from None

        cfg = GenerateConfig(max_tokens=2048, temperature=0.0, system_prompt=SYSTEM_PROMPT)
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        rows = list(ds)
        if self.max_samples:
            rows = rows[: self.max_samples]

        patches = []
        for row in rows:
            prompt = PROMPT_TEMPLATE.format(
                repo=row.get("repo", ""),
                problem_statement=row.get("problem_statement", ""),
                hints=row.get("hints_text", "")[:2000],
            )
            raw = model.complete(prompt, cfg)
            patch = extract_code_block(raw) if "```" in raw else raw.strip()
            patches.append(
                {
                    "instance_id": row["instance_id"],
                    "model_patch": patch,
                    "model_name_or_path": model.model_id,
                }
            )

        # Write patches for offline scoring with the official harness
        patches_path = RESULTS_DIR / f"swebench_patches_{model.model_id.replace('/', '_')}.jsonl"
        RESULTS_DIR.mkdir(exist_ok=True)
        with open(patches_path, "w") as f:
            for p in patches:
                f.write(json.dumps(p) + "\n")

        print(f"\n  SWE-bench patches written to: {patches_path}")
        print(
            f"  Score with: python -m swebench.harness.run_evaluation --predictions_path {patches_path}"
        )
        print("  (requires Docker + swebench package)\n")

        # Return -1 score as sentinel indicating patches generated but not yet scored
        return self._result(
            score=-1.0,
            details={
                "status": "patches_generated",
                "patches_file": str(patches_path),
                "num_instances": len(patches),
                "note": "Run offline scoring with swebench harness to get resolve_rate",
            },
        )
