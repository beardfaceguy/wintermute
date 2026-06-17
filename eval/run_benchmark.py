#!/usr/bin/env python3
"""
Wintermute Model Benchmarking Harness
======================================
Run one or more benchmark suites against a local or remote model.

Examples
--------
# Self-hosted vLLM (most common)
python eval/run_benchmark.py \\
    --target http://localhost:8010 \\
    --model wintermute-mistral-7b-sft \\
    --suite intelligence

# OpenAI
python eval/run_benchmark.py \\
    --target https://api.openai.com \\
    --model gpt-4o \\
    --api-key $OPENAI_API_KEY \\
    --suite coding

# Local HuggingFace checkpoint
python eval/run_benchmark.py \\
    --target hf:/mnt/checkpoints/my_sft_step3000 \\
    --suite all

# Fast smoke test (MMLU only, 10 questions per subject)
python eval/run_benchmark.py \\
    --target http://localhost:8010 \\
    --model wintermute-mistral-7b-sft \\
    --suite intelligence \\
    --fast

# List past runs
python eval/run_benchmark.py --list-runs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# Ensure repo root is on sys.path when run directly
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Load .env from repo root so HF_TOKEN and other secrets are available
# without requiring the caller to export them manually
_env_file = _REPO_ROOT / ".env"
if _env_file.exists():
    import os
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            # Force-set: .env values override empty shell vars (e.g. from shell rc files)
        _key, _val = _k.strip(), _v.strip()
        if _val and not os.environ.get(_key):
            os.environ[_key] = _val

from eval.model import make_backend, GenerateConfig
from eval.results import new_run, save_run, list_runs


# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------

def _load_benchmarks(suite: str, fast: bool) -> list:
    """Instantiate benchmark objects for the requested suite."""
    from eval.benchmarks.mmlu import MMLUBenchmark
    from eval.benchmarks.gpqa import GPQADiamondBenchmark
    from eval.benchmarks.ifeval import IFEvalBenchmark
    from eval.benchmarks.gsm8k import GSM8KBenchmark
    from eval.benchmarks.math_bench import MATHBenchmark
    from eval.benchmarks.arc import ARCChallengeBenchmark
    from eval.benchmarks.hellaswag import HellaSwagBenchmark
    from eval.benchmarks.winogrande import WinoGrandeBenchmark
    from eval.benchmarks.truthfulqa import TruthfulQABenchmark
    from eval.benchmarks.humaneval import HumanEvalBenchmark
    from eval.benchmarks.mbpp import MBPPBenchmark
    from eval.benchmarks.livecodebench import LiveCodeBenchmark
    from eval.benchmarks.xstest import XSTestBenchmark
    from eval.benchmarks.or_bench import ORBenchBenchmark
    from eval.benchmarks.locomo import LoCoMoBenchmark
    from eval.benchmarks.swebench import SWEBenchVerifiedBenchmark

    max_per = 10 if fast else 100
    max_s = 50 if fast else 0  # 0 = full dataset

    intelligence = [
        MMLUBenchmark(max_per_subject=max_per),
        GPQADiamondBenchmark(max_samples=max_s),
        IFEvalBenchmark(max_samples=max_s),
        GSM8KBenchmark(max_samples=max_s),
        MATHBenchmark(max_samples=500 if not fast else 50),
        ARCChallengeBenchmark(max_samples=max_s),
        HellaSwagBenchmark(max_samples=max_s),
        WinoGrandeBenchmark(max_samples=max_s),
    ]
    coding = [
        HumanEvalBenchmark(max_samples=max_s),
        MBPPBenchmark(max_samples=max_s),
        LiveCodeBenchmark(max_samples=max_s),
    ]
    agentic = [
        SWEBenchVerifiedBenchmark(max_samples=max_s),
    ]
    memory = [
        LoCoMoBenchmark(max_samples=max_s),
    ]
    groundedness = [
        TruthfulQABenchmark(max_samples=max_s),
    ]
    compliance = [
        XSTestBenchmark(max_samples=max_s),
        ORBenchBenchmark(max_per_tier=10 if fast else 50),
    ]
    personality: list = []

    registry = {
        "intelligence": intelligence,
        "coding": coding,
        "agentic": agentic,
        "memory": memory,
        "groundedness": groundedness,
        "compliance": compliance,
        "personality": personality,
    }

    if suite == "all":
        return [b for benchmarks in registry.values() for b in benchmarks]
    if suite not in registry:
        raise ValueError(f"Unknown suite '{suite}'. Choose from: {', '.join(registry)} or 'all'")
    return registry[suite]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Wintermute benchmark harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--target",
        help=(
            "Model target. Prefix with 'hf:' for local HuggingFace model/checkpoint, "
            "or provide an http(s) base URL for any OpenAI-compatible endpoint."
        ),
    )
    p.add_argument("--model", default="", help="Model name (required for API targets)")
    p.add_argument("--api-key", default="", help="API key (for remote APIs; omit for self-hosted)")
    p.add_argument("--device", default="auto", help="Device for HF local backend (auto/cpu/cuda/mps)")
    p.add_argument(
        "--suite",
        default="intelligence",
        choices=["intelligence", "coding", "agentic", "memory", "groundedness", "compliance", "personality", "all"],
        help="Benchmark suite to run (default: intelligence)",
    )
    p.add_argument("--fast", action="store_true", help="Quick run: 10 questions per benchmark (pipeline smoke test)")
    p.add_argument("--max-tokens", type=int, default=128, help="Max tokens for model responses")
    p.add_argument("--parallel", type=int, default=1,
                   help="Run N benchmarks concurrently. Only useful with vLLM/SageMaker backends "
                        "that support true batched inference. Ollama queues serially — N>1 adds no speedup there.")
    p.add_argument("--list-runs", action="store_true", help="List past runs and exit")
    return p.parse_args()


def _preflight_check(backend, target: str, retries: int = 3) -> None:
    """
    Send a trivial inference request before starting. If Ollama is down,
    try to restart it via SSH (works for pc-macbook) and retry.
    Aborts with a clear error if still unresponsive after retries.
    """
    import time, urllib.request

    base = target.rstrip("/")
    host = base.split("//")[-1].split(":")[0]

    from eval.model import GenerateConfig as _GC
    _probe_cfg = _GC(max_tokens=5, temperature=0.0)

    def _test() -> bool:
        try:
            backend.complete("Reply with just: OK", _probe_cfg)
            return True
        except Exception:
            return False

    # Fast path — model responds
    if _test():
        print("Pre-flight OK\n")
        return

    print(f"  Ollama unresponsive at {target} — attempting restart via SSH...", flush=True)
    try:
        import subprocess
        subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", f"beardface@{host}",
             "OLLAMA_HOST=0.0.0.0:11434 nohup ~/bin/ollama serve > /tmp/ollama_serve.log 2>&1 &"],
            timeout=15, check=False
        )
        time.sleep(12)
    except Exception as e:
        print(f"  SSH restart failed: {e}")

    for attempt in range(retries):
        if _test():
            print(f"  Ollama restarted OK (attempt {attempt + 1})\n")
            return
        time.sleep(10)

    print(f"ERROR: Model at {target} is unresponsive after {retries} retries. Aborting.")
    sys.exit(1)


def main():
    args = parse_args()

    if args.list_runs:
        runs = list_runs()
        if not runs:
            print("No runs recorded yet.")
            return
        print(f"{'RUN ID':<35} {'MODEL':<40} {'SUITE':<15} {'FINISHED'}")
        print("-" * 110)
        for r in runs:
            print(f"{r['run_id']:<35} {r['model_id']:<40} {r['suite']:<15} {r['finished_at'] or 'running'}")
        return

    if not args.target:
        print("error: --target is required (use --list-runs to see past runs)")
        sys.exit(1)

    # provider:model shortcuts (e.g. anthropic:claude-sonnet-4-6) embed the model in the target
    _has_embedded_model = ":" in args.target and not args.target.startswith(("http:", "https:", "hf:"))
    if not args.target.startswith("hf:") and not args.model and not _has_embedded_model:
        print("error: --model is required for API targets (or use provider:model shortcut)")
        sys.exit(1)

    # Build model backend
    backend = make_backend(args.target, model=args.model, api_key=args.api_key, device=args.device)
    cfg = GenerateConfig(max_tokens=args.max_tokens, temperature=0.0)

    # Pre-flight: verify the model actually responds before starting a multi-hour run
    if args.target.startswith("http"):
        _preflight_check(backend, args.target)

    benchmarks = _load_benchmarks(args.suite, fast=args.fast)
    if not benchmarks:
        print(f"No benchmarks implemented yet for suite '{args.suite}'.")
        print("Check eval/benchmarks/ and add adapters, then register them in run_benchmark.py.")
        sys.exit(0)

    if args.parallel > 1 and args.target.startswith("http") and "11434" in args.target:
        print(f"Note: --parallel {args.parallel} has no effect on Ollama (serial queue). "
              f"Use with vLLM/SageMaker for real speedup.\n")

    print(f"\nModel   : {backend.model_id}")
    print(f"Suite   : {args.suite}")
    print(f"Fast    : {args.fast}")
    print(f"Parallel: {args.parallel}")
    print(f"Benchmarks: {[b.name for b in benchmarks]}\n")

    record = new_run(model_id=backend.model_id, suite=args.suite)

    if args.parallel > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        _print_lock = threading.Lock()

        def _run_one(bench):
            try:
                result = bench.run(backend, cfg)
                with _print_lock:
                    print(f"  {bench.name}: {result.metric}={result.score:.4f}", flush=True)
                return result
            except Exception as e:
                with _print_lock:
                    print(f"  {bench.name}: FAILED: {e}", flush=True)
                return None

        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(_run_one, bench): bench for bench in benchmarks}
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None:
                    record.add(result)
    else:
        for bench in benchmarks:
            print(f"  Running {bench.name}...", end=" ", flush=True)
            try:
                result = bench.run(backend, cfg)
                record.add(result)
                print(f"{result.metric}={result.score:.4f}")
            except Exception as e:
                print(f"FAILED: {e}")

    out_path = save_run(record)
    print(f"\nResults saved → {out_path}")
    print(f"Run ID: {record.run_id}")

    # Print summary table
    print("\n── Summary ──────────────────────────────")
    print(f"{'Benchmark':<25} {'Suite':<15} {'Metric':<15} {'Score'}")
    print("-" * 65)
    for r in record.results:
        print(f"{r.benchmark:<25} {r.suite:<15} {r.metric:<15} {r.score:.4f}")

    _notify(backend.model_id, args.suite, record)


def _notify(model_id: str, suite: str, record) -> None:
    """Fire an ntfy.sh push notification when a benchmark run completes."""
    import os, urllib.request, urllib.error
    from pathlib import Path

    topic = os.environ.get("NTFY_TOPIC", "wintermute")

    # Load token from ~/.blue_rose/ntfy.env if present
    token = os.environ.get("NTFY_TOKEN", "")
    ntfy_env = Path.home() / ".blue_rose" / "ntfy.env"
    if not token and ntfy_env.exists():
        token = ntfy_env.read_text().strip()

    passed = [r for r in record.results if r.score >= 0]
    failed = len(record.results) - len(passed)
    lines = [f"{r.benchmark}: {r.score:.4f}" for r in passed]
    if failed:
        lines.append(f"({failed} benchmark(s) failed)")
    body = "\n".join(lines) if lines else "No results recorded."
    title = f"Benchmark done - {model_id} - {suite}"

    headers = {
        "Title": title,
        "Priority": "default",
        "Tags": "white_check_mark",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}",
            data=body.encode(),
            headers=headers,
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except (urllib.error.URLError, OSError):
        pass  # notification is best-effort — never block the run


if __name__ == "__main__":
    main()
