#!/usr/bin/env python3
"""
Compare two eval runs and print score deltas.

Usage
-----
python eval/compare.py --baseline <run_id_a> --candidate <run_id_b>

  Positive delta means candidate improved over baseline.
  Negative delta means regression.

List available runs:
  python eval/compare.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.results import list_runs, load_run


def _scores_by_benchmark(run: dict) -> dict[str, dict]:
    return {r["benchmark"]: r for r in run.get("results", [])}


def compare(baseline_id: str, candidate_id: str):
    baseline = load_run(baseline_id)
    candidate = load_run(candidate_id)

    b_scores = _scores_by_benchmark(baseline)
    c_scores = _scores_by_benchmark(candidate)

    all_benchmarks = sorted(set(b_scores) | set(c_scores))

    print(f"\nBaseline  : {baseline['model_id']}  ({baseline_id})")
    print(f"Candidate : {candidate['model_id']}  ({candidate_id})")
    print()
    print(f"{'Benchmark':<25} {'Metric':<12} {'Baseline':>10} {'Candidate':>10} {'Delta':>10} {''}  ")
    print("-" * 75)

    improved = regressed = unchanged = missing = 0

    for bench in all_benchmarks:
        b = b_scores.get(bench)
        c = c_scores.get(bench)

        if b is None or c is None:
            tag = "MISSING"
            missing += 1
            b_score_str = f"{b['score']:.4f}" if b else "—"
            c_score_str = f"{c['score']:.4f}" if c else "—"
            delta_str = "—"
        else:
            delta = c["score"] - b["score"]
            b_score_str = f"{b['score']:.4f}"
            c_score_str = f"{c['score']:.4f}"
            delta_str = f"{delta:+.4f}"
            if delta > 0.001:
                tag = "▲"
                improved += 1
            elif delta < -0.001:
                tag = "▼"
                regressed += 1
            else:
                tag = "="
                unchanged += 1

        metric = (b or c or {}).get("metric", "")
        print(f"{bench:<25} {metric:<12} {b_score_str:>10} {c_score_str:>10} {delta_str:>10}  {tag}")

    print()
    print(f"▲ improved: {improved}  ▼ regressed: {regressed}  = unchanged: {unchanged}  — missing: {missing}")


def main():
    p = argparse.ArgumentParser(description="Compare two eval runs")
    p.add_argument("--baseline", help="Run ID of the baseline")
    p.add_argument("--candidate", help="Run ID of the candidate")
    p.add_argument("--list", action="store_true", help="List available runs and exit")
    args = p.parse_args()

    if args.list:
        runs = list_runs()
        if not runs:
            print("No runs recorded yet.")
            return
        print(f"{'RUN ID':<35} {'MODEL':<40} {'SUITE':<15} {'FINISHED'}")
        print("-" * 110)
        for r in runs:
            print(f"{r['run_id']:<35} {r['model_id']:<40} {r['suite']:<15} {r['finished_at'] or 'running'}")
        return

    if not args.baseline or not args.candidate:
        p.print_help()
        sys.exit(1)

    compare(args.baseline, args.candidate)


if __name__ == "__main__":
    main()
