"""
Week 2 orchestrator. Runs all four phases in order with per-step timing.

Steps:
  1. run_judge_on_week1   : LLM judge scores the 10 Week 1 RAG outputs.
  2. consistency_test     : 3 base questions x 6 variants through the RAG pipeline + judge.
  3. compare_ragas_vs_judge : produces comparison CSVs + correlations + divergence cases.
  4. visualize_week2      : generates 3 PNGs under figures/.

Each step is isolated: a failure in one does not stop later steps (so that if the judge
phase produced partial artifacts, the comparison/visualization can still try to run).
"""

from __future__ import annotations

import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Callable

import pandas as pd

import run_judge_on_week1
import consistency_test
import compare_ragas_vs_judge
import visualize_week2


STEPS: list[tuple[str, Callable[[], None]]] = [
    ("run_judge_on_week1",     run_judge_on_week1.main),
    ("consistency_test",       consistency_test.main),
    ("compare_ragas_vs_judge", compare_ragas_vs_judge.main),
    ("visualize_week2",        visualize_week2.main),
]

JUDGE_SCORES_CSV = Path("llm_judge_scores.csv")
JUDGE_METRICS = ("faithfulness", "answer_relevance", "correctness", "context_relevance")


def _sanity_check_judge_scores() -> bool:
    """Inspect llm_judge_scores.csv after step 1.

    Prints the per-metric distribution of integer scores. Returns False (halt)
    iff all four metric columns contain ONLY the value 5 -- i.e., the judge
    showed zero discrimination. Returns True otherwise so the pipeline continues.
    """
    if not JUDGE_SCORES_CSV.exists():
        print(f"Sanity check skipped: {JUDGE_SCORES_CSV} not found.")
        return False

    df = pd.read_csv(JUDGE_SCORES_CSV)
    print("\nSanity check: judge score distribution")
    distributions: dict[str, Counter] = {}
    for metric in JUDGE_METRICS:
        if metric not in df.columns:
            print(f"  {metric}: column missing")
            distributions[metric] = Counter()
            continue
        counter: Counter = Counter()
        for v in df[metric].tolist():
            try:
                if pd.isna(v):
                    counter["null"] += 1
                else:
                    counter[int(v)] += 1
            except (TypeError, ValueError):
                counter["invalid"] += 1
        distributions[metric] = counter
        print(f"  {metric}: {dict(sorted(counter.items(), key=lambda kv: str(kv[0])))}")

    all_fives = all(
        set(distributions[m].keys()) == {5} and distributions[m][5] > 0
        for m in JUDGE_METRICS
    )
    if all_fives:
        print("\nWARNING: Judge produced all 5s. Rubric is too lenient. Halting pipeline.")
        return False
    return True


def _run_step(index: int, total: int, name: str, fn: Callable[[], None]) -> tuple[bool, float]:
    header = f"=== STEP {index}/{total}: {name} ==="
    print("\n" + "#" * len(header))
    print(header)
    print("#" * len(header))
    start = time.perf_counter()
    try:
        fn()
        elapsed = time.perf_counter() - start
        print(f"\n[{name}] Done in {elapsed:.1f}s")
        return True, elapsed
    except SystemExit as exc:
        elapsed = time.perf_counter() - start
        print(f"\n[{name}] Failed in {elapsed:.1f}s: SystemExit({exc.code})")
        return False, elapsed
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - start
        print(f"\n[{name}] Failed in {elapsed:.1f}s: {exc}")
        traceback.print_exc()
        return False, elapsed


def main() -> None:
    print("=" * 80)
    print(" WEEK 2 PIPELINE: LLM-as-a-Judge Evaluation")
    print("=" * 80)

    overall_start = time.perf_counter()
    results: list[tuple[str, bool, float]] = []
    total = len(STEPS)
    halted = False
    for i, (name, fn) in enumerate(STEPS, start=1):
        ok, elapsed = _run_step(i, total, name, fn)
        results.append((name, ok, elapsed))

        if name == "run_judge_on_week1" and ok:
            if not _sanity_check_judge_scores():
                halted = True
                print("\nStopping pipeline before expensive consistency test.")
                break

    overall = time.perf_counter() - overall_start

    print("\n" + "=" * 80)
    print("WEEK 2 PIPELINE SUMMARY")
    print("=" * 80)
    for name, ok, elapsed in results:
        status = "Done  " if ok else "Failed"
        print(f"  {status}  {name:28s}  {elapsed:7.1f}s")
    print(f"  Total elapsed: {overall:.1f}s")

    if halted:
        sys.exit(2)


if __name__ == "__main__":
    main()
