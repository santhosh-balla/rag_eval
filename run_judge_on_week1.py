"""
Run the LLM-as-a-judge over the Week 1 RAG outputs.

Reads: week1_artifacts/rag_outputs.json (10 entries with question, ground_truth,
context_domain, rag_answer, retrieved_contexts).

Writes:
  - llm_judge_outputs.json : full records including reasoning + raw_response.
  - llm_judge_scores.csv   : flat table keyed by question for joining with RAGAs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from llm_judge import score_batch


WEEK1_DIR = Path("week1_artifacts")
RAG_OUTPUTS_PATH = WEEK1_DIR / "rag_outputs.json"
JUDGE_OUTPUTS_JSON = Path("llm_judge_outputs.json")
JUDGE_SCORES_CSV = Path("llm_judge_scores.csv")


def _require_week1_artifacts() -> None:
    if not WEEK1_DIR.exists():
        print(
            "ERROR: week1_artifacts/ directory not found.\n"
            "Place the Week 1 files at ./week1_artifacts/:\n"
            "  - rag_outputs.json\n"
            "  - ragas_per_question_scores.csv\n"
            "  - ragas_domain_scorecard.csv"
        )
        sys.exit(1)
    if not RAG_OUTPUTS_PATH.exists():
        print(f"ERROR: {RAG_OUTPUTS_PATH} not found. Cannot run judge on Week 1 outputs.")
        sys.exit(1)


def _load_entries() -> list[dict]:
    with RAG_OUTPUTS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print(f"ERROR: {RAG_OUTPUTS_PATH} must contain a JSON array of entries.")
        sys.exit(1)
    required = ("question", "ground_truth", "context_domain", "rag_answer", "retrieved_contexts")
    for i, entry in enumerate(data):
        missing = [k for k in required if k not in entry]
        if missing:
            print(f"ERROR: entry {i} in {RAG_OUTPUTS_PATH} is missing keys: {missing}")
            sys.exit(1)
    return data


def main() -> None:
    print("=" * 80)
    print("LLM-as-a-Judge on Week 1 RAG outputs")
    print("=" * 80)

    _require_week1_artifacts()
    entries = _load_entries()
    print(f"Loaded {len(entries)} Week 1 entries from {RAG_OUTPUTS_PATH}")

    results = score_batch(entries)

    with JUDGE_OUTPUTS_JSON.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote full judge records to {JUDGE_OUTPUTS_JSON}")

    rows = [
        {
            "question": r.get("question", ""),
            "domain": r.get("context_domain", ""),
            "faithfulness": r.get("faithfulness"),
            "answer_relevance": r.get("answer_relevance"),
            "correctness": r.get("correctness"),
            "context_relevance": r.get("context_relevance"),
        }
        for r in results
    ]
    df = pd.DataFrame(rows, columns=[
        "question", "domain", "faithfulness", "answer_relevance",
        "correctness", "context_relevance",
    ])
    df.to_csv(JUDGE_SCORES_CSV, index=False)
    print(f"Wrote flat judge scores to {JUDGE_SCORES_CSV}")

    null_rows = df[df["faithfulness"].isna()]
    if not null_rows.empty:
        print(f"Note: {len(null_rows)} row(s) received null scores due to judge failures.")


if __name__ == "__main__":
    main()
