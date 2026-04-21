"""
Consistency test: paraphrase robustness of the RAG pipeline under LLM-judge scoring.

For each of 3 base questions (one per domain), we run the original plus 5 hand-written
paraphrases through the Week 1 RAG pipeline, then score each RAG output with the LLM
judge against the SAME ground truth as the base question.

3 base questions x 6 variants = 18 RAG runs + 18 judge calls.

Outputs:
  - consistency_test_results.csv : per-variant rag_answer + 4 judge scores.
  - consistency_summary.csv      : per base question, per metric, mean & std across 6 variants.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

from llm_judge import score_one
from rag_pipeline import get_or_create_chroma_store, get_rag_response


WEEK1_DIR = Path("week1_artifacts")
RAG_OUTPUTS_PATH = WEEK1_DIR / "rag_outputs.json"

RESULTS_CSV = Path("consistency_test_results.csv")
SUMMARY_CSV = Path("consistency_summary.csv")

METRICS = ("faithfulness", "answer_relevance", "correctness", "context_relevance")


PARAPHRASE_SET: Dict[str, Dict] = {
    "java_inheritance": {
        "domain": "java_textbook",
        "original": "What is inheritance in Java and how does it work?",
        "paraphrases": [
            "Can you explain the concept of inheritance in Java and describe the mechanism behind it?",
            "In Java, how does inheritance function, and what does it actually mean?",
            "Describe Java inheritance: what is it, and how is it implemented in practice?",
            "How do Java classes inherit from other classes, and what is inheritance conceptually?",
            "Walk me through what inheritance means in Java and the way it operates under the hood.",
        ],
    },
    "tc_limitations": {
        "domain": "terms_conditions",
        "original": "What are the limitations and disclaimers mentioned in the Terms and Conditions?",
        "paraphrases": [
            "Which disclaimers and limitations of liability does the Terms and Conditions document list?",
            "According to the T&C, what warranty disclaimers and limitation-of-liability clauses are present?",
            "List the limitations and disclaimers that appear in the Terms and Conditions.",
            "What does the Terms and Conditions say about warranties, disclaimers, and liability limits?",
            "Summarize the liability limitations and disclaimer statements included in the T&C.",
        ],
    },
    "cloudhost_features": {
        "domain": "product_catalog",
        "original": "What are the key features offered by CloudHost Business Plan?",
        "paraphrases": [
            "Which main features does the CloudHost Business Plan include?",
            "Describe the core capabilities bundled with the CloudHost Business Plan.",
            "What do you get with the CloudHost Business Plan; what are its headline features?",
            "List the primary features provided under the CloudHost Business Plan.",
            "Can you summarize the key offerings of the CloudHost Business Plan?",
        ],
    },
}


def _require_week1_artifacts() -> None:
    if not WEEK1_DIR.exists() or not RAG_OUTPUTS_PATH.exists():
        print(
            "ERROR: week1_artifacts/rag_outputs.json not found.\n"
            "Place the Week 1 files at ./week1_artifacts/ before running the consistency test."
        )
        sys.exit(1)


def _load_ground_truths() -> Dict[str, str]:
    """Return a mapping of original-question-text -> ground_truth from Week 1."""
    with RAG_OUTPUTS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    lookup = {entry["question"]: entry["ground_truth"] for entry in data}
    missing = [cfg["original"] for cfg in PARAPHRASE_SET.values() if cfg["original"] not in lookup]
    if missing:
        print("ERROR: The following base questions are not in Week 1 rag_outputs.json:")
        for q in missing:
            print(f"  - {q}")
        sys.exit(1)
    return lookup


def _build_variants(cfg: Dict) -> List[str]:
    return [cfg["original"], *cfg["paraphrases"]]


def main() -> None:
    print("=" * 80)
    print("Consistency test: 3 base questions x 6 variants = 18 runs")
    print("=" * 80)

    _require_week1_artifacts()
    ground_truths = _load_ground_truths()

    print("Loading (or creating) Chroma vector store...")
    vector_store = get_or_create_chroma_store()

    rows: List[Dict] = []
    total_variants = sum(len(_build_variants(cfg)) for cfg in PARAPHRASE_SET.values())
    counter = 0

    for base_id, cfg in PARAPHRASE_SET.items():
        domain = cfg["domain"]
        gt = ground_truths[cfg["original"]]
        variants = _build_variants(cfg)

        for variant_id, question in enumerate(variants):
            counter += 1
            preview = question[:60].replace("\n", " ")
            print(
                f"\n[{counter}/{total_variants}] base={base_id} variant={variant_id} :: "
                f"{preview}{'...' if len(question) > 60 else ''}"
            )

            try:
                rag_answer, retrieved_contexts = get_rag_response(question, vector_store)
            except Exception as exc:  # noqa: BLE001
                print(f"  RAG generation failed: {exc}. Recording empty answer.")
                rag_answer, retrieved_contexts = "", []

            judge_out = score_one(
                question=question,
                retrieved_contexts=retrieved_contexts,
                rag_answer=rag_answer,
                ground_truth=gt,
            )

            rows.append({
                "base_question_id": base_id,
                "variant_id": variant_id,
                "domain": domain,
                "question": question,
                "rag_answer": rag_answer,
                "faithfulness": judge_out["faithfulness"],
                "answer_relevance": judge_out["answer_relevance"],
                "correctness": judge_out["correctness"],
                "context_relevance": judge_out["context_relevance"],
            })

    results_df = pd.DataFrame(rows, columns=[
        "base_question_id", "variant_id", "domain", "question", "rag_answer",
        "faithfulness", "answer_relevance", "correctness", "context_relevance",
    ])
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nWrote per-variant results to {RESULTS_CSV}")

    summary_rows: List[Dict] = []
    print("\n" + "=" * 80)
    print("Consistency summary (mean / std across 6 variants per base question)")
    print("=" * 80)
    for base_id, cfg in PARAPHRASE_SET.items():
        subset = results_df[results_df["base_question_id"] == base_id]
        print(f"\n[{base_id}] domain={cfg['domain']}  n={len(subset)}")
        for metric in METRICS:
            col = pd.to_numeric(subset[metric], errors="coerce")
            mean = col.mean()
            std = col.std(ddof=1)
            mean_str = f"{mean:.3f}" if pd.notna(mean) else "nan"
            std_str = f"{std:.3f}" if pd.notna(std) else "nan"
            print(f"  {metric:20s} mean={mean_str}  std={std_str}")
            summary_rows.append({
                "base_question_id": base_id,
                "domain": cfg["domain"],
                "metric": metric,
                "mean": mean,
                "std": std,
            })

    summary_df = pd.DataFrame(summary_rows, columns=[
        "base_question_id", "domain", "metric", "mean", "std",
    ])
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nWrote consistency summary to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
