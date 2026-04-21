"""
Compare Week 1 RAGAs scores against the Week 2 LLM-as-a-judge scores.

Inputs:
  - week1_artifacts/ragas_per_question_scores.csv
  - llm_judge_scores.csv
  - llm_judge_outputs.json (for reasoning on divergence cases)

Outputs:
  - week2_comparison.csv        : per-question side-by-side, with normalized judge scores and deltas.
  - week2_domain_scorecard.csv  : per-domain means for RAGAs + judge metrics.
  - correlations.csv            : Pearson/Spearman r and p for faithfulness and context metrics.
  - divergence_cases.csv        : top 3 questions with largest judge-vs-RAGAs disagreement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


WEEK1_DIR = Path("week1_artifacts")
RAGAS_CSV = WEEK1_DIR / "ragas_per_question_scores.csv"
JUDGE_SCORES_CSV = Path("llm_judge_scores.csv")
JUDGE_OUTPUTS_JSON = Path("llm_judge_outputs.json")

COMPARISON_CSV = Path("week2_comparison.csv")
DOMAIN_CSV = Path("week2_domain_scorecard.csv")
CORRELATIONS_CSV = Path("correlations.csv")
DIVERGENCE_CSV = Path("divergence_cases.csv")

DIVERGENCE_THRESHOLD = 0.3


def _require_inputs() -> None:
    missing = [p for p in (RAGAS_CSV, JUDGE_SCORES_CSV) if not p.exists()]
    if missing:
        print("ERROR: Missing required inputs for comparison:")
        for p in missing:
            print(f"  - {p}")
        if not WEEK1_DIR.exists():
            print(
                "The week1_artifacts/ folder does not exist. Place Week 1 CSVs and JSON there "
                "(rag_outputs.json, ragas_per_question_scores.csv, ragas_domain_scorecard.csv)."
            )
        if not JUDGE_SCORES_CSV.exists():
            print("Run run_judge_on_week1.py first to produce llm_judge_scores.csv.")
        sys.exit(1)


def _load_ragas() -> pd.DataFrame:
    df = pd.read_csv(RAGAS_CSV)
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        norm = col.strip().lower().replace(" ", "_")
        if norm in ("answer_relevancy", "answer_relevance"):
            rename_map[col] = "ragas_answer_relevancy"
        elif norm == "faithfulness":
            rename_map[col] = "ragas_faithfulness"
        elif norm == "context_precision":
            rename_map[col] = "ragas_context_precision"
        elif norm == "context_recall":
            rename_map[col] = "ragas_context_recall"
        elif norm == "question":
            rename_map[col] = "question"
        elif norm in ("domain", "context_domain"):
            rename_map[col] = "ragas_domain"
    df = df.rename(columns=rename_map)
    return df


def _load_judge_reasoning() -> Dict[str, str]:
    if not JUDGE_OUTPUTS_JSON.exists():
        return {}
    with JUDGE_OUTPUTS_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {entry.get("question", ""): entry.get("reasoning", "") or "" for entry in data}


def _load_judge_rag_answers() -> Dict[str, str]:
    if not JUDGE_OUTPUTS_JSON.exists():
        return {}
    with JUDGE_OUTPUTS_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {entry.get("question", ""): entry.get("rag_answer", "") or "" for entry in data}


def _safe_corr(x: np.ndarray, y: np.ndarray) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return (None, None, None, None)
    pearson = stats.pearsonr(x[mask], y[mask])
    spearman = stats.spearmanr(x[mask], y[mask])
    return (float(pearson[0]), float(pearson[1]), float(spearman[0]), float(spearman[1]))


def main() -> None:
    print("=" * 80)
    print("Comparing RAGAs (Week 1) vs LLM Judge (Week 2)")
    print("=" * 80)

    _require_inputs()

    ragas_df = _load_ragas()
    judge_df = pd.read_csv(JUDGE_SCORES_CSV)

    merged = judge_df.merge(ragas_df, on="question", how="inner")
    if merged.empty:
        print("ERROR: No rows after joining judge scores with RAGAs on 'question'. "
              "Check that the 'question' column matches between files.")
        sys.exit(1)
    print(f"Joined {len(merged)} rows on question.")

    for col in ("faithfulness", "answer_relevance", "correctness", "context_relevance"):
        merged[f"judge_{col}_norm"] = pd.to_numeric(merged[col], errors="coerce") / 5.0

    for col in ("ragas_faithfulness", "ragas_context_precision", "ragas_context_recall", "ragas_answer_relevancy"):
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["delta_faithfulness"] = merged["judge_faithfulness_norm"] - merged.get("ragas_faithfulness")
    merged["delta_context"] = merged["judge_context_relevance_norm"] - merged.get("ragas_context_precision")

    comparison_cols = [
        "question", "domain",
        "ragas_faithfulness", "judge_faithfulness_norm", "delta_faithfulness",
        "ragas_context_precision", "judge_context_relevance_norm", "delta_context",
        "ragas_context_recall",
        "judge_answer_relevance_norm", "judge_correctness_norm",
    ]
    for col in comparison_cols:
        if col not in merged.columns:
            merged[col] = np.nan

    comparison_df = merged[comparison_cols].copy()
    comparison_df.to_csv(COMPARISON_CSV, index=False)
    print(f"Wrote per-question comparison to {COMPARISON_CSV}")

    domain_metrics = [
        "ragas_faithfulness", "judge_faithfulness_norm",
        "ragas_context_precision", "judge_context_relevance_norm",
        "judge_answer_relevance_norm", "judge_correctness_norm",
    ]
    domain_df = (
        comparison_df.groupby("domain", dropna=False)[domain_metrics]
        .mean(numeric_only=True)
        .reset_index()
    )
    domain_df.to_csv(DOMAIN_CSV, index=False)
    print(f"Wrote domain scorecard to {DOMAIN_CSV}")

    print("\nDomain scorecard:")
    print("-" * 80)
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}" if pd.notna(v) else "nan"):
        print(domain_df.to_string(index=False))
    print("-" * 80)

    corr_rows = []
    faith_pr, faith_pp, faith_sr, faith_sp = _safe_corr(
        comparison_df["ragas_faithfulness"].to_numpy(dtype=float),
        comparison_df["judge_faithfulness_norm"].to_numpy(dtype=float),
    )
    ctx_pr, ctx_pp, ctx_sr, ctx_sp = _safe_corr(
        comparison_df["ragas_context_precision"].to_numpy(dtype=float),
        comparison_df["judge_context_relevance_norm"].to_numpy(dtype=float),
    )
    corr_rows.append({
        "metric_pair": "ragas_faithfulness_vs_judge_faithfulness_norm",
        "pearson_r": faith_pr, "pearson_p": faith_pp,
        "spearman_r": faith_sr, "spearman_p": faith_sp,
    })
    corr_rows.append({
        "metric_pair": "ragas_context_precision_vs_judge_context_relevance_norm",
        "pearson_r": ctx_pr, "pearson_p": ctx_pp,
        "spearman_r": ctx_sr, "spearman_p": ctx_sp,
    })
    corr_df = pd.DataFrame(corr_rows, columns=[
        "metric_pair", "pearson_r", "pearson_p", "spearman_r", "spearman_p",
    ])
    corr_df.to_csv(CORRELATIONS_CSV, index=False)
    print(f"\nWrote correlations to {CORRELATIONS_CSV}")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}" if pd.notna(v) else "nan"):
        print(corr_df.to_string(index=False))

    diverged = comparison_df.copy()
    diverged["abs_delta_faith"] = diverged["delta_faithfulness"].abs()
    diverged["abs_delta_ctx"] = diverged["delta_context"].abs()
    diverged["max_abs_delta"] = diverged[["abs_delta_faith", "abs_delta_ctx"]].max(axis=1, skipna=True)
    flagged = diverged[
        (diverged["abs_delta_faith"] >= DIVERGENCE_THRESHOLD)
        | (diverged["abs_delta_ctx"] >= DIVERGENCE_THRESHOLD)
    ].sort_values("max_abs_delta", ascending=False).head(3)

    reasoning_lookup = _load_judge_reasoning()
    rag_answer_lookup = _load_judge_rag_answers()

    div_rows = []
    for _, row in flagged.iterrows():
        q = row["question"]
        div_rows.append({
            "question": q,
            "domain": row.get("domain"),
            "rag_answer": rag_answer_lookup.get(q, ""),
            "ragas_faithfulness": row.get("ragas_faithfulness"),
            "judge_faithfulness_norm": row.get("judge_faithfulness_norm"),
            "delta_faithfulness": row.get("delta_faithfulness"),
            "ragas_context_precision": row.get("ragas_context_precision"),
            "judge_context_relevance_norm": row.get("judge_context_relevance_norm"),
            "delta_context": row.get("delta_context"),
            "judge_reasoning": reasoning_lookup.get(q, ""),
        })

    div_df = pd.DataFrame(div_rows, columns=[
        "question", "domain", "rag_answer",
        "ragas_faithfulness", "judge_faithfulness_norm", "delta_faithfulness",
        "ragas_context_precision", "judge_context_relevance_norm", "delta_context",
        "judge_reasoning",
    ])
    div_df.to_csv(DIVERGENCE_CSV, index=False)
    print(f"\nWrote top-{len(div_df)} divergence cases to {DIVERGENCE_CSV} "
          f"(threshold |delta| >= {DIVERGENCE_THRESHOLD}).")


if __name__ == "__main__":
    main()
