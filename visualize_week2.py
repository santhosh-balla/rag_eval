"""
Generate Week 2 figures (matplotlib only). Saves PNGs to figures/.

Figures:
  - fig_ragas_vs_judge_scatter.png : RAGAs faithfulness vs Judge faithfulness (normalized), y=x ref.
  - fig_domain_comparison_bar.png  : grouped bar per domain of RAGAs vs Judge faithfulness means.
  - fig_consistency_heatmap.png    : std of judge scores across 6 paraphrase variants per base question.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURES_DIR = Path("figures")
COMPARISON_CSV = Path("week2_comparison.csv")
DOMAIN_CSV = Path("week2_domain_scorecard.csv")
CONSISTENCY_SUMMARY_CSV = Path("consistency_summary.csv")

SCATTER_PNG = FIGURES_DIR / "fig_ragas_vs_judge_scatter.png"
BAR_PNG = FIGURES_DIR / "fig_domain_comparison_bar.png"
HEATMAP_PNG = FIGURES_DIR / "fig_consistency_heatmap.png"

METRICS_ORDER = ("faithfulness", "answer_relevance", "correctness", "context_relevance")


def _ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _plot_scatter() -> None:
    if not COMPARISON_CSV.exists():
        print(f"Skipping scatter: {COMPARISON_CSV} not found.")
        return

    df = pd.read_csv(COMPARISON_CSV)
    df = df.dropna(subset=["ragas_faithfulness", "judge_faithfulness_norm"])
    if df.empty:
        print("Skipping scatter: no non-null rows in comparison file.")
        return

    domains = sorted(df["domain"].dropna().unique().tolist())
    cmap = plt.get_cmap("tab10")
    color_map = {d: cmap(i % 10) for i, d in enumerate(domains)}

    np.random.seed(42)

    fig, ax = plt.subplots(figsize=(7, 6))
    for d in domains:
        sub = df[df["domain"] == d]
        x = sub["ragas_faithfulness"].to_numpy(dtype=float)
        y = sub["judge_faithfulness_norm"].to_numpy(dtype=float)
        y_jittered = y + np.random.uniform(-0.01, 0.01, size=len(y))
        ax.scatter(
            x, y_jittered,
            label=d, color=color_map[d], s=120, alpha=0.6,
            edgecolors="black", linewidths=0.5,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="y = x (agreement)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("RAGAs Faithfulness (0-1)")
    ax.set_ylabel("Judge Faithfulness Normalized (0-1)")
    ax.set_title("RAGAs vs LLM Judge Faithfulness per Question")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(SCATTER_PNG, dpi=150)
    plt.close(fig)
    print(f"Saved {SCATTER_PNG}")


def _plot_domain_bar() -> None:
    if not DOMAIN_CSV.exists():
        print(f"Skipping bar: {DOMAIN_CSV} not found.")
        return

    df = pd.read_csv(DOMAIN_CSV)
    if df.empty:
        print("Skipping bar: domain scorecard is empty.")
        return

    required = {"domain", "ragas_faithfulness", "judge_faithfulness_norm"}
    missing = required - set(df.columns)
    if missing:
        print(f"Skipping bar: domain scorecard missing columns {missing}.")
        return

    domains = df["domain"].tolist()
    x = np.arange(len(domains))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, df["ragas_faithfulness"], width, label="RAGAs Faithfulness")
    ax.bar(x + width / 2, df["judge_faithfulness_norm"], width, label="Judge Faithfulness (norm)")
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean score (0-1)")
    ax.set_title("Per-Domain Faithfulness: RAGAs vs LLM Judge")
    ax.legend()
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(BAR_PNG, dpi=150)
    plt.close(fig)
    print(f"Saved {BAR_PNG}")


def _plot_consistency_heatmap() -> None:
    if not CONSISTENCY_SUMMARY_CSV.exists():
        print(f"Skipping heatmap: {CONSISTENCY_SUMMARY_CSV} not found.")
        return

    df = pd.read_csv(CONSISTENCY_SUMMARY_CSV)
    if df.empty:
        print("Skipping heatmap: consistency summary is empty.")
        return

    pivot = df.pivot(index="base_question_id", columns="metric", values="std")
    pivot = pivot.reindex(columns=[m for m in METRICS_ORDER if m in pivot.columns])
    base_ids = pivot.index.tolist()
    metrics = pivot.columns.tolist()
    values = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.8 * len(base_ids)))
    im = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(base_ids)))
    ax.set_yticklabels(base_ids)
    ax.set_title("Consistency across 6 paraphrase variants (std of judge scores)")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            txt = "nan" if np.isnan(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if (not np.isnan(v) and v > np.nanmax(values) / 2) else "black",
                    fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Std across 6 variants (lower = more consistent)")
    plt.tight_layout()
    plt.savefig(HEATMAP_PNG, dpi=150)
    plt.close(fig)
    print(f"Saved {HEATMAP_PNG}")


def main() -> None:
    print("=" * 80)
    print("Generating Week 2 figures")
    print("=" * 80)
    _ensure_dirs()

    failures = 0
    for fn in (_plot_scatter, _plot_domain_bar, _plot_consistency_heatmap):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"Figure step {fn.__name__} failed: {exc}")

    if failures:
        print(f"{failures} figure(s) failed to generate.")
        sys.exit(1 if failures == 3 else 0)


if __name__ == "__main__":
    main()
