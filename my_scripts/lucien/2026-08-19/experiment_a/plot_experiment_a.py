#!/usr/bin/env python3
"""Create simple diagnostic plots for Experiment A."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "actmof-matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
RESULTS_DIR = REPO_ROOT / "results" / "lucien" / "2026-08-19" / "experiment_a"
FIGURE_DIR = REPO_ROOT / "figures" / "lucien" / "2026-08-19" / "experiment_a"


def plot_curve(df: pd.DataFrame, metric: str, ylabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, group in df.groupby("method"):
        curve = group.groupby("iteration")[metric].agg(["mean", "std"]).reset_index()
        ax.plot(curve["iteration"], curve["mean"], label=method)
        ax.fill_between(
            curve["iteration"],
            curve["mean"] - curve["std"].fillna(0),
            curve["mean"] + curve["std"].fillna(0),
            alpha=0.18,
        )
    ax.set_xlabel("AL iteration")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS_DIR / "iteration_metrics.csv")
    plot_curve(df, "best_q", "Best q so far", "best_q.png")
    plot_curve(df, "normalized_regret", "Normalized regret", "normalized_regret.png")
    plot_curve(df, "hit_top0_1pct", "Top-0.1% hit rate", "top0_1pct_hit_rate.png")

    gate = df[df["method"].str.startswith("GP_Matern52_PI_GATE_") & (df["iteration"] > 0)].copy()
    gate["acq_suspicious_fraction"] = gate["n_suspicious_candidates"] / gate["n_acq_candidates"]
    plot_curve(gate, "acq_suspicious_fraction", "Suspicious fraction of acquisition pool", "acq_suspicious_fraction.png")
    print(f"Saved figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
