#!/usr/bin/env python3
"""Diagnostic plot for ActMOF leave-one-out q_true versus q_pred_mean.

The comparison is shown on log1p scale so multiplicative errors remain visible
across the strongly skewed q range. Catastrophic true-positive -> predicted-zero
cases are highlighted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "Error_data.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "figures" / "q_true_pred_diagnostic.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    required = ["q_true", "q_pred_mean", "y_log_abs_error"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    q_true = df["q_true"].to_numpy(dtype=float)
    q_pred = df["q_pred_mean"].to_numpy(dtype=float)
    x = np.log1p(q_true)
    y = np.log1p(q_pred)

    # Qualitatively severe failure: the held-out experiment is crystalline/nonzero,
    # while the emulator reconstruction collapses to q = 0.
    collapsed = (q_true > 0) & (q_pred <= 0)

    fig, ax = plt.subplots(figsize=(6.2, 5.6), constrained_layout=True)
    ax.scatter(x[~collapsed], y[~collapsed], s=48, alpha=0.78, label="Other experiments")
    ax.scatter(
        x[collapsed],
        y[collapsed],
        s=72,
        marker="x",
        linewidths=1.6,
        label="q_true > 0, q_pred = 0",
    )

    upper = max(float(x.max()), float(y.max())) * 1.03
    ax.plot([0, upper], [0, upper], linestyle="--", linewidth=1.1, label="Perfect prediction")
    ax.set_xlim(-0.15, upper)
    ax.set_ylim(-0.15, upper)
    ax.set_xlabel(r"$\log(1+q_{true})$")
    ax.set_ylabel(r"$\log(1+q_{pred})$")
    ax.set_title("ActMOF leave-one-out q diagnostic")
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(frameon=False, loc="best")

    mae_log = float(np.mean(np.abs(y - x)))
    pearson = float(pd.Series(x).corr(pd.Series(y), method="pearson"))
    spearman = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    ax.text(
        0.03,
        0.97,
        f"N = {len(df)}\nCollapsed-to-zero = {collapsed.sum()}\nMean log error = {mae_log:.3f}\n"
        f"Pearson r = {pearson:+.3f}\nSpearman rho = {spearman:+.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Experiments: {len(df)}")
    print(f"True q = 0: {(q_true == 0).sum()}")
    print(f"Predicted q = 0: {(q_pred == 0).sum()}")
    print(f"q_true > 0 but q_pred = 0: {collapsed.sum()}")
    print(f"Mean |log1p(q_pred)-log1p(q_true)|: {mae_log:.3f}")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
