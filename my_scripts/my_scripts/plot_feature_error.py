#!/usr/bin/env python3
"""Plot each error-prediction feature against ActMOF log reconstruction error.

Outputs seven PNG files plus a CSV containing Pearson and Spearman correlations.
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "feature_error"

TARGET = "y_log_abs_error"
FEATURES = [
    ("rf_intensity_variance", r"$V_I$", "continuous_log"),
    ("rf_fwhm_variance", r"$V_F$", "continuous_log"),
    ("nearest_chebyshev_distance", r"$d_{min}$", "discrete"),
    ("rule1_neighbor_count", r"$N_{R1}$", "discrete"),
    ("rule2_neighbor_count", r"$N_{R2}$", "discrete"),
    ("intensity_3000_cross_rate", r"$B_I$", "discrete"),
    ("fwhm_3_cross_rate", r"$B_F$", "discrete"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--catastrophic-threshold",
        type=float,
        default=5.0,
        help="Outline points with y_log_abs_error above this value (default: 5).",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def correlations(df: pd.DataFrame, feature: str) -> tuple[float, float]:
    pearson = df[[feature, TARGET]].corr(method="pearson").iloc[0, 1]
    spearman = df[[feature, TARGET]].corr(method="spearman").iloc[0, 1]
    return float(pearson), float(spearman)


def plot_one(
    df: pd.DataFrame,
    feature: str,
    label: str,
    kind: str,
    output: Path,
    catastrophic_threshold: float,
    dpi: int,
) -> tuple[float, float]:
    x = df[feature].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    pearson, spearman = correlations(df, feature)

    if kind == "continuous_log":
        x_plot = np.log10(1.0 + x)
        xlabel = rf"$\log_{{10}}(1+{label})$"
    else:
        # Deterministic jitter makes overlapping discrete observations visible.
        rng = np.random.default_rng(42)
        unique = np.unique(x)
        spacing = np.min(np.diff(unique)) if len(unique) > 1 else 1.0
        x_plot = x + rng.normal(0.0, 0.035 * spacing, size=len(x))
        xlabel = label

    catastrophic = y > catastrophic_threshold

    fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
    ax.scatter(x_plot[~catastrophic], y[~catastrophic], s=45, alpha=0.78)
    ax.scatter(
        x_plot[catastrophic],
        y[catastrophic],
        s=62,
        facecolors="none",
        edgecolors="black",
        linewidths=1.1,
        label=rf"$e_{{log}}>{catastrophic_threshold:g}$",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Log reconstruction error $e_{log}$")
    ax.set_title(f"{label} vs. log reconstruction error")
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.text(
        0.03,
        0.97,
        f"Pearson r = {pearson:+.3f}\nSpearman rho = {spearman:+.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
    )
    if catastrophic.any():
        ax.legend(frameon=False, loc="best")

    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return pearson, spearman


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)

    required = [TARGET] + [name for name, _, _ in FEATURES]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for feature, label, kind in FEATURES:
        output = args.output_dir / f"{feature}_vs_log_error.png"
        pearson, spearman = plot_one(
            df,
            feature,
            label,
            kind,
            output,
            args.catastrophic_threshold,
            args.dpi,
        )
        rows.append(
            {
                "feature": feature,
                "pearson_r": pearson,
                "spearman_rho": spearman,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "feature_error_correlations.csv", index=False)

    print(summary.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print(f"\nSaved 7 plots and correlation table to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
