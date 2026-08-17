#!/usr/bin/env python3
"""Create a correlation heatmap for the seven ActMOF error-prediction features."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FEATURES = {
    "rf_intensity_variance": r"$V_I$",
    "rf_fwhm_variance": r"$V_F$",
    "nearest_chebyshev_distance": r"$d_{min}$",
    "rule1_neighbor_count": r"$N_{R1}$",
    "rule2_neighbor_count": r"$N_{R2}$",
    "intensity_3000_cross_rate": r"$B_I$",
    "fwhm_3_cross_rate": r"$B_F$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("Error_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("figures/feature_correlation_heatmap.png"))
    parser.add_argument(
        "--method",
        choices=["spearman", "pearson"],
        default="spearman",
        help="Correlation used in the heatmap (default: spearman).",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    missing = [column for column in FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    corr = df[list(FEATURES)].corr(method=args.method)
    labels = list(FEATURES.values())

    fig, ax = plt.subplots(figsize=(7.0, 6.2), constrained_layout=True)
    image = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    fig.colorbar(image, ax=ax, shrink=0.82, label=f"{args.method.capitalize()} correlation")

    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.tick_params(axis="x", rotation=45)
    ax.set_title(f"Error-feature correlation structure ({args.method})")

    for row in range(len(labels)):
        for col in range(len(labels)):
            value = corr.iloc[row, col]
            text_color = "white" if abs(value) >= 0.55 else "black"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=9)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    csv_output = args.output.with_suffix(".csv")
    corr.rename(index=FEATURES, columns=FEATURES).to_csv(csv_output)

    print(corr.rename(index=FEATURES, columns=FEATURES).round(3).to_string())
    print(f"\nSaved heatmap: {args.output.resolve()}")
    print(f"Saved matrix:  {csv_output.resolve()}")


if __name__ == "__main__":
    main()
