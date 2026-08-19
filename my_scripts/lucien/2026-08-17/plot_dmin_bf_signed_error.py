#!/usr/bin/env python3
"""Plot Sam's d_min-B_F AND-gate with sign-aware q error coloring.

Threshold choice: this reuses the latest Sam follow-up diagnostic in
my_scripts/sam/2026-08-17/plot_interaction_three_views.py, where the
d_min x B_F risk zone is d_min >= 4 and B_F > 0. The catastrophic
threshold remains the existing exploratory e_log > 5 definition.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "actmof-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "Error_data.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "lucien" / "2026-08-18"

D_MIN_THRESHOLD = 4.0
B_F_THRESHOLD = 0.0
CATASTROPHIC_LOG_ERROR_THRESHOLD = 5.0


def main() -> None:
    df = pd.read_csv(DEFAULT_INPUT)
    required = {
        "q_true",
        "q_pred_mean",
        "nearest_chebyshev_distance",
        "fwhm_3_cross_rate",
        "y_signed_log_error",
        "y_log_abs_error",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns in {DEFAULT_INPUT}: {missing}")

    d_min = df["nearest_chebyshev_distance"]
    b_f = df["fwhm_3_cross_rate"]
    signed = df["y_signed_log_error"]
    inside = (d_min >= D_MIN_THRESHOLD) & (b_f > B_F_THRESHOLD)
    catastrophic = df["y_log_abs_error"] > CATASTROPHIC_LOG_ERROR_THRESHOLD

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    max_abs = float(np.nanmax(np.abs(signed)))
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    x_max = float(d_min.max()) + 0.8
    y_min, y_max = -0.04, 1.04

    fig, ax = plt.subplots(figsize=(6.7, 4.9))
    ax.fill_between(
        [D_MIN_THRESHOLD, x_max],
        B_F_THRESHOLD,
        y_max,
        color="#f2b8b5",
        alpha=0.22,
        zorder=0,
        label="AND-gate positive",
    )
    ax.axvline(D_MIN_THRESHOLD, color="#333333", lw=1.2, ls="--", zorder=1)
    ax.axhline(B_F_THRESHOLD, color="#333333", lw=1.2, ls="--", zorder=1)

    sc = ax.scatter(
        d_min,
        b_f,
        c=signed,
        cmap="RdBu_r",
        norm=norm,
        s=48,
        alpha=0.86,
        edgecolors="#2f2f2f",
        linewidths=0.35,
        zorder=2,
    )
    ax.scatter(
        d_min[catastrophic],
        b_f[catastrophic],
        facecolors="none",
        edgecolors="#111111",
        s=118,
        linewidths=1.1,
        zorder=3,
        label=rf"catastrophic ($e_{{log}}>{CATASTROPHIC_LOG_ERROR_THRESHOLD:g}$)",
    )

    ax.text(
        x_max - 0.15,
        0.97,
        "Risk region\n$d_{min}\\geq4$ and $B_F>0$",
        ha="right",
        va="top",
        fontsize=8.5,
        color="#6f1d1b",
    )
    ax.text(
        0.02,
        0.13,
        "red: overprediction / false optimism\nblue: underprediction / false pessimism",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#333333",
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Signed log error")
    ax.set_xlabel(r"$d_{\min}$ (Chebyshev distance to nearest experiment)")
    ax.set_ylabel(r"$B_f$ (FWHM boundary crossing rate)")
    ax.set_xlim(-0.25, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.tight_layout()

    png = DEFAULT_OUTPUT_DIR / "dmin_bf_signed_error.png"
    pdf = DEFAULT_OUTPUT_DIR / "dmin_bf_signed_error.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)

    def sign_counts(mask: pd.Series) -> tuple[int, int, int]:
        return (
            int((signed[mask] > 0).sum()),
            int((signed[mask] < 0).sum()),
            int((signed[mask] == 0).sum()),
        )

    inside_pos, inside_neg, inside_zero = sign_counts(inside)
    outside_pos, outside_neg, outside_zero = sign_counts(~inside)
    gate_cat = inside & catastrophic
    cat_over = int((gate_cat & (signed > 0)).sum())
    cat_under = int((gate_cat & (signed < 0)).sum())
    non_cat = int((inside & ~catastrophic).sum())

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
    print("\nAND-gate summary")
    print(f"total samples: {len(df)}")
    print(f"inside AND-gate: {int(inside.sum())}")
    print(f"outside AND-gate: {int((~inside).sum())}")
    print("\ninside gate:")
    print(f"  positive signed error: {inside_pos}")
    print(f"  negative signed error: {inside_neg}")
    print(f"  zero signed error: {inside_zero}")
    print("\noutside gate:")
    print(f"  positive signed error: {outside_pos}")
    print(f"  negative signed error: {outside_neg}")
    print(f"  zero signed error: {outside_zero}")
    print("\ninside gate, using existing catastrophic threshold y_log_abs_error > 5:")
    print(f"  catastrophic overpredictions: {cat_over}")
    print(f"  catastrophic underpredictions: {cat_under}")
    print(f"  non-catastrophic cases: {non_cat}")


if __name__ == "__main__":
    main()
