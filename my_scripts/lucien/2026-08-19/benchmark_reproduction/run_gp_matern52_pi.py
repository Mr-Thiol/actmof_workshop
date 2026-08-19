"""Run the original GP_Matern52_PI benchmark for ActMOF reproduction."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "actmof-matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd

from benchmark_core import (
    FAMILY,
    METHOD,
    RESULTS_DIR,
    load_design_grid,
    run_gp_matern52_pi_once,
    summarize_runs,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-repeats", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("[LOAD] Loading reproduced landscape if present, otherwise original split CSVs.")
    pool_df, best_row, total = load_design_grid()
    print(f"[LOAD] total rows scanned={total:,}; pool rows={len(pool_df):,}")
    print("[LOAD] best row:")
    print(pd.DataFrame([best_row]).to_string(index=False))

    all_rows = []
    for run in range(args.n_repeats):
        rows, _selected = run_gp_matern52_pi_once(pool_df, run)
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(RESULTS_DIR / "iteration_metrics.csv", index=False)

    iteration_df = pd.DataFrame(all_rows)
    run_summary, final_summary = summarize_runs(iteration_df)
    run_summary.to_csv(RESULTS_DIR / "run_summary.csv", index=False)
    final_summary.to_csv(RESULTS_DIR / "final_summary.csv", index=False)

    curve = iteration_df.groupby("iteration", as_index=False)["best_q"].agg(["mean", "std"]).reset_index()
    plt.figure(figsize=(7, 4))
    plt.plot(curve["iteration"], curve["mean"], label=METHOD)
    plt.fill_between(
        curve["iteration"],
        curve["mean"] - curve["std"].fillna(0),
        curve["mean"] + curve["std"].fillna(0),
        alpha=0.2,
    )
    plt.xlabel("AL iteration")
    plt.ylabel("Best q so far")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "gp_matern52_pi_best_q.png", dpi=150)
    plt.close()

    final = run_summary.sort_values("run")[["run", "final_best_q"]]
    mean = float(final["final_best_q"].mean())
    std = float(final["final_best_q"].std(ddof=1))
    ref = 187_320.0
    print(f"\n{METHOD} reproduction")
    for _, row in final.iterrows():
        print(f"Run {int(row['run']) + 1} final best q: {row['final_best_q']:.0f}")
    print(f"\nMean final best q: {mean:.1f}")
    print(f"Std final best q: {std:.1f}")
    print(f"\nReference mean: ~187,320")
    print(f"Absolute difference: {abs(mean - ref):.1f}")
    print(f"Relative difference: {abs(mean - ref) / ref * 100:.3f} %")
    print(f"\nOutputs written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
