#!/usr/bin/env python3
"""Create simple diagnostic plots for Experiment A."""

from __future__ import annotations

import os
import tempfile
import argparse
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "actmof-matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
BASE_RESULTS_DIR = REPO_ROOT / "results" / "lucien" / "2026-08-19" / "experiment_a"
BASE_FIGURE_DIR = REPO_ROOT / "figures" / "lucien" / "2026-08-19" / "experiment_a"
POLICY_NAMES = ("random", "gp_m52_pi", "gp_m52_ei", "pi_uncertainty", "imperfection_aware")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rounds", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--exploration", type=int)
    parser.add_argument("--policy", default="imperfection_aware")
    parser.add_argument("--compare-policies", action="store_true", help="overlay policy result directories")
    parser.add_argument("--policies", nargs="*", default=None, help="policies to include in comparison mode")
    parser.add_argument("--results-base-dir", type=Path, default=BASE_RESULTS_DIR)
    parser.add_argument("--figure-base-dir", type=Path, default=BASE_FIGURE_DIR)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    return parser.parse_args()


def settings_dir(base_dir: Path, policy: str, n_rounds: int, budget: int, batch_size: int, exploration: int) -> Path:
    return (
        base_dir
        / policy
        / f"n_rounds_{n_rounds}"
        / f"budget_{budget}"
        / f"batch_size_{batch_size}"
        / f"exploration_{exploration}"
    )


def comparison_dir(base_dir: Path, n_rounds: int, budget: int, batch_size: int, exploration: int | None) -> Path:
    out = (
        base_dir
        / "comparison"
        / f"n_rounds_{n_rounds}"
        / f"budget_{budget}"
        / f"batch_size_{batch_size}"
    )
    if exploration is not None:
        out = out / f"exploration_{exploration}"
    return out


def resolve_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.compare_policies:
        missing = [
            name for name in ["n_rounds", "budget", "batch_size"]
            if getattr(args, name) is None
        ]
        if missing:
            raise SystemExit(
                "Comparison mode requires "
                + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            )
        figure_dir = args.figure_dir or comparison_dir(
            args.figure_base_dir,
            args.n_rounds,
            args.budget,
            args.batch_size,
            args.exploration,
        )
        return args.results_base_dir, figure_dir

    if args.results_dir is not None:
        results_dir = args.results_dir
    else:
        missing = [
            name for name in ["n_rounds", "budget", "batch_size", "exploration"]
            if getattr(args, name) is None
        ]
        if missing:
            raise SystemExit(
                "Provide --results-dir, or provide all setting args: "
                + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            )
        results_dir = settings_dir(
            args.results_base_dir,
            args.policy,
            args.n_rounds,
            args.budget,
            args.batch_size,
            args.exploration,
        )

    if args.figure_dir is not None:
        figure_dir = args.figure_dir
    elif all(getattr(args, name) is not None for name in ["n_rounds", "budget", "batch_size", "exploration"]):
        figure_dir = settings_dir(
            args.figure_base_dir,
            args.policy,
            args.n_rounds,
            args.budget,
            args.batch_size,
            args.exploration,
        )
    else:
        figure_dir = args.figure_base_dir / results_dir.relative_to(args.results_base_dir)
    return results_dir, figure_dir


def _infer_policy_from_path(path: Path) -> str:
    for part in path.parts:
        if part in POLICY_NAMES:
            return part
    return "unknown"


def load_single_result(results_dir: Path, policy: str) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(results_dir / "iteration_metrics.csv")
    if "policy" not in df.columns:
        df["policy"] = _infer_policy_from_path(results_dir) or policy
    df["plot_label"] = df["policy"].astype(str)
    title = str(df["policy"].iloc[0]) if len(df) else policy
    return df, title


def load_comparison_results(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    policies = args.policies or list(POLICY_NAMES)
    frames = []
    read_paths = []
    for policy in policies:
        if args.exploration is None:
            setting_root = (
                args.results_base_dir
                / policy
                / f"n_rounds_{args.n_rounds}"
                / f"budget_{args.budget}"
                / f"batch_size_{args.batch_size}"
            )
            metric_paths = sorted(setting_root.glob("exploration_*/iteration_metrics.csv"))
        else:
            metric_paths = [
                settings_dir(
                    args.results_base_dir,
                    policy,
                    args.n_rounds,
                    args.budget,
                    args.batch_size,
                    args.exploration,
                )
                / "iteration_metrics.csv"
            ]
        for path in metric_paths:
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df["policy"] = df["policy"].astype(str) if "policy" in df.columns else policy
            exploration_label = path.parent.name.replace("exploration_", "")
            df["plot_label"] = df["policy"] if args.exploration is not None else df["policy"] + " / exp " + exploration_label
            frames.append(df)
            read_paths.append(path)

    if not frames:
        raise FileNotFoundError("No policy result files found for the requested comparison settings.")
    print("Read comparison metrics from:")
    for path in read_paths:
        print(f"  {path}")
    title = f"policy comparison, batch_size={args.batch_size}, budget={args.budget}"
    return pd.concat(frames, ignore_index=True), title


def plot_curve(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    filename: str,
    figure_dir: Path,
    title: str,
    label_col: str = "method",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, group in df.groupby(label_col, sort=False):
        curve = group.groupby("iteration")[metric].agg(["mean", "std"]).reset_index()
        ax.plot(curve["iteration"], curve["mean"], label=label)
        ax.fill_between(
            curve["iteration"],
            curve["mean"] - curve["std"].fillna(0),
            curve["mean"] + curve["std"].fillna(0),
            alpha=0.18,
    )
    ax.set_xlabel("AL iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / filename, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir, figure_dir = resolve_dirs(args)
    figure_dir.mkdir(parents=True, exist_ok=True)
    if args.compare_policies:
        df, title = load_comparison_results(args)
        label_col = "plot_label"
    else:
        df, title = load_single_result(results_dir, args.policy)
        label_col = "method"

    plot_curve(df, "best_q", "Best q so far", "best_q.png", figure_dir, title, label_col)
    plot_curve(df, "normalized_regret", "Normalized regret", "normalized_regret.png", figure_dir, title, label_col)
    plot_curve(df, "hit_top0_1pct", "Top-0.1% hit rate", "top0_1pct_hit_rate.png", figure_dir, title, label_col)

    gate = df[df["method"].str.startswith("GP_Matern52_PI_GATE_") & (df["iteration"] > 0)].copy()
    if not gate.empty:
        gate["acq_suspicious_fraction"] = gate["n_suspicious_candidates"] / gate["n_acq_candidates"]
        plot_curve(
            gate,
            "acq_suspicious_fraction",
            "Suspicious fraction of acquisition pool",
            "acq_suspicious_fraction.png",
            figure_dir,
            title,
            label_col,
        )
    if not args.compare_policies:
        print(f"Read metrics from {results_dir}")
    print(f"Saved figures to {figure_dir}")


if __name__ == "__main__":
    main()
