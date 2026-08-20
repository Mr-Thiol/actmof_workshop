#!/usr/bin/env python3
"""Aggregate Experiment-B split shards into CSV summaries and manuscript tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results" / "lucien" / "2026-08-20" / "experiment_b"
METHOD_ORDER = [
    "random",
    "gp_m52_pi",
    "gp_m52_ei",
    "pi_random_exploration",
    "pi_uncertainty",
    "imperfection_aware",
]
METHOD_LABELS = {
    "random": "Random",
    "gp_m52_pi": "GP M52 + PI",
    "gp_m52_ei": "GP M52 + EI",
    "pi_random_exploration": "GP M52 PI + Random Exploration",
    "pi_uncertainty": "PI + Uncertainty",
    "imperfection_aware": "Imperfection-aware",
}
ROUND_LABELS = {0: "Initial 60", 1: "+5", 2: "+10", 3: "+15", 4: "+20"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args(argv)


def read_shards(results_dir: Path, stem: str) -> pd.DataFrame:
    files = sorted((results_dir / "shards").glob(f"{stem}_seed_*.csv"))
    if not files:
        raise FileNotFoundError(f"No shard files found for {stem} in {results_dir / 'shards'}")
    return pd.concat((pd.read_csv(path) for path in files), ignore_index=True)


def sem(x: pd.Series) -> float:
    return float(x.std(ddof=1) / np.sqrt(x.count())) if x.count() > 1 else 0.0


def aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    metrics["method_order"] = metrics["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    agg_map = {
        "catastrophic_error_rate": ["mean", "std", sem],
        "n_catastrophic_errors": ["mean", "std", sem],
        "mean_abs_log_error": ["mean", "std", sem],
        "median_abs_log_error": ["mean", "std", sem],
        "max_abs_log_error": ["mean", "std", sem],
        "rmse_log_error": ["mean", "std", sem],
        "signed_log_error_mean": ["mean", "std", sem],
        "gate_fallback_count": ["mean", "max"],
    }
    out = metrics.groupby(["method", "method_label", "round"], as_index=False).agg(agg_map)
    out.columns = [
        "_".join(c).rstrip("_") if isinstance(c, tuple) else c
        for c in out.columns.to_flat_index()
    ]
    out = out.rename(columns={
        "catastrophic_error_rate_sem": "catastrophic_error_rate_sem",
        "n_catastrophic_errors_sem": "n_catastrophic_errors_sem",
    })
    for metric in [
        "catastrophic_error_rate",
        "n_catastrophic_errors",
        "mean_abs_log_error",
        "median_abs_log_error",
        "max_abs_log_error",
        "rmse_log_error",
        "signed_log_error_mean",
    ]:
        out[f"{metric}_ci95_low"] = out[f"{metric}_mean"] - 1.96 * out[f"{metric}_sem"]
        out[f"{metric}_ci95_high"] = out[f"{metric}_mean"] + 1.96 * out[f"{metric}_sem"]
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    out["method_order"] = out["method"].map(order)
    return out.sort_values(["method_order", "round"]).drop(columns=["method_order"]).reset_index(drop=True)


def add_error_auc(metrics: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    auc = (
        metrics[metrics["round"] > 0]
        .groupby(["split_seed", "method", "method_label"], as_index=False)["catastrophic_error_rate"]
        .mean()
        .rename(columns={"catastrophic_error_rate": "error_auc"})
    )
    auc_summary = auc.groupby(["method", "method_label"], as_index=False).agg(
        error_auc_mean=("error_auc", "mean"),
        error_auc_std=("error_auc", "std"),
        error_auc_sem=("error_auc", sem),
    )
    auc_summary["error_auc_ci95_low"] = auc_summary["error_auc_mean"] - 1.96 * auc_summary["error_auc_sem"]
    auc_summary["error_auc_ci95_high"] = auc_summary["error_auc_mean"] + 1.96 * auc_summary["error_auc_sem"]
    return summary.merge(auc_summary, on=["method", "method_label"], how="left")


def fmt_mean_sem(mean: float, se: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} +/- {se:.{digits}f}"


def round_labels(metrics: pd.DataFrame) -> dict[int, str]:
    labels = {}
    by_round = (
        metrics.groupby("round", as_index=False)[["n_initial_train", "n_acquired"]]
        .first()
        .sort_values("round")
    )
    for _, row in by_round.iterrows():
        round_id = int(row["round"])
        if round_id == 0:
            labels[round_id] = f"Initial {int(row['n_initial_train'])}"
        else:
            labels[round_id] = f"+{int(row['n_acquired'])}"
    return labels


def main_table(summary: pd.DataFrame, labels: dict[int, str]) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        sub = summary[summary["method"] == method]
        if sub.empty:
            continue
        row = {"Method": METHOD_LABELS[method]}
        for round_id, label in labels.items():
            r = sub[sub["round"] == round_id].iloc[0]
            row[label] = fmt_mean_sem(r["catastrophic_error_rate_mean"], r["catastrophic_error_rate_sem"])
        r0 = sub.iloc[0]
        row["Error AUC"] = fmt_mean_sem(r0["error_auc_mean"], r0["error_auc_sem"])
        rows.append(row)
    return pd.DataFrame(rows)


def split_description(metrics: pd.DataFrame) -> str:
    first = metrics.sort_values(["split_seed", "method", "round"]).iloc[0]
    pool = int(metrics.groupby(["split_seed", "method"])["n_acquired"].max().iloc[0])
    return f"{int(first['n_initial_train'])}/{pool}/{int(first['n_test'])}"


def write_markdown_table(table: pd.DataFrame, path: Path, threshold: float, split_desc: str, n_test: int) -> None:
    caption = (
        f"Experiment B headline metric: catastrophic test-error rate "
        f"(abs signed log error >= {threshold:g}) on the untouched {n_test}-point test set. "
        f"Cells are mean +/- SEM over repeated {split_desc} splits. Error AUC is the mean "
        "catastrophic-error rate over all post-acquisition rounds; lower is better."
    )
    path.write_text(caption + "\n\n" + table.to_markdown(index=False) + "\n")


def write_latex_table(table: pd.DataFrame, path: Path, threshold: float) -> None:
    latex = table.to_latex(index=False, escape=True, column_format="l" + "r" * (len(table.columns) - 1))
    caption = (
        f"% Catastrophic test-error rate: abs(log1p(q_pred)-log1p(q_true)) >= {threshold:g}. "
        "Cells are mean +/- SEM; Error AUC averages all post-acquisition rounds.\n"
    )
    path.write_text(caption + latex)


def validate(metrics: pd.DataFrame, selected: pd.DataFrame, splits: pd.DataFrame, preds: pd.DataFrame) -> None:
    first_seed = sorted(splits["split_seed"].unique())[0]
    expected_counts = splits[splits["split_seed"] == first_seed]["split_role"].value_counts().to_dict()
    required_roles = {"initial_train", "acquisition_pool", "test"}
    if set(expected_counts) != required_roles:
        raise RuntimeError(f"Split roles must be {required_roles}; found {set(expected_counts)}")
    batch_size = int(selected.groupby(["split_seed", "method", "round"])["experiment_id"].size().iloc[0])
    final_round = int(metrics["round"].max())
    for seed, g in splits.groupby("split_seed"):
        counts = g["split_role"].value_counts().to_dict()
        if counts != expected_counts:
            raise RuntimeError(f"Split-size sanity check failed for seed {seed}: {counts}")
        role_sets = {
            role: set(sg["experiment_id"])
            for role, sg in g.groupby("split_role")
        }
        if len(set.union(*role_sets.values())) != 95:
            raise RuntimeError(f"Split union sanity check failed for seed {seed}")
        if role_sets["test"] & set(selected.loc[selected["split_seed"] == seed, "experiment_id"]):
            raise RuntimeError(f"Test experiment selected for acquisition in seed {seed}")

    batch_counts = selected.groupby(["split_seed", "method", "round"])["experiment_id"].agg(["count", "nunique"])
    bad = batch_counts[(batch_counts["count"] != batch_size) | (batch_counts["nunique"] != batch_size)]
    if len(bad):
        raise RuntimeError(f"Batch sanity check failed:\n{bad.head()}")
    exhausted = selected.groupby(["split_seed", "method"])["experiment_id"].nunique()
    if not (exhausted == expected_counts["acquisition_pool"]).all():
        raise RuntimeError("At least one method did not acquire all acquisition-pool points")

    role_counts = selected.groupby(["split_seed", "method", "round", "selection_role"]).size().unstack(fill_value=0)
    for key, row in role_counts.reset_index().iterrows():
        method = row["method"]
        if method in {"pi_random_exploration", "pi_uncertainty", "imperfection_aware"}:
            exploit = int(row.get("exploitation", 0))
            explore = int(row.drop(labels=["split_seed", "method", "round"], errors="ignore").sum()) - exploit
            if exploit != batch_size - 1 or explore != 1:
                raise RuntimeError(f"Exploration batch role sanity check failed: {row.to_dict()}")

    metric_cols = [
        "n_catastrophic_errors",
        "catastrophic_error_rate",
        "mean_abs_log_error",
        "median_abs_log_error",
        "max_abs_log_error",
        "rmse_log_error",
        "signed_log_error_mean",
    ]
    for round_id in [0, final_round]:
        spread = metrics[metrics["round"] == round_id].groupby("split_seed")[metric_cols].nunique()
        if (spread > 1).any().any():
            raise RuntimeError(f"Endpoint equality sanity check failed at round {round_id}")

    pred_counts = preds.groupby(["split_seed", "method", "round"])["experiment_id"].nunique()
    if not (pred_counts == expected_counts["test"]).all():
        raise RuntimeError("Test prediction file does not contain the configured number of test points per split/method/round")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_shards(args.results_dir, "experiment_b_per_split_round")
    selected = read_shards(args.results_dir, "experiment_b_selected_points")
    preds = read_shards(args.results_dir, "experiment_b_test_predictions")
    splits = read_shards(args.results_dir, "experiment_b_splits")
    validate(metrics, selected, splits, preds)

    metrics.to_csv(args.results_dir / "experiment_b_per_split_round.csv", index=False)
    selected.to_csv(args.results_dir / "experiment_b_selected_points.csv", index=False)
    preds.to_csv(args.results_dir / "experiment_b_test_predictions.csv", index=False)
    splits.to_csv(args.results_dir / "experiment_b_splits.csv", index=False)

    summary = add_error_auc(metrics, aggregate(metrics))
    summary.to_csv(args.results_dir / "experiment_b_summary.csv", index=False)
    labels = round_labels(metrics)
    table = main_table(summary, labels)
    threshold = float(metrics["catastrophic_threshold"].iloc[0])
    n_test = int(metrics["n_test"].iloc[0])
    write_markdown_table(table, args.results_dir / "experiment_b_main_table.md", threshold, split_description(metrics), n_test)
    write_latex_table(table, args.results_dir / "experiment_b_main_table.tex", threshold)

    meta_path = args.results_dir / "experiment_b_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {}
    meta.update({
        "aggregated_split_seeds": sorted(int(x) for x in metrics["split_seed"].unique()),
        "n_aggregated_splits": int(metrics["split_seed"].nunique()),
        "summary_files": [
            "experiment_b_per_split_round.csv",
            "experiment_b_selected_points.csv",
            "experiment_b_test_predictions.csv",
            "experiment_b_splits.csv",
            "experiment_b_summary.csv",
            "experiment_b_main_table.md",
            "experiment_b_main_table.tex",
        ],
        "error_auc_definition": "Mean catastrophic-error rate over all post-acquisition rounds only.",
    })
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[DONE] Aggregated {metrics['split_seed'].nunique()} split(s) into {args.results_dir}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
