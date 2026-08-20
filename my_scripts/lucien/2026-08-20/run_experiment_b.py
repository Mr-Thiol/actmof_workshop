#!/usr/bin/env python3
"""Run Experiment B: scarce-observation acquisition for ActMOF simulator repair."""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used with `sklearn\.utils\.parallel\.Parallel`",
    category=UserWarning,
    module=r"sklearn\.utils\.parallel",
)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
REPRO_DIR = SCRIPT_DIR.parent / "2026-08-19" / "benchmark_reproduction"
EXPA_DIR = SCRIPT_DIR.parent / "2026-08-19" / "experiment_a"
sys.path.insert(0, str(REPRO_DIR))
sys.path.insert(0, str(EXPA_DIR))

from benchmark_core import (  # noqa: E402
    DIVERSITY_LAMBDA,
    FEATURES,
    MIN_STD,
    RF_N_TREES,
    RF_RANDOM_STATE,
    ConstantSurrogate,
    SklearnGPSurrogate,
    apply_rules_batch,
    calc_q_scalar,
    gaussian_ei,
    gaussian_pi,
    greedy_diverse_batch,
    postprocess_and_round_vec,
    predict_in_chunks,
    stable_int,
    y_to_model_scale,
)
from gate import classify_gate, compute_boundary_features, compute_dmin  # noqa: E402
from policies import PolicyContext, make_policy  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "lucien" / "2026-08-20" / "experiment_b" / "15-40-40"
ERROR_DATA = REPO_ROOT / "Error_data.csv"
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
SHARD_STEMS = {
    "metrics": "experiment_b_per_split_round",
    "selected": "experiment_b_selected_points",
    "preds": "experiment_b_test_predictions",
    "splits": "experiment_b_splits",
}
SELECTION_LABELS = {
    "random": "random",
    "gp_m52_pi": "exploitation",
    "gp_m52_ei": "exploitation",
    "pi_random_exploration": "mixed",
    "pi_uncertainty": "mixed",
    "imperfection_aware": "mixed",
}
GATE_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_INITIAL_TRAIN_SIZE = 15
DEFAULT_ACQUISITION_POOL_SIZE = 40
DEFAULT_TEST_SIZE = 40


@dataclass
class ModelState:
    q_pred: np.ndarray
    intensity_pred: np.ndarray
    fwhm_pred: np.ndarray
    d_min: np.ndarray
    b_i: np.ndarray
    b_f: np.ndarray
    suspicious: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-seed", type=int, help="run one split seed shard")
    parser.add_argument("--n-splits", type=int, default=20, help="number of split seeds when --split-seed is omitted")
    parser.add_argument("--initial-train-size", type=int, default=DEFAULT_INITIAL_TRAIN_SIZE)
    parser.add_argument("--acquisition-pool-size", type=int, default=DEFAULT_ACQUISITION_POOL_SIZE)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--exploration-slots", type=int, default=1)
    parser.add_argument("--catastrophic-threshold", type=float, default=1.0)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--error-data", type=Path, default=ERROR_DATA)
    parser.add_argument("--rf-n-trees", type=int, default=RF_N_TREES)
    parser.add_argument("--rf-n-jobs", type=int, default=-1)
    parser.add_argument("--gp-n-restarts", type=int, default=1)
    parser.add_argument("--methods", nargs="+", choices=METHOD_ORDER, default=METHOD_ORDER)
    parser.add_argument("--aggregate", action="store_true", help="run summarization after split shard(s) finish")
    parser.add_argument("--force", action="store_true", help="recompute split shards even when complete outputs already exist")
    return parser.parse_args()


def load_experimental_data(path: Path) -> pd.DataFrame:
    usecols = ["experiment_id", *FEATURES, "Intensity_exp", "FWHM_exp", "q_true"]
    data = pd.read_csv(path, usecols=usecols)
    data = data.sort_values("experiment_id").reset_index(drop=True)
    if len(data) != 95:
        raise RuntimeError(f"Expected 95 canonical experiments, found {len(data)}")
    data["q_true"] = data["q_true"].astype(float)
    q_check = [calc_q_scalar(i, f) for i, f in zip(data.Intensity_exp, data.FWHM_exp)]
    if not np.array_equal(data["q_true"].to_numpy(dtype=int), np.asarray(q_check, dtype=int)):
        raise RuntimeError("q_true is not consistent with measured Intensity_exp/FWHM_exp")
    return data


def split_indices(
    n: int,
    split_seed: int,
    initial_train_size: int,
    acquisition_pool_size: int,
    test_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    train_end = initial_train_size
    pool_end = train_end + acquisition_pool_size
    train = np.sort(perm[:train_end])
    pool = np.sort(perm[train_end:pool_end])
    test = np.sort(perm[pool_end:pool_end + test_size])
    if len(set(train) | set(pool) | set(test)) != n:
        raise RuntimeError(
            f"{initial_train_size}/{acquisition_pool_size}/{test_size} split integrity failed"
        )
    return train, pool, test


def train_rf_pair(x: np.ndarray, y_i: np.ndarray, y_f: np.ndarray, seed: int, n_trees: int, n_jobs: int):
    kwargs = dict(n_estimators=n_trees, random_state=seed, n_jobs=n_jobs)
    return (
        RandomForestRegressor(**kwargs).fit(x, y_i),
        RandomForestRegressor(**kwargs).fit(x, y_f),
    )


def build_state(
    data: pd.DataFrame,
    train_idx: list[int],
    eval_idx: np.ndarray,
    args: argparse.Namespace,
    compute_gate: bool,
) -> ModelState:
    train = data.iloc[np.sort(np.asarray(train_idx, dtype=int))]
    x_train = train[FEATURES].to_numpy(dtype=float)
    y_i = train["Intensity_exp"].to_numpy(dtype=float)
    y_f = train["FWHM_exp"].to_numpy(dtype=float)
    x_eval = data.iloc[eval_idx][FEATURES].to_numpy(dtype=float)

    rf_i, rf_f = train_rf_pair(x_train, y_i, y_f, RF_RANDOM_STATE, args.rf_n_trees, args.rf_n_jobs)
    i_pre, f_pre, _flags = apply_rules_batch(x_eval, rf_i.predict(x_eval), rf_f.predict(x_eval), train)
    i_final, f_final, q_pred = postprocess_and_round_vec(i_pre, f_pre)

    if compute_gate:
        gate_i, gate_f = [], []
        for seed in GATE_SEEDS:
            model_i, model_f = train_rf_pair(x_train, y_i, y_f, seed, args.rf_n_trees, args.rf_n_jobs)
            gate_i.append(model_i)
            gate_f.append(model_f)
        d_min = compute_dmin(x_eval, x_train, chunk_size=50_000)
        b_i, b_f = compute_boundary_features(x_eval, x_train, y_i, y_f, gate_i, gate_f)
        suspicious = classify_gate(d_min, b_i, b_f)
    else:
        d_min = np.full(len(eval_idx), np.nan, dtype=float)
        b_i = np.full(len(eval_idx), np.nan, dtype=float)
        b_f = np.full(len(eval_idx), np.nan, dtype=float)
        suspicious = np.zeros(len(eval_idx), dtype=bool)
    return ModelState(q_pred.astype(float), i_final.astype(float), f_final.astype(float), d_min, b_i, b_f, suspicious)


def fit_gp(data: pd.DataFrame, train_idx: list[int], scaler: MinMaxScaler, split_seed: int, method: str, round_id: int, n_restarts: int):
    x_all = scaler.transform(data[FEATURES].to_numpy(dtype=float))
    y_all = data["q_true"].to_numpy(dtype=float)
    ordered_train = np.sort(np.asarray(train_idx, dtype=int))
    x_train = x_all[ordered_train]
    y_train = y_to_model_scale(y_all[ordered_train])
    seed = (7919 + 1009 * split_seed + 53 * round_id + stable_int(method)) % (2**31 - 1)
    if np.nanstd(y_train) < 1e-12:
        model: Any = ConstantSurrogate().fit(x_train, y_train)
    else:
        model = SklearnGPSurrogate(random_state=seed, n_restarts=n_restarts).fit(x_train, y_train)
    return model, x_all, y_train


def acquisition_context(
    data: pd.DataFrame,
    train_idx: list[int],
    remaining_pool: list[int],
    gate_state: ModelState,
    scaler: MinMaxScaler,
    split_seed: int,
    method: str,
    round_id: int,
    batch_size: int,
    exploration_slots: int,
    rng: np.random.Generator,
    n_restarts: int,
) -> tuple[PolicyContext, pd.DataFrame]:
    model, x_all, y_train = fit_gp(data, train_idx, scaler, split_seed, method, round_id, n_restarts)
    candidates = np.asarray(remaining_pool, dtype=np.int64)
    mu, sd = predict_in_chunks(model, x_all[candidates])
    pi = gaussian_pi(mu, sd, float(y_train.max()), xi=0.01)
    ei = gaussian_ei(mu, sd, float(y_train.max()), xi=0.01)
    suspicious = np.zeros(len(data), dtype=bool)
    suspicious[candidates] = gate_state.suspicious
    diag = pd.DataFrame({
        "experiment_id": data.iloc[candidates]["experiment_id"].to_numpy(),
        "pool_idx": candidates,
        "GP_mean": mu,
        "GP_std": np.maximum(sd, MIN_STD),
        "PI": pi,
        "EI": ei,
        "d_min": gate_state.d_min,
        "B_I": gate_state.b_i,
        "B_F": gate_state.b_f,
        "suspicious": gate_state.suspicious,
    })
    ctx = PolicyContext(
        candidate_idx=candidates,
        remaining_idx=candidates,
        x_scaled=x_all,
        suspicious_mask=suspicious,
        mu=mu,
        sd=sd,
        pi=pi,
        ei=ei,
        batch_size=batch_size,
        exploration_slots=exploration_slots,
        rng=rng,
        diversity_lambda=DIVERSITY_LAMBDA,
    )
    return ctx, diag


def select_pi_random(context: PolicyContext) -> tuple[np.ndarray, dict]:
    n_explore = min(context.exploration_slots, context.batch_size)
    n_pi = context.batch_size - n_explore
    selected = np.asarray(greedy_diverse_batch(context.candidate_idx, context.pi, context.x_scaled, n_pi, context.rng), dtype=np.int64)
    if n_explore:
        eligible = context.candidate_idx[~np.isin(context.candidate_idx, selected)]
        if len(eligible):
            extra = context.rng.choice(eligible, size=min(n_explore, len(eligible)), replace=False)
            selected = np.concatenate([selected, np.asarray(extra, dtype=np.int64)])
    return selected, {"gate_fallback": False, "fallback_slots": "", "requested_trustworthy": n_pi, "requested_suspicious": n_explore}


def select_batch(method: str, context: PolicyContext) -> tuple[np.ndarray, dict]:
    if method == "pi_random_exploration":
        return select_pi_random(context)
    exploration = context.exploration_slots if method in {"pi_uncertainty", "imperfection_aware"} else 0
    policy = make_policy(method, context.batch_size, exploration)
    result = policy.select_batch(context)
    return np.asarray(result.indices, dtype=np.int64), dict(result.diag)


def roles_for(method: str, selected: list[int], diag: dict, batch_size: int, exploration_slots: int) -> list[str]:
    if method == "random":
        return ["random"] * len(selected)
    if method in {"gp_m52_pi", "gp_m52_ei"}:
        return ["exploitation"] * len(selected)
    n_explore = min(exploration_slots, batch_size)
    n_exploit = max(0, batch_size - n_explore)
    roles = ["exploitation"] * min(n_exploit, len(selected))
    if len(selected) > len(roles):
        if method == "pi_random_exploration":
            roles.extend(["random_exploration"] * (len(selected) - len(roles)))
        elif method == "pi_uncertainty":
            roles.extend(["uncertainty_exploration"] * (len(selected) - len(roles)))
        else:
            roles.extend(["suspicious_exploration"] * (len(selected) - len(roles)))
    return roles[:len(selected)]


def metric_rows(
    data: pd.DataFrame,
    split_seed: int,
    method: str,
    round_id: int,
    initial_train_size: int,
    train_idx: list[int],
    pool_remaining: list[int],
    test_idx: np.ndarray,
    state: ModelState,
    threshold: float,
    gate_fallback_count: int,
) -> tuple[dict, list[dict]]:
    test = data.iloc[test_idx].reset_index(drop=True)
    signed = np.log1p(state.q_pred) - np.log1p(test["q_true"].to_numpy(dtype=float))
    abs_err = np.abs(signed)
    is_cat = abs_err >= threshold
    row = {
        "split_seed": split_seed,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "round": round_id,
        "n_initial_train": initial_train_size,
        "n_acquired": len(train_idx) - initial_train_size,
        "n_train_total": len(train_idx),
        "n_pool_remaining": len(pool_remaining),
        "n_test": len(test_idx),
        "catastrophic_threshold": threshold,
        "n_catastrophic_errors": int(is_cat.sum()),
        "catastrophic_error_rate": float(is_cat.mean()),
        "mean_abs_log_error": float(abs_err.mean()),
        "median_abs_log_error": float(np.median(abs_err)),
        "max_abs_log_error": float(abs_err.max()),
        "rmse_log_error": float(np.sqrt(np.mean(signed**2))),
        "signed_log_error_mean": float(signed.mean()),
        "gate_fallback_count": gate_fallback_count,
    }
    pred_rows = []
    for j, idx in enumerate(test_idx):
        pred_rows.append({
            "split_seed": split_seed,
            "method": method,
            "round": round_id,
            "experiment_id": data.loc[idx, "experiment_id"],
            "q_true": float(data.loc[idx, "q_true"]),
            "q_pred": float(state.q_pred[j]),
            "Intensity_pred": float(state.intensity_pred[j]),
            "FWHM_pred": float(state.fwhm_pred[j]),
            "signed_log_error": float(signed[j]),
            "abs_log_error": float(abs_err[j]),
            "is_catastrophic": bool(is_cat[j]),
        })
    return row, pred_rows


def fill_unique(selected: np.ndarray, context: PolicyContext, target_k: int) -> tuple[list[int], bool]:
    out = [int(i) for i in np.asarray(selected).reshape(-1) if int(i) in set(map(int, context.remaining_idx))]
    out = list(dict.fromkeys(out))
    fallback = False
    if len(out) < target_k:
        eligible = context.remaining_idx[~np.isin(context.remaining_idx, out)]
        missing = min(target_k - len(out), len(eligible))
        if missing:
            out.extend(int(i) for i in context.rng.choice(eligible, size=missing, replace=False))
            fallback = True
    return out[:target_k], fallback


def shard_paths(results_dir: Path, split_seed: int) -> dict[str, Path]:
    shard_dir = results_dir / "shards"
    return {
        key: shard_dir / f"{stem}_seed_{split_seed}.csv"
        for key, stem in SHARD_STEMS.items()
    }


def split_shard_status(split_seed: int, args: argparse.Namespace) -> tuple[bool, str]:
    paths = shard_paths(args.results_dir, split_seed)
    missing = [path.name for path in paths.values() if not path.exists()]
    if missing:
        return False, f"missing {', '.join(missing)}"

    try:
        metrics = pd.read_csv(paths["metrics"])
        selected = pd.read_csv(paths["selected"])
        preds = pd.read_csv(paths["preds"])
        splits = pd.read_csv(paths["splits"])
    except Exception as exc:
        return False, f"could not read existing shard files: {exc}"

    requested_methods = set(args.methods)
    n_acquisition_rounds = args.acquisition_pool_size // args.batch_size
    expected_split_total = args.initial_train_size + args.acquisition_pool_size + args.test_size
    checks = [
        ("metrics split_seed", metrics, {"split_seed"}),
        ("selected split_seed", selected, {"split_seed"}),
        ("preds split_seed", preds, {"split_seed"}),
        ("splits split_seed", splits, {"split_seed"}),
    ]
    for label, df, required in checks:
        if not required.issubset(df.columns):
            return False, f"{label} file missing columns {sorted(required - set(df.columns))}"
        if set(df["split_seed"].unique()) != {split_seed}:
            return False, f"{label} file contains split_seed values other than {split_seed}"

    if not requested_methods.issubset(set(metrics.get("method", pd.Series(dtype=object)).unique())):
        return False, "existing metrics shard does not contain all requested methods"
    if not requested_methods.issubset(set(selected.get("method", pd.Series(dtype=object)).unique())):
        return False, "existing selected-points shard does not contain all requested methods"
    if not requested_methods.issubset(set(preds.get("method", pd.Series(dtype=object)).unique())):
        return False, "existing predictions shard does not contain all requested methods"

    metric_sub = metrics[metrics["method"].isin(requested_methods)].copy()
    selected_sub = selected[selected["method"].isin(requested_methods)].copy()
    preds_sub = preds[preds["method"].isin(requested_methods)].copy()
    if metric_sub.empty or selected_sub.empty or preds_sub.empty:
        return False, "existing shard has no rows for requested methods"

    required_metric_cols = {
        "round",
        "n_initial_train",
        "n_acquired",
        "n_test",
        "catastrophic_threshold",
    }
    if not required_metric_cols.issubset(metric_sub.columns):
        return False, f"metrics shard missing columns {sorted(required_metric_cols - set(metric_sub.columns))}"
    expected_metric_rounds = set(range(n_acquisition_rounds + 1))
    if set(metric_sub["round"].unique()) != expected_metric_rounds:
        return False, "existing metrics shard has a different round count"
    metric_round_counts = metric_sub.groupby("method")["round"].nunique()
    if not (metric_round_counts == len(expected_metric_rounds)).all():
        return False, "existing metrics shard is missing at least one method/round"
    if not (metric_sub["n_initial_train"] == args.initial_train_size).all():
        return False, "existing metrics shard has a different initial train size"
    if not (metric_sub["n_test"] == args.test_size).all():
        return False, "existing metrics shard has a different test size"
    if not np.isclose(metric_sub["catastrophic_threshold"].astype(float), args.catastrophic_threshold).all():
        return False, "existing metrics shard has a different catastrophic threshold"
    final_n_acquired = metric_sub.groupby("method")["n_acquired"].max()
    if not (final_n_acquired == args.acquisition_pool_size).all():
        return False, "existing metrics shard has a different acquisition pool size"

    required_split_cols = {"split_role", "experiment_id"}
    if not required_split_cols.issubset(splits.columns):
        return False, f"splits shard missing columns {sorted(required_split_cols - set(splits.columns))}"
    split_counts = splits["split_role"].value_counts().to_dict()
    expected_split_counts = {
        "initial_train": args.initial_train_size,
        "acquisition_pool": args.acquisition_pool_size,
        "test": args.test_size,
    }
    if split_counts != expected_split_counts or len(splits) != expected_split_total:
        return False, "existing splits shard has different train/pool/test sizes"

    required_selected_cols = {"method", "round", "experiment_id"}
    if not required_selected_cols.issubset(selected_sub.columns):
        return False, f"selected-points shard missing columns {sorted(required_selected_cols - set(selected_sub.columns))}"
    selected_counts = selected_sub.groupby(["method", "round"])["experiment_id"].agg(["count", "nunique"])
    if selected_counts.empty or not ((selected_counts["count"] == args.batch_size) & (selected_counts["nunique"] == args.batch_size)).all():
        return False, "existing selected-points shard has incomplete or duplicate acquisition batches"
    expected_selected_rounds = set(range(1, n_acquisition_rounds + 1))
    selected_round_counts = selected_sub.groupby("method")["round"].nunique()
    if (
        set(selected_sub["round"].unique()) != expected_selected_rounds
        or not (selected_round_counts == len(expected_selected_rounds)).all()
    ):
        return False, "existing selected-points shard is missing at least one method/round"

    required_pred_cols = {"method", "round", "experiment_id"}
    if not required_pred_cols.issubset(preds_sub.columns):
        return False, f"predictions shard missing columns {sorted(required_pred_cols - set(preds_sub.columns))}"
    pred_counts = preds_sub.groupby(["method", "round"])["experiment_id"].nunique()
    if pred_counts.empty or not (pred_counts == args.test_size).all():
        return False, "existing predictions shard has incomplete test predictions"
    pred_round_counts = preds_sub.groupby("method")["round"].nunique()
    if set(preds_sub["round"].unique()) != expected_metric_rounds or not (pred_round_counts == len(expected_metric_rounds)).all():
        return False, "existing predictions shard is missing at least one method/round"

    return True, "complete"


def run_split(split_seed: int, data: pd.DataFrame, args: argparse.Namespace) -> None:
    if args.exploration_slots != 1:
        raise SystemExit("Experiment B exploration-enabled policies require --exploration-slots 1")
    split_total = args.initial_train_size + args.acquisition_pool_size + args.test_size
    if split_total != len(data):
        raise SystemExit(
            f"Split sizes must sum to {len(data)} canonical experiments; got {split_total}"
        )
    if min(args.initial_train_size, args.acquisition_pool_size, args.test_size) < 1:
        raise SystemExit("Split sizes must all be positive")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.acquisition_pool_size % args.batch_size != 0:
        raise SystemExit("--acquisition-pool-size must be divisible by --batch-size")
    n_acquisition_rounds = args.acquisition_pool_size // args.batch_size
    train0, pool0, test_idx = split_indices(
        len(data),
        split_seed,
        args.initial_train_size,
        args.acquisition_pool_size,
        args.test_size,
    )
    scaler = MinMaxScaler().fit(data[FEATURES].to_numpy(dtype=float))
    split_rng_base = 104729 + 100003 * split_seed

    split_rows = []
    for role, idxs in [("initial_train", train0), ("acquisition_pool", pool0), ("test", test_idx)]:
        for pos, idx in enumerate(idxs):
            split_rows.append({
                "split_seed": split_seed,
                "split_role": role,
                "position": pos,
                "experiment_id": data.loc[idx, "experiment_id"],
            })

    metric_all, selected_all, pred_all = [], [], []
    for method in args.methods:
        train_idx = [int(i) for i in train0]
        remaining = [int(i) for i in pool0]
        gate_fallback_count = 0

        for round_id in range(n_acquisition_rounds + 1):
            test_state = build_state(data, train_idx, test_idx, args, compute_gate=False)
            metric, pred_rows = metric_rows(
                data, split_seed, method, round_id, args.initial_train_size, train_idx, remaining, test_idx,
                test_state, args.catastrophic_threshold, gate_fallback_count,
            )
            metric_all.append(metric)
            pred_all.extend(pred_rows)
            if not remaining:
                break

            gate_state = build_state(data, train_idx, np.asarray(remaining, dtype=int), args, compute_gate=True)
            rng = np.random.default_rng(split_rng_base + 1009 * round_id + stable_int(method))
            context, cand_diag = acquisition_context(
                data, train_idx, remaining, gate_state, scaler, split_seed, method,
                round_id + 1, args.batch_size, args.exploration_slots, rng, args.gp_n_restarts,
            )
            selected, diag = select_batch(method, context)
            selected, fill_fallback = fill_unique(selected, context, args.batch_size)
            diag["gate_fallback"] = bool(diag.get("gate_fallback", False) or fill_fallback)
            if diag["gate_fallback"]:
                gate_fallback_count += 1
            roles = roles_for(method, selected, diag, args.batch_size, args.exploration_slots)
            cand_by_idx = cand_diag.set_index("pool_idx")
            for pos, (idx, role) in enumerate(zip(selected, roles), start=1):
                c = cand_by_idx.loc[idx]
                selected_all.append({
                    "split_seed": split_seed,
                    "method": method,
                    "round": round_id + 1,
                    "batch_position": pos,
                    "experiment_id": data.loc[idx, "experiment_id"],
                    "selection_role": role,
                    "PI": float(c["PI"]),
                    "EI": float(c["EI"]),
                    "GP_mean": float(c["GP_mean"]),
                    "GP_std": float(c["GP_std"]),
                    "d_min": float(c["d_min"]),
                    "B_I": float(c["B_I"]),
                    "B_F": float(c["B_F"]),
                    "suspicious": bool(c["suspicious"]),
                    "gate_fallback": bool(diag["gate_fallback"]),
                    "q_true": float(data.loc[idx, "q_true"]),
                })
            if len(set(selected)) != args.batch_size:
                raise RuntimeError(f"{method} split {split_seed} round {round_id + 1} selected duplicate points")
            if set(selected) & set(test_idx):
                raise RuntimeError("Test experiment leaked into selected acquisition batch")
            train_idx.extend(selected)
            remaining = [idx for idx in remaining if idx not in set(selected)]
        if len(train_idx) != args.initial_train_size + args.acquisition_pool_size or remaining:
            raise RuntimeError(f"{method} split {split_seed} did not exhaust the acquisition pool")

    out = args.results_dir / "shards"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_all).to_csv(out / f"experiment_b_per_split_round_seed_{split_seed}.csv", index=False)
    pd.DataFrame(selected_all).to_csv(out / f"experiment_b_selected_points_seed_{split_seed}.csv", index=False)
    pd.DataFrame(pred_all).to_csv(out / f"experiment_b_test_predictions_seed_{split_seed}.csv", index=False)
    pd.DataFrame(split_rows).to_csv(out / f"experiment_b_splits_seed_{split_seed}.csv", index=False)
    print(f"[DONE] split_seed={split_seed} rows={len(metric_all)} selections={len(selected_all)}")


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    data = load_experimental_data(args.error_data)
    seeds = [args.split_seed] if args.split_seed is not None else list(range(args.n_splits))
    started = time.perf_counter()
    skipped_seeds = []
    computed_seeds = []
    for seed in seeds:
        reusable, reason = split_shard_status(int(seed), args)
        if reusable and not args.force:
            print(f"[SKIP] split_seed={seed} existing shard is complete; reusing {args.results_dir / 'shards'}")
            skipped_seeds.append(int(seed))
            continue
        if not reusable:
            print(f"[RUN] split_seed={seed} existing shard not reusable: {reason}")
        elif args.force:
            print(f"[RUN] split_seed={seed} --force set; recomputing existing shard")
        run_split(int(seed), data, args)
        computed_seeds.append(int(seed))
    metadata = {
        "experiment": "Experiment B",
        "date": "2026-08-20",
        "n_splits_requested": len(seeds),
        "split_seeds": seeds,
        "computed_split_seeds": computed_seeds,
        "reused_split_seeds": skipped_seeds,
        "reuse_existing_shards": not args.force,
        "batch_size": args.batch_size,
        "acquisition_rounds": args.acquisition_pool_size // args.batch_size,
        "split_sizes": {
            "initial_train": args.initial_train_size,
            "acquisition_pool": args.acquisition_pool_size,
            "test": args.test_size,
        },
        "catastrophic_threshold": args.catastrophic_threshold,
        "catastrophic_definition": "abs(log1p(q_pred) - log1p(q_true)) >= catastrophic_threshold",
        "gate_definition": "nearest Chebyshev d_min > 4 and max(B_I, B_F) > 0",
        "error_data_usage": "Only experiment_id, reaction coordinates, Intensity_exp, FWHM_exp, and q_true are loaded from Error_data.csv.",
        "rf_n_trees": args.rf_n_trees,
        "rf_random_state": RF_RANDOM_STATE,
        "gate_rf_seeds": list(GATE_SEEDS),
        "methods": args.methods,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.results_dir / "experiment_b_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.aggregate:
        import summarize_experiment_b

        summarize_experiment_b.main(["--results-dir", str(args.results_dir)])


if __name__ == "__main__":
    main()
