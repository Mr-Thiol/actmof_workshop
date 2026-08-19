#!/usr/bin/env python3
"""Run configurable paired vanilla versus fixed-gate ActMOF AL repeats."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.preprocessing import MinMaxScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
REPRO_DIR = SCRIPT_DIR.parent / "benchmark_reproduction"
sys.path.insert(0, str(REPRO_DIR))

from benchmark_core import (  # noqa: E402
    ACQ_CANDIDATES_PER_ITER,
    ConstantSurrogate,
    DIVERSITY_LAMBDA,
    FEATURES,
    INITIAL_EXPERIMENTS,
    METHOD,
    MIN_STD,
    RANDOM_STATE,
    SklearnGPSurrogate,
    TARGET,
    VALIDATION_SIZE,
    gaussian_crps,
    gaussian_nll,
    gaussian_pi,
    greedy_diverse_batch,
    kth_top_threshold,
    load_design_grid,
    predict_in_chunks,
    safe_spearman,
    stable_int,
    summarize_runs,
    y_from_model_scale,
    y_to_model_scale,
)
from policies import select_gate_batch  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "lucien" / "2026-08-19" / "experiment_a"
GATE_METADATA = RESULTS_DIR / "gate_metadata.npz"
BASELINE_RESULTS = REPO_ROOT / "results" / "lucien" / "2026-08-19" / "benchmark_reproduction" / "iteration_metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-rounds", "--n-repeats", dest="n_rounds", type=int, default=5,
        help="number of paired independent runs/repeats (default: 5)",
    )
    parser.add_argument("--batch-size", type=int, default=3, help="experiments selected per AL batch")
    parser.add_argument("--budget", type=int, default=150, help="total experiments per run, including initialization")
    parser.add_argument(
        "--exploration", type=int, default=1,
        help="suspicious-region PI slots per gated batch (default: 1)",
    )
    parser.add_argument("--gate-metadata", type=Path, default=GATE_METADATA)
    return parser.parse_args()


def evaluate_surrogate(model: Any, x_pool: np.ndarray, y_raw: np.ndarray, val_idx: np.ndarray) -> dict:
    if model is None:
        return {k: np.nan for k in [
            "surrogate_rmse_raw", "surrogate_mae_raw", "surrogate_spearman_raw",
            "surrogate_crps", "surrogate_nll", "surrogate_coverage95", "surrogate_mean_sd",
        ]}
    mu, sd = predict_in_chunks(model, x_pool[val_idx])
    y_model = y_to_model_scale(y_raw[val_idx])
    pred_raw = y_from_model_scale(mu)
    return {
        "surrogate_rmse_raw": float(np.sqrt(np.mean((pred_raw - y_raw[val_idx]) ** 2))),
        "surrogate_mae_raw": float(np.mean(np.abs(pred_raw - y_raw[val_idx]))),
        "surrogate_spearman_raw": safe_spearman(pred_raw, y_raw[val_idx]),
        "surrogate_crps": float(np.mean(gaussian_crps(y_model, mu, sd))),
        "surrogate_nll": float(np.mean(gaussian_nll(y_model, mu, sd))),
        "surrogate_coverage95": float(np.mean((y_model >= mu - 1.96 * sd) & (y_model <= mu + 1.96 * sd))),
        "surrogate_mean_sd": float(np.mean(np.maximum(sd, MIN_STD))),
    }


def run_once(
    pool_df: pd.DataFrame,
    suspicious_mask: np.ndarray,
    run: int,
    method: str,
    batch_size: int,
    budget: int,
    exploration_slots: int,
) -> tuple[list[dict], list[int], list[int]]:
    x_raw = pool_df[FEATURES].to_numpy(dtype=np.float64)
    y_raw = pool_df[TARGET].to_numpy(dtype=np.float64)
    x_pool = MinMaxScaler().fit_transform(x_raw)
    oracle_best, oracle_min = float(y_raw.max()), float(y_raw.min())
    top1, top01 = kth_top_threshold(y_raw, 0.01), kth_top_threshold(y_raw, 0.001)

    # Both methods intentionally use the successfully reproduced vanilla seed.
    seed = (RANDOM_STATE + 10_000 * run + stable_int(METHOD)) % (2**31 - 1)
    rng = np.random.default_rng(seed)
    tried = np.zeros(len(pool_df), dtype=bool)
    initial_idx = rng.choice(len(pool_df), size=INITIAL_EXPERIMENTS, replace=False)
    selected = [int(i) for i in initial_idx]
    tried[selected] = True
    val_idx = rng.choice(len(pool_df), size=min(VALIDATION_SIZE, len(pool_df)), replace=False)

    def metric_row(iteration, last_batch, fit_s, acq_s, cum_s, status, model, gate_diag):
        y_sel = y_raw[np.asarray(selected)]
        best_pool_idx = int(selected[int(np.argmax(y_sel))])
        best_q = float(y_raw[best_pool_idx])
        row = {
            "method": method,
            "family": "Acquisition Ablation" if method == METHOD else "Imperfection-aware",
            "run": run,
            "iteration": iteration,
            "configured_batch_size": batch_size,
            "configured_budget": budget,
            "configured_exploration": exploration_slots,
            "n_experiments": len(selected),
            "batch_size_added": len(last_batch),
            "best_q": best_q,
            "best_pool_idx": best_pool_idx,
            "simple_regret": oracle_best - best_q,
            "normalized_regret": (oracle_best - best_q) / max(oracle_best - oracle_min, 1e-12),
            "best_rank_percentile": float(100.0 * np.mean(y_raw <= best_q)),
            "hit_top1pct": float(best_q >= top1),
            "hit_top0_1pct": float(best_q >= top01),
            "success_fraction_selected": float(np.mean(y_sel > 0.0)),
            "mean_q_selected": float(np.mean(y_sel)),
            "last_batch_best_q": float(y_raw[np.asarray(last_batch)].max()) if last_batch else np.nan,
            "mean_pairwise_batch_distance": float(np.mean(pdist(x_pool[np.asarray(last_batch)]))) if len(last_batch) >= 2 else np.nan,
            "fit_seconds": fit_s,
            "acq_seconds": acq_s,
            "cum_seconds": cum_s,
            "status": status,
            "pool_oracle_best_q": oracle_best,
            "full_grid_best_q": oracle_best,
            **evaluate_surrogate(model, x_pool, y_raw, val_idx),
            **gate_diag,
        }
        return row

    initial_s = int(suspicious_mask[initial_idx].sum())
    initial_diag = {
        "n_acq_candidates": 0,
        "n_trustworthy_candidates": np.nan,
        "n_suspicious_candidates": np.nan,
        "selected_trustworthy": len(initial_idx) - initial_s,
        "selected_suspicious": initial_s,
        "requested_trustworthy": np.nan,
        "requested_suspicious": np.nan,
        "gate_fallback": False,
        "fallback_slots": "",
        "selected_suspicious_fraction": initial_s / len(initial_idx),
    }
    rows = [metric_row(0, list(initial_idx), 0.0, 0.0, 0.0, "initial", None, initial_diag)]
    cum_seconds = 0.0
    iteration = 0
    while len(selected) < min(budget, len(pool_df)):
        iteration += 1
        remaining = np.flatnonzero(~tried)
        batch_k = min(batch_size, budget - len(selected), len(remaining))
        x_train = x_pool[np.asarray(selected)]
        y_train = y_to_model_scale(y_raw[np.asarray(selected)])

        t0 = time.perf_counter()
        if np.nanstd(y_train) < 1e-12:
            model: Any = ConstantSurrogate().fit(x_train, y_train)
            status = "constant_target_fallback"
        else:
            model = SklearnGPSurrogate(random_state=seed + iteration, n_restarts=1).fit(x_train, y_train)
            status = "ok"
        fit_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        acq_idx = rng.choice(remaining, size=min(ACQ_CANDIDATES_PER_ITER, len(remaining)), replace=False)
        mu, sd = predict_in_chunks(model, x_pool[acq_idx])
        pi = gaussian_pi(mu, sd, float(y_train.max()), xi=0.01)
        n_suspicious = int(suspicious_mask[acq_idx].sum())
        if method == METHOD:
            next_idx = greedy_diverse_batch(acq_idx, pi, x_pool, batch_k, rng)
            policy_diag = {
                "requested_trustworthy": np.nan,
                "requested_suspicious": np.nan,
                "gate_fallback": False,
                "fallback_slots": "",
            }
        else:
            next_idx, policy_diag = select_gate_batch(
                acq_idx, pi, x_pool, suspicious_mask, batch_k,
                min(exploration_slots, batch_k), rng, DIVERSITY_LAMBDA,
            )
        next_idx = [int(i) for i in np.asarray(next_idx).reshape(-1) if not tried[int(i)]]
        if len(next_idx) < batch_k:
            available = remaining[~np.isin(remaining, next_idx)]
            missing = batch_k - len(next_idx)
            next_idx.extend(int(i) for i in rng.choice(available, size=missing, replace=False))
            policy_diag["gate_fallback"] = True
            policy_diag["fallback_slots"] = ";".join(filter(None, [policy_diag["fallback_slots"], "global_fill"]))
        acq_s = time.perf_counter() - t1

        for idx in next_idx:
            tried[idx] = True
            selected.append(idx)
        selected_s = int(suspicious_mask[np.asarray(next_idx)].sum())
        gate_diag = {
            "n_acq_candidates": len(acq_idx),
            "n_trustworthy_candidates": len(acq_idx) - n_suspicious,
            "n_suspicious_candidates": n_suspicious,
            "selected_trustworthy": len(next_idx) - selected_s,
            "selected_suspicious": selected_s,
            **policy_diag,
            "selected_suspicious_fraction": selected_s / len(next_idx),
        }
        # Actual counts take precedence over the policy's pre-fill counts.
        gate_diag["selected_trustworthy"] = len(next_idx) - selected_s
        gate_diag["selected_suspicious"] = selected_s
        cum_seconds += fit_s + acq_s
        rows.append(metric_row(iteration, next_idx, fit_s, acq_s, cum_seconds, status, model, gate_diag))
        print(f"[{method} run {run}] iter={iteration:02d} n={len(selected)} best_q={rows[-1]['best_q']:.0f}")
    return rows, selected, [int(i) for i in initial_idx]


def regression_check(vanilla: pd.DataFrame, batch_size: int) -> None:
    if batch_size != 3:
        print("[SKIP] Validated vanilla reference uses batch_size=3; batch-size regression is not applicable.")
        return
    if not BASELINE_RESULTS.exists():
        raise FileNotFoundError(f"Validated baseline output not found: {BASELINE_RESULTS}")
    reference = pd.read_csv(BASELINE_RESULTS)
    keys = ["run", "iteration"]
    actual = vanilla.merge(reference[keys], on=keys, how="inner").sort_values(keys)
    reference = reference.merge(vanilla[keys], on=keys, how="inner").sort_values(keys)
    if not len(actual):
        raise RuntimeError("No overlapping rows with the validated vanilla reference")
    for col in ["best_q", "best_pool_idx"]:
        if not np.array_equal(actual[col].to_numpy(), reference[col].to_numpy()):
            raise RuntimeError(f"Vanilla regression failed for {col}")
    print(f"[PASS] Vanilla best_q and best_pool_idx match {len(actual)} overlapping validated iteration rows.")


def pool_feature_hash(pool_df: pd.DataFrame, chunk_size: int = 250_000) -> str:
    digest = hashlib.sha256()
    for start in range(0, len(pool_df), chunk_size):
        x = pool_df.iloc[start:start + chunk_size][FEATURES].to_numpy(dtype=np.int16)
        digest.update(np.ascontiguousarray(x).tobytes())
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.n_rounds < 1:
        raise SystemExit("--n-rounds must be positive")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.budget < INITIAL_EXPERIMENTS:
        raise SystemExit(f"--budget must be at least INITIAL_EXPERIMENTS={INITIAL_EXPERIMENTS}")
    if not 0 <= args.exploration <= args.batch_size:
        raise SystemExit("--exploration must be between 0 and --batch-size")
    n_trustworthy = args.batch_size - args.exploration
    gate_method = f"GP_Matern52_PI_GATE_{n_trustworthy}T{args.exploration}S"
    gate_label = f"Gate-{n_trustworthy}T{args.exploration}S"
    print(
        f"[CONFIG] paired_runs={args.n_rounds} batch_size={args.batch_size} "
        f"budget={args.budget} exploration_slots={args.exploration} method={gate_method}"
    )
    metadata = np.load(args.gate_metadata)
    suspicious = metadata["suspicious"].astype(bool)
    pool_idx = metadata["pool_idx"]
    pool_df, _best, _total = load_design_grid()
    if len(pool_df) != len(suspicious) or not np.array_equal(pool_idx, np.arange(len(pool_df))):
        raise RuntimeError("Gate metadata is not aligned with benchmark pool_idx")
    expected_hash = str(metadata["features_sha256"].item())
    if pool_feature_hash(pool_df) != expected_hash:
        raise RuntimeError("Gate metadata feature order does not match benchmark pool_df")
    print(f"[PASS] Gate metadata feature order matches all {len(pool_df):,} benchmark pool rows.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    vanilla_rows = []
    vanilla_initial = {}
    for run in range(args.n_rounds):
        rows, _, initial = run_once(
            pool_df, suspicious, run, METHOD,
            args.batch_size, args.budget, args.exploration,
        )
        vanilla_initial[run] = initial
        vanilla_rows.extend(rows)
        pd.DataFrame(vanilla_rows).to_csv(RESULTS_DIR / "iteration_metrics.csv", index=False)
    vanilla = pd.DataFrame(vanilla_rows)
    regression_check(vanilla, args.batch_size)

    gate_rows = []
    for run in range(args.n_rounds):
        rows, _, initial = run_once(
            pool_df, suspicious, run, gate_method,
            args.batch_size, args.budget, args.exploration,
        )
        if initial != vanilla_initial[run]:
            raise RuntimeError(f"Paired initialization failed for run {run}")
        gate_rows.extend(rows)
        pd.concat([vanilla, pd.DataFrame(gate_rows)], ignore_index=True).to_csv(
            RESULTS_DIR / "iteration_metrics.csv", index=False
        )

    metrics = pd.concat([vanilla, pd.DataFrame(gate_rows)], ignore_index=True)
    run_summary, final_summary = summarize_runs(metrics)
    run_summary.to_csv(RESULTS_DIR / "run_summary.csv", index=False)
    final_summary.to_csv(RESULTS_DIR / "final_summary.csv", index=False)

    finals = run_summary.pivot(index="run", columns="method", values="final_best_q")
    differences = finals[gate_method] - finals[METHOD]
    gate_iters = metrics[(metrics.method == gate_method) & (metrics.iteration > 0)]
    fallbacks = int(gate_iters["gate_fallback"].astype(bool).sum())
    print("\nExperiment A\n")
    print(f"Run    Vanilla final q    {gate_label} final q    Difference")
    for run, row in finals.iterrows():
        print(f"{run + 1:<7}{row[METHOD]:<19.0f}{row[gate_method]:<22.0f}{differences.loc[run]:.0f}")
    for method, label in [(METHOD, "Vanilla"), (gate_method, gate_label)]:
        values = finals[method]
        print(f"{label} mean ± SD: {values.mean():.1f} ± {values.std(ddof=1):.1f}")
    print(f"Mean paired difference: {differences.mean():.1f}")
    print(f"Total gate fallbacks: {fallbacks}")
    print(f"Fraction of gate iterations requiring fallback: {fallbacks / len(gate_iters):.6f}")


if __name__ == "__main__":
    main()
