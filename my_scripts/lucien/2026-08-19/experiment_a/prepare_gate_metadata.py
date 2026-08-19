#!/usr/bin/env python3
"""Precompute the frozen Experiment-A gate over the benchmark candidate pool."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

SCRIPT_DIR = Path(__file__).resolve().parent
REPRO_DIR = SCRIPT_DIR.parent / "benchmark_reproduction"
sys.path.insert(0, str(REPRO_DIR))

from benchmark_core import (  # noqa: E402
    EXPECTED_N_CANDIDATES,
    FEATURES,
    RF_N_TREES,
    experimental_dataframe,
    landscape_files,
)
from gate import classify_gate, compute_boundary_features, compute_dmin  # noqa: E402

REPO_ROOT = SCRIPT_DIR.parents[3]
RESULTS_DIR = REPO_ROOT / "results" / "lucien" / "2026-08-19" / "experiment_a"
DEFAULT_OUTPUT = RESULTS_DIR / "gate_metadata.npz"
SEEDS = (0, 1, 2, 3, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def train_boundary_ensemble(exp: pd.DataFrame):
    x = exp[FEATURES].to_numpy(dtype=np.int16)
    y_i = exp["Intensity_exp"].to_numpy(dtype=float)
    y_f = exp["FWHM_exp"].to_numpy(dtype=float)
    models_i, models_f = [], []
    for seed in SEEDS:
        kwargs = dict(n_estimators=RF_N_TREES, random_state=seed, n_jobs=-1)
        models_i.append(RandomForestRegressor(**kwargs).fit(x, y_i))
        models_f.append(RandomForestRegressor(**kwargs).fit(x, y_f))
    return models_i, models_f


def feature_block(
    x: np.ndarray,
    exp: pd.DataFrame,
    models_i: list,
    models_f: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    exp_x = exp[FEATURES].to_numpy(dtype=np.int16)
    exp_i = exp["Intensity_exp"].to_numpy(dtype=np.float32)
    exp_f = exp["FWHM_exp"].to_numpy(dtype=np.float32)
    d_min = compute_dmin(x, exp_x, chunk_size=len(x))
    b_i, b_f = compute_boundary_features(x, exp_x, exp_i, exp_f, models_i, models_f)
    return d_min, b_i, b_f


def main() -> None:
    args = parse_args()
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be positive")
    paths = landscape_files()
    if not paths:
        raise FileNotFoundError("No reproduced or original design_grid_part*.csv files found")

    exp = experimental_dataframe()
    if len(exp) != 95:
        raise RuntimeError(f"Expected 95 unique experiments, found {len(exp)}")
    print("[TRAIN] Fitting the established five-seed frozen RF ensemble (seeds 0..4).")
    models_i, models_f = train_boundary_ensemble(exp)

    d_min = np.empty(EXPECTED_N_CANDIDATES, dtype=np.float32)
    b_i = np.empty(EXPECTED_N_CANDIDATES, dtype=np.float32)
    b_f = np.empty(EXPECTED_N_CANDIDATES, dtype=np.float32)
    q_blocks: list[np.ndarray] = []
    feature_hash = hashlib.sha256()
    cursor = 0
    scanned = 0
    started = time.perf_counter()

    # Benchmark v3 moves Q rows to the end via append + keep="last" dedup.
    # Stream non-Q rows first and append the 95 Q rows to reproduce that pool_idx.
    for path in paths:
        for chunk in pd.read_csv(path, usecols=FEATURES + ["label_id"], chunksize=args.chunk_size):
            x_all = chunk[FEATURES].to_numpy(dtype=np.int16)
            q_mask = chunk["label_id"].astype(str).str.startswith("Q").to_numpy()
            if q_mask.any():
                q_blocks.append(x_all[q_mask])
            x = x_all[~q_mask]
            if len(x):
                feature_hash.update(np.ascontiguousarray(x).tobytes())
                d, bi, bf = feature_block(x, exp, models_i, models_f)
                stop = cursor + len(x)
                d_min[cursor:stop], b_i[cursor:stop], b_f[cursor:stop] = d, bi, bf
                cursor = stop
            scanned += len(chunk)
            elapsed = time.perf_counter() - started
            print(f"[GATE] scanned={scanned:,}/{EXPECTED_N_CANDIDATES:,} pool_rows={cursor:,} elapsed={elapsed:.1f}s")

    q_x = np.concatenate(q_blocks)
    feature_hash.update(np.ascontiguousarray(q_x).tobytes())
    d, bi, bf = feature_block(q_x, exp, models_i, models_f)
    stop = cursor + len(q_x)
    d_min[cursor:stop], b_i[cursor:stop], b_f[cursor:stop] = d, bi, bf
    cursor = stop
    if scanned != EXPECTED_N_CANDIDATES or cursor != EXPECTED_N_CANDIDATES or len(q_x) != len(exp):
        raise RuntimeError(f"Pool alignment failed: scanned={scanned}, written={cursor}, Q rows={len(q_x)}")

    boundary_risk = np.maximum(b_i, b_f)
    suspicious = classify_gate(d_min, b_i, b_f)
    rng = np.random.default_rng(42)
    check = rng.choice(len(suspicious), size=10_000, replace=False)
    expected = (d_min[check] > 4.0) & (boundary_risk[check] > 0.0)
    if not np.array_equal(suspicious[check], expected):
        raise RuntimeError("Gate equality sanity check failed")
    if suspicious[d_min == 4.0].any() or suspicious[boundary_risk == 0.0].any():
        raise RuntimeError("Strict-boundary gate sanity check failed")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        pool_idx=np.arange(len(d_min), dtype=np.int32),
        d_min=d_min,
        B_I=b_i,
        B_F=b_f,
        boundary_risk=boundary_risk,
        suspicious=suspicious,
        features_sha256=np.asarray(feature_hash.hexdigest()),
    )

    summary = {
        "total_candidates": len(suspicious),
        "trustworthy_candidates": int((~suspicious).sum()),
        "suspicious_candidates": int(suspicious.sum()),
        "suspicious_fraction": float(suspicious.mean()),
        "d_min_min": float(d_min.min()), "d_min_median": float(np.median(d_min)), "d_min_max": float(d_min.max()),
        "B_I_min": float(b_i.min()), "B_I_median": float(np.median(b_i)), "B_I_max": float(b_i.max()),
        "B_F_min": float(b_f.min()), "B_F_median": float(np.median(b_f)), "B_F_max": float(b_f.max()),
        "boundary_risk_min": float(boundary_risk.min()),
        "boundary_risk_median": float(np.median(boundary_risk)),
        "boundary_risk_max": float(boundary_risk.max()),
        "d_min_eq_4": int((d_min == 4.0).sum()),
        "d_min_gt_4": int((d_min > 4.0).sum()),
        "boundary_risk_eq_0": int((boundary_risk == 0.0).sum()),
        "boundary_risk_gt_0": int((boundary_risk > 0.0).sum()),
        "features_sha256": feature_hash.hexdigest(),
    }
    pd.DataFrame([summary]).to_csv(args.output.parent / "gate_summary.csv", index=False)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"[DONE] {args.output}")


if __name__ == "__main__":
    main()
