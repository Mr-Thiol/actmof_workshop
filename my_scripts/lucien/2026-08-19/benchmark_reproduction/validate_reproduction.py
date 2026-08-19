"""Validate reproduced ActMOF landscape against original split CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_core import EXPECTED_N_CANDIDATES, FEATURES, LANDSCAPE_DIR, ORIGINAL_DIR, TARGET


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--new-dir", type=Path, default=LANDSCAPE_DIR)
    p.add_argument("--original-dir", type=Path, default=ORIGINAL_DIR)
    p.add_argument("--sample-rows", type=int, default=20_000)
    return p.parse_args()


def files(base: Path):
    return sorted(base.glob("design_grid_part*.csv"))


def scan(paths: list[Path]):
    usecols = FEATURES + ["Intensity_final", "FWHM_final", TARGET, "rule_applied"]
    total = zeros = 0
    q_values = []
    rule_counts = {}
    mins = {c: np.inf for c in FEATURES}
    maxs = {c: -np.inf for c in FEATURES}
    uniques = {c: set() for c in FEATURES}
    top = []
    for path in paths:
        for chunk in pd.read_csv(path, usecols=lambda c: c in usecols, chunksize=250_000, low_memory=False):
            total += len(chunk)
            q = pd.to_numeric(chunk[TARGET], errors="coerce").fillna(0).to_numpy()
            zeros += int(np.sum(q == 0))
            q_values.append(q)
            if "rule_applied" in chunk.columns:
                for k, v in chunk["rule_applied"].value_counts().items():
                    rule_counts[int(k)] = rule_counts.get(int(k), 0) + int(v)
            for c in FEATURES:
                vals = pd.to_numeric(chunk[c], errors="coerce").dropna()
                mins[c] = min(mins[c], float(vals.min()))
                maxs[c] = max(maxs[c], float(vals.max()))
                uniques[c].update(vals.astype(int).unique().tolist())
            top.append(chunk.nlargest(min(10, len(chunk)), TARGET))
    q_all = np.concatenate(q_values)
    top_df = pd.concat(top, ignore_index=True).nlargest(10, TARGET)
    return {
        "total_rows": total,
        "feature_min": mins,
        "feature_max": maxs,
        "feature_nunique": {k: len(v) for k, v in uniques.items()},
        "q_min": float(np.min(q_all)),
        "q_max": float(np.max(q_all)),
        "q_zero_count": zeros,
        "q_zero_proportion": zeros / total,
        "q_quantiles": {str(q): float(v) for q, v in zip([0, .5, .9, .95, .99, .999, 1], np.quantile(q_all, [0, .5, .9, .95, .99, .999, 1]))},
        "top1pct_threshold": float(np.partition(q_all, len(q_all) - int(np.ceil(0.01 * len(q_all))))[len(q_all) - int(np.ceil(0.01 * len(q_all)))]),
        "top0_1pct_threshold": float(np.partition(q_all, len(q_all) - int(np.ceil(0.001 * len(q_all))))[len(q_all) - int(np.ceil(0.001 * len(q_all)))]),
        "rule_counts": rule_counts,
        "top10": top_df,
    }


def print_stats(name: str, stats: dict):
    print(f"\n{name}")
    for key in [
        "total_rows", "feature_min", "feature_max", "feature_nunique", "q_min", "q_max",
        "q_zero_count", "q_zero_proportion", "q_quantiles", "top1pct_threshold",
        "top0_1pct_threshold", "rule_counts",
    ]:
        print(f"{key}: {stats[key]}")
    print("top10:")
    print(stats["top10"].to_string(index=False))


def compare_sample(new_paths: list[Path], old_paths: list[Path], n: int):
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(EXPECTED_N_CANDIDATES, size=min(n, EXPECTED_N_CANDIDATES), replace=False))
    usecols = FEATURES + ["Intensity_final", "FWHM_final", TARGET, "rule_applied"]

    def rows_at(paths):
        frames = []
        offset = 0
        wanted_pos = 0
        for path in paths:
            rows = sum(1 for _ in open(path)) - 1
            local = idx[(idx >= offset) & (idx < offset + rows)] - offset
            if len(local):
                df = pd.read_csv(path, usecols=lambda c: c in usecols)
                frames.append(df.iloc[local])
                wanted_pos += len(local)
            offset += rows
        return pd.concat(frames, ignore_index=True)

    a = rows_at(new_paths)
    b = rows_at(old_paths)
    neq = (a.reset_index(drop=True).astype(str) != b.reset_index(drop=True).astype(str)).any(axis=1)
    print(f"\nsampled row comparison: {len(a) - int(neq.sum()):,}/{len(a):,} exact string-matched rows")
    if neq.any():
        print("first mismatches:")
        print(pd.concat({"new": a[neq].head(), "original": b[neq].head()}, axis=1).to_string(index=False))


def main():
    args = parse_args()
    new_paths = files(args.new_dir)
    old_paths = files(args.original_dir)
    if not old_paths:
        raise FileNotFoundError(f"No original design_grid_part*.csv files under {args.original_dir}")
    if not new_paths:
        print(f"No reproduced files found under {args.new_dir}; reporting original statistics only.")
        old = scan(old_paths)
        print_stats("original", old)
        return

    new = scan(new_paths)
    old = scan(old_paths)
    print_stats("reproduced", new)
    print_stats("original", old)
    print("\nsummary differences:")
    for key in ["total_rows", "q_min", "q_max", "q_zero_count", "q_zero_proportion", "top1pct_threshold", "top0_1pct_threshold"]:
        print(f"{key}: reproduced={new[key]} original={old[key]} diff={new[key] - old[key]}")
    compare_sample(new_paths, old_paths, args.sample_rows)


if __name__ == "__main__":
    main()
