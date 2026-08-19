"""Reconstruct the ActMOF 6,101,172-row virtual landscape."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_core import (
    EXPECTED_N_CANDIDATES,
    FEATURES,
    LANDSCAPE_DIR,
    apply_rules_batch,
    experimental_dataframe,
    grid_chunk_from_linear,
    postprocess_and_round_vec,
    train_landscape_rfs,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--chunk-size", type=int, default=100_000)
    p.add_argument("--rows-per-file", type=int, default=1_000_000)
    p.add_argument("--output-dir", type=Path, default=LANDSCAPE_DIR)
    p.add_argument("--minimal", action="store_true", help="write only five features plus q_final")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("design_grid_part*.csv"):
        old.unlink()

    print(f"[INFO] Full-grid total conditions: {EXPECTED_N_CANDIDATES:,}")
    print("[INFO] Training RF emulators from original 96-run experimental table.")
    rf_i, rf_f, exp = train_landscape_rfs()
    exp_lookup = {
        tuple(int(v) for v in row): (lab, float(i), float(f), int(q))
        for row, lab, i, f, q in zip(
            exp[FEATURES].to_numpy(),
            exp["label_id"],
            exp["Intensity_exp"],
            exp["FWHM_exp"],
            exp["q_exp"],
        )
    }

    y_counter = 1
    file_idx = 1
    rows_in_file = 0
    rows_written = 0
    t0 = time.time()

    for start in range(0, EXPECTED_N_CANDIDATES, args.chunk_size):
        end = min(start + args.chunk_size, EXPECTED_N_CANDIDATES)
        keys = grid_chunk_from_linear(start, end)
        b = len(keys)
        is_exp = np.array([tuple(int(x) for x in row) in exp_lookup for row in keys], dtype=bool)

        i_final = np.zeros(b, dtype=np.int64)
        f_final = np.zeros(b, dtype=np.float64)
        q_final = np.zeros(b, dtype=np.int64)
        rule = np.zeros(b, dtype=np.int8)
        label = np.empty(b, dtype=object)

        if np.any(is_exp):
            idx = np.where(is_exp)[0]
            vals = [exp_lookup[tuple(int(x) for x in row)] for row in keys[idx]]
            label[idx] = [v[0] for v in vals]
            i_final[idx] = np.rint([v[1] for v in vals]).astype(np.int64)
            f_final[idx] = np.asarray([v[2] for v in vals], dtype=np.float64)
            q_final[idx] = np.asarray([v[3] for v in vals], dtype=np.int64)

        if np.any(~is_exp):
            idx = np.where(~is_exp)[0]
            k_syn = keys[idx]
            rf_i_pred = rf_i.predict(k_syn)
            rf_f_pred = rf_f.predict(k_syn)
            i_blend, f_blend, flags = apply_rules_batch(k_syn, rf_i_pred, rf_f_pred, exp)
            i_pp, f_pp, q_pp = postprocess_and_round_vec(i_blend, f_blend)
            i_final[idx] = i_pp
            f_final[idx] = f_pp
            q_final[idx] = q_pp
            rule[idx] = flags
            label[idx] = [f"Y{n:07d}" for n in range(y_counter, y_counter + len(idx))]
            y_counter += len(idx)

        df = pd.DataFrame({c: keys[:, j].astype(int) for j, c in enumerate(FEATURES)})
        if not args.minimal:
            df["label_id"] = label
            df["Intensity_exp"] = np.where(is_exp, i_final, np.nan)
            df["FWHM_exp"] = np.where(is_exp, f_final, np.nan)
            df["q_exp"] = np.where(is_exp, q_final, np.nan)
            df["Intensity_final"] = i_final
            df["FWHM_final"] = np.where(np.isclose(f_final, 30.0), 30, f_final)
        df["q_final"] = q_final
        if not args.minimal:
            df["rule_applied"] = rule

        if rows_in_file + b > args.rows_per_file:
            file_idx += 1
            rows_in_file = 0
        out = out_dir / f"design_grid_part{file_idx:03d}.csv"
        df.to_csv(out, index=False, mode="a" if rows_in_file else "w", header=(rows_in_file == 0))
        rows_in_file += b
        rows_written += b

        elapsed = time.time() - t0
        rate = rows_written / max(elapsed, 1e-9)
        eta = (EXPECTED_N_CANDIDATES - rows_written) / max(rate, 1e-9)
        print(
            f"[WRITE] rows {rows_written:,}/{EXPECTED_N_CANDIDATES:,} | file {file_idx} | "
            f"elapsed {elapsed:.1f}s | ETA {eta:.1f}s | rate {rate:,.0f} rows/s"
        )

    if rows_written != EXPECTED_N_CANDIDATES:
        raise RuntimeError(f"wrote {rows_written:,}, expected {EXPECTED_N_CANDIDATES:,}")
    print(f"[DONE] Landscape written to {out_dir}")


if __name__ == "__main__":
    main()
