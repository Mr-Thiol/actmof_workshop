#!/usr/bin/env python3
"""Re-simulate the ActMOF LOO reconstruction (same code path as build_error_dataset.py)
to recover the PRE-threshold continuous intensity/FWHM predictions, then derive:
  10.1 continuous boundary margins M_I, M_F
  10.2 continuous local-support / density features (k-NN distances, KDE)
"""
from __future__ import annotations
import argparse
import time
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "figures" / "followup" / "continuous_features.csv"

FEATURES = ["metal_amount", "modulator", "add_solvent", "reaction_time", "reaction_temperature"]
SEEDS = [0, 1, 2, 3, 4]
N_TREES = 400

# verbatim from build_error_dataset.py
EXPERIMENTS_CSV = """35,5,22,9,14,0,30
68,7,15,12,24,0,30
12,9,1,7,29,785,2.2
36,10,24,3,18,0,30
55,11,8,2,15,0,30
69,6,23,1,20,605,30
14,12,2,11,15,0,30
42,7,27,6,17,0,30
9,14,7,4,11,182,30
45,6,11,8,28,0,30
27,12,1,2,25,880,3.9
41,14,5,3,16,0,30
5,5,0,1,30,1457,0.32
5,15,0,1,30,0,30
75,15,0,1,30,0,30
16,7,22,3,20,2345,0.37
16,9,23,4,21,960,1.1
15,7,12,11,19,16751,0.22
6,5,29,11,27,7575,0.18
62,7,26,11,26,0,30
5,15,2,11,16,0,30
33,10,7,11,18,8060,0.25
9,6,1,12,19,1318,0.4
48,14,3,12,23,920,4.6
19,8,12,11,16,0,30
15,7,14,11,24,23969,0.21
12,8,10,11,18,7305,0.15
15,7,13,11,18,7703,0.27
14,7,11,11,20,12434,0.24
7,6,20,11,15,0,30
15,7,14,11,24,23969,0.21
15,6,10,11,14,0,30
15,7,10,11,12,0,30
15,8,12,11,19,1103,0.8
9,8,13,7,20,0,30
21,6,13,12,19,17269,0.21
6,6,12,11,20,1077,0.48
15,8,9,12,19,903,2
75,12,6,11,18,0,30
25,13,14,7,24,0,30
17,13,16,12,24,0,30
38,13,16,8,24,0,30
65,5,0,12,24,0,30
74,7,23,1,16,0,30
5,7,21,12,20,0,30
15,7,16,12,29,31082,0.2
16,7,15,11,18,12791,0.3
62,10,12,10,17,0,30
11,7,12,11,24,28179,0.26
17,7,16,11,27,29258,0.29
13,7,10,12,25,24240,0.24
13,6,13,11,24,37307,0.18
15,5,0,12,24,15489,0.28
21,5,26,11,23,9906,0.26
16,7,8,11,24,21878,0.21
28,7,0,12,30,11411,0.22
10,7,0,12,24,688,0.6
23,7,30,11,24,18169,0.2
20,7,30,11,29,10048,0.22
23,7,14,11,25,12005,0.24
16,7,18,12,25,35900,0.22
17,7,18,12,24,35967,0.21
24,7,18,12,24,13410,0.22
16,7,14,11,30,2331,0.3
15,7,13,9,23,23301,0.29
15,7,7,7,26,11663,0.23
13,6,25,12,24,23658,0.28
13,7,11,11,23,17558,0.26
14,7,7,11,26,7418,0.29
15,6,23,12,24,32110,0.3
15,5,8,12,23,20121,0.21
15,6,14,12,21,19118,0.25
13,5,10,11,24,34595,0.14
17,6,11,11,22,17880,0.22
19,5,2,12,24,6978,0.16
6,6,19,12,30,0,30
25,7,18,11,24,8241,0.26
27,7,27,12,26,7910,0.23
15,5,8,11,23,26277,0.22
15,5,11,10,21,18112,0.29
24,5,30,12,24,4394,0.31
13,7,15,12,25,27452,0.24
16,7,11,12,24,16348,0.28
14,6,15,12,26,30033,0.23
14,7,15,11,26,28764,0.29
18,7,10,12,23,28299,0.22
10,7,13,12,29,1082,0.41
23,7,2,12,19,20334,0.27
18,7,1,4,19,19300,0.32
19,6,4,12,25,12874,0.21
17,7,10,12,25,28836,0.26
16,6,6,5,23,16918,0.31
20,6,0,12,23,11668,0.3
17,7,20,11,21,26346,0.28
24,7,22,8,21,17895,0.31
25,7,19,10,27,11657,0.34"""


def calculate_q(intensity: float, fwhm: float) -> int:
    if fwhm == 30 or fwhm <= 0 or intensity <= 0:
        return 0
    return int(round(float(intensity) / float(fwhm)))


def load_and_aggregate_experiments() -> pd.DataFrame:
    columns = FEATURES + ["Intensity_exp", "FWHM_exp"]
    raw = pd.read_csv(StringIO(EXPERIMENTS_CSV), header=None, names=columns)
    data = raw.groupby(FEATURES, as_index=False).agg(
        Intensity_exp=("Intensity_exp", "mean"), FWHM_exp=("FWHM_exp", "mean"))
    data["q_true"] = [calculate_q(i, f) for i, f in zip(data.Intensity_exp, data.FWHM_exp)]
    data = data.sort_values(FEATURES).reset_index(drop=True)
    data.insert(0, "experiment_id", [f"Q{i:03d}" for i in range(1, len(data) + 1)])
    return data


def compute_neighbor_features(point, remaining):
    diffs = np.abs(remaining - point[0])
    rule1 = np.all(diffs <= 1, axis=1)
    at_two = diffs == 2
    rule2 = (np.all(diffs <= 2, axis=1) & np.isin(at_two.sum(axis=1), (1, 2))
             & np.all((diffs <= 1) | at_two, axis=1))
    return diffs, rule1, rule2


def apply_actmof_rules(rf_i, rf_f, rule1, rule2, intensity, fwhm):
    if rule1.any():
        return 0.8 * intensity[rule1].mean() + 0.2 * rf_i, 0.8 * fwhm[rule1].mean() + 0.2 * rf_f, 1
    if rule2.any():
        return 0.5 * intensity[rule2].mean() + 0.5 * rf_i, 0.5 * fwhm[rule2].mean() + 0.5 * rf_f, 2
    return rf_i, rf_f, 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_and_aggregate_experiments()
    assert len(data) == 95, len(data)

    t0 = time.perf_counter()
    records = []
    for held_out in data.index:
        point = data.loc[held_out, FEATURES].to_numpy(dtype=float).reshape(1, -1)
        train = data.drop(index=held_out)
        x = train[FEATURES].to_numpy(dtype=float)
        y_i = train.Intensity_exp.to_numpy(dtype=float)
        y_f = train.FWHM_exp.to_numpy(dtype=float)
        diffs, rule1, rule2 = compute_neighbor_features(point, x)

        i_pres, f_pres, rf_is, rf_fs, var_is, var_fs = [], [], [], [], [], []
        for seed in SEEDS:
            kwargs = dict(n_estimators=N_TREES, random_state=seed, n_jobs=1)
            rfi_model = RandomForestRegressor(**kwargs).fit(x, y_i)
            rff_model = RandomForestRegressor(**kwargs).fit(x, y_f)
            vi = np.fromiter((t.predict(point)[0] for t in rfi_model.estimators_), dtype=float)
            vf = np.fromiter((t.predict(point)[0] for t in rff_model.estimators_), dtype=float)
            rf_i_mean, rf_i_var = float(vi.mean()), float(vi.var())
            rf_f_mean, rf_f_var = float(vf.mean()), float(vf.var())
            i_pre, f_pre, rule = apply_actmof_rules(rf_i_mean, rf_f_mean, rule1, rule2, y_i, y_f)
            i_pres.append(i_pre); f_pres.append(f_pre)
            rf_is.append(rf_i_mean); rf_fs.append(rf_f_mean)
            var_is.append(rf_i_var); var_fs.append(rf_f_var)

        i_pres = np.array(i_pres); f_pres = np.array(f_pres)

        # ---- 10.2 continuous local-support / density (same raw-unit Chebyshev metric as d_min) ----
        cheb = diffs.max(axis=1)
        d_sorted = np.sort(cheb)
        d1, d2, d3 = d_sorted[0], d_sorted[1], d_sorted[2]
        mean_k3 = d_sorted[:3].mean()
        mean_k5 = d_sorted[:5].mean()
        h = np.median(cheb)  # bandwidth: median-heuristic
        kde_cheb = float(np.exp(-(cheb ** 2) / (2 * h ** 2)).sum())

        records.append(dict(
            experiment_id=data.loc[held_out, "experiment_id"],
            I_pre_mean=i_pres.mean(), I_pre_std=i_pres.std(ddof=0),
            F_pre_mean=f_pres.mean(), F_pre_std=f_pres.std(ddof=0),
            rf_i_mean_check=np.mean(rf_is), rf_f_mean_check=np.mean(rf_fs),
            rf_intensity_variance_check=np.mean(var_is), rf_fwhm_variance_check=np.mean(var_fs),
            d_min_check=d1, d2=d2, d3=d3, mean_d_k3=mean_k3, mean_d_k5=mean_k5, kde_cheb=kde_cheb,
        ))
        if (held_out + 1) % 20 == 0:
            print(f"  ...{held_out + 1}/95 done ({time.perf_counter()-t0:.1f}s)")

    feat_df = pd.DataFrame(records)
    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(out_path, index=False)
    print(f"Done in {time.perf_counter()-t0:.1f}s. Saved: {out_path}")


if __name__ == "__main__":
    main()
