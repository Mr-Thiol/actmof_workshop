"""Minimal ActMOF benchmark reproduction core.

Source of truth:
- ``ActMOF_code_original/construct dataset with time tracker.ipynb`` cell 0
  defines the 5D full grid, 96 raw experimental runs, RF emulators, Rule 1/2
  neighbor blending, hard post-processing, and q_final.
- ``ActMOF_code_original/Benchmark v3.ipynb`` main benchmark cell defines
  ``GP_Matern52_PI`` and active-learning settings.
"""

from __future__ import annotations

import glob
import hashlib
import math
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import norm, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


REPO_ROOT = Path(__file__).resolve().parents[4]
ORIGINAL_DIR = REPO_ROOT / "ActMOF_code_original"
RESULTS_DIR = REPO_ROOT / "results" / "lucien" / "2026-08-19" / "benchmark_reproduction"
LANDSCAPE_DIR = RESULTS_DIR / "landscape"

FEATURES = [
    "metal_amount",
    "modulator",
    "add_solvent",
    "reaction_time",
    "reaction_temperature",
]
TARGET = "q_final"
BOUNDS = {
    "metal_amount": (5, 75, 1),
    "modulator": (5, 15, 1),
    "add_solvent": (0, 30, 1),
    "reaction_time": (1, 12, 1),
    "reaction_temperature": (10, 30, 1),
}
GRID_VALUES = [np.arange(lo, hi + 1, step, dtype=np.int16) for lo, hi, step in BOUNDS.values()]
EXPECTED_N_CANDIDATES = int(np.prod([len(v) for v in GRID_VALUES]))

RF_N_TREES = 400
RF_RANDOM_STATE = 42

BATCH_SIZE = 3
BUDGET_TOTAL_EXPERIMENTS = 300
INITIAL_EXPERIMENTS = 3
USE_EXISTING_Q_ROWS_AS_INITIAL = False
N_REPEATS = 5
MAX_CANDIDATE_POOL = None
ACQ_CANDIDATES_PER_ITER = 15_000
VALIDATION_SIZE = 5_000
LOAD_CHUNK_SIZE = 250_000
PREDICT_CHUNK_SIZE = 50_000
RANDOM_STATE = 42
USE_LOG1P_TARGET = True
MIN_STD = 1e-9
DIVERSITY_LAMBDA = 0.03
METHOD = "GP_Matern52_PI"
FAMILY = "Acquisition Ablation"


EXPERIMENTAL_X = np.array([
    [35,5,22,9,14],[68,7,15,12,24],[12,9,1,7,29],[36,10,24,3,18],[55,11,8,2,15],
    [69,6,23,1,20],[14,12,2,11,15],[42,7,27,6,17],[9,14,7,4,11],[45,6,11,8,28],
    [27,12,1,2,25],[41,14,5,3,16],
    [5,5,0,1,30],[5,15,0,1,30],[75,15,0,1,30],
    [16,7,22,3,20],[16,9,23,4,21],[15,7,12,11,19],
    [6,5,29,11,27],[62,7,26,11,26],[5,15,2,11,16],
    [33,10,7,11,18],[9,6,1,12,19],[48,14,3,12,23],
    [19,8,12,11,16],[15,7,14,11,24],[12,8,10,11,18],
    [15,7,13,11,18],[14,7,11,11,20],[7,6,20,11,15],
    [15,7,14,11,24],[15,6,10,11,14],[15,7,10,11,12],
    [15,8,12,11,19],[9,8,13,7,20],[21,6,13,12,19],
    [6,6,12,11,20],[15,8,9,12,19],[75,12,6,11,18],
    [25,13,14,7,24],[17,13,16,12,24],[38,13,16,8,24],
    [65,5,0,12,24],[74,7,23,1,16],[5,7,21,12,20],
    [15,7,16,12,29],[16,7,15,11,18],[62,10,12,10,17],
    [11,7,12,11,24],[17,7,16,11,27],[13,7,10,12,25],
    [13,6,13,11,24],[15,5,0,12,24],[21,5,26,11,23],
    [16,7,8,11,24],[28,7,0,12,30],[10,7,0,12,24],
    [23,7,30,11,24],[20,7,30,11,29],[23,7,14,11,25],
    [16,7,18,12,25],[17,7,18,12,24],[24,7,18,12,24],
    [16,7,14,11,30],[15,7,13,9,23],[15,7,7,7,26],
    [13,6,25,12,24],[13,7,11,11,23],[14,7,7,11,26],
    [15,6,23,12,24],[15,5,8,12,23],[15,6,14,12,21],
    [13,5,10,11,24],[17,6,11,11,22],[19,5,2,12,24],
    [6,6,19,12,30],[25,7,18,11,24],[27,7,27,12,26],
    [15,5,8,11,23],[15,5,11,10,21],[24,5,30,12,24],
    [13,7,15,12,25],[16,7,11,12,24],[14,6,15,12,26],
    [14,7,15,11,26],[18,7,10,12,23],[10,7,13,12,29],
    [23,7,2,12,19],[18,7,1,4,19],[19,6,4,12,25],
    [17,7,10,12,25],[16,6,6,5,23],[20,6,0,12,23],
    [17,7,20,11,21],[24,7,22,8,21],[25,7,19,10,27]
], dtype=np.int16)

EXPERIMENTAL_INTENSITY = np.array([
    0,0,785,0,0,605,0,0,182,0,880,0,
    1457,0,0,2345,960,16751,7575,0,0,8060,1318,920,
    0,23969,7305,7703,12434,0,23969,0,0,1103,0,17269,
    1077,903,0,0,0,0,0,0,0,31082,12791,0,28179,29258,24240,
    37307,15489,9906,21878,11411,688,18169,10048,12005,35900,35967,13410,
    2331,23301,11663,23658,17558,7418,32110,20121,19118,34595,17880,6978,
    0,8241,7910,26277,18112,4394,27452,16348,30033,28764,28299,1082,
    20334,19300,12874,28836,16918,11668,26346,17895,11657
], dtype=np.float32)

EXPERIMENTAL_FWHM = np.array([
    30,30,2.2,30,30,30,30,30,30,30,3.9,30,
    0.32,30,30,0.37,1.1,0.22,0.18,30,30,0.25,0.40,4.6,
    30,0.21,0.15,0.27,0.24,30,0.21,30,30,0.8,30,0.21,
    0.48,2,30,30,30,30,30,30,30,0.2,0.3,30,0.26,0.29,0.24,
    0.18,0.28,0.26,0.21,0.22,0.6,0.2,0.22,0.24,0.22,0.21,0.22,
    0.3,0.29,0.23,0.28,0.26,0.29,0.3,0.21,0.25,0.14,0.22,0.16,
    30,0.26,0.23,0.22,0.29,0.31,0.24,0.28,0.23,0.29,0.22,0.41,
    0.27,0.32,0.21,0.26,0.31,0.30,0.28,0.31,0.34
], dtype=np.float32)


def calc_q_scalar(intensity: float, fwhm: float) -> int:
    if fwhm == 30 or intensity <= 0 or fwhm <= 0:
        return 0
    return int(round(float(intensity) / float(fwhm)))


def experimental_dataframe() -> pd.DataFrame:
    exp = pd.DataFrame(EXPERIMENTAL_X, columns=FEATURES)
    exp["Intensity_exp"] = EXPERIMENTAL_INTENSITY.astype(float)
    exp["FWHM_exp"] = EXPERIMENTAL_FWHM.astype(float)
    agg = exp.groupby(FEATURES, as_index=False).agg(
        Intensity_exp=("Intensity_exp", "mean"),
        FWHM_exp=("FWHM_exp", "mean"),
    )
    agg["q_exp"] = [calc_q_scalar(i, f) for i, f in zip(agg.Intensity_exp, agg.FWHM_exp)]
    agg = agg.sort_values(FEATURES).reset_index(drop=True)
    agg["label_id"] = ["Q" + str(i + 1).zfill(3) for i in range(len(agg))]
    return agg


def train_landscape_rfs():
    exp = experimental_dataframe()
    x = exp[FEATURES].to_numpy()
    rf_i = RandomForestRegressor(n_estimators=RF_N_TREES, random_state=RF_RANDOM_STATE, n_jobs=-1).fit(
        x, exp["Intensity_exp"].to_numpy()
    )
    rf_f = RandomForestRegressor(n_estimators=RF_N_TREES, random_state=RF_RANDOM_STATE, n_jobs=-1).fit(
        x, exp["FWHM_exp"].to_numpy()
    )
    return rf_i, rf_f, exp


def postprocess_and_round_vec(intensity: np.ndarray, fwhm: np.ndarray):
    intensity = intensity.astype(np.float64, copy=True)
    fwhm = fwhm.astype(np.float64, copy=True)
    low_i = intensity < 3000.0
    intensity[low_i] = 0.0
    fwhm[low_i] = 30.0
    hi_f = (fwhm > 3.0) & (~low_i)
    fwhm[hi_f] = 30.0

    i_final = np.rint(intensity).astype(np.int64)
    f_final = np.empty_like(fwhm)
    m30 = fwhm == 30.0
    mlt1 = (fwhm < 1.0) & (~m30)
    other = (~mlt1) & (~m30)
    f_final[m30] = 30
    f_final[mlt1] = np.round(fwhm[mlt1], 2)
    f_final[other] = np.round(fwhm[other], 1)

    q = np.zeros_like(i_final)
    valid = (f_final != 30) & (i_final > 0)
    q[valid] = np.rint(i_final[valid] / f_final[valid]).astype(np.int64)
    return i_final, f_final, q


def apply_rules_batch(keys: np.ndarray, rf_i: np.ndarray, rf_f: np.ndarray, exp: pd.DataFrame):
    exp_params = exp[FEATURES].to_numpy(dtype=np.int16)
    exp_i = exp["Intensity_exp"].to_numpy(dtype=np.float32)
    exp_f = exp["FWHM_exp"].to_numpy(dtype=np.float32)
    diffs = np.abs(keys[:, None, :].astype(np.float32) - exp_params[None, :, :].astype(np.float32))

    r1 = np.all(diffs <= 1.0 + 1e-9, axis=2)
    r1_has = np.any(r1, axis=1)
    r1_counts = np.maximum(r1.sum(axis=1), 1)
    r1_i = (r1 @ exp_i) / r1_counts
    r1_f = (r1 @ exp_f) / r1_counts

    le2 = np.all(diffs <= 2.0 + 1e-9, axis=2)
    eq2 = np.isclose(diffs, 2.0)
    count2 = np.sum(eq2, axis=2)
    non2_max = np.max(np.where(~eq2, diffs, -np.inf), axis=2)
    r2 = le2 & (non2_max <= 1.0 + 1e-9) & ((count2 == 1) | (count2 == 2))
    r2_has = np.any(r2, axis=1)
    r2_counts = np.maximum(r2.sum(axis=1), 1)
    r2_i = (r2 @ exp_i) / r2_counts
    r2_f = (r2 @ exp_f) / r2_counts

    i_blend = rf_i.copy()
    f_blend = rf_f.copy()
    flags = np.zeros(keys.shape[0], dtype=np.int8)

    idx1 = np.where(r1_has)[0]
    i_blend[idx1] = 0.8 * r1_i[idx1] + 0.2 * rf_i[idx1]
    f_blend[idx1] = 0.8 * r1_f[idx1] + 0.2 * rf_f[idx1]
    flags[idx1] = 1

    idx2 = np.where(~r1_has & r2_has)[0]
    i_blend[idx2] = 0.5 * r2_i[idx2] + 0.5 * rf_i[idx2]
    f_blend[idx2] = 0.5 * r2_f[idx2] + 0.5 * rf_f[idx2]
    flags[idx2] = 2
    return i_blend, f_blend, flags


def grid_chunk_from_linear(start: int, end: int) -> np.ndarray:
    dims = [len(v) for v in GRID_VALUES]
    lin = np.arange(start, end, dtype=np.int64)
    out = np.empty((end - start, 5), dtype=np.int16)
    rem = lin
    for j in range(4, -1, -1):
        idx = rem % dims[j]
        rem = rem // dims[j]
        out[:, j] = GRID_VALUES[j][idx]
    return out


def stable_int(text: str, modulo: int = 2**31 - 1) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16) % modulo


def y_to_model_scale(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return np.log1p(np.maximum(y, 0.0)) if USE_LOG1P_TARGET else y.copy()


def y_from_model_scale(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return np.expm1(y) if USE_LOG1P_TARGET else y.copy()


def kth_top_threshold(y: np.ndarray, frac: float) -> float:
    k = max(1, int(np.ceil(frac * len(y))))
    return float(np.partition(y, len(y) - k)[len(y) - k])


def landscape_files(base: Path | None = None) -> list[Path]:
    base = base or LANDSCAPE_DIR
    files = sorted(base.glob("design_grid_part*.csv"))
    if files:
        return files
    return sorted(ORIGINAL_DIR.glob("design_grid_part*.csv"))


def load_design_grid(files: list[Path] | None = None) -> tuple[pd.DataFrame, pd.Series, int]:
    files = files or landscape_files()
    if not files:
        raise FileNotFoundError("No design_grid_part*.csv files found in reproduced or original locations.")

    optional = ["label_id", "rule_applied", "Intensity_final", "FWHM_final"]
    header = pd.read_csv(files[0], nrows=0).columns.tolist()
    usecols = FEATURES + [TARGET] + [c for c in optional if c in header]
    frames = []
    q_frames = []
    total = 0
    best_val = -np.inf
    best_row = None
    for path in files:
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=LOAD_CHUNK_SIZE, low_memory=False):
            for col in FEATURES + [TARGET]:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0)
            chunk[FEATURES] = chunk[FEATURES].astype(np.float32)
            chunk[TARGET] = chunk[TARGET].astype(np.float32)
            total += len(chunk)
            local_idx = chunk[TARGET].astype(float).idxmax()
            local_val = float(chunk.loc[local_idx, TARGET])
            if local_val > best_val:
                best_val = local_val
                best_row = chunk.loc[local_idx].copy()
            # Benchmark v3 appends the experimental Q rows before keep="last"
            # deduplication. This deliberately moves them to the end of the pool;
            # preserving that order is essential because seeded choices use pool_idx.
            if "label_id" in chunk.columns:
                q_mask = chunk["label_id"].astype(str).str.startswith("Q")
                if q_mask.any():
                    q_frames.append(chunk.loc[q_mask].copy())
            frames.append(chunk)
    pool = pd.concat(frames + q_frames, ignore_index=True)
    pool = pool.drop_duplicates(subset=FEATURES, keep="last").reset_index(drop=True)
    pool["_pool_idx"] = np.arange(len(pool), dtype=np.int64)
    return pool, best_row, total


class ConstantSurrogate:
    def fit(self, x: np.ndarray, y: np.ndarray):
        self.mean_ = float(np.mean(y)) if len(y) else 0.0
        s = float(np.std(y, ddof=1)) if len(y) > 1 else 1.0
        self.sd_ = max(s, 1.0)
        return self

    def predict(self, x: np.ndarray):
        return np.full(x.shape[0], self.mean_), np.full(x.shape[0], self.sd_)


class SklearnGPSurrogate:
    def __init__(self, random_state: int, n_restarts: int = 1):
        self.random_state = random_state
        self.n_restarts = n_restarts

    def fit(self, x: np.ndarray, y: np.ndarray):
        p = x.shape[1]
        kernel = (
            C(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(p) * 0.5, length_scale_bounds=(1e-2, 10.0), nu=2.5)
            + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e0))
        )
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-8,
            normalize_y=True,
            n_restarts_optimizer=self.n_restarts,
            random_state=self.random_state,
        )
        self.model.fit(x, y)
        return self

    def predict(self, x: np.ndarray):
        mu, sd = self.model.predict(x, return_std=True)
        return np.asarray(mu, dtype=float), np.maximum(np.asarray(sd, dtype=float), MIN_STD)


def predict_in_chunks(model: Any, x: np.ndarray, chunk_size: int = PREDICT_CHUNK_SIZE):
    mus, sds = [], []
    for start in range(0, x.shape[0], chunk_size):
        mu, sd = model.predict(x[start:start + chunk_size])
        mus.append(np.asarray(mu, dtype=float).reshape(-1))
        sds.append(np.asarray(sd, dtype=float).reshape(-1))
    return np.concatenate(mus), np.concatenate(sds)


def gaussian_pi(mu: np.ndarray, sd: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    sd = np.maximum(np.asarray(sd, dtype=float), MIN_STD)
    return norm.cdf((np.asarray(mu, dtype=float) - best - xi) / sd)


def greedy_diverse_batch(candidate_idx: np.ndarray, acq: np.ndarray, x_scaled: np.ndarray, k: int, rng: np.random.Generator):
    candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
    acq = np.asarray(acq, dtype=float)
    if len(candidate_idx) <= k:
        return candidate_idx
    acq = np.where(np.isfinite(acq), acq, -np.inf) + rng.normal(0.0, 1e-12, size=len(acq))
    acq_z = (acq - np.nanmean(acq)) / (np.nanstd(acq) + 1e-12)
    chosen_local: list[int] = []
    remaining = np.arange(len(candidate_idx))
    for _ in range(k):
        if len(chosen_local) == 0 or DIVERSITY_LAMBDA <= 0:
            best_local = remaining[np.argmax(acq_z[remaining])]
        else:
            x_chosen = x_scaled[candidate_idx[np.asarray(chosen_local)]]
            x_rem = x_scaled[candidate_idx[remaining]]
            dist = np.sqrt(((x_rem[:, None, :] - x_chosen[None, :, :]) ** 2).sum(axis=2))
            score = acq_z[remaining] + DIVERSITY_LAMBDA * dist.min(axis=1)
            best_local = remaining[np.argmax(score)]
        chosen_local.append(int(best_local))
        remaining = remaining[remaining != best_local]
    return candidate_idx[np.asarray(chosen_local, dtype=int)]


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    val = spearmanr(a, b).correlation
    return float(0.0 if np.isnan(val) else val)


def gaussian_crps(y: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    sd = np.maximum(np.asarray(sd, dtype=float), MIN_STD)
    z = (np.asarray(y, dtype=float) - np.asarray(mu, dtype=float)) / sd
    return sd * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))


def gaussian_nll(y: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    sd = np.maximum(np.asarray(sd, dtype=float), MIN_STD)
    return 0.5 * np.log(2.0 * np.pi * sd**2) + 0.5 * (
        (np.asarray(y, dtype=float) - np.asarray(mu, dtype=float)) / sd
    ) ** 2


def run_gp_matern52_pi_once(pool_df: pd.DataFrame, run: int) -> tuple[list[dict[str, Any]], list[int]]:
    x_raw = pool_df[FEATURES].to_numpy(dtype=np.float64)
    y_raw = pool_df[TARGET].to_numpy(dtype=np.float64)
    x_pool = MinMaxScaler().fit_transform(x_raw)
    oracle_best = float(np.max(y_raw))
    oracle_min = float(np.min(y_raw))
    top1 = kth_top_threshold(y_raw, 0.01)
    top01 = kth_top_threshold(y_raw, 0.001)

    seed = (RANDOM_STATE + 10_000 * run + stable_int(METHOD)) % (2**31 - 1)
    rng = np.random.default_rng(seed)
    n_pool = len(pool_df)
    tried = np.zeros(n_pool, dtype=bool)
    initial_idx = rng.choice(n_pool, size=min(INITIAL_EXPERIMENTS, BUDGET_TOTAL_EXPERIMENTS, n_pool), replace=False)
    selected = [int(x) for x in initial_idx]
    tried[selected] = True
    val_idx = rng.choice(n_pool, size=min(VALIDATION_SIZE, n_pool), replace=False)

    def metrics(iteration: int, last_batch: list[int], fit_s: float, acq_s: float, cum_s: float, status: str, model=None):
        y_sel = y_raw[np.asarray(selected, dtype=int)]
        best_pos = int(np.argmax(y_sel))
        best_pool_idx = int(selected[best_pos])
        best_q = float(y_raw[best_pool_idx])
        denom = max(oracle_best - oracle_min, 1e-12)
        row = {
            "method": METHOD,
            "family": FAMILY,
            "run": run,
            "iteration": iteration,
            "n_experiments": len(selected),
            "batch_size_added": len(last_batch),
            "best_q": best_q,
            "best_pool_idx": best_pool_idx,
            "simple_regret": float(oracle_best - best_q),
            "normalized_regret": float((oracle_best - best_q) / denom),
            "best_rank_percentile": float(100.0 * np.mean(y_raw <= best_q)),
            "hit_top1pct": float(best_q >= top1),
            "hit_top0_1pct": float(best_q >= top01),
            "success_fraction_selected": float(np.mean(y_sel > 0.0)),
            "mean_q_selected": float(np.mean(y_sel)),
            "last_batch_best_q": float(np.max(y_raw[np.asarray(last_batch, dtype=int)])) if last_batch else np.nan,
            "mean_pairwise_batch_distance": float(np.mean(pdist(x_pool[np.asarray(last_batch, dtype=int)]))) if len(last_batch) >= 2 else np.nan,
            "fit_seconds": fit_s,
            "acq_seconds": acq_s,
            "cum_seconds": cum_s,
            "status": status,
            "pool_oracle_best_q": oracle_best,
            "full_grid_best_q": oracle_best,
        }
        if model is None:
            row.update({
                "surrogate_rmse_raw": np.nan,
                "surrogate_mae_raw": np.nan,
                "surrogate_spearman_raw": np.nan,
                "surrogate_crps": np.nan,
                "surrogate_nll": np.nan,
                "surrogate_coverage95": np.nan,
                "surrogate_mean_sd": np.nan,
            })
        else:
            mu, sd = predict_in_chunks(model, x_pool[val_idx])
            y_val_model = y_to_model_scale(y_raw[val_idx])
            pred_raw = y_from_model_scale(mu)
            row.update({
                "surrogate_rmse_raw": float(np.sqrt(np.mean((pred_raw - y_raw[val_idx]) ** 2))),
                "surrogate_mae_raw": float(np.mean(np.abs(pred_raw - y_raw[val_idx]))),
                "surrogate_spearman_raw": safe_spearman(pred_raw, y_raw[val_idx]),
                "surrogate_crps": float(np.mean(gaussian_crps(y_val_model, mu, sd))),
                "surrogate_nll": float(np.mean(gaussian_nll(y_val_model, mu, sd))),
                "surrogate_coverage95": float(np.mean(
                    (y_val_model >= mu - 1.96 * sd) & (y_val_model <= mu + 1.96 * sd)
                )),
                "surrogate_mean_sd": float(np.mean(sd)),
            })
        return row

    rows = [metrics(0, list(initial_idx), 0.0, 0.0, 0.0, "initial")]
    cum_seconds = 0.0
    iteration = 0
    while len(selected) < min(BUDGET_TOTAL_EXPERIMENTS, n_pool):
        iteration += 1
        remaining = np.flatnonzero(~tried)
        batch_k = min(BATCH_SIZE, BUDGET_TOTAL_EXPERIMENTS - len(selected), len(remaining))
        x_train = x_pool[np.asarray(selected, dtype=int)]
        y_train = y_to_model_scale(y_raw[np.asarray(selected, dtype=int)])

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
        acq = gaussian_pi(mu, sd, float(np.max(y_train)), xi=0.01)
        next_idx = greedy_diverse_batch(acq_idx, acq, x_pool, batch_k, rng)
        acq_s = time.perf_counter() - t1

        next_idx = [int(x) for x in np.asarray(next_idx, dtype=int).reshape(-1) if not tried[int(x)]]
        if len(next_idx) < batch_k:
            fill = [int(x) for x in rng.choice(np.flatnonzero(~tried), size=batch_k - len(next_idx), replace=False)]
            next_idx.extend(fill)
        next_idx = next_idx[:batch_k]
        for idx in next_idx:
            tried[idx] = True
            selected.append(idx)
        cum_seconds += fit_s + acq_s
        rows.append(metrics(iteration, next_idx, fit_s, acq_s, cum_seconds, status, model))
        print(f"[{METHOD} run {run}] iter {iteration:02d} n={len(selected)} best_q={rows[-1]['best_q']:.0f}")
    return rows, selected


def summarize_runs(iteration_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows = []
    for (method, run), g in iteration_df.groupby(["method", "run"]):
        g = g.sort_values("iteration").reset_index(drop=True)
        final = g.iloc[-1]
        final_best = float(g["best_q"].max())
        first_final = g.loc[g["best_q"] >= final_best - 1e-12].iloc[0]
        oracle = float(final["pool_oracle_best_q"])
        oracle_min = oracle - float(final["simple_regret"]) - float(final["best_q"])
        denom = max(oracle - oracle_min, 1e-12)
        hit90 = g.loc[g["best_q"] >= 0.90 * oracle]
        norm_best = (g["best_q"].to_numpy(float) - oracle_min) / denom
        x = g["iteration"].to_numpy(float)
        auc = float(np.trapezoid(norm_best, x) / (x[-1] - x[0]))
        run_rows.append({
            "method": method,
            "family": str(final["family"]),
            "run": int(run),
            "final_best_q": final_best,
            "final_simple_regret": oracle - final_best,
            "final_normalized_regret": (oracle - final_best) / denom,
            "final_best_rank_percentile": float(final["best_rank_percentile"]),
            "final_hit_top1pct": float(final["hit_top1pct"]),
            "final_hit_top0_1pct": float(final["hit_top0_1pct"]),
            "final_success_fraction": float(final["success_fraction_selected"]),
            "first_final_best_iteration": float(first_final["iteration"]),
            "first_final_best_n_experiments": float(first_final["n_experiments"]),
            "experiments_to_90pct_oracle": float(hit90.iloc[0]["n_experiments"]) if len(hit90) else np.nan,
            "iteration_to_90pct_oracle": float(hit90.iloc[0]["iteration"]) if len(hit90) else np.nan,
            "auc_normalized_best": auc,
            "total_fit_seconds": float(g["fit_seconds"].sum()),
            "total_acq_seconds": float(g["acq_seconds"].sum()),
            "total_seconds": float(g["fit_seconds"].sum() + g["acq_seconds"].sum()),
            **{f"last_{c}": float(final[c]) for c in [
                "surrogate_rmse_raw", "surrogate_mae_raw", "surrogate_spearman_raw",
                "surrogate_crps", "surrogate_nll", "surrogate_coverage95",
            ]},
            "pool_oracle_best_q": oracle,
            "full_grid_best_q": float(final["full_grid_best_q"]),
        })
    run_summary = pd.DataFrame(run_rows)
    final_summary = run_summary.groupby(["method", "family"], as_index=False).agg(
        n_runs=("run", "nunique"),
        final_best_q_mean=("final_best_q", "mean"),
        final_best_q_std=("final_best_q", "std"),
        final_simple_regret_mean=("final_simple_regret", "mean"),
        final_normalized_regret_mean=("final_normalized_regret", "mean"),
        final_best_rank_percentile_mean=("final_best_rank_percentile", "mean"),
        hit_top1pct_rate=("final_hit_top1pct", "mean"),
        hit_top0_1pct_rate=("final_hit_top0_1pct", "mean"),
        success_fraction_mean=("final_success_fraction", "mean"),
        first_final_best_iteration_mean=("first_final_best_iteration", "mean"),
        first_final_best_n_experiments_mean=("first_final_best_n_experiments", "mean"),
        experiments_to_90pct_oracle_mean=("experiments_to_90pct_oracle", "mean"),
        iteration_to_90pct_oracle_mean=("iteration_to_90pct_oracle", "mean"),
        auc_normalized_best_mean=("auc_normalized_best", "mean"),
        total_fit_seconds_mean=("total_fit_seconds", "mean"),
        total_acq_seconds_mean=("total_acq_seconds", "mean"),
        total_seconds_mean=("total_seconds", "mean"),
        last_surrogate_rmse_raw_mean=("last_surrogate_rmse_raw", "mean"),
        last_surrogate_mae_raw_mean=("last_surrogate_mae_raw", "mean"),
        last_surrogate_spearman_raw_mean=("last_surrogate_spearman_raw", "mean"),
        last_surrogate_crps_mean=("last_surrogate_crps", "mean"),
        last_surrogate_nll_mean=("last_surrogate_nll", "mean"),
        last_surrogate_coverage95_mean=("last_surrogate_coverage95", "mean"),
    )
    return run_summary, final_summary
