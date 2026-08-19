import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                              roc_auc_score, average_precision_score)
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "Error_data.csv"
DEFAULT_CONTINUOUS = PROJECT_ROOT / "figures" / "sam" / "2026-08-17" / "followup" / "continuous_features.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "sam" / "2026-08-17" / "followup"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--continuous-features", type=Path, default=DEFAULT_CONTINUOUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    err = pd.read_csv(args.input)
    cont = pd.read_csv(args.continuous_features)

    df = err.merge(cont, on="experiment_id", how="inner", validate="one_to_one")
    assert len(df) == 95, len(df)
    df["catastrophic"] = (df["y_log_abs_error"] > 5).astype(int)
    cat_mask = df["catastrophic"].astype(bool)

    # ---------------------------------------------------------------------
    # 0. Sanity-check the re-simulation against the original Error_data.csv
    # ---------------------------------------------------------------------
    print("=" * 70)
    print("0. VALIDATION: re-simulated quantities vs original Error_data.csv")
    print("=" * 70)
    dmin_diff = (df["d_min_check"] - df["nearest_chebyshev_distance"]).abs()
    print(f"d_min match: max abs diff = {dmin_diff.max():.4f} "
          f"({(dmin_diff < 1e-9).sum()}/95 exact matches)")
    qpred_resim = np.where(
        (df["I_pre_mean"] < 3000), 0,
        np.where(df["F_pre_mean"] > 3, 0, np.rint(df["I_pre_mean"] / df["F_pre_mean"].round(1).replace(0, np.nan)))
    )
    print("(q_pred re-derivation is approximate/seed-mean-based; not expected to match exactly, "
          "d_min match above is the real correctness check.)")

    # ---------------------------------------------------------------------
    # 1. CONTINUOUS BOUNDARY MARGINS  (10.1)
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("1. Continuous boundary margins M_I, M_F")
    print("=" * 70)
    eps = 1e-6
    df["M_I"] = (df["I_pre_mean"] - 3000) / (df["I_pre_std"] + eps)
    df["M_F"] = (df["F_pre_mean"] - 3) / (df["F_pre_std"] + eps)
    # clip extreme values from near-zero std (deterministic across all 5 seeds) for readability
    df["M_I_clip"] = df["M_I"].clip(-50, 50)
    df["M_F_clip"] = df["M_F"].clip(-50, 50)
    
    print(df[["experiment_id", "I_pre_mean", "I_pre_std", "M_I",
              "F_pre_mean", "F_pre_std", "M_F", "y_log_abs_error", "catastrophic"]]
          .sort_values("catastrophic", ascending=False).head(15).to_string(index=False))
    
    for c in ["M_I", "M_F"]:
        pr, _ = pearsonr(df[c], df["y_log_abs_error"])
        sr, _ = spearmanr(df[c], df["y_log_abs_error"])
        print(f"{c} vs e_log (all 95):        Pearson={pr:+.3f}  Spearman={sr:+.3f}")
        pr2, _ = pearsonr(df.loc[~cat_mask, c], df.loc[~cat_mask, "y_log_abs_error"])
        sr2, _ = spearmanr(df.loc[~cat_mask, c], df.loc[~cat_mask, "y_log_abs_error"])
        print(f"{c} vs e_log (non-cat, n=85): Pearson={pr2:+.3f}  Spearman={sr2:+.3f}")
    
    # figure: M_I vs M_F, colored by e_log, catastrophic labeled
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sc = ax.scatter(df["M_I_clip"], df["M_F_clip"], c=df["y_log_abs_error"], cmap="viridis",
                     s=42, edgecolor="white", linewidth=0.4, zorder=2)
    ax.scatter(df.loc[cat_mask, "M_I_clip"], df.loc[cat_mask, "M_F_clip"], s=150,
               facecolor="none", edgecolor="#d62728", linewidth=1.8, zorder=3)
    for _, r in df[cat_mask].iterrows():
        ax.annotate(r["experiment_id"], (r["M_I_clip"], r["M_F_clip"]), fontsize=8, color="#8b0000",
                    xytext=(5, 5), textcoords="offset points")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel(r"$M_I = (\hat{I}_{pre}-3000)/\sigma_{I,pre}$  (clipped at $\pm$50)")
    ax.set_ylabel(r"$M_F = (\hat{F}_{pre}-3)/\sigma_{F,pre}$  (clipped at $\pm$50)")
    ax.set_title("Continuous boundary margins — red = catastrophic ($e_{\\log}>5$)")
    cb = fig.colorbar(sc, ax=ax); cb.set_label(r"$e_{\log}$")
    fig.tight_layout()
    fig.savefig(outdir / "continuous_margins_MI_MF.png", dpi=170)
    plt.close(fig)
    
    # ---------------------------------------------------------------------
    # 2. CONTINUOUS LOCAL SUPPORT / DENSITY  (10.2)
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. Continuous local-support / density features")
    print("=" * 70)
    density_feats = ["d2", "d3", "mean_d_k3", "mean_d_k5", "kde_cheb"]
    rows = []
    for c in density_feats + ["nearest_chebyshev_distance"]:
        pr, _ = pearsonr(df[c], df["y_log_abs_error"])
        sr, _ = spearmanr(df[c], df["y_log_abs_error"])
        rows.append((c, pr, sr))
    dens_corr = pd.DataFrame(rows, columns=["feature", "pearson_r", "spearman_rho"])
    print(dens_corr.round(3).to_string(index=False))
    dens_corr.to_csv(outdir / "density_feature_correlations.csv", index=False)
    
    # figure: kde_cheb (smooth local density) vs e_log
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(df["kde_cheb"], df["y_log_abs_error"], c=df["mean_d_k3"], cmap="viridis",
                     s=42, edgecolor="white", linewidth=0.4)
    ax.scatter(df.loc[cat_mask, "kde_cheb"], df.loc[cat_mask, "y_log_abs_error"], s=150,
               facecolor="none", edgecolor="#d62728", linewidth=1.8)
    for _, r in df[cat_mask].iterrows():
        ax.annotate(r["experiment_id"], (r["kde_cheb"], r["y_log_abs_error"]), fontsize=8, color="#8b0000",
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Gaussian KDE local density (Chebyshev, median-bandwidth)")
    ax.set_ylabel(r"$e_{\log}$")
    ax.set_title("Local experimental density vs reconstruction error")
    cb = fig.colorbar(sc, ax=ax); cb.set_label(r"mean dist. to 3 nearest ($mean\_d_{k3}$)")
    fig.tight_layout()
    fig.savefig(outdir / "density_vs_error.png", dpi=170)
    plt.close(fig)
    
    # ---------------------------------------------------------------------
    # 3. DOES REPLACING BINARY FEATURES WITH CONTINUOUS ONES IMPROVE THE
    #    LOOCV CLASSIFIER FROM LAST ROUND?
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. LOOCV classifier comparison: binary vs continuous features")
    print("=" * 70)
    df["boundary_active"] = ((df["intensity_3000_cross_rate"] > 0) | (df["fwhm_3_cross_rate"] > 0)).astype(int)
    df["min_abs_margin"] = np.minimum(df["M_I"].abs(), df["M_F"].abs()).clip(upper=50)
    
    def loocv_logit(X, y):
        loo = LeaveOneOut()
        probs = np.zeros(len(y)); preds = np.zeros(len(y))
        for tr, te in loo.split(X):
            sc_ = StandardScaler().fit(X[tr])
            clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000)
            clf.fit(sc_.transform(X[tr]), y[tr])
            probs[te] = clf.predict_proba(sc_.transform(X[te]))[:, 1]
            preds[te] = clf.predict(sc_.transform(X[te]))
        return probs, preds
    
    y = df["catastrophic"].values
    variants = {
        "A: binary [d_min, boundary_active]  (previous round)":
            df[["nearest_chebyshev_distance", "boundary_active"]].values,
        "B: continuous [mean_d_k3, min_abs_margin]":
            df[["mean_d_k3", "min_abs_margin"]].values,
        "C: continuous [d_min, min_abs_margin]":
            df[["nearest_chebyshev_distance", "min_abs_margin"]].values,
        "D: continuous [kde_cheb, min_abs_margin]":
            df[["kde_cheb", "min_abs_margin"]].values,
        "E: continuous 3-feat [mean_d_k3, M_I, M_F]":
            df[["mean_d_k3", "M_I", "M_F"]].clip(-50, 50).values,
    }
    print(f"{'variant':50s} {'recall':>7s} {'prec':>7s} {'F1':>7s} {'ROC-AUC':>8s} {'PR-AUC':>7s}  FP")
    for name, X in variants.items():
        probs, preds = loocv_logit(X, y)
        cm = confusion_matrix(y, preds)
        p, r, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
        auc = roc_auc_score(y, probs)
        ap = average_precision_score(y, probs)
        n_fp = int(cm[0, 1])
        print(f"{name:50s} {r:7.3f} {p:7.3f} {f1:7.3f} {auc:8.3f} {ap:7.3f}  {n_fp}")
    
    print("\nSaved figures/tables to:", outdir)


if __name__ == "__main__":
    main()
