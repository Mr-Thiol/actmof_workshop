import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                              roc_auc_score, average_precision_score, r2_score,
                              mean_absolute_error)
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "Error_data.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "followup"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df["row_idx"] = np.arange(1, len(df) + 1)

    FEATS = {
        "rf_intensity_variance": "V_I",
        "rf_fwhm_variance": "V_F",
        "nearest_chebyshev_distance": "d_min",
        "rule1_neighbor_count": "N_R1",
        "rule2_neighbor_count": "N_R2",
        "intensity_3000_cross_rate": "B_I",
        "fwhm_3_cross_rate": "B_F",
    }
    feat_cols = list(FEATS.keys())
    df["catastrophic"] = (df["y_log_abs_error"] > 5).astype(int)
    cat_mask = df["catastrophic"].astype(bool)
    CAT_IDS = df.loc[cat_mask, "experiment_id"].tolist()
    print("Catastrophic set (n=%d):" % cat_mask.sum(), CAT_IDS)
    
    # ======================================================================
    # A. PAIRWISE (d1, d2, e) ANALYSIS — priority pairs, explicit labeling
    # ======================================================================
    PRIORITY_PAIRS = [
        ("nearest_chebyshev_distance", "fwhm_3_cross_rate", r"$d_{min}$", r"$B_F$"),
        ("nearest_chebyshev_distance", "intensity_3000_cross_rate", r"$d_{min}$", r"$B_I$"),
        ("intensity_3000_cross_rate", "fwhm_3_cross_rate", r"$B_I$", r"$B_F$"),
        ("nearest_chebyshev_distance", "rf_fwhm_variance", r"$d_{min}$", r"$V_F$"),
        ("nearest_chebyshev_distance", "rf_intensity_variance", r"$d_{min}$", r"$V_I$"),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    axes = axes.ravel()
    for ax, (xc, yc, xl, yl) in zip(axes, PRIORITY_PAIRS):
        x, y = df[xc], df[yc]
        # small jitter for discrete features so overlapping points are visible
        jx = np.random.RandomState(0).uniform(-0.06, 0.06, size=len(df)) if df[xc].nunique() < 15 else 0
        jy = np.random.RandomState(1).uniform(-0.02, 0.02, size=len(df)) if df[yc].nunique() < 15 else 0
        sc = ax.scatter(x + jx, y + jy, c=df["y_log_abs_error"], cmap="viridis", s=38,
                         edgecolor="white", linewidth=0.4, zorder=2)
        if yc == "rf_fwhm_variance" or yc == "rf_intensity_variance":
            ax.set_yscale("symlog")
        ax.scatter(x[cat_mask] + (jx[cat_mask] if hasattr(jx, "__len__") else 0),
                   y[cat_mask] + (jy[cat_mask] if hasattr(jy, "__len__") else 0),
                   s=150, facecolor="none", edgecolor="#d62728", linewidth=1.8, zorder=3)
        for _, r in df[cat_mask].iterrows():
            xv = r[xc] + (np.random.RandomState(int(r["row_idx"])).uniform(-0.06, 0.06) if df[xc].nunique() < 15 else 0)
            yv = r[yc]
            ax.annotate(r["experiment_id"], (xv, yv), fontsize=7, color="#8b0000",
                        xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"{xl} vs {yl}", fontsize=10)
    cb = fig.colorbar(sc, ax=axes, shrink=0.7, pad=0.01)
    cb.set_label(r"$e_{\log}$")
    axes[-1].axis("off")
    fig.suptitle("Priority pairwise-feature scatter — red circle + label = catastrophic ($e_{\\log}>5$)", fontsize=12)
    fig.savefig(outdir / "priority_pairs_scatter_labeled.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    
    # ---- binned conditional failure-rate heatmaps for all 5 priority pairs ----
    def make_bins(series, kind):
        if kind == "dmin":
            edges = [0.5, 2.5, 4.5, 12]
            labels = ["1-2", "3-4", ">=5"]
        elif kind == "boundary":
            edges = [-0.01, 0.001, 0.5, 1.01]
            labels = ["0", "(0,0.5]", "(0.5,1]"]
        elif kind == "variance":
            # tertiles (data-driven, since V_I/V_F are continuous & heavily skewed)
            q = series.quantile([0, 1/3, 2/3, 1]).values
            q[0] -= 1e-6
            edges = list(q)
            labels = ["low(T1)", "mid(T2)", "high(T3)"]
        return edges, labels
    
    def binned_heatmap(xcol, ycol, xkind, ykind, xlabel, ylabel, fname):
        xedges, xlabels = make_bins(df[xcol], xkind)
        yedges, ylabels = make_bins(df[ycol], ykind)
        dfx = df.copy()
        dfx["xb"] = pd.cut(dfx[xcol], bins=xedges, include_lowest=True)
        dfx["yb"] = pd.cut(dfx[ycol], bins=yedges, include_lowest=True)
        x_cats = list(dfx["xb"].cat.categories)
        y_cats = list(dfx["yb"].cat.categories)
        rate = dfx.groupby(["yb", "xb"], observed=False)["catastrophic"].mean().reindex(
            pd.MultiIndex.from_product([y_cats, x_cats])).unstack()
        n = dfx.groupby(["yb", "xb"], observed=False)["catastrophic"].size().reindex(
            pd.MultiIndex.from_product([y_cats, x_cats])).unstack()
        rate = rate.reindex(index=y_cats, columns=x_cats)
        n = n.reindex(index=y_cats, columns=x_cats)
    
        fig, ax = plt.subplots(figsize=(6.2, 5))
        im = ax.imshow(rate.values, cmap="viridis", vmin=0, vmax=1, origin="lower", aspect="auto")
        for i in range(rate.shape[0]):
            for j in range(rate.shape[1]):
                v = rate.values[i, j]
                cnt = n.values[i, j]
                txt = "-" if np.isnan(v) else f"{v:.2f}\n(n={int(cnt)})"
                color = "black" if (not np.isnan(v) and v > 0.55) else "white"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
        ax.set_xticks(range(rate.shape[1])); ax.set_xticklabels(xlabels, rotation=20, ha="right", fontsize=8)
        ax.set_yticks(range(rate.shape[0])); ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(f"P(catastrophic | {xlabel}, {ylabel})")
        cb = fig.colorbar(im, ax=ax, shrink=0.85); cb.set_label("catastrophic rate")
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=170)
        plt.close(fig)
        return rate, n
    
    print("\n" + "=" * 70)
    print("A. BINNED CONDITIONAL FAILURE-RATE TABLES (5 priority pairs)")
    print("=" * 70)
    pair_specs = [
        ("nearest_chebyshev_distance", "fwhm_3_cross_rate", "dmin", "boundary", "d_min", "B_F", "heatmap2_dmin_BF.png"),
        ("nearest_chebyshev_distance", "intensity_3000_cross_rate", "dmin", "boundary", "d_min", "B_I", "heatmap2_dmin_BI.png"),
        ("intensity_3000_cross_rate", "fwhm_3_cross_rate", "boundary", "boundary", "B_I", "B_F", "heatmap2_BI_BF.png"),
        ("nearest_chebyshev_distance", "rf_fwhm_variance", "dmin", "variance", "d_min", "V_F", "heatmap2_dmin_VF.png"),
        ("nearest_chebyshev_distance", "rf_intensity_variance", "dmin", "variance", "d_min", "V_I", "heatmap2_dmin_VI.png"),
    ]
    for xcol, ycol, xk, yk, xl, yl, fname in pair_specs:
        rate, n = binned_heatmap(xcol, ycol, xk, yk, xl, yl, fname)
        print(f"\n--- {xl} x {yl} -> {fname} ---")
        print(rate.round(2).to_string())
    
    # ======================================================================
    # C. SMALL CATASTROPHIC-ERROR CLASSIFIER, STRICT LEAVE-ONE-OUT
    # ======================================================================
    print("\n" + "=" * 70)
    print("C1. Fixed rule baseline (no fitting): d_min>=4 AND N_R1=N_R2=0 AND (B_I>0 or B_F>0)")
    print("=" * 70)
    rule_pred = ((df["nearest_chebyshev_distance"] >= 4) &
                 (df["rule1_neighbor_count"] == 0) &
                 (df["rule2_neighbor_count"] == 0) &
                 ((df["intensity_3000_cross_rate"] > 0) | (df["fwhm_3_cross_rate"] > 0))).astype(int)
    cm = confusion_matrix(df["catastrophic"], rule_pred)
    p, r, f1, _ = precision_recall_fscore_support(df["catastrophic"], rule_pred, average="binary", zero_division=0)
    print("Confusion matrix [ [TN FP] [FN TP] ]:\n", cm)
    print(f"precision={p:.3f}  recall={r:.3f}  F1={f1:.3f}")
    false_pos = df.loc[(rule_pred == 1) & (df["catastrophic"] == 0), "experiment_id"].tolist()
    false_neg = df.loc[(rule_pred == 0) & (df["catastrophic"] == 1), "experiment_id"].tolist()
    print("False positives:", false_pos)
    print("False negatives:", false_neg)
    
    print("\n" + "=" * 70)
    print("C2. Minimal 2-feature logistic regression, strict Leave-One-Out CV")
    print("    features = [d_min, boundary_active=1{B_I>0 or B_F>0}]")
    print("=" * 70)
    df["boundary_active"] = ((df["intensity_3000_cross_rate"] > 0) | (df["fwhm_3_cross_rate"] > 0)).astype(int)
    X = df[["nearest_chebyshev_distance", "boundary_active"]].values
    y = df["catastrophic"].values
    
    loo = LeaveOneOut()
    probs = np.zeros(len(y))
    preds = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        scaler = StandardScaler().fit(X[train_idx])
        Xtr = scaler.transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000)
        clf.fit(Xtr, y[train_idx])
        probs[test_idx] = clf.predict_proba(Xte)[:, 1]
        preds[test_idx] = clf.predict(Xte)
    
    cm2 = confusion_matrix(y, preds)
    p2, r2_, f12, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
    auc = roc_auc_score(y, probs)
    ap = average_precision_score(y, probs)
    print("LOOCV confusion matrix [ [TN FP] [FN TP] ]:\n", cm2)
    print(f"LOOCV precision={p2:.3f}  recall={r2_:.3f}  F1={f12:.3f}  ROC-AUC={auc:.3f}  PR-AUC={ap:.3f}")
    fp2 = df.loc[(preds == 1) & (y == 0), "experiment_id"].tolist()
    fn2 = df.loc[(preds == 0) & (y == 1), "experiment_id"].tolist()
    print("LOOCV false positives:", fp2)
    print("LOOCV false negatives:", fn2)
    
    # fit on full data once just to report the learned coefficients (for interpretability)
    scaler_full = StandardScaler().fit(X)
    clf_full = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000).fit(scaler_full.transform(X), y)
    print("Full-data-fit coefficients (standardized):", dict(zip(["d_min", "boundary_active"], clf_full.coef_[0].round(3))),
          "intercept:", round(clf_full.intercept_[0], 3))
    
    # ======================================================================
    # D. MODEL ORDINARY ERROR SEPARATELY (non-catastrophic subset, n=85)
    # ======================================================================
    print("\n" + "=" * 70)
    print("D1. Feature correlations vs e_log, restricted to NON-catastrophic points")
    print("=" * 70)
    non_cat = df.loc[~cat_mask]
    rows = []
    for c in feat_cols:
        pr, _ = pearsonr(non_cat[c], non_cat["y_log_abs_error"])
        sr, _ = spearmanr(non_cat[c], non_cat["y_log_abs_error"])
        rows.append((FEATS[c], pr, sr))
    corr_df = pd.DataFrame(rows, columns=["feature", "pearson_r_noncat", "spearman_rho_noncat"])
    print(corr_df.round(3).to_string(index=False))
    corr_df.to_csv(outdir / "correlations_noncatastrophic.csv", index=False)
    
    print("\n" + "=" * 70)
    print("D2. LOOCV linear regression on non-catastrophic subset only (n=%d)" % len(non_cat))
    print("=" * 70)
    Xnc = non_cat[feat_cols].values
    ync = non_cat["y_log_abs_error"].values
    loo2 = LeaveOneOut()
    pred_nc = np.zeros(len(ync))
    for tr, te in loo2.split(Xnc):
        sc_ = StandardScaler().fit(Xnc[tr])
        lr = LinearRegression().fit(sc_.transform(Xnc[tr]), ync[tr])
        pred_nc[te] = lr.predict(sc_.transform(Xnc[te]))
    r2_nc = r2_score(ync, pred_nc)
    mae_nc = mean_absolute_error(ync, pred_nc)
    print(f"7-feature linear regression, LOOCV on non-catastrophic-only: R2={r2_nc:.3f}  MAE={mae_nc:.3f}")
    print(f"(compare to non-catastrophic mean e_log={ync.mean():.3f}, std={ync.std():.3f})")
    
    # ======================================================================
    # E. FULL 7-FEATURE REGRESSION ON ALL 95 (baseline for comparison)
    # ======================================================================
    print("\n" + "=" * 70)
    print("E1. 7-feature linear regression, LOOCV on ALL 95 points (single-stage baseline)")
    print("=" * 70)
    Xall = df[feat_cols].values
    yall = df["y_log_abs_error"].values
    loo3 = LeaveOneOut()
    pred_all = np.zeros(len(yall))
    for tr, te in loo3.split(Xall):
        sc_ = StandardScaler().fit(Xall[tr])
        lr = LinearRegression().fit(sc_.transform(Xall[tr]), yall[tr])
        pred_all[te] = lr.predict(sc_.transform(Xall[te]))
    r2_all = r2_score(yall, pred_all)
    mae_all = mean_absolute_error(yall, pred_all)
    print(f"7-feature linear regression, LOOCV on ALL 95: R2={r2_all:.3f}  MAE={mae_all:.3f}")
    print(f"(compare to overall mean e_log={yall.mean():.3f}, std={yall.std():.3f})")
    
    print("\nSUMMARY TABLE")
    print("-" * 70)
    print(f"{'model':45s} {'R2':>8s} {'MAE':>8s}")
    print(f"{'single-stage 7-feat regression (all 95)':45s} {r2_all:8.3f} {mae_all:8.3f}")
    print(f"{'stage-B 7-feat regression (86 non-cat only)':45s} {r2_nc:8.3f} {mae_nc:8.3f}")
    print(f"{'stage-A classifier (2-feat, LOOCV)':45s} {'precision='+format(p2,'.2f'):>8s} {'recall='+format(r2_,'.2f'):>8s}")
    
    print("\nSaved figures/tables to:", outdir)


if __name__ == "__main__":
    main()
