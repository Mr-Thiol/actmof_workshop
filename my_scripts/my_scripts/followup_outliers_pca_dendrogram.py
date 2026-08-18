import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

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
    df["row_idx"] = np.arange(1, len(df) + 1)  # 1-indexed row-in-file (matches CSV line - header)

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
    short = [FEATS[c] for c in feat_cols]
    
    df["catastrophic"] = df["y_log_abs_error"] > 5
    
    # ----------------------------------------------------------------------
    # 1. OUTLIERS
    # ----------------------------------------------------------------------
    print("=" * 70)
    print("1a. CATASTROPHIC ERROR OUTLIERS (y_log_abs_error > 5)")
    print("=" * 70)
    cat = df[df["catastrophic"]].sort_values("y_log_abs_error", ascending=False)
    print(cat[["row_idx", "experiment_id", "q_true", "q_pred_mean", "y_log_abs_error",
               "nearest_chebyshev_distance", "rule1_neighbor_count", "rule2_neighbor_count",
               "intensity_3000_cross_rate", "fwhm_3_cross_rate"]].to_string(index=False))
    
    print()
    print("=" * 70)
    print("1b. UNIVARIATE IQR OUTLIERS PER FEATURE (Tukey 1.5xIQR fence)")
    print("=" * 70)
    iqr_flags = pd.DataFrame(index=df.index)
    for c in feat_cols + ["y_log_abs_error"]:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_flags[c] = (df[c] < lo) | (df[c] > hi)
        n = iqr_flags[c].sum()
        if n:
            ids = df.loc[iqr_flags[c], "experiment_id"].tolist()
            print(f"{c:28s}: {n:2d} outliers -> {ids}")
    
    df["n_feature_outliers"] = iqr_flags[feat_cols].sum(axis=1)
    print()
    print("Rows flagged as outlier on >=3 of the 7 features:")
    multi = df[df["n_feature_outliers"] >= 3].sort_values("n_feature_outliers", ascending=False)
    print(multi[["row_idx", "experiment_id", "n_feature_outliers", "y_log_abs_error"]].to_string(index=False))
    
    print()
    print("=" * 70)
    print("1c. MULTIVARIATE OUTLIERS IN 7-D FEATURE SPACE (Mahalanobis distance)")
    print("=" * 70)
    # V_I, V_F are heavily right-skewed (span ~1e6-1e8); log1p them before standardizing
    # so a handful of huge-variance points don't dominate the covariance estimate.
    X_for_maha = df[feat_cols].copy()
    X_for_maha["rf_intensity_variance"] = np.log1p(X_for_maha["rf_intensity_variance"])
    X_for_maha["rf_fwhm_variance"] = np.log1p(X_for_maha["rf_fwhm_variance"])
    Xs_m = StandardScaler().fit_transform(X_for_maha.values)
    pca_m = PCA().fit(Xs_m)
    scores_m = pca_m.transform(Xs_m)
    # standard (non-robust) squared Mahalanobis distance = sum of whitened PCA scores^2
    md2 = (scores_m ** 2 / pca_m.explained_variance_).sum(axis=1)
    thresh = chi2.ppf(0.975, df=len(feat_cols))
    df["mahalanobis_d2"] = md2
    mv_out = df[md2 > thresh].sort_values("mahalanobis_d2", ascending=False)
    print(f"(log1p applied to V_I, V_F before standardizing; Chi2(df={len(feat_cols)}) 97.5% threshold = {thresh:.2f})")
    print(mv_out[["row_idx", "experiment_id", "mahalanobis_d2", "y_log_abs_error", "catastrophic"]].to_string(index=False))
    
    df.to_csv(outdir / "Error_data_with_outlier_flags.csv", index=False)
    
    # ----------------------------------------------------------------------
    # 2. DIMENSION REDUCTION: PCA + feature clustering
    #    (reuse the log1p(V_I), log1p(V_F)-standardized matrix from step 1c —
    #     same rationale: two heavily right-skewed variance features shouldn't
    #     dominate the covariance structure)
    # ----------------------------------------------------------------------
    pca = pca_m
    scores = scores_m
    evr = pca.explained_variance_ratio_
    print()
    print("=" * 70)
    print("2a. PCA explained variance ratio")
    print("=" * 70)
    for i, v in enumerate(evr, 1):
        print(f"PC{i}: {v:.3f}  (cumulative {evr[:i].sum():.3f})")
    
    loadings = pd.DataFrame(pca.components_[:3].T, index=short, columns=["PC1", "PC2", "PC3"])
    print()
    print("Loadings (top 3 PCs):")
    print(loadings.round(3).to_string())
    
    # PCA scatter, PC1 vs PC2, colored by log-error
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, color_col, cbar_label, title in [
        (axes[0], "y_log_abs_error", r"$e_{\log}$", "PCA of 7 reliability features — colored by $e_{\\log}$"),
        (axes[1], "q_true", r"$\log_{10}(1+q_{true})$", "PCA of 7 reliability features — colored by $q_{true}$"),
    ]:
        c = np.log10(1 + df["q_true"]) if color_col == "q_true" else df[color_col]
        sc = ax.scatter(scores[:, 0], scores[:, 1], c=c, cmap="viridis", s=42,
                         edgecolor="white", linewidth=0.4, zorder=2)
        cat_mask = df["catastrophic"].values
        ax.scatter(scores[cat_mask, 0], scores[cat_mask, 1], s=140, facecolor="none",
                   edgecolor="#d62728", linewidth=1.8, zorder=3, label="catastrophic ($e_{\\log}>5$)")
        ax.set_xlabel(f"PC1 ({evr[0]*100:.0f}% var)")
        ax.set_ylabel(f"PC2 ({evr[1]*100:.0f}% var)")
        ax.set_title(title, fontsize=10)
        cb = fig.colorbar(sc, ax=ax, shrink=0.85)
        cb.set_label(cbar_label)
        ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "pca_2d.png", dpi=170)
    plt.close(fig)
    
    # Feature clustering dendrogram (1 - |spearman rho| distance)
    corr = df[feat_cols].corr(method="spearman")
    dist = 1 - corr.abs()
    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method="average")
    fig, ax = plt.subplots(figsize=(7, 4))
    dendrogram(Z, labels=short, ax=ax, color_threshold=0.5)
    ax.set_ylabel("1 - |Spearman rho|  (distance)")
    ax.set_title("Feature redundancy clustering (7 features)")
    fig.tight_layout()
    fig.savefig(outdir / "feature_dendrogram.png", dpi=170)
    plt.close(fig)
    
    # ----------------------------------------------------------------------
    # 3. BINNED HEATMAP: catastrophic-rate over (d_min, B_F) and (d_min, B_I)
    # ----------------------------------------------------------------------
    def binned_catastrophic_heatmap(xcol, ycol, xbins, ybins, xlabel, ylabel, fname, xlabels=None, ylabels=None):
        dfx = df.copy()
        dfx["xb"] = pd.cut(dfx[xcol], bins=xbins, include_lowest=True)
        dfx["yb"] = pd.cut(dfx[ycol], bins=ybins, include_lowest=True)
        x_cats = list(dfx["xb"].cat.categories)
        y_cats = list(dfx["yb"].cat.categories)
        rate = dfx.groupby(["yb", "xb"], observed=False)["catastrophic"].mean().unstack()
        n = dfx.groupby(["yb", "xb"], observed=False)["catastrophic"].size().unstack()
        rate = rate.reindex(index=y_cats, columns=x_cats)
        n = n.reindex(index=y_cats, columns=x_cats)
    
        fig, ax = plt.subplots(figsize=(6.2, 5))
        im = ax.imshow(rate.values, cmap="viridis", vmin=0, vmax=1, origin="lower", aspect="auto")
        for i in range(rate.shape[0]):
            for j in range(rate.shape[1]):
                v = rate.values[i, j]
                cnt = n.values[i, j]
                if np.isnan(v):
                    txt = "-"
                else:
                    txt = f"{v:.2f}\n(n={int(cnt)})"
                color = "black" if (not np.isnan(v) and v > 0.55) else "white"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
        ax.set_xticks(range(rate.shape[1]))
        ax.set_xticklabels(xlabels or [str(iv) for iv in rate.columns], rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(rate.shape[0]))
        ax.set_yticklabels(ylabels or [str(iv) for iv in rate.index], fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"P(catastrophic | {xlabel}, {ylabel})")
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label("catastrophic rate")
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=170)
        plt.close(fig)
    
    # d_min: use its actual integer values as bins; B_F: 0 / (0,0.5] / (0.5,1]
    dmin_edges = [0.5, 2.5, 4.5, 12]  # groups: 1-2 (supported), 3-4, 5+
    bf_edges = [-0.01, 0.001, 0.5, 1.01]
    binned_catastrophic_heatmap(
        "nearest_chebyshev_distance", "fwhm_3_cross_rate",
        dmin_edges, bf_edges,
        r"$d_{min}$ (binned)", r"$B_F$ (binned)",
        "heatmap_dmin_BF.png",
        xlabels=["1-2", "3-4", ">=5"], ylabels=["0", "(0,0.5]", "(0.5,1]"],
    )
    binned_catastrophic_heatmap(
        "nearest_chebyshev_distance", "intensity_3000_cross_rate",
        dmin_edges, bf_edges,
        r"$d_{min}$ (binned)", r"$B_I$ (binned)",
        "heatmap_dmin_BI.png",
        xlabels=["1-2", "3-4", ">=5"], ylabels=["0", "(0,0.5]", "(0.5,1]"],
    )
    
    # ----------------------------------------------------------------------
    # 4. FULL PAIRWISE FEATURE GRID (lower triangle), colored by y_log_abs_error and by q_true
    # ----------------------------------------------------------------------
    def pairwise_grid(color_col, cbar_label, fname, log_color=False):
        n = len(feat_cols)
        fig = plt.figure(figsize=(2.05 * n, 2.05 * n))
        gs = GridSpec(n, n, figure=fig, wspace=0.08, hspace=0.08)
        c = np.log10(1 + df[color_col]) if log_color else df[color_col]
        vmin, vmax = c.min(), c.max()
        cat_mask = df["catastrophic"].values
        last_sc = None
        for i in range(n):  # row = y feature
            for j in range(n):  # col = x feature
                if j > i:
                    continue
                ax = fig.add_subplot(gs[i, j])
                if i == j:
                    ax.hist(df[feat_cols[i]], bins=15, color="#4c72b0", alpha=0.75)
                    ax.set_yticks([])
                else:
                    sc = ax.scatter(df[feat_cols[j]], df[feat_cols[i]], c=c, cmap="viridis",
                                     vmin=vmin, vmax=vmax, s=22, edgecolor="white", linewidth=0.3)
                    ax.scatter(df.loc[cat_mask, feat_cols[j]], df.loc[cat_mask, feat_cols[i]],
                               s=70, facecolor="none", edgecolor="#d62728", linewidth=1.3)
                    last_sc = sc
                if i == n - 1:
                    ax.set_xlabel(short[j], fontsize=9)
                else:
                    ax.set_xticklabels([])
                if j == 0:
                    ax.set_ylabel(short[i], fontsize=9)
                else:
                    ax.set_yticklabels([])
                ax.tick_params(labelsize=7)
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cb = fig.colorbar(last_sc, cax=cax)
        cb.set_label(cbar_label, fontsize=10)
        fig.suptitle(f"Pairwise feature grid — dot color = {cbar_label}; red circle = catastrophic ($e_{{\\log}}>5$)",
                     fontsize=11, y=0.995)
        fig.savefig(outdir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    pairwise_grid("y_log_abs_error", r"$e_{\log}$", "pairwise_grid_by_logerror.png", log_color=False)
    pairwise_grid("q_true", r"$\log_{10}(1+q_{true})$", "pairwise_grid_by_qtrue.png", log_color=True)
    
    print()
    print("Saved figures to:", outdir)


if __name__ == "__main__":
    main()
