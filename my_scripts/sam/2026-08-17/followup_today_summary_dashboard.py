import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

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
    df = err.merge(cont, on="experiment_id", how="inner")
    df["catastrophic"] = (df["y_log_abs_error"] > 5).astype(int)
    cat = df["catastrophic"].astype(bool)

    FEATS = ["rf_intensity_variance", "rf_fwhm_variance", "nearest_chebyshev_distance",
             "rule1_neighbor_count", "rule2_neighbor_count",
             "intensity_3000_cross_rate", "fwhm_3_cross_rate"]
    
    # ---- shared prep: PCA ----
    Xlog = df[FEATS].copy()
    Xlog["rf_intensity_variance"] = np.log1p(Xlog["rf_intensity_variance"])
    Xlog["rf_fwhm_variance"] = np.log1p(Xlog["rf_fwhm_variance"])
    Xs = StandardScaler().fit_transform(Xlog.values)
    pca = PCA().fit(Xs)
    scores = pca.transform(Xs)
    evr = pca.explained_variance_ratio_
    
    # ---- shared prep: margins ----
    eps = 1e-6
    df["M_I"] = (df["I_pre_mean"] - 3000) / (df["I_pre_std"] + eps)
    df["M_F"] = (df["F_pre_mean"] - 3) / (df["F_pre_std"] + eps)
    df["M_I_clip"] = df["M_I"].clip(-50, 50)
    df["M_F_clip"] = df["M_F"].clip(-50, 50)
    df["min_abs_margin"] = np.minimum(df["M_I"].abs(), df["M_F"].abs()).clip(upper=50)
    df["boundary_active"] = ((df["intensity_3000_cross_rate"] > 0) | (df["fwhm_3_cross_rate"] > 0)).astype(int)
    
    # ---- shared prep: binned heatmap d_min x B_F ----
    dmin_edges = [0.5, 2.5, 4.5, 12]
    bf_edges = [-0.01, 0.001, 0.5, 1.01]
    dfx = df.copy()
    dfx["xb"] = pd.cut(dfx["nearest_chebyshev_distance"], bins=dmin_edges, include_lowest=True)
    dfx["yb"] = pd.cut(dfx["fwhm_3_cross_rate"], bins=bf_edges, include_lowest=True)
    x_cats = list(dfx["xb"].cat.categories); y_cats = list(dfx["yb"].cat.categories)
    rate = dfx.groupby(["yb", "xb"], observed=False)["catastrophic"].mean().reindex(
        pd.MultiIndex.from_product([y_cats, x_cats])).unstack().reindex(index=y_cats, columns=x_cats)
    n = dfx.groupby(["yb", "xb"], observed=False)["catastrophic"].size().reindex(
        pd.MultiIndex.from_product([y_cats, x_cats])).unstack().reindex(index=y_cats, columns=x_cats)
    
    # ---- shared prep: LOOCV classifier comparison ----
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
    metrics = {}
    for label, cols in [("Binary\n(d_min, boundary_active)", ["nearest_chebyshev_distance", "boundary_active"]),
                         ("Continuous\n(d_min, min_abs_margin)", ["nearest_chebyshev_distance", "min_abs_margin"])]:
        X = df[cols].values
        probs, preds = loocv_logit(X, y)
        p, r, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
        auc = roc_auc_score(y, probs)
        metrics[label] = dict(recall=r, precision=p, f1=f1, auc=auc)
    
    # =====================================================================
    # DASHBOARD: 2x2
    # =====================================================================
    COLOR_SAFE_LOW = "#440154"   # viridis low (from cmap, used only for consistency reference)
    CAT_EDGE = "#d62728"
    COL_BINARY = "#8a8f98"      # muted gray — "previous / binary"
    COL_CONT = "#2a9d8f"        # teal — "today / continuous"
    
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)
    
    # --- Panel A: PCA ---
    axA = fig.add_subplot(gs[0, 0])
    sc = axA.scatter(scores[:, 0], scores[:, 1], c=df["y_log_abs_error"], cmap="viridis", s=48,
                      edgecolor="white", linewidth=0.5, zorder=2)
    axA.scatter(scores[cat, 0], scores[cat, 1], s=170, facecolor="none", edgecolor=CAT_EDGE,
                linewidth=2, zorder=3)
    for i, r in df[cat].iterrows():
        axA.annotate(r["experiment_id"], (scores[i, 0], scores[i, 1]), fontsize=7.5, color="#8b0000",
                     xytext=(5, 4), textcoords="offset points")
    axA.set_xlabel(f"PC1 ({evr[0]*100:.0f}% var)")
    axA.set_ylabel(f"PC2 ({evr[1]*100:.0f}% var)")
    axA.set_title("A. PCA of the 7 reliability features\n8/10 catastrophic points separate unsupervised", fontsize=11)
    cb = fig.colorbar(sc, ax=axA, shrink=0.85); cb.set_label(r"$e_{\log}$")
    axA.legend(handles=[Line2D([0], [0], marker="o", color="none", markeredgecolor=CAT_EDGE,
                                markerfacecolor="none", markersize=11, markeredgewidth=2,
                                label="catastrophic ($e_{\\log}>5$)")], loc="lower left", fontsize=8, frameon=False)
    
    # --- Panel B: binned heatmap ---
    axB = fig.add_subplot(gs[0, 1])
    im = axB.imshow(rate.values, cmap="viridis", vmin=0, vmax=1, origin="lower", aspect="auto")
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            v = rate.values[i, j]; c_ = n.values[i, j]
            txt = "-" if np.isnan(v) else f"{v:.2f}\n(n={int(c_)})"
            color = "black" if (not np.isnan(v) and v > 0.55) else "white"
            axB.text(j, i, txt, ha="center", va="center", fontsize=9.5, color=color)
    axB.set_xticks(range(rate.shape[1])); axB.set_xticklabels(["1-2", "3-4", ">=5"])
    axB.set_yticks(range(rate.shape[0])); axB.set_yticklabels(["0", "(0,0.5]", "(0.5,1]"])
    axB.set_xlabel(r"$d_{min}$ (binned)"); axB.set_ylabel(r"$B_F$ (binned)")
    axB.set_title("B. Interaction confirmed: risk only where\nsparse support AND boundary instability coincide", fontsize=11)
    cb2 = fig.colorbar(im, ax=axB, shrink=0.85); cb2.set_label("P(catastrophic)")
    
    # --- Panel C: continuous margins ---
    axC = fig.add_subplot(gs[1, 0])
    sc3 = axC.scatter(df["M_I_clip"], df["M_F_clip"], c=df["y_log_abs_error"], cmap="viridis", s=48,
                       edgecolor="white", linewidth=0.5, zorder=2)
    axC.scatter(df.loc[cat, "M_I_clip"], df.loc[cat, "M_F_clip"], s=170, facecolor="none",
                edgecolor=CAT_EDGE, linewidth=2, zorder=3)
    for _, r in df[cat].iterrows():
        axC.annotate(r["experiment_id"], (r["M_I_clip"], r["M_F_clip"]), fontsize=7.5, color="#8b0000",
                     xytext=(5, 4), textcoords="offset points")
    axC.axvline(0, color="gray", lw=0.8, ls="--"); axC.axhline(0, color="gray", lw=0.8, ls="--")
    axC.set_xlabel(r"$M_I=(\hat I_{pre}-3000)/\sigma_{I,pre}$ (clipped $\pm$50)")
    axC.set_ylabel(r"$M_F=(\hat F_{pre}-3)/\sigma_{F,pre}$ (clipped $\pm$50)")
    axC.set_title("C. New today (§10.1): continuous margins reveal\ntwo distinct collapse sub-mechanisms", fontsize=11)
    cb3 = fig.colorbar(sc3, ax=axC, shrink=0.85); cb3.set_label(r"$e_{\log}$")
    
    # --- Panel D: classifier improvement bar chart ---
    axD = fig.add_subplot(gs[1, 1])
    metric_names = ["recall", "precision", "f1", "auc"]
    metric_labels = ["Recall", "Precision", "F1", "ROC-AUC"]
    x = np.arange(len(metric_names))
    w = 0.36
    vals_bin = [metrics["Binary\n(d_min, boundary_active)"][m] for m in metric_names]
    vals_con = [metrics["Continuous\n(d_min, min_abs_margin)"][m] for m in metric_names]
    b1 = axD.bar(x - w/2, vals_bin, width=w, color=COL_BINARY, label="binary: [d_min, boundary_active]\n(before today)")
    b2 = axD.bar(x + w/2, vals_con, width=w, color=COL_CONT, label="continuous: [d_min, min_abs_margin]\n(added today, §10.1)")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            axD.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width()/2, h), textcoords="offset points",
                         xytext=(0, 3), ha="center", fontsize=8.5)
    axD.set_xticks(x); axD.set_xticklabels(metric_labels)
    axD.set_ylim(0, 1.15)
    axD.set_ylabel("LOOCV score")
    axD.set_title("D. Continuous boundary margin improves the\ncatastrophic-failure classifier (perfect recall kept)", fontsize=11)
    axD.legend(loc="upper right", fontsize=7.8, frameon=False)
    for spine in ["top", "right"]:
        axD.spines[spine].set_visible(False)
    
    fig.suptitle("ActMOF emulator-error follow-up — 17 Aug 2026 summary", fontsize=15, y=0.995)
    fig.savefig(outdir / "today_summary_dashboard.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", outdir / "today_summary_dashboard.png")
    print(pd.DataFrame(metrics).T.round(3).to_string())


if __name__ == "__main__":
    main()
