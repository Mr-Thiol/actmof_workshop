"""
Three visualizations of the sparse-support × boundary-instability interaction effect.
Saves to figures/sam/2026-08-17/followup/.
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── paths ────────────────────────────────────────────────────────────────────
BASE = pathlib.Path(__file__).resolve().parents[3]
DATA_CSV   = BASE / "Error_data.csv"
OUT_DIR    = BASE / "figures" / "sam" / "2026-08-17" / "followup"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load & derive columns ─────────────────────────────────────────────────────
df = pd.read_csv(DATA_CSV)
df = df.rename(columns={
    "nearest_chebyshev_distance":  "d_min",
    "intensity_3000_cross_rate":   "B_I",
    "fwhm_3_cross_rate":           "B_F",
    "rule1_neighbor_count":        "N_R1",
    "rule2_neighbor_count":        "N_R2",
    "y_log_abs_error":             "e_log",
})

THRESH = 5.0
df["catastrophic"]    = df["e_log"] > THRESH
df["boundary_active"] = (df["B_I"] > 0) | (df["B_F"] > 0)

# collapse sub-type for catastrophic points
def sub_type(row):
    if not row["catastrophic"]:
        return "normal"
    if row["B_F"] > 0:
        return "fwhm_collapse"
    return "intensity_collapse"

df["sub_type"] = df.apply(sub_type, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — And-Gate quadrant scatter
# X = d_min, Y = max(B_I, B_F), four quadrants coloured by risk
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5.5))

df["B_max"] = df[["B_I", "B_F"]].max(axis=1)

d_thresh = 4.0
ax.axvspan(0, d_thresh, ymin=0, ymax=1, color="#d4e6f1", alpha=0.45, zorder=0)
ax.axvspan(d_thresh, df["d_min"].max() + 1, ymin=0, ymax=0.5,
           color="#d5e8d4", alpha=0.45, zorder=0)
ax.axvspan(d_thresh, df["d_min"].max() + 1, ymin=0.5, ymax=1,
           color="#f8cecc", alpha=0.55, zorder=0)

ax.text(1.5, 0.97, "Safe zone\n(well-supported)", ha="center", va="top",
        color="#1a5276", fontsize=8, transform=ax.get_xaxis_transform())
ax.text(7.5, 0.03, "Sparse only\n(still safe)", ha="center", va="bottom",
        color="#196f3d", fontsize=8, transform=ax.get_xaxis_transform())
ax.text(7.5, 0.97, "Risk zone\n(sparse + boundary)", ha="center", va="top",
        color="#922b21", fontsize=8, transform=ax.get_xaxis_transform())

mask_n = ~df["catastrophic"]
ax.scatter(df.loc[mask_n, "d_min"], df.loc[mask_n, "B_max"],
           c=df.loc[mask_n, "e_log"], cmap="YlOrBr", vmin=0, vmax=THRESH,
           s=40, alpha=0.7, edgecolors="grey", linewidths=0.4, zorder=2)

mask_c = df["catastrophic"]
sc = ax.scatter(df.loc[mask_c, "d_min"], df.loc[mask_c, "B_max"],
                c=df.loc[mask_c, "e_log"], cmap="YlOrBr", vmin=0, vmax=THRESH,
                s=130, marker="*", edgecolors="red", linewidths=1.2, zorder=3)
for _, row in df[mask_c].iterrows():
    ax.annotate(row["experiment_id"], (row["d_min"], row["B_max"]),
                textcoords="offset points", xytext=(5, 4), fontsize=6.5,
                color="darkred")

ax.axvline(d_thresh, color="#555", lw=1.2, ls="--", zorder=1)
ax.axhline(0.0, color="#777", lw=0.8, ls=":", zorder=1)

cbar = plt.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label("$e_{\\log}$", fontsize=10)

ax.set_xlabel("$d_{\\min}$ (Chebyshev distance to nearest experiment; larger = sparser support)", fontsize=9)
ax.set_ylabel("$\\max(B_I, B_F)$ (boundary crossing rate; larger = more unstable)", fontsize=9)
ax.set_title("AND-Gate: catastrophic failures require BOTH sparse support AND boundary instability", fontsize=9)
ax.set_xlim(-0.3, df["d_min"].max() + 0.8)
ax.set_ylim(-0.05, 1.08)

legend_elems = [
    mpatches.Patch(color="#d4e6f1", alpha=0.7, label="Safe zone (well-supported)"),
    mpatches.Patch(color="#d5e8d4", alpha=0.7, label="Sparse only (still safe)"),
    mpatches.Patch(color="#f8cecc", alpha=0.7, label="Risk zone (both conditions met)"),
    Line2D([0], [0], marker="*", color="w", markerfacecolor="#e67e22",
           markeredgecolor="red", markersize=11, label="Catastrophic failure (e_log > 5)"),
]
ax.legend(handles=legend_elems, fontsize=8, loc="upper left")

plt.tight_layout()
out1 = OUT_DIR / "interaction_andgate_scatter.png"
plt.savefig(out1, dpi=150)
plt.close()
print(f"Saved: {out1}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Conditional effect line plot
# X = d_min bins, Y = P(catastrophic), two lines: B_F=0 vs B_F>0
# ─────────────────────────────────────────────────────────────────────────────
bins   = [0, 2, 4, df["d_min"].max() + 1]
labels = ["1-2", "3-4", ">=5"]
df["d_bin"] = pd.cut(df["d_min"], bins=bins, labels=labels)

groups = {
    "$B_F = 0$ (no FWHM boundary crossing)": df[df["B_F"] == 0],
    "$B_F > 0$ (boundary active)":            df[df["B_F"]  > 0],
}

fig, ax = plt.subplots(figsize=(6.5, 4.5))

colors  = ["#2980b9", "#e74c3c"]
markers = ["o", "^"]
for (label, sub), col, mkr in zip(groups.items(), colors, markers):
    rates = sub.groupby("d_bin", observed=True)["catastrophic"].agg(
        rate="mean", n="count"
    ).reset_index()
    ax.plot(rates["d_bin"], rates["rate"], color=col, marker=mkr,
            linewidth=2, markersize=8, label=label, zorder=3)
    for _, r in rates.iterrows():
        ax.annotate(f"n={int(r['n'])}", (r["d_bin"], r["rate"]),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color=col)

ax.axhline(0, color="grey", lw=0.8, ls=":")
ax.set_xlabel("$d_{\\min}$ bin (experimental support sparsity)", fontsize=11)
ax.set_ylabel("$P(e_{\\log}>5)$ catastrophic failure rate", fontsize=11)
ax.set_title("Conditional effect: $d_{\\min}$ raises failure rate ONLY when $B_F > 0$", fontsize=10)
ax.set_ylim(-0.05, 1.0)
ax.legend(fontsize=10)

ax.annotate("Both conditions met\n-> failure rate 0.56-0.67",
            xy=(">=5", 0.56), xytext=("3-4", 0.72),
            arrowprops=dict(arrowstyle="->", color="#922b21"),
            color="#922b21", fontsize=9, ha="center")

plt.tight_layout()
out2 = OUT_DIR / "interaction_conditional_lines.png"
plt.savefig(out2, dpi=150)
plt.close()
print(f"Saved: {out2}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Annotated d_min × B_F scatter (all 95 points)
# Color = catastrophic/normal, shape = collapse sub-type
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 5.5))

style_map = {
    "normal":             dict(marker="o", color="#85c1e9", s=45,  edgecolors="#2471a3", lw=0.5, zorder=2),
    "fwhm_collapse":      dict(marker="^", color="#e74c3c", s=140, edgecolors="#922b21", lw=1.2, zorder=4),
    "intensity_collapse": dict(marker="s", color="#f39c12", s=140, edgecolors="#784212", lw=1.2, zorder=4),
}

for stype, style in style_map.items():
    sub = df[df["sub_type"] == stype]
    ax.scatter(sub["d_min"], sub["B_F"], **style, alpha=0.85)

for _, row in df[df["catastrophic"]].iterrows():
    ax.annotate(row["experiment_id"], (row["d_min"], row["B_F"]),
                textcoords="offset points", xytext=(6, 4),
                fontsize=7, color="darkred",
                arrowprops=dict(arrowstyle="-", color="darkred", lw=0.5))

ax.axvline(4, color="#555", lw=1.2, ls="--", alpha=0.7)
ax.axhline(0, color="#777", lw=0.8, ls=":", alpha=0.7)

ax.axvspan(4, df["d_min"].max() + 0.8, ymin=0.02, ymax=1,
           color="#f8cecc", alpha=0.25, zorder=0)
ax.text(df["d_min"].max() - 0.5, 0.95, "Risk zone", ha="right", va="top",
        color="#922b21", fontsize=9)

ax.set_xlabel("$d_{\\min}$ (Chebyshev distance to nearest experiment)", fontsize=11)
ax.set_ylabel("$B_F$ (FWHM boundary crossing rate)", fontsize=11)
ax.set_title("All 95 LOO points: $d_{\\min}$ vs $B_F$\n(shape = collapse sub-type; labels = catastrophic IDs)", fontsize=10)
ax.set_xlim(-0.3, df["d_min"].max() + 1)
ax.set_ylim(-0.05, 1.12)

handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#85c1e9",
           markeredgecolor="#2471a3", markersize=9, label="Normal (n=85)"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#e74c3c",
           markeredgecolor="#922b21", markersize=11, label="Catastrophic: FWHM-collapse (n=7)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#f39c12",
           markeredgecolor="#784212", markersize=11, label="Catastrophic: Intensity-collapse (n=3)"),
    Line2D([0], [0], color="#555", lw=1.5, ls="--", label="$d_{\\min}=4$ threshold"),
]
ax.legend(handles=handles, fontsize=9, loc="upper left")

plt.tight_layout()
out3 = OUT_DIR / "interaction_annotated_scatter.png"
plt.savefig(out3, dpi=150)
plt.close()
print(f"Saved: {out3}")

print("\nAll 3 figures done.")
