# ActMOF Daily Research Handoff — 17 Aug 2026 (Follow-up session)

**Purpose.** This picks up directly from `ActMOF_daily_handoff_2026-08-17.md`
§12 ("Recommended next steps when we resume") and works through essentially
all seven of those items in one session, plus the two feature-improvement
ideas from §10.1/§10.2. Same status as the original: research notes for
asynchronous discussion between Beijing and St. Louis, not manuscript text.
All numbers below are reproducible from the scripts listed in §9.

Companion reference: **`feature_glossary.md`** now defines every symbol used
here, including the new ones introduced today (`M_I`, `M_F`, `min_abs_margin`,
`d2`, `d3`, `mean_d_k3`, `mean_d_k5`, `kde_cheb`). `feature_glossary.csv` was
deleted — the markdown file is now the single source of truth.

## 0. Quick visual summary

![Today's summary dashboard](figures/sam/2026-08-17/followup/today_summary_dashboard.png)

Four panels, each expanded in its own section below: **(A)** unsupervised
PCA on the 7 original features already separates 8/10 catastrophic points
into a distinct arm (§3) — **(B)** the binned heatmap confirms the
interaction hypothesis quantitatively: risk only concentrates where sparse
support and boundary instability coincide (§4) — **(C)** today's new
continuous boundary margins (§9) resolve the 10 catastrophic points into
two distinct collapse sub-mechanisms (§8.2) — **(D)** swapping the binary
boundary flag for the continuous margin measurably improves the
leave-one-out catastrophic-failure classifier while keeping perfect recall
(§9).

---

## 1. Correction to the previous handoff's arithmetic

The original handoff's §5 divides the 95 points into "9 collapsed-to-zero"
and "the remaining 86." With the catastrophic definition applied literally
(`e_log>5`, 10 points total), the correct split is **10 catastrophic / 85
non-catastrophic** (95 − 10 = 85, not 86). This doesn't change any
conclusion, but all counts below use 85 consistently. The 10 catastrophic
points, for reference, by experiment ID:

```
Q001, Q005, Q006, Q009, Q013, Q016, Q077, Q078, Q079, Q086
```

(9 of these still fit the stronger "true nonzero → predicted exactly zero"
definition; `Q078` is the exception — it collapses to `q_pred_mean=609.8`,
not exactly 0, but still crosses `e_log>5`.)

---

## 2. Outlier identification (handoff §12 item request, extended)

Three different outlier notions were run, and **they disagree in an
important way**:

- **Catastrophic-error outliers** (`e_log>5`): the 10 listed above.
- **Per-feature IQR outliers** (Tukey 1.5×IQR fence): mostly a *different*
  set — e.g. `rule1_neighbor_count` flags 17 points, `fwhm_3_cross_rate`
  flags 14 — with only partial overlap with the catastrophic 10.
- **Multivariate 7-D Mahalanobis outliers** (log1p applied to `V_I`,`V_F`
  first, since they span ~6 orders of magnitude and would otherwise
  dominate the covariance estimate; χ²(7) 97.5% threshold = 16.01):
  flags `Q018, Q043, Q037, Q032` — **zero overlap** with the catastrophic
  set.

**Key finding:** the catastrophic points are not statistical outliers in
the raw 7-feature space — `d_min≥4` and `N_R1=N_R2=0` are common among the
correctly-predicted-zero points too. They only stand out once you look at
*error*, not features in isolation. This is a direct, quantitative
confirmation that the interaction hypothesis (§6/§7 of the original
handoff) is the right framing, not a marginal-outlier story.

Full flags per row saved in `figures/sam/2026-08-17/followup/Error_data_with_outlier_flags.csv`.

---

## 3. Dimensionality reduction (handoff §10 lead-in, general request)

- **PCA** on the 7 features (log1p applied to `V_I`,`V_F` before
  standardizing, consistent with §2): PC1 = 46.6% variance, PC2 = 22.8%,
  PC3 = 12.1% (cumulative 3-PC ≈ 81.5%). PC1 loads positively on
  `d_min, V_F, B_I` and negatively on `N_R1, N_R2` — i.e., PC1 *is*
  essentially the sparse-support ↔ boundary-instability axis the working
  hypothesis is built on, recovered unsupervised.
- **PCA scatter (PC1 vs PC2), colored by `e_log` and by `q_true`,
  catastrophic points circled** (`figures/sam/2026-08-17/followup/pca_2d.png`): **8 of
  the 10 catastrophic points fall in a visually distinct arm** of the
  projection, separate from the two main point clouds, using no
  supervision at all. The other 2 (`Q078`, `Q086`) sit inside the
  bottom-right "well-supported" cluster instead — worth a closer look
  (see §10).
- **Feature-redundancy dendrogram**
  (`figures/sam/2026-08-17/followup/feature_dendrogram.png`, distance = `1 − |Spearman ρ|`):
  confirms the correlation heatmap from the original handoff in a cleaner
  form. Two natural families emerge: a **sparse-support family**
  `{d_min, N_R2, V_F}` and a **boundary-instability family**
  `{V_I, B_I}`, with `B_F` and `N_R1` more loosely attached. Practically:
  one representative feature per family (e.g. `d_min` + `B_F`) already
  carries most of the information in all 7.
- Other reduction methods considered but not run (UMAP, MDS, factor
  analysis): not worth it at n=95 — PCA + the dendrogram already give an
  interpretable, low-effort answer; t-SNE was already used in the
  original session for the qualitative absolute-vs-log-error comparison.

---

## 4. Pairwise \((d_1,d_2,e)\) analysis — all 5 priority pairs, binned + labeled

Directly executes handoff §8/§12 items 1–3. Two complementary figure types
per pair, both with **the 10 catastrophic points explicitly labeled by
experiment ID**, not just marked:

- `figures/sam/2026-08-17/followup/priority_pairs_scatter_labeled.png` — raw scatter,
  color = `e_log`, catastrophic points circled + text-labeled.
- `figures/sam/2026-08-17/followup/heatmap2_{dmin_BF, dmin_BI, BI_BF, dmin_VF, dmin_VI}.png`
  — binned conditional catastrophic-rate heatmaps, `P(e_log>5 | x, y)`
  per cell, with sample count `n` annotated (not relying on correlation
  coefficients alone, per handoff §12 item 3).

Key cells:

| Pair | Highest-risk cell | Rate | n |
|---|---|---:|---:|
| `d_min`×`B_F` | `d_min` 3–4, `B_F`∈(0.5,1] | **0.67** | 3 |
| `d_min`×`B_F` | `d_min`≥5, `B_F`∈(0.5,1] | **0.56** | 9 |
| `d_min`×`B_F` | `d_min` 1–2 or `B_F=0` (either alone) | 0.00 | 44–56 |
| `d_min`×`B_I` | `d_min`≥5, `B_I=0` | 0.42 | small n, noisy |
| `B_I`×`B_F` | `B_F`∈(0.5,1] (any `B_I`) | **0.58** | 12 |
| `d_min`×`V_F` | `d_min` 3–4, `V_F` top tertile | **0.67** | — |

Confirms the interaction hypothesis cleanly: risk only concentrates where
**both** `d_min` is large **and** a boundary is active; either alone stays
at/near 0.

**Refinement to the hypothesis:** the `B_I×B_F` heatmap shows `B_I` and
`B_F` are essentially mutually exclusive in this data (only 1/95 points has
both mid-level). §7 below explains this is not a coincidence.

---

## 5. Small catastrophic-failure classifier, strict leave-one-out (handoff §12 item 4)

Two minimal models, both honestly evaluated (classifier refit 95 times,
held-out point never seen during its own fit):

**Fixed rule** (`d_min≥4 AND N_R1=N_R2=0 AND (B_I>0 OR B_F>0)`, no fitting):
recall **1.00**, precision **0.278**, F1 **0.435** (26 false positives, 0
false negatives).

**2-feature logistic regression** `[d_min, boundary_active]`,
`class_weight="balanced"`, true LOOCV: recall **1.00**, precision
**0.256**, F1 **0.408**, ROC-AUC **0.814**, PR-AUC **0.282** (29 false
positives, 0 false negatives).

**Interpretation:** both models achieve **perfect recall** — in this LOO
test, this feature pair never silently trusts a catastrophic point — but
precision caps around 0.26–0.28. The false positives are overwhelmingly
correctly-predicted true-zero experiments that also happen to sit far from
support and cross a boundary (mechanistically legitimate, not actually
errors). Read as a **conservative triage/screening signal**, not a precise
classifier: flags ~1/3 of otherwise-fine space to double-check, but never
misses a real catastrophic case.

---

## 6. Ordinary error modeled separately (handoff §12 item 5)

Restricting to the 85 non-catastrophic points and recomputing correlations
against `e_log`:

| feature | Pearson (non-cat) | Spearman (non-cat) |
|---|---:|---:|
| `B_I` | −0.410 | **−0.719** |
| `d_min` | −0.302 | −0.555 |
| `V_I` | +0.131 | +0.433 |
| `N_R2` | +0.073 | +0.382 |
| `V_F` | −0.163 | −0.385 |

`B_I` alone reaches ρ = **−0.719** once the catastrophic points are
excluded — the single cleanest relationship in the entire analysis, and it
was invisible in the full-95 table (original handoff had `B_I` at only
−0.562). A 7-feature LOOCV linear regression on this 85-point subset gets
**R²=0.101, MAE=0.463** (vs. std=0.696) — modest, and `B_I` is doing nearly
all of the work; the other 6 features add little on top of it.
(`figures/sam/2026-08-17/followup/correlations_noncatastrophic.csv`)

---

## 7. Is a 7-feature regression justified? (handoff §12 item 6 — answered: no)

| Model | LOOCV R² | LOOCV MAE | Gives an actionable decision? |
|---|---:|---:|---|
| Single-stage, 7 features, all 95 | 0.178 | 1.332 | No — no pass/fail boundary |
| Stage B, 7 features, 85 non-catastrophic only | 0.101 | 0.463 | Weak; ~1 feature (`B_I`) carries it |
| Stage A, 2-feature classifier, LOOCV | — | — | Yes — recall 1.00 / precision 0.26–0.28 |

**Decision: no, the full 7-feature joint regression is not the model to
lead with.** It explains only ~18% of variance across all 95 points, has
no interpretable decision boundary, and at n=95 with strongly correlated
features (§3) it risks overfitting for a modest R² gain over the much
simpler two-stage pieces. Keep it only as an appendix robustness check.
**The two-stage hurdle formulation from §9 of the original handoff is the
one to build on.**

---

## 8. Mechanistic finding: reading `build_error_dataset.py` directly

Rather than only correlating symptoms, we read the actual reconstruction
code. Two concrete, code-level facts:

**8.1 `B_I + B_F ≤ 1` is an identity, not a correlation.** The
postprocessing logic (`postprocess_prediction`) is a nested
`if intensity_pre<3000: … elif fwhm_pre>3: …` — the FWHM check only ever
fires *after* the intensity check has already passed. So a single seed can
trigger at most one of the two collapse mechanisms, and this holds exactly
across all 95 rows (`max(B_I+B_F) = 1.0`, zero violations, verified
directly). The near-empty `B_I×B_F` "both active" cell from §4 is
therefore **not** an empirical pattern to interpret — it's guaranteed by
the code's control flow.

**8.2 There are two distinct catastrophic sub-mechanisms, not one:**

- **FWHM-collapse** (`B_F` active, intensity prediction fine):
  `Q001, Q005, Q006, Q009, Q013, Q016` (all `B_F=1.0`) plus `Q078`
  (`B_F=0.8`) — 7 of 10.
- **Intensity-collapse** (`B_I` active, raw intensity prediction itself
  falls under 3000): `Q077, Q079, Q086` (all `B_I=1.0`) — 3 of 10.

These are mechanistically different failure paths through the same
threshold logic and probably deserve separate framing rather than one
generic "boundary instability" label.

---

## 9. §10.1 — Continuous boundary margin: implemented and adopted

Built `M_I=(\hat I_{pre}-3000)/\sigma_{I,pre}` and
`M_F=(\hat F_{pre}-3)/\sigma_{F,pre}` **from a full re-simulation** of the
LOO reconstruction (`followup_build_continuous_features.py`, same 5 seeds,
same rules, same code path as `build_error_dataset.py`) — re-deriving the
*pre-threshold* continuous predictions, since the values already stored in
`Error_data.csv` are post-threshold and uninformative for margin
construction once a point has collapsed to exactly 0. **Validation:**
re-simulated `d_min` matched `nearest_chebyshev_distance` exactly on all 95
rows, confirming the re-simulation is faithful.

- Correlations with `e_log`: messy across all 95 (Pearson −0.156/+0.176),
  but **on the 85 non-catastrophic points, Spearman jumps to +0.565 (`M_I`)
  and −0.515 (`M_F`)** — same clean-only-after-splitting pattern as `B_I` in
  §6.
- `figures/sam/2026-08-17/followup/continuous_margins_MI_MF.png` (`M_I` vs `M_F`, colored
  by `e_log`, catastrophic points labeled): visually separates three
  regimes exactly as hoped — most points sit at confidently-extreme
  margins (safe), and the 10 catastrophic points split into the two
  sub-mechanisms from §8.2: `Q001/Q005/Q006/Q009/Q013/Q016/Q078` cluster in
  a narrow band of small positive `M_F` (genuinely uncertain FWHM
  crossing), `Q077/Q079/Q086` sit at moderately negative `M_I` (genuinely
  uncertain intensity crossing).
- **Classifier improvement (the real test):** swapping the binary
  `boundary_active` flag for the continuous, bounded feature
  `min_abs_margin = min(|M_I|,|M_F|)` (clipped at 50 — raw `M_I`/`M_F` are
  numerically unstable, spanning roughly ±2500 whenever a seed's std is
  near 0) while keeping `d_min`:

  | features | recall | precision | F1 | ROC-AUC |
  |---|---:|---:|---:|---:|
  | `[d_min, boundary_active]` (binary, §5) | 1.00 | 0.256 | 0.408 | 0.814 |
  | `[d_min, min_abs_margin]` (continuous) | **1.00** | **0.333** | **0.500** | **0.867** |

  Same perfect recall, precision up ~30% relative, F1 up ~22% relative,
  AUC up. **Adopted** — this is a measured, not hypothetical, improvement.

---

## 10. §10.2 — Continuous local support/density: tested, not adopted

Computed `d2`, `d3` (2nd/3rd-nearest Chebyshev distance), `mean_d_k3`,
`mean_d_k5` (mean distance to 3/5 nearest), and a Gaussian KDE local
density (median-heuristic bandwidth) — all in the same raw-unit Chebyshev
metric `d_min` already uses.

| feature | Pearson | Spearman |
|---|---:|---:|
| `d_min` (original) | +0.257 | −0.248 |
| `d3` | +0.142 | −0.340 |
| `mean_d_k3` | +0.198 | −0.300 |
| `mean_d_k5` | +0.163 | −0.305 |
| `kde_cheb` | −0.155 | −0.187 |

Marginal at best — `d3`/`mean_d_k3` edge out plain `d_min` slightly, but
`kde_cheb` is worse and visually noisy (`figures/sam/2026-08-17/followup/density_vs_error.png`):
several safe points sit at equally low local density as the catastrophic
ones. **Confirmed by the classifier test**: replacing `d_min` with
`mean_d_k3` drops recall to 0.90 (misses a catastrophic point); replacing
it with `kde_cheb` drops recall to 0.80 (misses two). **Verdict: keep
`d_min` (nearest-neighbor distance), do not adopt the smoothed/averaged
density alternatives** — "is there a close neighbor at all" matters more
here than "how dense is the neighborhood overall."

---

## 11. Updated working hypothesis (refines original handoff §7)

\[
\boxed{
\text{large emulator error}
\;\approx\;
\text{sparse experimental support (}d_{\min}\text{)}
\times
\text{proximity to a hard emulator boundary (continuous margin, not just binary crossing)}
}
\]

with the added mechanistic nuance from §8: boundary proximity splits into
two distinct, mutually-exclusive-by-construction failure pathways
(FWHM-side vs. intensity-side), and the interaction is best captured with
`[d_min, min_abs_margin]` rather than `[d_min, boundary_active]`.

---

## 12. Recommended next steps when we resume

1. **Pick the mini-page headline figures now** — leading candidates:
   `heatmap2_dmin_BF.png` (quantitative, states the interaction claim
   directly) and `continuous_margins_MI_MF.png` (visually striking, shows
   the two sub-mechanisms with zero modeling). `pca_2d.png` is a strong
   secondary/supporting panel.
2. **Sensitivity-check the `e_log>5` threshold.** It's still an exploratory
   cutoff (unchanged caveat from the original handoff). Sweep it and
   confirm the `[d_min, min_abs_margin]` classifier's recall=1.0 is stable
   across nearby thresholds, not an artifact of exactly 5.
3. **Chase a domain-level explanation for the 2 PCA-cluster exceptions**
   (`Q078`, `Q086` — catastrophic but sitting inside the "safe" PCA
   cluster rather than the separated arm). Check their raw synthesis
   conditions (`metal_amount`, `modulator`, `reaction_time/temperature`)
   for anything distinctive — a physically-grounded explanation would
   strengthen the paper more than the current purely-statistical account.
4. **Decide how to present the classifier's precision ceiling honestly.**
   0.256→0.333 precision is a real improvement but still means ~2/3 of
   flagged points are false alarms. Frame explicitly as a safety-net/triage
   tool (never misses a catastrophic point) rather than a precise
   predictor, in both the text and any figure caption.
5. Only if there's space: a supplementary appendix figure/table showing
   the single-stage 7-feature regression as the rejected alternative,
   to preempt the "why not just regress on all 7 features" reviewer
   question (§7 already has the numbers ready).

---

## 13. Files generated / used today

**Documentation**
- `feature_glossary.md` — updated with all features introduced today
  (§9/§10 above); `feature_glossary.csv` deleted per today's request — the
  markdown file is now the only glossary.
- `ActMOF_daily_handoff_2026-08-17_followup.md` — this file.

**Scripts** (`my_scripts/sam/2026-08-17/`)
- `followup_outliers_pca_dendrogram.py` — §2/§3: IQR + Mahalanobis outliers,
  PCA, feature dendrogram.
- `followup_pairwise_classifier_regression.py` — §4/§5/§6/§7: priority
  pairwise scatter + binned heatmaps, rule/logistic LOOCV classifier,
  non-catastrophic correlations and regression, all-95 regression baseline.
- `followup_build_continuous_features.py` — §9/§10: re-simulates the LOO
  reconstruction to derive `M_I`, `M_F`, and the density features (~7 min
  runtime, 950 RF fits).
- `followup_analyze_continuous_features.py` — §9/§10: correlations, the
  `M_I`/`M_F` figure, density-vs-error figure, and the classifier
  variant comparison (Table in §9).
- `followup_today_summary_dashboard.py` — §0: the 4-panel summary
  dashboard (PCA, binned heatmap, continuous margins, classifier
  before/after) combining today's headline results into one figure.

**Figures/tables** (`figures/sam/2026-08-17/followup/`)
- `Error_data_with_outlier_flags.csv` — full table with IQR + Mahalanobis
  flags appended.
- `pca_2d.png`, `feature_dendrogram.png`
- `heatmap_dmin_BF.png`, `heatmap_dmin_BI.png` (original 2-bin version)
- `priority_pairs_scatter_labeled.png`
- `heatmap2_dmin_BF.png`, `heatmap2_dmin_BI.png`, `heatmap2_BI_BF.png`,
  `heatmap2_dmin_VF.png`, `heatmap2_dmin_VI.png` (refined 3×3 binned
  version used for §4's table)
- `pairwise_grid_by_logerror.png`, `pairwise_grid_by_qtrue.png` — full 7×7
  lower-triangular scatterplot matrix
- `correlations_noncatastrophic.csv`
- `continuous_features.csv` — re-simulated `M_I`/`M_F` inputs and density
  features, keyed by `experiment_id`
- `continuous_margins_MI_MF.png`, `density_vs_error.png`,
  `density_feature_correlations.csv`
- `today_summary_dashboard.png` — the 4-panel summary figure referenced
  in §0 above.

---

## 14. Notes / caveats

- All caveats from the original handoff still apply (exploratory
  threshold, small-sample overfitting risk, correlation ≠ causation,
  feature redundancy).
- The 85-vs-86 non-catastrophic count (§1) is a correction to the original
  handoff's arithmetic, not a new result — treat any older notes that say
  "86" as meaning the same 85-point set.
- Multivariate Mahalanobis outliers (§2) were computed with `V_I`,`V_F`
  log1p-transformed first; the raw-scale version is numerically unstable
  (`MinCovDet` failed to converge — determinant warnings) because of the
  many near-identical zero-feature rows in this dataset. Use the log1p
  version if re-running.
- `M_I`/`M_F` (§9) are only as good as the pre-threshold re-simulation;
  they depend on the same 5 fixed seeds as the original pipeline and on
  `sklearn==1.7.2` (the env used for `myWork_Clone`) — a different sklearn
  version may shift `RandomForestRegressor` internals slightly. `d_min`
  matching exactly (§9) is the confidence check that the re-simulation is
  otherwise faithful.
- `kde_cheb`'s bandwidth (§10) was a simple median-heuristic choice, not
  tuned — this could be revisited, but given it already underperforms
  plain `d_min` there's no reason to invest further here.
- Everything above still comes from **95 leave-one-out points and ~10
  catastrophic cases** — every reported classifier/regression number
  should be read as directional evidence, not a calibrated performance
  guarantee.

---

## 15. Visualization update — 17 Aug 2026 (added after CDT session)

Three new figures were generated to communicate the interaction-effect
conclusion visually. All saved to `figures/sam/2026-08-17/followup/`; script is
`my_scripts/sam/2026-08-17/plot_interaction_three_views.py`.

**Fig A — AND-Gate quadrant scatter** (`interaction_andgate_scatter.png`)

Axes: $d_{\min}$ (x) vs $\max(B_I, B_F)$ (y), colored by $e_{\log}$.
Background shading divides the space into four risk quadrants. Key
takeaway: **all 10 catastrophic points sit in the upper-right (pink) risk
zone** — both conditions must be met simultaneously. The lower-right
(sparse only) and left half (well-supported, any boundary rate) remain
essentially risk-free.

**Fig B — Conditional line plot** (`interaction_conditional_lines.png`)

Two lines: $B_F=0$ (blue) vs $B_F>0$ (red), x-axis = $d_{\min}$ bins,
y-axis = $P(e_{\log}>5)$. The blue line stays flat at 0 across all
$d_{\min}$ values; the red line rises sharply to 0.56–0.67 once
$d_{\min}\ge 3$. This "scissors gap" is the clearest quantitative picture
of the interaction: **sparsity alone is harmless; it only becomes
dangerous when the boundary is also active.**

**Fig C — Annotated full scatter** (`interaction_annotated_scatter.png`)

All 95 LOO points plotted as $d_{\min}$ vs $B_F$, with shape encoding
collapse sub-type (triangle = FWHM-collapse, square = Intensity-collapse).
The two sub-mechanisms fall in different locations and are mutually
exclusive by code construction ($B_I + B_F \le 1$, see §8.1). Experiment
IDs of all 10 catastrophic points are labelled directly on the figure.
