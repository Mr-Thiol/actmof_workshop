# ActMOF Daily Research Handoff — 17 Aug 2026

**Purpose.** Working summary of today's ActMOF emulator-error analysis, intended for asynchronous discussion between Beijing and St. Louis. This is a research handoff rather than manuscript-ready text; numerical claims below should be treated as analysis notes until the corresponding scripts/figures are finalized.

---

## 1. Question we tried to answer

The broader question is:

> **Can we predict when the ActMOF pseudo-laboratory / emulator is likely to be wrong, using only information available from the emulator and the local experimental geometry?**

The immediate motivation is the Sim2Science-style setting: the emulator is useful as a large synthetic benchmark, but it is imperfect because it is ultimately reconstructed from a very sparse set of real laboratory measurements. Instead of treating all simulated points as equally reliable, we want to learn the *structure of emulator error*.

The first proposed error-prediction feature vector was

\[
x_i = [V_I, V_F, d_{\min}, N_{R1}, N_{R2}, B_I, B_F],
\]

where:

- \(V_I\): RF-ensemble variance for predicted intensity;
- \(V_F\): RF-ensemble variance for predicted FWHM;
- \(d_{\min}\): Chebyshev distance to the nearest retained experimental condition;
- \(N_{R1}\): number of Rule-1 neighbors;
- \(N_{R2}\): number of Rule-2 neighbors;
- \(B_I\): fraction/rate of emulator predictions crossing the hard `Intensity = 3000` boundary;
- \(B_F\): fraction/rate of emulator predictions crossing the hard `FWHM = 3` boundary.

The primary error targets are:

\[
e_{\mathrm{abs}} = |q_{\mathrm{pred}}-q_{\mathrm{true}}|
\]

and

\[
e_{\log} = \left|\log(1+q_{\mathrm{pred}})-\log(1+q_{\mathrm{true}})\right|.
\]

Because ActMOF's crystallinity score is highly skewed, today's analysis increasingly favored \(e_{\log}\) over raw absolute error.

---

## 2. Error dataset constructed today

We built a **leave-one-experiment-out reconstruction dataset** from the 95 experimental conditions currently used by the ActMOF emulator analysis.

For each experimental condition \(i\):

1. remove experiment \(i\);
2. rebuild/reconstruct the emulator using the remaining experiments;
3. predict the held-out condition;
4. compare the reconstructed \(q_{\mathrm{pred}}\) with the real \(q_{\mathrm{true}}\);
5. record the seven proposed reliability/error features above;
6. record both absolute and log reconstruction errors.

This produces `Error_data.csv` with **95 rows** and the current analysis fields, including the seven features, `q_true`, `q_pred_mean`, `y_abs_error`, and `y_log_abs_error`.

This is conceptually different from ordinary surrogate-model validation: each row asks how wrong the *pseudo-laboratory itself* would have been if that laboratory observation had not existed.

---

## 3. First observation: raw absolute error is probably not the best target

Earlier today we visualized the 95 leave-one-out points with t-SNE and colored the same embedding using several quantities, including the two error definitions and the crystallinity score \(q\).

The important qualitative observation was:

- the coloring by **absolute error** resembled the coloring by \(q\) rather strongly;
- this is expected because a fixed relative mismatch at large \(q\) automatically produces a much larger absolute error;
- the **log error** is therefore more meaningful if we want to study *relative emulator reliability* rather than simply identify high-\(q\) points.

Working decision from today:

> **Use \(e_{\log}\) as the main error quantity for subsequent structure analysis.**

---

## 4. Analyses run after constructing the dataset

Three small diagnostic analyses were implemented.

### 4.1 Seven feature-vs-log-error plots

For each of the seven proposed features, we plotted the feature against `y_log_abs_error` and calculated both Pearson and Spearman correlations.

Current results:

| Feature | Pearson \(r\) | Spearman \(\rho\) |
|---|---:|---:|
| \(V_I\) | +0.072 | +0.307 |
| \(V_F\) | +0.438 | -0.092 |
| \(d_{\min}\) | +0.257 | -0.248 |
| \(N_{R1}\) | -0.097 | +0.103 |
| \(N_{R2}\) | -0.192 | +0.158 |
| \(B_I\) | -0.107 | -0.562 |
| \(B_F\) | +0.544 | +0.148 |

The plots do **not** show a clean one-dimensional relationship of the form

\[
e_{\log}=f(d_j)
\]

for any single feature.

A particularly useful clue is that Pearson and Spearman sometimes differ strongly, even in sign. This is consistent with outliers, mixtures of regimes, threshold effects, discreteness, and/or feature interactions rather than a simple monotonic relationship.

### 4.2 Feature-correlation heatmap

The seven features are not independent. Several pairs have substantial Spearman correlations:

- \(d_{\min}\) vs. \(N_{R2}\): **-0.748**
- \(V_I\) vs. \(B_I\): **-0.687**
- \(d_{\min}\) vs. \(N_{R1}\): **-0.672**
- \(d_{\min}\) vs. \(V_F\): **+0.669**
- \(d_{\min}\) vs. \(B_I\): **+0.650**
- \(V_F\) vs. \(N_{R2}\): **-0.605**

This is not necessarily a problem; it is physically/algorithmically understandable. Several variables are different ways of describing a common latent concept:

> **How weakly is this condition supported by nearby experiments, and how unstable is the emulator around its rule/threshold boundaries?**

However, it means that fitting an ordinary seven-variable regression and interpreting each coefficient independently would be risky.

### 4.3 \(q_{\mathrm{true}}\) vs. \(q_{\mathrm{pred}}\) leave-one-out diagnostic

On the `log1p(q)` scale, the overall reconstruction is actually fairly strong:

- \(N = 95\)
- Pearson \(r = 0.886\)
- Spearman \(\rho = 0.895\)
- mean log error shown by the diagnostic: **1.169**
- **9** cases have \(q_{\mathrm{true}}>0\) but collapse to \(q_{\mathrm{pred}}=0\)

So the emulator is not globally poor. It preserves the broad ordering of crystallinity quite well, but it contains a small number of qualitatively severe failures.

This became the most important observation of today's analysis.

---

## 5. Error distribution: evidence for two qualitatively different regimes

Using \(e_{\log}>5\) as an exploratory definition of a catastrophic error:

- there are **10 catastrophic points** among the 95 leave-one-out cases;
- **9 of those 10** are exactly the cases where a truly nonzero/crystalline experimental result is reconstructed as \(q_{\mathrm{pred}}=0\);
- the mean `y_log_abs_error` among the 9 collapsed-to-zero cases is approximately **7.75**;
- the mean among the remaining 86 cases is only approximately **0.57**.

This suggests that the current target distribution is not simply "a continuous range of regression errors." It may be better thought of as:

1. a large **ordinary-error / reasonably reconstructed** regime; and
2. a small **catastrophic emulator-failure** regime.

This distinction may be more scientifically useful than trying to predict one continuous error value from the beginning.

---

## 6. Strongest structural clue found today

The 10 points with \(e_{\log}>5\) have a surprisingly consistent local structure.

### 6.1 Sparse experimental support

For **all 10 catastrophic points**:

\[
d_{\min}\ge 4,
\]

and their Rule-1 and Rule-2 neighbor counts are both zero:

\[
N_{R1}=N_{R2}=0.
\]

So catastrophic failures occur away from direct local experimental support.

### 6.2 Boundary instability

For **all 10 catastrophic points**, at least one hard-boundary crossing feature is active:

\[
B_I>0 \quad \text{or} \quad B_F>0.
\]

By contrast:

- there are **56** points with \(B_I=0\) and \(B_F=0\);
- **none** of those 56 points has \(e_{\log}>5\).

For the FWHM-boundary feature specifically:

- 12 observations have \(B_F\ge 0.8\);
- 7 of these 12 are catastrophic.

These counts are much more suggestive than the simple one-feature correlations.

---

## 7. Main working hypothesis at the end of today

The current evidence suggests:

\[
\boxed{
\text{large emulator error}
\;\approx\;
\text{sparse experimental support}
\times
\text{instability near a hard emulator boundary}
}
\]

In words:

> **Being far from experimental data alone may not be sufficient to make the emulator fail. Being near a hard boundary alone may also not be sufficient. The severe failures may arise when both conditions occur simultaneously.**

This would explain why the seven separate feature-vs-error plots do not show a simple relationship.

It also suggests that the simulator's imperfection may be **localized in particular geometric/decision-boundary regimes**, instead of being smoothly distributed throughout the synthesis space.

This is currently the most interesting interpretation of the error dataset.

---

## 8. Consequence: the next analysis should focus on feature interactions

The natural next step is to study

\[
(d_1,d_2,e)
\]

rather than only

\[
(d,e).
\]

However, instead of blindly plotting all \( {7\choose2}=21 \) pairs, the highest-priority combinations should be:

1. \((d_{\min}, B_F, e_{\log})\)
2. \((d_{\min}, B_I, e_{\log})\)
3. \((B_I, B_F, e_{\log})\)
4. \((d_{\min}, V_F, e_{\log})\)
5. \((d_{\min}, V_I, e_{\log})\)

The first two directly test the "sparse support × boundary instability" hypothesis.

A potentially more interpretable visualization than a generic 3-D scatterplot would be:

- x-axis: one feature;
- y-axis: a second feature;
- color: \(e_{\log}\), or the empirical probability of \(e_{\log}>5\);
- catastrophic points explicitly marked.

For example, a binned heatmap of

\[
P(e_{\log}>5 \mid d_{\min}, B_F)
\]

would directly show whether a high-risk region exists.

Pairs such as \(d_{\min}\) and \(N_{R2}\) are lower priority because they are already strongly anticorrelated and likely contain redundant information.

---

## 9. Possible reformulation of the prediction task

Instead of immediately fitting

\[
[V_I,V_F,d_{\min},N_{R1},N_{R2},B_I,B_F]
\rightarrow e_{\log},
\]

we now think a **two-stage / hurdle formulation** may be more appropriate.

### Stage A — catastrophic failure classification

Predict

\[
C_i=\mathbf{1}(e_{\log}>5)
\]

or, even more mechanistically,

\[
C_i=\mathbf{1}(q_{\mathrm{true}}>0,\;q_{\mathrm{pred}}=0).
\]

Question:

> **Is this condition likely to be qualitatively misrepresented by the pseudo-laboratory?**

### Stage B — ordinary error regression

For points that are not catastrophic, model

\[
e_{\log}\mid C_i=0.
\]

This may be a better match to the observed data distribution and a more useful reliability model for downstream simulated experimentation.

Any classifier must be evaluated very conservatively because there are currently only about 10 positive catastrophic cases.

---

## 10. Feature improvements worth considering later

The present seven features are intentionally simple and interpretable, but two families may be too coarse.

### 10.1 Continuous boundary margin

Instead of only recording whether bootstrap/emulator predictions cross the hard boundaries, construct a continuous normalized signed margin, e.g.

\[
M_I=\frac{\hat I-3000}{\sigma_I}
\]

and an analogous quantity for the FWHM threshold.

This would distinguish:

- confidently far from the boundary;
- barely on one side;
- genuinely uncertain/crossing the boundary.

### 10.2 Continuous local support / density

In addition to \(d_{\min}\) and Rule-1/Rule-2 counts, consider:

- distance to the \(k\)-th nearest experiment;
- average distance to several nearest experiments;
- kernel/local-density measures.

These may describe experimental support more smoothly than discrete neighbor counts.

---

## 11. Current interpretation in one paragraph

Today's error analysis does **not** support the idea that ActMOF reconstruction error is a simple monotonic function of any single proposed uncertainty feature. The overall leave-one-out \(q\) reconstruction is actually strong, but a small number of conditions fail catastrophically, especially by collapsing a real nonzero \(q\) to zero. These severe failures all occur without local Rule-1/Rule-2 experimental support and while at least one of the emulator's hard intensity/FWHM boundaries is unstable. Our current hypothesis is therefore that emulator error is interaction-driven: **sparse experimental support combined with boundary instability creates a high-risk region**. The next analysis should test this explicitly in two-feature error maps before attempting a general seven-feature regression model.

---

## 12. Recommended next steps when we resume

1. Generate pairwise error maps for the five priority feature pairs above.
2. Mark the 10 \(e_{\log}>5\) points explicitly.
3. Plot binned/conditional catastrophic-failure rates, especially for \(d_{\min}\times B_F\) and \(d_{\min}\times B_I\).
4. Test a minimal catastrophic-failure classifier using strict leave-one-experiment-out validation.
5. Separately model ordinary error for non-catastrophic points.
6. Only after this, decide whether a richer seven-feature regression is justified.
7. If necessary, replace discrete boundary-crossing variables with continuous boundary margins and enrich local-density features.

---

## 13. Files generated / used today

- `Error_data.csv` — 95-row leave-one-out error-prediction dataset.
- `plot_feature_error.py` — seven feature-vs-log-error plots plus Pearson/Spearman correlation table.
- `feature_error.zip` — rendered feature/error plots and `feature_error_correlations.csv`.
- `feature_correlation_heatmap.py` — feature-correlation heatmap and CSV matrix.
- `feature_correlation_heatmap.csv` — current Spearman feature-correlation matrix.
- `q_true_pred_diagnostic.py` — leave-one-out `q_true` vs. `q_pred_mean` diagnostic.
- `q_true_pred_diagnostic.png` — diagnostic showing the nine nonzero-to-zero collapses.
- t-SNE figures/scripts from the earlier exploratory step — used to compare absolute error, log error, and \(q\)-related colorings.

---

## 14. Notes / caveats

- The catastrophic threshold \(e_{\log}>5\) is currently an **exploratory diagnostic threshold**, not a scientifically calibrated cutoff.
- With only 95 leave-one-out points and roughly 10 catastrophic cases, any predictive model can overfit easily.
- Correlations are descriptive, not causal.
- Several proposed features are strongly dependent on one another by construction.
- The current conclusions are therefore best treated as **hypotheses to test with pairwise structure and strict cross-validation**, rather than final claims.

