# ActMOF Error-Analysis Feature Glossary

Defines every symbol/abbreviation used in `Error_data.csv`, the daily handoffs,
and the follow-up analysis in `figures/followup/`.

## Identifiers

| Symbol | CSV column | Meaning |
|---|---|---|
| — | `experiment_id` | Real-lab experiment ID (`Q001`…`Q095`), 95 total. |
| — | `metal_amount`, `modulator`, `add_solvent`, `reaction_time`, `reaction_temperature` | The 5 real synthesis-condition inputs to the RF emulator. |
| — | `Intensity_exp`, `FWHM_exp` | Measured XRD peak intensity and full-width-half-max for that experiment (raw lab readout, used to derive `q_true`). |

## Reconstruction target

| Symbol | CSV column | Meaning |
|---|---|---|
| `q_true` | `q_true` | Crystallinity score computed from the real measurement for the held-out experiment. |
| `q_pred` | `q_pred_mean` | Crystallinity score the leave-one-out-reconstructed emulator predicts for that same held-out experiment. |
| — | `intensity_pred_mean`, `fwhm_pred_mean` | The reconstructed emulator's mean intensity/FWHM prediction (inputs to `q_pred`). |

## The 7 reliability/error features (`x_i` in the handoff)

| Symbol | CSV column | Meaning |
|---|---|---|
| `V_I` | `rf_intensity_variance` | RF-ensemble (bootstrap tree) variance of the predicted **intensity** at this condition — a measure of the emulator's *own* uncertainty about intensity. |
| `V_F` | `rf_fwhm_variance` | RF-ensemble variance of the predicted **FWHM** — emulator's own uncertainty about peak width. |
| `d_min` | `nearest_chebyshev_distance` | Chebyshev (L∞ / max-coordinate) distance in synthesis-condition space to the *nearest retained real experiment*. Large `d_min` = this condition is far from any experiment the emulator was actually built from ⇒ **sparse experimental support**. |
| `N_R1` | `rule1_neighbor_count` | Count of neighboring experiments satisfying "Rule 1" (a defined local-support/similarity rule used by the emulator's rule-based branch). |
| `N_R2` | `rule2_neighbor_count` | Count of neighboring experiments satisfying "Rule 2" (a second, distinct local-support rule). |
| `B_I` | `intensity_3000_cross_rate` | Fraction of bootstrap/ensemble predictions whose intensity crosses the hard decision boundary **Intensity = 3000** — i.e., how often this condition sits right on the amorphous/crystalline intensity threshold rather than confidently on one side. |
| `B_F` | `fwhm_3_cross_rate` | Same idea for the hard boundary **FWHM = 3**. |

Also present in the CSV but not part of the core 7: `rule_applied` (which
rule branch — 0/1/2 — the emulator's reconstruction logic actually used for
this condition).

## Error targets

| Symbol | CSV column | Meaning |
|---|---|---|
| `e_abs` | `y_abs_error` | `\|q_pred − q_true\|` — raw absolute reconstruction error. Tracks `q_true` itself (skewed), so **not** the primary target. |
| `e_log` | `y_log_abs_error` | `\|log(1+q_pred) − log(1+q_true)\|` — log-scale reconstruction error. **Primary error quantity** used for all structure analysis (relative, not absolute, reliability). |

## Derived flags used in the follow-up analysis (not in the original CSV)

| Symbol | Definition | Meaning |
|---|---|---|
| catastrophic | `e_log > 5` | Exploratory threshold marking a qualitatively severe reconstruction failure (currently 10 of 95 points). Not a calibrated cutoff — a diagnostic one. |
| `boundary_active` | `1{B_I>0 or B_F>0}` | Binary flag: at least one hard emulator boundary is unstable for this condition. Used as the second feature in the minimal LOOCV classifier. |
| `n_feature_outliers` | count of the 7 features on which a row exceeds its Tukey 1.5×IQR fence | Crude multi-feature marginal-outlier score (see `Error_data_with_outlier_flags.csv`). |
| `mahalanobis_d2` | robust 7-D squared Mahalanobis distance (V_I, V_F log1p-transformed first) | Multivariate outlier score in the full reliability-feature space; **not** the same set as the catastrophic-error points (see analysis notes). |

## Continuous features added 17 Aug 2026 (follow-up, §10.1/§10.2 of the original handoff)

Built by `my_scripts/my_scripts/followup_build_continuous_features.py`, which
**re-simulates** the exact LOO reconstruction from `build_error_dataset.py`
(same 5 seeds, same rules, same postprocessing) to recover the *pre-threshold*
continuous predictions — the original CSV only stores the final,
already-thresholded values, which are uninformative for margin construction
once a point has collapsed to 0. Saved to
`figures/followup/continuous_features.csv`, keyed by `experiment_id`. Re-simulated
`d_min` matched the original `nearest_chebyshev_distance` exactly on all 95 rows
(validates the re-simulation).

### 10.1 Continuous boundary margins

| Symbol | Column | Meaning |
|---|---|---|
| — | `I_pre_mean`, `I_pre_std` | Mean and cross-seed (5-seed) std. of the **pre-threshold** predicted intensity, i.e. the value *before* the `intensity<3000 → 0` hard cutoff is applied. |
| — | `F_pre_mean`, `F_pre_std` | Same for the pre-threshold predicted FWHM, before the `fwhm>3 → 30` cutoff. |
| `M_I` | derived: `(I_pre_mean − 3000) / I_pre_std` | Continuous, signed, normalized margin to the intensity boundary. Large positive = confidently crystalline-side; large negative = confidently amorphous-side; near 0 = genuinely uncertain crossing. |
| `M_F` | derived: `(F_pre_mean − 3) / F_pre_std` | Same for the FWHM boundary. |
| `min_abs_margin` | derived: `min(\|M_I\|, \|M_F\|)` clipped to 50 | Distance (in σ-units) to whichever boundary is closest — the robust, bounded feature that actually improved the LOOCV classifier (see follow-up handoff §7). Raw unclipped `M_I`/`M_F` are numerically unstable (span roughly ±2500) whenever a seed's cross-seed std is near 0, so use `min_abs_margin`, not raw `M_I`/`M_F`, as a model input. |

### 10.2 Continuous local-support / density

All computed in the same **raw-unit Chebyshev** metric as `d_min` (no
normalization — matches `build_error_dataset.py`'s own convention), over the
remaining 94 experiments for each held-out point.

| Symbol | Column | Meaning |
|---|---|---|
| `d2`, `d3` | `d2`, `d3` | Distance to the 2nd- and 3rd-nearest retained experiment (Chebyshev). `d_min` itself is the 1st-nearest. |
| — | `mean_d_k3`, `mean_d_k5` | Mean distance to the nearest 3 / nearest 5 experiments — a smoothed local-support measure. |
| — | `kde_cheb` | Gaussian kernel density estimate at the held-out point, Chebyshev distance, median-heuristic bandwidth. Higher = denser local experimental support. |

**Verdict (see follow-up handoff §8 for full results): 10.1 is adopted, 10.2 is not.** `min_abs_margin` measurably improves the LOOCV catastrophic-failure classifier over the binary `boundary_active` flag; none of `d2/d3/mean_d_k3/mean_d_k5/kde_cheb` beat plain `d_min`, and swapping `d_min` for any of them in the classifier reduces recall below 1.0.

## Working hypothesis (for reference)

> Large emulator error ≈ sparse experimental support (`d_min` large, `N_R1=N_R2=0`)
> **combined with** instability near a hard emulator boundary (`B_I>0` or `B_F>0`).
> Neither condition alone is sufficient; their interaction is what predicts risk.
