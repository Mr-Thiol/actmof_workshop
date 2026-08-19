# Experiment A: Fixed Imperfection-Aware Batch Allocation

## Scientific definition

The fixed gate uses the established reliability features from
`my_scripts/lucien/2026-08-17/build_error_dataset.py`:

```text
Suspicious: d_min > 4 AND max(B_I, B_F) > 0
Trustworthy: otherwise
```

`d_min` is raw-unit five-dimensional Chebyshev distance to the nearest of the
95 unique experiments. `B_I` and `B_F` are the five-RF-seed crossing rates
after the original Rule-1/Rule-2 blend and before hard postprocessing:
`I_pre < 3000`, or (when intensity does not cross) `F_pre > 3`.

The 95-row error analysis calculated these features under leave-one-out fits.
For the fixed landscape-wide Experiment-A gate, the exact same definitions are
applied with all 95 experiments retained in each frozen RF fit. The RF ensemble,
gate features, and mask are computed once and never updated during AL.

Supporting references are `feature_glossary.md` (scientific meanings and raw
Chebyshev convention),
`my_scripts/sam/2026-08-17/followup_build_continuous_features.py` (independent
re-simulation check), and
`my_scripts/sam/2026-08-17/plot_interaction_three_views.py` (the latest
`max(B_I, B_F)` interaction analysis). Some historical plots used `d_min >= 4`;
Experiment A intentionally follows its specified strict `d_min > 4` gate.

## Batch policies

```text
Vanilla: 3 from all acquisition candidates.
Proposed: 2 trustworthy + 1 suspicious.
```

Both methods use the same GP, Matérn-5/2 kernel, log1p target, PI values,
15,000-candidate acquisition sample, and diversity-aware greedy score. The gate
only restricts slot eligibility after PI is computed; it never rescales PI.
Each paired run uses the vanilla `GP_Matern52_PI` seed and identical initial
three experiments.

The policy module also includes the future `2 trustworthy + 1 uniform random`
exploration comparator, but Experiment A does not execute it.

## Commands

From the repository root:

```bash
python my_scripts/lucien/2026-08-19/experiment_a/prepare_gate_metadata.py \
  --chunk-size 50000

python my_scripts/lucien/2026-08-19/experiment_a/run_experiment_a.py \
  --batch-size 3 \
  --n-rounds 5 \
  --budget 150 \
  --exploration 1

python my_scripts/lucien/2026-08-19/experiment_a/plot_experiment_a.py
```

Numerical outputs are written to `results/lucien/2026-08-19/experiment_a/`;
plots are written to `figures/lucien/2026-08-19/experiment_a/`.

`--n-rounds` is the number of paired independent runs/repeats. `--budget` is
the total number of experiments in each run, including the three initial
experiments. `--exploration` is the number of suspicious-region, PI-ranked
slots in each gated batch; the other `batch_size - exploration` slots are
trustworthy. Thus the defaults reproduce 2T+1S, while
`--batch-size 5 --exploration 1` runs 4T+1S. A non-divisible budget produces a
smaller final batch.

## First five paired repeats

The fixed gate contains 1,361,004 trustworthy and 4,740,168 suspicious
candidates (77.6927449% suspicious). All 245 gated AL iterations obtained the
requested 2 trustworthy + 1 suspicious allocation; there were zero fallbacks.

Final best q values:

| Run | Vanilla | Gate 2T+1S | Paired difference |
|---:|---:|---:|---:|
| 1 | 159,714 | 155,545 | -4,169 |
| 2 | 150,817 | 151,318 | +501 |
| 3 | 217,713 | 218,787 | +1,074 |
| 4 | 222,233 | 57,722 | -164,511 |
| 5 | 186,121 | 149,913 | -36,208 |

Vanilla: 187,319.6 ± 32,552.5. Gate: 146,657.0 ± 57,496.7. The mean
paired difference (gate minus vanilla) is -40,662.6. Thresholds were not changed
after observing this result.

The vanilla regression check matched `best_q` and `best_pool_idx` on all 250
rows of the validated reproduction. The metadata stores a SHA-256 digest of
the ordered five-feature matrix, which the runner checks against `pool_df` to
guard against silent pool-index misalignment.
