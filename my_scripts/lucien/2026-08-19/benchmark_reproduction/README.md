# ActMOF Benchmark Reproduction

This folder reproduces only the original ActMOF benchmark components needed before
adding any new method.

## Original References

- `ActMOF_code_original/construct dataset with time tracker.ipynb`, notebook cell 0:
  full 5D grid, experimental table, RF emulators, Rule 1 / Rule 2 local-neighbor
  corrections, hard post-processing, q calculation, and split `design_grid_part*.csv`
  landscape writing.
- `ActMOF_code_original/Benchmark v3.ipynb`, notebook cell 8:
  `GP_Matern52_PI` method definition and benchmark loop.
- `ActMOF_code_original/bo_benchmark_outputs/bo_benchmark_final_summary.csv`:
  saved reference result for `GP_Matern52_PI`, mean final best q = 187,319.6 over
  5 runs.

## Constants Preserved

- Feature order:
  `metal_amount`, `modulator`, `add_solvent`, `reaction_time`, `reaction_temperature`
- Grid bounds:
  `5..75`, `5..15`, `0..30`, `1..12`, `10..30`, all step 1
- Candidate count: `6,101,172`
- Landscape RFs: `RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)`
- Rule 1: all five feature differences <= 1, blend `80% experimental neighbor mean + 20% RF`
- Rule 2: all differences <= 2 with exactly one or two dimensions at distance 2 and others <= 1,
  blend `50% experimental neighbor mean + 50% RF`
- Hard post-processing:
  intensity `< 3000` gives intensity `0`, FWHM `30`; FWHM `> 3` gives FWHM `30`
- q: rounded `Intensity_final / FWHM_final`, except zero when FWHM is 30 or intensity <= 0
- Benchmark:
  `BATCH_SIZE=3`, `BUDGET_TOTAL_EXPERIMENTS=150`, `INITIAL_EXPERIMENTS=3`,
  `N_REPEATS=5`, `ACQ_CANDIDATES_PER_ITER=15000`, `USE_LOG1P_TARGET=True`,
  `RANDOM_STATE=42`, `DIVERSITY_LAMBDA=0.03`
- `GP_Matern52_PI`:
  scikit-learn GP with `ConstantKernel * Matern(nu=2.5) + WhiteKernel`,
  `alpha=1e-8`, `normalize_y=True`, `n_restarts_optimizer=1`, PI `xi=0.01`

## Workflow

```bash
python my_scripts/lucien/2026-08-19/benchmark_reproduction/reproduce_landscape.py \
  --chunk-size 100000

python my_scripts/lucien/2026-08-19/benchmark_reproduction/validate_reproduction.py

python my_scripts/lucien/2026-08-19/benchmark_reproduction/run_gp_matern52_pi.py \
  --n-repeats 5
```

If the reproduced landscape is not present, the benchmark runner falls back to the
read-only original split CSVs so the active-learning reproduction can still be checked.

The original loader appends all experimental `Q...` rows and then performs
`drop_duplicates(..., keep="last")`. This relocates those rows to the end of the
candidate pool. The reproduction intentionally preserves that seemingly incidental
ordering because the seeded active-learning trajectory indexes into this pool.

## Verified Reproduction (2026-08-19)

The regenerated landscape has 6,101,172 rows and matches the original on all
reported statistics. A fixed-seed sample of 20,000 rows matched exactly as CSV
values for all five features, intensity, FWHM, `q_final`, and rule flag.

- `q_final` range: 0 to 247,107
- zero count: 5,716,993 (93.7031934%)
- 95th / 99th / 99.9th percentiles: 4,171 / 42,477 / 100,352
- top-1% / top-0.1% thresholds: 42,477 / 100,352
- Rule 0 / Rule 1 / Rule 2 counts: 5,998,688 / 16,119 / 86,365

The five `GP_Matern52_PI` final best-q values are 159,714; 150,817; 217,713;
222,233; and 186,121. Their mean is 187,319.6 and sample SD is 32,552.457,
exactly matching the original saved final results (difference from the rounded
187,320 reference: 0.4, or 0.000214%). The complete best-q trajectory and best
pool index match on all 250 saved run/iteration rows.

One secondary trajectory difference remains: run 2 selects a different batch
starting at iteration 32, although its best-q-so-far trajectory and final result
remain identical. This affects selected-set statistics such as mean selected q;
the likely cause is a near-tie resolved differently by GP optimizer/prediction
floating-point behavior in the current environment (Python 3.13.5, NumPy 2.1.3,
SciPy 1.15.3, scikit-learn 1.6.1). Tiny surrogate diagnostic differences before
that branch support this explanation. No benchmark parameters were adjusted.
