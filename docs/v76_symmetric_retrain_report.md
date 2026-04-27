# V7.6 — Symmetric LightGBM retrain on Kaggle (negative result)

## Goal

The V7.5 final report flagged "symmetric-loss V7 retrain on Linux / Kaggle
GPU" as the most promising single lever remaining after V7.5's LAD
stacker was in place.  On macOS the symmetric retrain had stalled because
of OpenMP/thread-affinity issues, so the experiment was deferred to
Kaggle.

## What was trained

Kaggle kernel: [`mykhailokozyrev/bpm-v76-symmetric`](https://www.kaggle.com/code/mykhailokozyrev/bpm-v76-symmetric)
attached to dataset `mykhailokozyrev/bpm-v6-abt` (v2 — now carries
`abt_v7_cached.parquet`).

Five symmetric regression variants were trained on the full V7 feature
set (112 features), each with the **same** classifier stage, differing
only in the regression objective:

| variant             | LightGBM objective                | fit time |
|---------------------|-----------------------------------|---------:|
| `v7sym_tweedie`     | tweedie, variance_power = 1.3     |   22.6 s |
| `v7sym_tweedie15`   | tweedie, variance_power = 1.5     |   21.1 s |
| `v7sym_mae`         | regression_l1                     |   17.5 s |
| `v7sym_huber`       | huber, alpha = 0.9                |   43.5 s |
| `v7sym_l2`          | regression                        |   14.6 s |

Predictions are persisted as `output/preds_v7sym_<variant>_{val,test}.csv`
and also bundled in `output/v76_kaggle/v7sym_bundle.zip`.

## Standalone scores (held-out test set)

| model              | SIMSCORE |  WAPE  | Agg bias % |
|--------------------|---------:|-------:|-----------:|
| V7.5 LAD (champion)|   0.4875 | 0.4255 |     −1.54  |
| `v7sym_mae`        |   0.4920 | 0.4279 |     −2.23  |
| `v7sym_tweedie15`  |   0.5284 | 0.4473 |     +1.04  |
| `v7sym_tweedie`    |   0.5372 | 0.4520 |     +0.96  |
| `v7sym_huber`      |   0.5652 | 0.4501 |    −10.49  |
| `v7sym_l2`         |   0.6293 | 0.4824 |    +12.46  |

`mae` is the best single symmetric base and nearly matches V7.5's LAD
(which is itself a blend of six LightGBMs); `tweedie` variants have
near-zero aggregate bias but WAPE 5–7 % worse than the stack.

## Did adding them to the LAD pool help?

`scripts/v76_lad_stack.py` re-ran the V7.5 LAD recipe across three pools
× four fit modes (per-channel raw + reconcile at shrink 0.5 / 0.8 / 1.0):

| pool              | rows fed in |
|-------------------|-------------|
| `compact`         | V7.5 pool (v4..v72_champion)                         |
| `sym`             | compact + {tweedie, tweedie15, mae}                  |
| `sym+analytical`  | sym + {ewma6, ewma12, median12, yoyTrend}            |

### OOF CV (3 rolling origins over the validation window)

Top of table (best SIMSCORE first):

```
v76_lad_sym+analytical_reconcile_0.8   OOF=0.4521  in=0.4389  gap=+0.0132  ← CV winner
v76_lad_sym_reconcile_0.5              OOF=0.4534  in=0.4486  gap=+0.0048
v76_lad_sym+analytical_reconcile_0.5   OOF=0.4536  in=0.4457  gap=+0.0079
v76_lad_sym_reconcile_0.8              OOF=0.4546  in=0.4411  gap=+0.0135
v76_lad_compact_reconcile_0.8          OOF=0.4589  in=0.4450  gap=+0.0139  ← V7.5 shape
v76_lad_compact_reconcile_0.5          OOF=0.4596  in=0.4521  gap=+0.0075
```

CV says: adding the symmetric bases lowers OOF SIMSCORE by ≈0.007
(−1.5 %) and reconcile_0.8 stays the best shrink.

### Held-out TEST set (reality check)

Exact same candidates, refit on the full val window, scored on the
untouched test window:

| pool              | per_channel |  rec_0.5 |  rec_0.8 |  rec_1.0 |
|-------------------|------------:|---------:|---------:|---------:|
| compact           |      0.5034 |   0.4932 | **0.4875** |   0.4844 |
| sym               |      0.5069 |   0.4979 |   0.4940 |   0.4980 |
| sym + analytical  |      0.5052 |   0.4967 |   0.4930 |   0.4947 |

The **pool-compact / reconcile_0.8** variant (= V7.5 LAD) remains the
best test SIMSCORE at 0.4875.  Both symmetric-augmented pools degrade
test SIMSCORE by +0.005 to +0.01.  The CV → test gap materialized
exactly at the magnitude the +0.0132 CV gap predicted.

### Diagnosis — why the symmetric bases didn't lift the stack

* `mae` is good enough standalone (0.492) to influence the LAD weights,
  but it is *correlated* with the existing pinball LightGBMs at the row
  level.  It therefore steals weight from v7/v71/v72 without adding
  orthogonal information.  Aggregate bias improves (−0.44 % vs −1.54 %),
  but **monthly WAPE gets worse** (0.1243 vs 0.1085), which dominates
  the SIMSCORE λ-weighted term.
* `tweedie` / `tweedie15` have useful near-zero bias, but their raw
  row-level WAPE is ~5 % worse; once the LAD weight on them is positive,
  they drag up row-level error faster than they fix monthly totals.
* Reconciliation already handles aggregate bias — it does not need
  additional low-bias bases to do its job.

## Decision

**Do not promote V7.6.**  V7.5 remains the production champion
(`v75_lad_compact_reconcile_0.8`, test SIMSCORE 0.4875).

Artefacts kept for posterity:

* `output/preds_v7sym_{tweedie,tweedie15,mae,huber,l2}_{val,test}.csv`
* `output/v76_kaggle/`  (full kernel output dump + `v7sym_metrics.csv`)
* `output/v76/lad_cv.csv`, `output/v76/lad_champion.json`
* `notebooks/v76_symmetric_retrain.ipynb`
* `scripts/v76_lad_stack.py`

## What this rules out

* Adding more LightGBM base learners trained on the *same V7 feature
  set* to the LAD pool.  The stack is saturated — we have six
  LightGBMs + four analytical baselines already, and the new ones
  brought no orthogonal signal.
* Symmetric objectives as a silver bullet.  At V7.5's level the gain
  from switching objective families is ≤ ensemble noise.

## What's now the most promising direction

1. **Feature-engineering fork, not objective fork.**  Retrain a V7
   variant on a *different* feature subspace (e.g., no promo-lifecycle
   features, or cohort-only features) so its errors decorrelate from
   the existing bases.  A base with higher row-level WAPE but lower
   correlation to v7/v71/v72 is more valuable than one with lower WAPE
   and high correlation.
2. **Hierarchical model beyond channel×month** — add a per-brand or
   per-ABC-class reconciliation axis.  Reconciliation is currently the
   single biggest lever (V7.5's 0.4589 CV → 0.4521 would have been a
   win had it generalized; channel×month alone is constrained).
3. **Conformal-aware stacker.**  Instead of point-LAD, fit quantile
   LAD (τ=0.45, τ=0.5, τ=0.55) per channel and blend quantiles with an
   in-sample-calibrated τ.  This explicitly controls the asymmetric
   penalty instead of relying on reconciliation as a post-hoc patch.
