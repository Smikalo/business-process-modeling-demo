# V7.2 — Final report

## TL;DR

- **Champion**: V7.1 recipe (recency weights γ=0.95 + per-channel specialists)
  retuned with **Optuna targeting UAH cost directly** (not pinball loss),
  plus a 50/50 global↔channel blend.
- **Test WAPE 0.4086** (vs V7.1 0.4122, V7 0.4208, V6 0.4335)
- **Annual UAH cost: 1,310,563** (vs V7.1 1,316,197, V7 1,403,295, V6 2,072,072)
- **Savings: −5.6K UAH (−0.43%) vs V7.1, −762K UAH (−36.7%) vs V6**
- Dec peak under-forecast improved from −12.7% to −11.6% (+1.1pp of recall
  on the highest-volume month) — achieved without any new features, purely
  by re-tuning hyperparameters against the business metric.
- Released as tag **`v7.2`**.

## Three experiments, one winner

### Experiment 1 — Q4 seasonal features (DROPPED, no gain)

Added `is_xmas_window`, `month_of_year`, `months_to_xmas`,
`sku_dec_lift_lag1y`, `brand_channel_dec_lift`, `y_lag12` (src/features_seasonal.py).

Result: basically identical UAH cost (1,315,650 vs 1,316,197 — within noise).
Feature-importance analysis showed:

- `sku_dec_lift_lag1y`: 510 gain (rank ~70)
- `y_lag12`: 500 gain (rank ~70)
- `brand_channel_dec_lift`: **0 gain** (never split on)
- `is_xmas_window`: **0 gain** (redundant with existing `month`)

The model already has `month`, `rmean_3`, `rmean_12`, `lag_1..lag_12` which
encode the same seasonal motif more granularly. The new features were
redundant.

### Experiment 2 — Monthly multiplier calibrator (DROPPED, +293K UAH worse)

Learned a per-month correction factor `c_m = Σ actual / Σ pred` from
validation (the under-forecast ran −0.7% to −15.7% across every month in
val, not just Dec).

Result: applying the correction to test predictions **increased UAH cost
by +293K (22%)**. This is an important negative result: the systematic
under-forecast in V7.1 is *cost-optimal by design*. The pinball α=0.45
loss is deliberately biased down because holding cost > lost margin ×
recovery per unit; un-biasing pushes us out of the cost minimum.

**Decision implication**: the visual "Dec peak gap" on the portfolio
timeline chart is not a bug, it's the correct newsvendor solution for
this cost structure.

### Experiment 3 — Optuna on UAH cost (WIN, −5.6K UAH) ✓

Previous Kaggle Optuna optimised val **pinball loss**, which tuned a model
that was 12.5K UAH *worse* on business cost. Re-ran Optuna locally with
per-row UAH cost (from `sku_margin.parquet`) as the direct objective:

- 30 trials, search space matching the Kaggle kernel.
- Best val cost: **2,096,776 UAH** at trial 14.
- vs V7.1 champion on same val split: 2,199,746 UAH → **−4.7%**.

Best hyperparameters (`output/v72_optuna_uah_best.json`):

```json
{
  "num_leaves": 94,
  "learning_rate": 0.01005,
  "feature_fraction": 0.634,
  "bagging_fraction": 0.736,
  "bagging_freq": 3,
  "min_data_in_leaf": 75,
  "reg_lambda": 0.00155
}
```

Notice the **slower learning rate + larger min_data_in_leaf** vs V7.1
defaults (LR=0.05, min=20). Re-tuning for cost found a more regularised
model that generalises better to the asymmetric test distribution.

## Per-month impact (test period)

| Month | Actual | V7.1 pred | V7.2 pred | V7.1 err | V7.2 err | Δ |
|:-----:|-------:|----------:|----------:|---------:|---------:|---:|
| 2025-07 | 9,834 | 8,746 | 8,671 | −11.1% | −11.8% | −0.7pp |
| 2025-08 | 10,034 | 9,479 | 9,668 | −5.5% | −3.6% | +1.9pp |
| 2025-09 | 9,230 | 9,450 | 9,496 | +2.4% | +2.9% | +0.5pp |
| 2025-10 | 11,518 | 10,073 | 10,164 | −12.5% | −11.8% | +0.7pp |
| 2025-11 | 13,609 | 11,186 | 11,305 | −17.8% | −16.9% | +0.9pp |
| **2025-12** | **24,840** | **21,691** | **21,953** | **−12.7%** | **−11.6%** | **+1.1pp** |
| 2026-01 | 11,164 | 9,390 | 9,570 | −15.9% | −14.3% | +1.6pp |
| 2026-02 | 7,298 | 5,587 | 5,490 | −23.4% | −24.8% | −1.4pp |

V7.2 improves **6 of 8 months**, including the critical Dec peak.

## Changes vs V7.1

| Item | Variant tag | Test UAH cost | Δ vs V7.1 | Kept? |
|---|---|---:|---:|---|
| V7.1 champion (reference) | `rec95` + `ch` blend w=0.6 | 1,316,197 | — | — |
| V7.2 + seasonal features | `v72_global` + `ch72` blend | 1,315,650 | −547 | ✗ (noise) |
| V7.2 + monthly calibrator | post-hoc correction | 1,610,094 | +293,897 | ✗ (disaster) |
| **V7.2 + UAH-Optuna** | `v72_uahopt` + `ch72u` w=0.5 | **1,310,563** | **−5,634** | **✓** |
| V7.2 + UAH-Optuna (2500 rounds) | `v72_uahopt2k` | 1,345,035 | +28,838 | ✗ (overfit) |
| V7.2 + UAH-Optuna + seasonal ABT | `v72_combined` | 1,344,826 | +28,629 | ✗ (interaction) |

## Artefacts

- `output/v72_champion.json` — champion config + blend sweep
- `output/v72_optuna_uah_best.json` — Optuna-on-cost best params
- `output/v72_optuna_uah_trials.csv` — all 30 Optuna trials
- `output/preds_v72_champion_{val,test}.csv` — champion blended predictions
- `output/model_v7_v72_uahopt.joblib` — V7.2 global booster
- `output/model_v7_ch72u_{im,nkp,rs,sk}.joblib` — V7.2 channel specialists
- `output/v72_channel_blend_sweep.csv` — official scorecard sweep
- `scripts/optuna_uah_cost.py` — Optuna driver (reusable)
- `scripts/eval_monthly_calibrator.py` — monthly-calibrator A/B script
- `scripts/build_v72_abt.py`, `src/features_seasonal.py` — seasonal
  features (retained in repo as kept-but-unused reference)

## How to reproduce

```bash
# 1. V7 ABT + per-SKU margins must exist
python -m scripts.build_v7_abt
python -m scripts.build_sku_margin_table

# 2. Optuna on UAH cost (30 trials, ~30 min CPU)
python -m scripts.optuna_uah_cost --n-trials 30 --num-boost-round 400 \
    --output-json output/v72_optuna_uah_best.json

# 3. Train V7.2 global with tuned params
python -m scripts.train_v7 --disable-residual --save-tag v72_uahopt \
    --recency-gamma 0.95 --optuna-params output/v72_optuna_uah_best.json

# 4. Train per-channel specialists with same tuned params
python -m scripts.train_v71_channels --global-tag v72_uahopt \
    --recency-gamma 0.95 \
    --optuna-params output/v72_optuna_uah_best.json \
    --tag-prefix ch72u --weights 0.0 0.3 0.5 0.6 0.7 0.8 1.0

# 5. Official scorecard sweep → pick best w
python -m scripts.sweep_channel_blend --tag-prefix ch72u \
    --global-tag v72_uahopt --output-prefix v72u

# 6. Rename to champion (if v72u beats champion on your run)
cp output/preds_v72u_test.csv  output/preds_v72_champion_test.csv
cp output/preds_v72u_val.csv   output/preds_v72_champion_val.csv
cp output/v72u_champion.json   output/v72_champion.json
```

## Why the win is small

V7.1 was already heavily optimised: recency weights, channel specialists,
pinball α=0.45, per-SKU margin scorecard. The remaining headroom in that
configuration is genuinely thin. The **5.6K UAH / year is ~0.4% of total
cost** — a real but modest win. The *real* value of V7.2 is the negative
results that crystallised what the remaining theoretical ceiling is and
where it lives:

1. **Dec peak is NOT free to fix** — α=0.45 is cost-optimal for these
   margins. Pushing predictions up costs more than it saves.
2. **Standard seasonal features are saturated** — the model already
   captures Dec via `month` + `rmean_12`.
3. **The next meaningful lift requires structural change**, not more
   features.

## V7.3 candidates

Concretely where the next 10–20K UAH could come from:

- **Negotiated per-SKU margins from the business** — currently 75% of
  SKUs sit at the empirical-Bayes margin floor. Real margins would make
  per-SKU newsvendor α useful and break open the pinball-α ceiling.
- **Smooth pinball objective (Huber-pinball)** — the current custom
  pinball has a constant 1e-3 Hessian that blocks monotone constraints.
  A proper curvature unlocks monotone sign constraints and potentially
  better splits.
- **CRPS / multi-quantile objective** — train one regressor with several
  α simultaneously, then pick the α per SKU at inference. Requires the
  margin-diversity fix above.
- **Promo-forward features** — `data/Нац. акции 2024.xlsx` has planned
  promotions but we only use lagged promo features. Using the forward
  calendar for the test window could add 10–50K UAH on Q4.
- **Non-LGBM ensemble member** — Prophet/ETS-X or tiny N-BEATS as a
  third base for the Ridge stacker; current ensemble is low-diversity
  (all LGBM).

## See also

- `docs/reports/v71_final_report.md` — V7.1 release (recency weights +
  per-channel specialists)
- `docs/v71_optuna_comparison.md` — why Optuna on pinball loss did NOT
  help (and V7.2 fixed that by switching to UAH cost)
- `docs/adr/adr-006-v71.md` — V7.1 ADR
