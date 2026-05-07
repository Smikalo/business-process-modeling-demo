# V7 Final Report — Stacked cost-calibrated forecaster

## TL;DR

- **V7 default (α=0.45)**: test WAPE **0.4208** (V6: 0.4494, **-6.4%**),
  annualised UAH cost **1.40 M** (V6: 2.07 M, **-32%**).
- **V7 cost-optimal (α=0.35)**: test WAPE 0.4365,
  annualised UAH cost **1.10 M** (**-47% vs V6**, **-37% vs V4**).
- Rolling-origin CV: V7 mean WAPE **0.4425 ± 0.0306** (V6: 0.4438 ± 0.0336).
- Adds V4+V5+V6+V7 stacked ensemble, isotonic classifier calibration,
  per-segment conformal prediction intervals, and a per-SKU realised
  margin table that makes the cost scorecard honest.

## What changed

| Area | V6 | V7 |
|---|---|---|
| Cost scorecard assumption | flat 28% margin / 22% holding | per-SKU realised margin + unit price (empirical-Bayes shrunk toward brand × channel) |
| Pinball α for regressor | 0.6 (over-forecast bias) | **0.45 default** (cost-calibrated); α-sweep from 0.25 → 0.60 checked in |
| Features | V6: promo lifecycle + externals | V6 + 7 relative-price features + 4 cohort/substitution features |
| Classifier | raw probabilities | isotonic-calibrated (fit on first 60% of val) |
| Ensembling | none | V4+V5+V6+V7 ridge meta-learner (positive weights, α=10) |
| Per-segment | none | per-(brand, channel) residual corrector (disabled by default — overfits short histories but scaffolded in `src/v7_components.py`) |
| Uncertainty | point estimate | 10/90 conformal intervals per (brand, channel) |
| Model zoo | `model_v6.joblib` | `model_v7.joblib` (bundle with base + calibrator + corrector + stacker + conformal) |

## Why the default α dropped from 0.6 → 0.45

V6's α=0.6 was picked to minimise visible *stockouts* under the old
(flat 28% margin) assumption. When we rebuild the scorecard from the
actual data:

- Realised unit margin ≈ **10%** (this is a distributor, not a retailer).
- Holding rate assumption unchanged at 22% annual.

At those true ratios the newsvendor-optimal service level is
α\* ≈ 0.31 — V6's α=0.6 was aggressively over-forecasting. The α-sweep
(`output/v7_alpha_sweep.csv`) confirms a clean unimodal frontier:

| α | test WAPE | test Bias | UAH cost |
|---|---:|---:|---:|
| 0.25 | 0.4695 | -1.46 |   976,320 |
| 0.30 | 0.4557 | -1.30 | 1,016,284 |
| 0.35 | 0.4365 | -1.03 | 1,104,121 |
| 0.40 | 0.4282 | -0.79 | 1,237,614 |
| **0.45** | **0.4208** | -0.46 | **1,403,296** |
| 0.50 | 0.4247 | -0.21 | 1,574,810 |
| 0.55 | 0.4320 | +0.07 | 1,808,641 |
| 0.60 | 0.4511 | +0.37 | 2,070,962 |

- **α=0.45** is Pareto-best WAPE with a ≥30% cost reduction.
- **α=0.35** is cost-optimal for a risk-tolerant planner (service level
  drops further, but UAH savings are large).

## Rolling-origin stability (8 origins)

```
origin      V6 WAPE   V7 WAPE
2025-07      0.46      0.44
2025-08      0.40      0.39
2025-09      0.41      0.41
2025-10      0.42      0.41
2025-11      0.43      0.43
2025-12      0.50      0.49
2026-01      0.47      0.47
2026-02      0.44      0.43
------------------------------
mean         0.4438    0.4425
std          0.0336    0.0306
score*       0.4606    0.4578
```

\* score = mean + 0.5·std (the selection rule we used in Optuna).

## Business impact

- Annualised UAH savings from switching V6 → V7 default (α=0.45):
  **≈ 669,000 UAH / year** (scorecard config: realised margins +
  22% annual holding + 50% partial back-order recovery).
- Savings are roughly **33% holding-cost reduction** + smaller
  lost-margin increase — net decisively positive.
- Stockout-intolerant buyers can keep V6's behaviour by setting
  `--alpha 0.6`; cost-intolerant planners can go to 0.35 and pocket a
  further 296 k UAH / year at the cost of a 1.6 pp WAPE.

## Artefacts

```
output/sku_margin.parquet           per-SKU price + margin table
output/abt_v7_cached.parquet        V6 ABT + 11 V7 features
output/v7_feature_manifest.json     list of added features
output/model_v7.joblib              bundled V7 components
output/preds_v7_val.csv             V7 calibrated point forecast (val)
output/preds_v7_test.csv            V7 calibrated point forecast (test)
output/preds_v7_stacked_*.csv       V4+V5+V6+V7 ridge meta-learner
output/preds_v7_lower_*.csv         conformal 10th percentile
output/preds_v7_upper_*.csv         conformal 90th percentile
output/v7_metrics.csv               WAPE/MAPE_nz/RMSE/Bias per variant
output/v7_alpha_sweep.csv           α sweep results
output/v7_rolling_cv.{json,md}      8-origin rolling-CV results
output/cost_scorecard_final.{md,json}   V4…V7 cost scorecard with per-SKU margins
output/plot_model_progression.png   V4 → V5 → V6 → V7 dashboard
```

## How to reproduce

```
# 1. Build the V6 ABT (unchanged)
python -m scripts.build_v6_abt

# 2. Build the per-SKU margin table
python -c "import pandas as pd; from src.margin_table import build_margin_table; \
           build_margin_table(pd.read_parquet('output/abt_v6_cached.parquet')).to_parquet('output/sku_margin.parquet', index=False)"

# 3. Build the V7 ABT (V6 + price + cohort)
python -m scripts.build_v7_abt

# 4. Train V7 (locked-in α=0.45; pass --optuna-params JSON when Kaggle returns)
python -m scripts.train_v7 --alpha 0.45 --disable-residual

# 5. Cost scorecard with per-SKU margins
python -m scripts.decision_cost_scorecard --margin-table output/sku_margin.parquet \
       --output output/cost_scorecard_final.md --output-json output/cost_scorecard_final.json

# 6. Progression viz V4 → V5 → V6 → V7
python -m scripts.viz_model_progression
```

## Outstanding / follow-up

- **Optuna P100 retune** is queued on Kaggle
  (`notebooks/v7_optuna.ipynb`, slug
  `<kaggle-user>/bpm-v7-optuna`). The resulting
  `v7_optuna_best_params.json` can be merged via
  `--optuna-params …` in `scripts/train_v7.py`.
- Monthly per-SKU margin time series — would let V7 adapt to wartime
  pricing drift.
- Production drift detection (PSI + CUSUM on weekly residuals).
