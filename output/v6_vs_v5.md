# V6 vs V5 — imputation + promo-lifecycle + cost-calibrated loss

Objective: `pinball` {'alpha': 0.6}  | Target: `target_qty_imputed`

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val  | V5 | 0.4722 | 0.5430 | 3.6169 | — |
| val  | V6 | 0.4400 | 0.6004 | 4.2010 | +0.338 |
| test | V5 | 0.5113 | 0.5734 | 5.1460 | — |
| test | V6 | 0.4494 | 0.6484 | 4.7599 | +0.413 |

## Deltas (V6 − V5, negative = better)

- val  WAPE Δ = **-0.0322**
- test WAPE Δ = **-0.0619**
- test MAPE_nz Δ = **+0.0750**

## Features added in V6

`was_censored`, `promo_duration_months`, `promo_depth_pct_current`, `months_since_last_promo`, `months_until_next_promo`, `post_promo_depletion_flag`, `sku_promo_sensitivity`.