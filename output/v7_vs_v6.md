# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.4233 | 0.5215 | 4.328 | -0.324 |
| test | V7_base | 0.4250 | 0.5420 | 5.050 | -0.416 |
| val | V7_cal | 0.4200 | 0.5308 | 4.319 | -0.374 |
| test | V7_cal | 0.4208 | 0.5509 | 5.032 | -0.459 |
| val | V7 | 0.4200 | 0.5308 | 4.319 | -0.374 |
| test | V7 | 0.4208 | 0.5509 | 5.032 | -0.459 |
| val | V7_stacked | 0.4249 | 0.5729 | 4.073 | +0.136 |
| test | V7_stacked | 0.4445 | 0.6128 | 4.562 | +0.151 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0049** (V7_stacked 0.4445 vs V6 0.4494)
- MAPE_nz Δ = **-0.0356**
- Bias   V7_stacked +0.151 / V6 +0.413