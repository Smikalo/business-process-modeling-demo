# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.3977 | 0.5117 | 4.562 | -0.619 |
| test | V7_base | 0.4257 | 0.5636 | 5.553 | -0.854 |
| val | V7_cal | 0.3980 | 0.5023 | 4.554 | -0.545 |
| test | V7_cal | 0.4221 | 0.5531 | 5.497 | -0.776 |
| val | V7 | 0.3366 | 0.4439 | 3.979 | -0.327 |
| test | V7 | 0.4204 | 0.5850 | 5.162 | -0.473 |
| val | V7_stacked | 0.3766 | 0.4939 | 3.761 | +0.040 |
| test | V7_stacked | 0.4162 | 0.5630 | 4.450 | -0.034 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0332** (V7_stacked 0.4162 vs V6 0.4494)
- MAPE_nz Δ = **-0.0854**
- Bias   V7_stacked -0.034 / V6 +0.413