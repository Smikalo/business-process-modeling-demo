# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.4267 | 0.5125 | 4.412 | -0.403 |
| test | V7_base | 0.4236 | 0.5329 | 5.036 | -0.480 |
| val | V7_cal | 0.4226 | 0.5220 | 4.394 | -0.449 |
| test | V7_cal | 0.4193 | 0.5399 | 5.016 | -0.528 |
| val | V7 | 0.4226 | 0.5220 | 4.394 | -0.449 |
| test | V7 | 0.4193 | 0.5399 | 5.016 | -0.528 |
| val | V7_stacked | 0.4232 | 0.5656 | 4.061 | +0.128 |
| test | V7_stacked | 0.4377 | 0.6001 | 4.521 | +0.115 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0117** (V7_stacked 0.4377 vs V6 0.4494)
- MAPE_nz Δ = **-0.0483**
- Bias   V7_stacked +0.115 / V6 +0.413