# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.4258 | 0.5172 | 4.371 | -0.352 |
| test | V7_base | 0.4196 | 0.5359 | 4.984 | -0.377 |
| val | V7_cal | 0.4220 | 0.5284 | 4.354 | -0.387 |
| test | V7_cal | 0.4153 | 0.5459 | 4.958 | -0.412 |
| val | V7 | 0.4220 | 0.5284 | 4.354 | -0.387 |
| test | V7 | 0.4153 | 0.5459 | 4.958 | -0.412 |
| val | V7_stacked | 0.4255 | 0.5751 | 4.077 | +0.141 |
| test | V7_stacked | 0.4459 | 0.6163 | 4.564 | +0.163 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0035** (V7_stacked 0.4459 vs V6 0.4494)
- MAPE_nz Δ = **-0.0321**
- Bias   V7_stacked +0.163 / V6 +0.413