# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.3976 | 0.5165 | 4.639 | -0.553 |
| test | V7_base | 0.4303 | 0.5678 | 5.705 | -0.814 |
| val | V7_cal | 0.3979 | 0.5081 | 4.634 | -0.499 |
| test | V7_cal | 0.4277 | 0.5602 | 5.664 | -0.757 |
| val | V7 | 0.3394 | 0.4388 | 4.062 | -0.384 |
| test | V7 | 0.4239 | 0.5806 | 5.288 | -0.510 |
| val | V7_stacked | 0.3800 | 0.5024 | 3.787 | +0.083 |
| test | V7_stacked | 0.4190 | 0.5725 | 4.478 | +0.029 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0304** (V7_stacked 0.4190 vs V6 0.4494)
- MAPE_nz Δ = **-0.0759**
- Bias   V7_stacked +0.029 / V6 +0.413