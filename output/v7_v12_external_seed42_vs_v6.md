# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.4034 | 0.5092 | 4.758 | -0.656 |
| test | V7_base | 0.4351 | 0.5571 | 5.833 | -0.921 |
| val | V7_cal | 0.4038 | 0.4997 | 4.753 | -0.605 |
| test | V7_cal | 0.4316 | 0.5436 | 5.791 | -0.861 |
| val | V7 | 0.3377 | 0.4400 | 4.040 | -0.331 |
| test | V7 | 0.4213 | 0.5801 | 5.275 | -0.467 |
| val | V7_stacked | 0.3741 | 0.4906 | 3.764 | +0.023 |
| test | V7_stacked | 0.4135 | 0.5620 | 4.477 | -0.047 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0359** (V7_stacked 0.4135 vs V6 0.4494)
- MAPE_nz Δ = **-0.0864**
- Bias   V7_stacked -0.047 / V6 +0.413