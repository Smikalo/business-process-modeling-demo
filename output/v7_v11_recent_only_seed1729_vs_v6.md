# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.3983 | 0.5212 | 4.598 | -0.540 |
| test | V7_base | 0.4218 | 0.5659 | 5.488 | -0.711 |
| val | V7_cal | 0.3989 | 0.5097 | 4.596 | -0.507 |
| test | V7_cal | 0.4198 | 0.5513 | 5.471 | -0.674 |
| val | V7 | 0.3377 | 0.4370 | 4.030 | -0.410 |
| test | V7 | 0.4198 | 0.5812 | 5.153 | -0.441 |
| val | V7_stacked | 0.3814 | 0.5023 | 3.780 | +0.094 |
| test | V7_stacked | 0.4211 | 0.5737 | 4.454 | +0.082 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0283** (V7_stacked 0.4211 vs V6 0.4494)
- MAPE_nz Δ = **-0.0747**
- Bias   V7_stacked +0.082 / V6 +0.413