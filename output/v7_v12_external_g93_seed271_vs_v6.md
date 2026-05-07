# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.4120 | 0.5053 | 4.936 | -0.787 |
| test | V7_base | 0.4405 | 0.5434 | 5.937 | -1.085 |
| val | V7_cal | 0.4120 | 0.4946 | 4.928 | -0.729 |
| test | V7_cal | 0.4363 | 0.5306 | 5.894 | -1.020 |
| val | V7 | 0.3422 | 0.4413 | 4.140 | -0.322 |
| test | V7 | 0.4254 | 0.5691 | 5.379 | -0.557 |
| val | V7_stacked | 0.3898 | 0.5165 | 3.848 | +0.047 |
| test | V7_stacked | 0.4199 | 0.5704 | 4.466 | -0.020 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0295** (V7_stacked 0.4199 vs V6 0.4494)
- MAPE_nz Δ = **-0.0780**
- Bias   V7_stacked -0.020 / V6 +0.413