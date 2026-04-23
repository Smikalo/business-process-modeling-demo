# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.4247 | 0.5081 | 4.484 | -0.449 |
| test | V7_base | 0.4162 | 0.5275 | 5.078 | -0.467 |
| val | V7_cal | 0.4211 | 0.5189 | 4.467 | -0.490 |
| test | V7_cal | 0.4125 | 0.5366 | 5.059 | -0.513 |
| val | V7 | 0.4211 | 0.5189 | 4.467 | -0.490 |
| test | V7 | 0.4125 | 0.5366 | 5.059 | -0.513 |
| val | V7_stacked | 0.4258 | 0.5797 | 4.097 | +0.171 |
| test | V7_stacked | 0.4442 | 0.6197 | 4.549 | +0.177 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0052** (V7_stacked 0.4442 vs V6 0.4494)
- MAPE_nz Δ = **-0.0287**
- Bias   V7_stacked +0.177 / V6 +0.413