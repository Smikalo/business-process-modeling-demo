# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.7548 | 0.6315 | 3.771 | -0.030 |
| test | V7_base | 0.9462 | 0.6480 | 2.365 | +0.172 |
| val | V7_cal | 0.7201 | 0.6417 | 3.751 | -0.205 |
| test | V7_cal | 0.8927 | 0.6621 | 2.364 | -0.042 |
| val | V7 | 0.7201 | 0.6417 | 3.751 | -0.205 |
| test | V7 | 0.8927 | 0.6621 | 2.364 | -0.042 |
| val | V7_stacked | 0.7513 | 0.6976 | 3.436 | +0.247 |
| test | V7_stacked | 0.9594 | 0.6941 | 2.433 | +0.252 |

## V7_stacked vs V6 on test

- WAPE   Δ = **+0.5100** (V7_stacked 0.9594 vs V6 0.4494)
- MAPE_nz Δ = **+0.0457**
- Bias   V7_stacked +0.252 / V6 +0.413