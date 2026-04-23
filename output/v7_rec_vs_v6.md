# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.4252 | 0.5207 | 4.369 | -0.344 |
| test | V7_base | 0.4216 | 0.5383 | 4.984 | -0.421 |
| val | V7_cal | 0.4212 | 0.5317 | 4.349 | -0.379 |
| test | V7_cal | 0.4169 | 0.5477 | 4.955 | -0.457 |
| val | V7 | 0.4212 | 0.5317 | 4.349 | -0.379 |
| test | V7 | 0.4169 | 0.5477 | 4.955 | -0.457 |
| val | V7_stacked | 0.4232 | 0.5678 | 4.057 | +0.121 |
| test | V7_stacked | 0.4388 | 0.6033 | 4.525 | +0.117 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0106** (V7_stacked 0.4388 vs V6 0.4494)
- MAPE_nz Δ = **-0.0451**
- Bias   V7_stacked +0.117 / V6 +0.413