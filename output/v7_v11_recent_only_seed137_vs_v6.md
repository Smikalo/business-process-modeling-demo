# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.3986 | 0.5140 | 4.649 | -0.601 |
| test | V7_base | 0.4334 | 0.5836 | 5.677 | -0.785 |
| val | V7_cal | 0.3989 | 0.5068 | 4.640 | -0.533 |
| test | V7_cal | 0.4317 | 0.5795 | 5.634 | -0.719 |
| val | V7 | 0.3388 | 0.4404 | 4.041 | -0.365 |
| test | V7 | 0.4257 | 0.6002 | 5.271 | -0.449 |
| val | V7_stacked | 0.3738 | 0.4897 | 3.751 | +0.052 |
| test | V7_stacked | 0.4193 | 0.5735 | 4.511 | +0.007 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0301** (V7_stacked 0.4193 vs V6 0.4494)
- MAPE_nz Δ = **-0.0749**
- Bias   V7_stacked +0.007 / V6 +0.413