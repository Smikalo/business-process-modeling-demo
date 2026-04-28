# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.3847 | 0.4995 | 4.224 | -0.430 |
| test | V7_base | 0.4195 | 0.5871 | 5.165 | -0.448 |
| val | V7_cal | 0.3832 | 0.4995 | 4.222 | -0.447 |
| test | V7_cal | 0.4189 | 0.5889 | 5.172 | -0.465 |
| val | V7 | 0.3314 | 0.4381 | 3.818 | -0.306 |
| test | V7 | 0.4238 | 0.6135 | 4.991 | -0.260 |
| val | V7_stacked | 0.3689 | 0.4855 | 3.723 | +0.054 |
| test | V7_stacked | 0.4215 | 0.5841 | 4.487 | +0.070 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0279** (V7_stacked 0.4215 vs V6 0.4494)
- MAPE_nz Δ = **-0.0643**
- Bias   V7_stacked +0.070 / V6 +0.413