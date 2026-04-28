# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.3809 | 0.4975 | 4.152 | -0.408 |
| test | V7_base | 0.4216 | 0.5983 | 5.088 | -0.397 |
| val | V7_cal | 0.3794 | 0.4984 | 4.149 | -0.424 |
| test | V7_cal | 0.4212 | 0.6017 | 5.094 | -0.412 |
| val | V7 | 0.3289 | 0.4333 | 3.779 | -0.335 |
| test | V7 | 0.4234 | 0.6182 | 4.978 | -0.270 |
| val | V7_stacked | 0.3685 | 0.4877 | 3.721 | +0.119 |
| test | V7_stacked | 0.4244 | 0.5948 | 4.504 | +0.142 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0250** (V7_stacked 0.4244 vs V6 0.4494)
- MAPE_nz Δ = **-0.0536**
- Bias   V7_stacked +0.142 / V6 +0.413