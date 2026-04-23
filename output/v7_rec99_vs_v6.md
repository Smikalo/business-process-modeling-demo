# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.4232 | 0.5194 | 4.376 | -0.353 |
| test | V7_base | 0.4260 | 0.5389 | 5.098 | -0.451 |
| val | V7_cal | 0.4200 | 0.5262 | 4.369 | -0.410 |
| test | V7_cal | 0.4222 | 0.5447 | 5.084 | -0.499 |
| val | V7 | 0.4200 | 0.5262 | 4.369 | -0.410 |
| test | V7 | 0.4222 | 0.5447 | 5.084 | -0.499 |
| val | V7_stacked | 0.4247 | 0.5722 | 4.071 | +0.135 |
| test | V7_stacked | 0.4445 | 0.6118 | 4.561 | +0.147 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0049** (V7_stacked 0.4445 vs V6 0.4494)
- MAPE_nz Δ = **-0.0366**
- Bias   V7_stacked +0.147 / V6 +0.413