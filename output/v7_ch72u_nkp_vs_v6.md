# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.4460 | 0.5425 | 3.113 | -0.517 |
| test | V7_base | 0.4447 | 0.5482 | 3.310 | -0.514 |
| val | V7_cal | 0.4451 | 0.5387 | 3.103 | -0.492 |
| test | V7_cal | 0.4418 | 0.5428 | 3.301 | -0.505 |
| val | V7 | 0.4451 | 0.5387 | 3.103 | -0.492 |
| test | V7 | 0.4418 | 0.5428 | 3.301 | -0.505 |
| val | V7_stacked | 0.4324 | 0.5785 | 2.789 | +0.099 |
| test | V7_stacked | 0.4693 | 0.6061 | 3.166 | +0.229 |

## V7_stacked vs V6 on test

- WAPE   Δ = **+0.0199** (V7_stacked 0.4693 vs V6 0.4494)
- MAPE_nz Δ = **-0.0423**
- Bias   V7_stacked +0.229 / V6 +0.413