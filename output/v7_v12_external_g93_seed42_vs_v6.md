# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.4117 | 0.5039 | 4.953 | -0.816 |
| test | V7_base | 0.4404 | 0.5465 | 5.956 | -1.054 |
| val | V7_cal | 0.4120 | 0.4913 | 4.949 | -0.777 |
| test | V7_cal | 0.4362 | 0.5306 | 5.915 | -1.003 |
| val | V7 | 0.3401 | 0.4403 | 4.098 | -0.348 |
| test | V7 | 0.4230 | 0.5728 | 5.310 | -0.488 |
| val | V7_stacked | 0.3846 | 0.5094 | 3.814 | +0.050 |
| test | V7_stacked | 0.4182 | 0.5690 | 4.464 | +0.001 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0312** (V7_stacked 0.4182 vs V6 0.4494)
- MAPE_nz Δ = **-0.0794**
- Bias   V7_stacked +0.001 / V6 +0.413