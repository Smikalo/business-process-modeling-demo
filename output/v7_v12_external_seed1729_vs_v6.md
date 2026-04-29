# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.3998 | 0.5116 | 4.612 | -0.594 |
| test | V7_base | 0.4299 | 0.5592 | 5.670 | -0.815 |
| val | V7_cal | 0.4002 | 0.4986 | 4.610 | -0.559 |
| test | V7_cal | 0.4268 | 0.5422 | 5.643 | -0.771 |
| val | V7 | 0.3375 | 0.4380 | 4.002 | -0.347 |
| test | V7 | 0.4251 | 0.5779 | 5.299 | -0.487 |
| val | V7_stacked | 0.3843 | 0.5054 | 3.801 | +0.064 |
| test | V7_stacked | 0.4199 | 0.5680 | 4.456 | +0.013 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0295** (V7_stacked 0.4199 vs V6 0.4494)
- MAPE_nz Δ = **-0.0804**
- Bias   V7_stacked +0.013 / V6 +0.413