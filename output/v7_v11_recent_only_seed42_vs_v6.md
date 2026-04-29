# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.3996 | 0.5160 | 4.696 | -0.567 |
| test | V7_base | 0.4263 | 0.5654 | 5.598 | -0.762 |
| val | V7_cal | 0.4001 | 0.5057 | 4.692 | -0.526 |
| test | V7_cal | 0.4232 | 0.5527 | 5.561 | -0.714 |
| val | V7 | 0.3389 | 0.4372 | 4.091 | -0.377 |
| test | V7 | 0.4237 | 0.5794 | 5.269 | -0.453 |
| val | V7_stacked | 0.3848 | 0.5074 | 3.813 | +0.086 |
| test | V7_stacked | 0.4236 | 0.5737 | 4.490 | +0.058 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0258** (V7_stacked 0.4236 vs V6 0.4494)
- MAPE_nz Δ = **-0.0747**
- Bias   V7_stacked +0.058 / V6 +0.413