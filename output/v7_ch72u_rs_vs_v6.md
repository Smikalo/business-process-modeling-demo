# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.5554 | 0.4624 | 2.879 | -0.320 |
| test | V7_base | 0.5117 | 0.4704 | 3.562 | -0.589 |
| val | V7_cal | 0.5430 | 0.4973 | 2.831 | -0.357 |
| test | V7_cal | 0.4989 | 0.5078 | 3.498 | -0.618 |
| val | V7 | 0.5430 | 0.4973 | 2.831 | -0.357 |
| test | V7 | 0.4989 | 0.5078 | 3.498 | -0.618 |
| val | V7_stacked | 0.5351 | 0.4821 | 2.614 | -0.356 |
| test | V7_stacked | 0.5077 | 0.5134 | 3.331 | -0.551 |

## V7_stacked vs V6 on test

- WAPE   Δ = **+0.0583** (V7_stacked 0.5077 vs V6 0.4494)
- MAPE_nz Δ = **-0.1350**
- Bias   V7_stacked -0.551 / V6 +0.413