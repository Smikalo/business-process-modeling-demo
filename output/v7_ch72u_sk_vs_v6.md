# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.3923 | 0.5040 | 5.658 | -0.628 |
| test | V7_base | 0.3728 | 0.5095 | 6.521 | -0.613 |
| val | V7_cal | 0.3905 | 0.5037 | 5.641 | -0.665 |
| test | V7_cal | 0.3709 | 0.5081 | 6.503 | -0.655 |
| val | V7 | 0.3905 | 0.5037 | 5.641 | -0.665 |
| test | V7 | 0.3709 | 0.5081 | 6.503 | -0.655 |
| val | V7_stacked | 0.3836 | 0.5499 | 4.861 | +0.239 |
| test | V7_stacked | 0.4070 | 0.6001 | 5.574 | +0.275 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0424** (V7_stacked 0.4070 vs V6 0.4494)
- MAPE_nz Δ = **-0.0483**
- Bias   V7_stacked +0.275 / V6 +0.413