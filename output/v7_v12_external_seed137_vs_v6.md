# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.3994 | 0.5093 | 4.666 | -0.660 |
| test | V7_base | 0.4270 | 0.5556 | 5.658 | -0.895 |
| val | V7_cal | 0.3998 | 0.4999 | 4.659 | -0.591 |
| test | V7_cal | 0.4226 | 0.5425 | 5.609 | -0.816 |
| val | V7 | 0.3372 | 0.4398 | 4.014 | -0.340 |
| test | V7 | 0.4166 | 0.5760 | 5.178 | -0.476 |
| val | V7_stacked | 0.3712 | 0.4877 | 3.756 | +0.011 |
| test | V7_stacked | 0.4111 | 0.5613 | 4.478 | -0.059 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0383** (V7_stacked 0.4111 vs V6 0.4494)
- MAPE_nz Δ = **-0.0871**
- Bias   V7_stacked -0.059 / V6 +0.413