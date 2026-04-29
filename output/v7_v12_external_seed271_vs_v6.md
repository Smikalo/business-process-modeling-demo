# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.4037 | 0.5103 | 4.745 | -0.672 |
| test | V7_base | 0.4300 | 0.5587 | 5.677 | -0.916 |
| val | V7_cal | 0.4039 | 0.4999 | 4.740 | -0.617 |
| test | V7_cal | 0.4264 | 0.5459 | 5.640 | -0.855 |
| val | V7 | 0.3392 | 0.4405 | 4.063 | -0.349 |
| test | V7 | 0.4206 | 0.5785 | 5.249 | -0.536 |
| val | V7_stacked | 0.3770 | 0.4973 | 3.776 | +0.039 |
| test | V7_stacked | 0.4147 | 0.5654 | 4.454 | -0.046 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0347** (V7_stacked 0.4147 vs V6 0.4494)
- MAPE_nz Δ = **-0.0830**
- Bias   V7_stacked -0.046 / V6 +0.413