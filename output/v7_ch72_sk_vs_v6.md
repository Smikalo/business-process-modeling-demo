# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.3934 | 0.5055 | 5.650 | -0.645 |
| test | V7_base | 0.3779 | 0.5058 | 6.676 | -0.743 |
| val | V7_cal | 0.3911 | 0.5038 | 5.636 | -0.694 |
| test | V7_cal | 0.3758 | 0.5035 | 6.658 | -0.793 |
| val | V7 | 0.3911 | 0.5038 | 5.636 | -0.694 |
| test | V7 | 0.3758 | 0.5035 | 6.658 | -0.793 |
| val | V7_stacked | 0.3826 | 0.5489 | 4.850 | +0.210 |
| test | V7_stacked | 0.4046 | 0.6002 | 5.571 | +0.272 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0448** (V7_stacked 0.4046 vs V6 0.4494)
- MAPE_nz Δ = **-0.0482**
- Bias   V7_stacked +0.272 / V6 +0.413