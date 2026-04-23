# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.3927 | 0.5081 | 5.519 | -0.602 |
| test | V7_base | 0.3764 | 0.5080 | 6.480 | -0.686 |
| val | V7_cal | 0.3905 | 0.5060 | 5.501 | -0.641 |
| test | V7_cal | 0.3745 | 0.5058 | 6.462 | -0.729 |
| val | V7 | 0.3905 | 0.5060 | 5.501 | -0.641 |
| test | V7 | 0.3745 | 0.5058 | 6.462 | -0.729 |
| val | V7_stacked | 0.3827 | 0.5485 | 4.851 | +0.213 |
| test | V7_stacked | 0.4052 | 0.5999 | 5.574 | +0.272 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0442** (V7_stacked 0.4052 vs V6 0.4494)
- MAPE_nz Δ = **-0.0485**
- Bias   V7_stacked +0.272 / V6 +0.413