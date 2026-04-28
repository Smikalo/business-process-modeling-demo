# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.3945 | 0.5213 | 4.153 | -0.371 |
| test | V7_base | 0.4050 | 0.5429 | 4.854 | -0.452 |
| val | V7_cal | 0.3928 | 0.5247 | 4.146 | -0.387 |
| test | V7_cal | 0.4030 | 0.5466 | 4.843 | -0.463 |
| val | V7 | 0.3403 | 0.4639 | 3.791 | -0.307 |
| test | V7 | 0.4135 | 0.5822 | 4.790 | -0.345 |
| val | V7_stacked | 0.3815 | 0.5070 | 3.775 | +0.090 |
| test | V7_stacked | 0.4270 | 0.5805 | 4.479 | +0.059 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0224** (V7_stacked 0.4270 vs V6 0.4494)
- MAPE_nz Δ = **-0.0679**
- Bias   V7_stacked +0.059 / V6 +0.413