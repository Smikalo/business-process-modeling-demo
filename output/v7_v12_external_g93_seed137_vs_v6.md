# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.4057 | 0.5093 | 4.826 | -0.712 |
| test | V7_base | 0.4414 | 0.5543 | 5.863 | -1.032 |
| val | V7_cal | 0.4059 | 0.4999 | 4.820 | -0.652 |
| test | V7_cal | 0.4383 | 0.5456 | 5.815 | -0.969 |
| val | V7 | 0.3392 | 0.4410 | 4.103 | -0.358 |
| test | V7 | 0.4264 | 0.5798 | 5.308 | -0.563 |
| val | V7_stacked | 0.3808 | 0.5027 | 3.799 | +0.048 |
| test | V7_stacked | 0.4177 | 0.5680 | 4.455 | -0.036 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0317** (V7_stacked 0.4177 vs V6 0.4494)
- MAPE_nz Δ = **-0.0804**
- Bias   V7_stacked -0.036 / V6 +0.413