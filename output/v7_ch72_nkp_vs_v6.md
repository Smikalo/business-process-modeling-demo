# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.4426 | 0.5577 | 2.983 | -0.402 |
| test | V7_base | 0.4440 | 0.5578 | 3.223 | -0.437 |
| val | V7_cal | 0.4422 | 0.5520 | 2.975 | -0.384 |
| test | V7_cal | 0.4420 | 0.5492 | 3.218 | -0.435 |
| val | V7 | 0.4422 | 0.5520 | 2.975 | -0.384 |
| test | V7 | 0.4420 | 0.5492 | 3.218 | -0.435 |
| val | V7_stacked | 0.4337 | 0.5798 | 2.810 | +0.110 |
| test | V7_stacked | 0.4710 | 0.6070 | 3.181 | +0.240 |

## V7_stacked vs V6 on test

- WAPE   Δ = **+0.0216** (V7_stacked 0.4710 vs V6 0.4494)
- MAPE_nz Δ = **-0.0414**
- Bias   V7_stacked +0.240 / V6 +0.413