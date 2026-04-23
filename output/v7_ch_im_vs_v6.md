# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.7517 | 0.6576 | 3.722 | +0.011 |
| test | V7_base | 0.9426 | 0.6501 | 2.394 | +0.153 |
| val | V7_cal | 0.7180 | 0.6720 | 3.711 | -0.164 |
| test | V7_cal | 0.8855 | 0.6675 | 2.385 | -0.064 |
| val | V7 | 0.7180 | 0.6720 | 3.711 | -0.164 |
| test | V7 | 0.8855 | 0.6675 | 2.385 | -0.064 |
| val | V7_stacked | 0.7643 | 0.7064 | 3.487 | +0.292 |
| test | V7_stacked | 0.9712 | 0.6946 | 2.465 | +0.280 |

## V7_stacked vs V6 on test

- WAPE   Δ = **+0.5218** (V7_stacked 0.9712 vs V6 0.4494)
- MAPE_nz Δ = **+0.0462**
- Bias   V7_stacked +0.280 / V6 +0.413