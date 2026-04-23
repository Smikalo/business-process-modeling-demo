# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.4200 | 0.5099 | 4.315 | -0.389 |
| test | V7_base | 0.4177 | 0.5310 | 4.954 | -0.409 |
| val | V7_cal | 0.4162 | 0.5201 | 4.299 | -0.433 |
| test | V7_cal | 0.4139 | 0.5401 | 4.936 | -0.454 |
| val | V7 | 0.4162 | 0.5201 | 4.299 | -0.433 |
| test | V7 | 0.4139 | 0.5401 | 4.936 | -0.454 |
| val | V7_stacked | 0.4205 | 0.5558 | 4.044 | +0.102 |
| test | V7_stacked | 0.4329 | 0.5881 | 4.521 | +0.090 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0165** (V7_stacked 0.4329 vs V6 0.4494)
- MAPE_nz Δ = **-0.0603**
- Bias   V7_stacked +0.090 / V6 +0.413