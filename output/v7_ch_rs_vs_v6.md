# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.5616 | 0.4559 | 2.819 | -0.308 |
| test | V7_base | 0.5209 | 0.4601 | 3.576 | -0.593 |
| val | V7_cal | 0.5461 | 0.4863 | 2.766 | -0.352 |
| test | V7_cal | 0.5059 | 0.4927 | 3.512 | -0.631 |
| val | V7 | 0.5461 | 0.4863 | 2.766 | -0.352 |
| test | V7 | 0.5059 | 0.4927 | 3.512 | -0.631 |
| val | V7_stacked | 0.5351 | 0.4821 | 2.614 | -0.356 |
| test | V7_stacked | 0.5078 | 0.5134 | 3.332 | -0.551 |

## V7_stacked vs V6 on test

- WAPE   Δ = **+0.0584** (V7_stacked 0.5078 vs V6 0.4494)
- MAPE_nz Δ = **-0.1350**
- Bias   V7_stacked -0.551 / V6 +0.413