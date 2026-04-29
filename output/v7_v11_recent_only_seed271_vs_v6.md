# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.3982 | 0.5135 | 4.647 | -0.601 |
| test | V7_base | 0.4314 | 0.5651 | 5.723 | -0.837 |
| val | V7_cal | 0.3983 | 0.5036 | 4.639 | -0.541 |
| test | V7_cal | 0.4288 | 0.5556 | 5.676 | -0.774 |
| val | V7 | 0.3386 | 0.4401 | 4.045 | -0.368 |
| test | V7 | 0.4260 | 0.5832 | 5.319 | -0.476 |
| val | V7_stacked | 0.3842 | 0.5079 | 3.807 | +0.077 |
| test | V7_stacked | 0.4236 | 0.5740 | 4.504 | +0.036 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0258** (V7_stacked 0.4236 vs V6 0.4494)
- MAPE_nz Δ = **-0.0744**
- Bias   V7_stacked +0.036 / V6 +0.413