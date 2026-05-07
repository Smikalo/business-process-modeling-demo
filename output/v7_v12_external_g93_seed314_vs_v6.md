# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.4086 | 0.5014 | 4.864 | -0.777 |
| test | V7_base | 0.4440 | 0.5467 | 6.006 | -1.092 |
| val | V7_cal | 0.4087 | 0.4921 | 4.857 | -0.722 |
| test | V7_cal | 0.4408 | 0.5365 | 5.964 | -1.034 |
| val | V7 | 0.3400 | 0.4406 | 4.121 | -0.362 |
| test | V7 | 0.4285 | 0.5805 | 5.424 | -0.579 |
| val | V7_stacked | 0.3829 | 0.5046 | 3.820 | +0.028 |
| test | V7_stacked | 0.4177 | 0.5671 | 4.504 | -0.051 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0317** (V7_stacked 0.4177 vs V6 0.4494)
- MAPE_nz Δ = **-0.0813**
- Bias   V7_stacked -0.051 / V6 +0.413