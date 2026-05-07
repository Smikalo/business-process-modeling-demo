# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.4067 | 0.4983 | 4.843 | -0.768 |
| test | V7_base | 0.4433 | 0.5468 | 6.016 | -1.066 |
| val | V7_cal | 0.4066 | 0.4872 | 4.837 | -0.719 |
| test | V7_cal | 0.4398 | 0.5338 | 5.966 | -1.011 |
| val | V7 | 0.3417 | 0.4427 | 4.109 | -0.333 |
| test | V7 | 0.4310 | 0.5775 | 5.458 | -0.576 |
| val | V7_stacked | 0.3848 | 0.5072 | 3.822 | +0.030 |
| test | V7_stacked | 0.4207 | 0.5691 | 4.506 | -0.056 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0287** (V7_stacked 0.4207 vs V6 0.4494)
- MAPE_nz Δ = **-0.0793**
- Bias   V7_stacked -0.056 / V6 +0.413