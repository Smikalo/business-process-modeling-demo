# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.3949 | 0.5150 | 4.187 | -0.401 |
| test | V7_base | 0.4102 | 0.5413 | 5.002 | -0.458 |
| val | V7_cal | 0.3932 | 0.5181 | 4.182 | -0.419 |
| test | V7_cal | 0.4083 | 0.5443 | 4.994 | -0.469 |
| val | V7 | 0.3932 | 0.5181 | 4.182 | -0.419 |
| test | V7 | 0.4083 | 0.5443 | 4.994 | -0.469 |
| val | V7_stacked | 0.4041 | 0.5367 | 3.946 | +0.059 |
| test | V7_stacked | 0.4242 | 0.5695 | 4.555 | +0.018 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0252** (V7_stacked 0.4242 vs V6 0.4494)
- MAPE_nz Δ = **-0.0789**
- Bias   V7_stacked +0.018 / V6 +0.413