# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal

Base: pinball α=0.45 on `target_qty_imputed` | Features: V6 + price(7) + cohort(4)

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| val | V7_base | 0.3960 | 0.5210 | 4.567 | -0.533 |
| test | V7_base | 0.4230 | 0.5670 | 5.553 | -0.778 |
| val | V7_cal | 0.3965 | 0.5095 | 4.563 | -0.478 |
| test | V7_cal | 0.4197 | 0.5529 | 5.522 | -0.719 |
| val | V7 | 0.3383 | 0.4394 | 4.052 | -0.385 |
| test | V7 | 0.4191 | 0.5760 | 5.221 | -0.498 |
| val | V7_stacked | 0.3810 | 0.5034 | 3.795 | +0.088 |
| test | V7_stacked | 0.4189 | 0.5722 | 4.476 | +0.039 |

## V7_stacked vs V6 on test

- WAPE   Δ = **-0.0305** (V7_stacked 0.4189 vs V6 0.4494)
- MAPE_nz Δ = **-0.0762**
- Bias   V7_stacked +0.039 / V6 +0.413