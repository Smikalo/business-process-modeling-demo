# V5 ensemble (V4 + V5) — tuning results

Best validation blend: **w_V4 = 0.07, w_V5 = 0.93**

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE |
|---|---|---:|---:|---:|
| val  | V4 alone  | 0.4869 | 0.5277 | 3.7201 |
| val  | V5 alone  | 0.4722 | 0.5430 | 3.6169 |
| val  | Ensemble  | 0.4721 | 0.5407 | 3.6120 |
| test | V4 alone  | 0.5022 | 0.5322 | 5.0443 |
| test | V5 alone  | 0.5113 | 0.5734 | 5.1460 |
| test | Ensemble  | 0.5097 | 0.5693 | 5.1251 |

## Deltas (Ensemble vs V4; negative is better)

- val  WAPE Δ = **-0.0148**
- test WAPE Δ = **+0.0075**