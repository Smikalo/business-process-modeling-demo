# V5 vs V4 — enriched-signals comparison

V5 keeps only sources that passed the decision gate:
`conflict_ua`, `gtrends_ua`, `holidays_ua`, `nbu_fx`, `tmdb_movies`, `world_bank_ua`.

## Metrics

| Split | Model | WAPE | MAPE_nz | RMSE |
|---|---|---:|---:|---:|
| val  | V4 | 0.4869 | 0.5277 | 3.7201 |
| val  | V5 | 0.4722 | 0.5430 | 3.6169 |
| test | V4 | 0.5022 | 0.5322 | 5.0443 |
| test | V5 | 0.5113 | 0.5734 | 5.1460 |

## Deltas (V5 − V4, negative = better)

- val  WAPE Δ = **-0.0147**
- val  MAPE_nz Δ = **+0.0153**
- test WAPE Δ = **+0.0091**
- test MAPE_nz Δ = **+0.0412**

## Features added

V4 = 61, V5 = 94 (+33 external).