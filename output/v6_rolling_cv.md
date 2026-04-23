# Rolling-origin CV summary

ABT: `abt_v6_cached.parquet`  
Target: `target_qty_imputed` | Objective: `pinball` {'alpha': 0.6}

## Per-origin metrics

| origin | n_train | n_test | WAPE | MAPE_nz | RMSE | Bias | sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025-09 | 217,817 | 3,251 | 0.5045 | 0.6234 | 3.15 | +0.61 | 13.2 |
| 2025-10 | 227,528 | 3,346 | 0.4409 | 0.5404 | 3.58 | -0.05 | 10.8 |
| 2025-11 | 241,224 | 3,496 | 0.4235 | 0.5478 | 5.28 | -0.38 | 18.4 |
| 2025-12 | 252,910 | 3,613 | 0.4020 | 0.7403 | 8.79 | +0.85 | 16.8 |
| 2026-01 | 268,309 | 3,779 | 0.4071 | 0.5762 | 3.71 | -0.18 | 33.5 |
| 2026-02 | 288,648 | 4,009 | 0.4231 | 0.4813 | 2.48 | -0.36 | 18.1 |

## Aggregates
- mean WAPE: **0.4335**  (std 0.0341)
- selection score (mean + 0.5σ): **0.4506**
- mean MAPE_nz: 0.5849
- mean RMSE: 4.50
- mean Bias: +0.082 (std 0.476)
