# Limitations, Assumptions & Next Steps

## Status summary (as of V4)

| Area | V1 | V2 | V3 | V4 |
|------|----|----|----|----|
| Test WAPE | 0.886 | 0.492 | 0.509 | **0.490** |
| Test MAPE (active SKUs) | 0.683 | 0.527 | 0.537 | **0.509** |
| Test RMSE | 12.52 | 5.11 | 5.19 | **5.13** |
| Training time (laptop CPU) | ~40 s | ~70 s | ~90 s | ~3 min |

## Known Limitations

### Data

1. **Extreme sparsity (partially mitigated).** ~91% of (Partner, SKU, Month) cells were zero pre-filtering. V2+ addresses this with **active-pair filtering** (require ≥3 nonzero months in trailing 12), raising the nonzero rate to 59%. Remaining sparsity is inherent to monthly intermittent demand and is the main ceiling on accuracy.
2. **No true stock-out correction.** When demand is zero due to stockout (not lack of demand), we flag it (`stockout_orc`, `stockout_tt`) but don't impute counterfactual demand. Production models need censored demand estimation (e.g., Tobit regression or Kaplan-Meier-style survival adjustment).
3. **Returns as negative values (mitigated).** V2+ clips `target_qty` to ≥0 before training; a secondary `returns` feature could be added if return prediction becomes important.
4. **Dual-agreement partners.** 8 partners have both Выкуп and Комиссионер agreements. Current code keeps only one row per partner (alphabetical last = Комиссионер). Some transactions may be misclassified.
5. **Price files had corrupt xlsx.** Required `calamine` engine fallback. Some price records may be missing if the corruption affected data rows.
6. **Short validation window.** Only 2 months of validation data (Jul 2024 – Jun 2025 split), which limits the complexity of post-hoc calibration methods we can reliably deploy (see V4 isotonic and GBDT meta-learner failures in `docs/v4-creative-approaches.md`).

### Modeling

1. **Addressed in V4:** ~~No category encoding~~ — V2+ uses native LightGBM categoricals for Бренд, Канал, Группа_товара, Сегмент_ABC, Тип_соглашения.
2. **Addressed in V4:** ~~Rolling features approximate~~ — V2+ uses proper expanding-window rolling via `groupby().rolling()` on shifted series.
3. **Addressed in V4:** ~~Heavy-tail under-prediction~~ — V4 blends in a `log1p(qty)` regressor to correct high-volume bias.
4. **Still open: single-step multi-horizon.** h=3 and h=6 forecasts are naive multiples of h=1, not recursive or direct multi-step models.
5. **Still open: small test set.** Only 8 months (Jul 2025 – Feb 2026). The V3→V4 improvement of 3.8% WAPE is within plausible between-month variance; larger held-out windows would strengthen the claim.
6. **Still open: wartime regime shift.** Feb 2022 structural break is flagged but not explicitly modeled (e.g., no regime-switching model, no pre-war/post-war separate models).
7. **Still open: no external regressors.** No macro data (inflation, UAH/USD FX, GDP), no Google Trends, no competitor pricing.

### Infrastructure

1. **No MLOps.** No experiment tracking (MLflow), no model registry, no automated retraining.
2. **No data validation.** No Great Expectations or similar; relies on assertions in code.
3. **No CI/CD.** Manual pipeline execution only.

## Assumptions

- Monthly granularity is sufficient for procurement decisions (confirmed by client: 2-month lead time).
- Выкуп partners' demand ≈ shipment quantity (they buy in bulk, so shipment is a cleaner demand signal than downstream resale).
- Комиссионер partners' demand ≈ retail sales quantity (commission-based; shipment reflects inventory movement, not demand).
- Forward-filling RRP prices is reasonable between price change events.
- Safety-stock formula using q90−q50 spread is appropriate for this demand profile.
- Two months of validation is sufficient for weight selection of a 4-component convex ensemble (validated empirically — see `docs/adr/adr-002`).

## Recommended Next Steps

### Short term (if PoC approved)

1. **Expand test set** — add ≥4 more months of holdout data; re-confirm V3→V4 improvement.
2. **Censored demand estimation** for stockout periods — the largest remaining source of reducible error per error analysis.
3. **Proper multi-step forecasting** (recursive for h=3, direct per-horizon models for h=6) — replace the current naive scaling.
4. **External features v1**: Ukrainian holidays calendar, wartime intensity proxy (e.g., UAF casualty reports), FX rate (USD/UAH).
5. **Expand Optuna budget** to 200+ trials for each base model; currently V4 components use hand-tuned configs because Optuna on V3 triggered a categorical-monotone LightGBM bug (see past conversation).

### Medium term

1. **Add more brands** — currently 3, client has 20+ in their 1C system. More data per category will reduce sparsity.
2. **Google Trends integration** — Ukrainian search volume for toy categories is publicly available; likely a strong leading indicator.
3. **MLflow experiment tracking** — formalize the V1→V2→V3→V4 progression for reproducibility.
4. **Automated monthly retraining** pipeline (Airflow or Prefect).
5. **Re-enable segmented isotonic calibration** (Approach 4 from V4) once 6+ months of validation data accumulates — it showed a clear validation win but overfit a 2-month window.
6. **Re-enable GBDT meta-learner stacking** (Approach 5 from V4) once 12+ months of validation is available.

### Long term

1. **Scale to all 20+ brands.**
2. **Deep learning experiments** (N-BEATS, TFT, Temporal Fusion Transformer) via Kaggle free GPU if tabular models plateau further.
3. **Hierarchical reconciliation at brand/category level** — failed at SKU×Partner granularity in V4 but may work at a coarser level where the aggregate model has more signal.
4. **Decision-metric training objective** — replace WAPE minimization with direct inventory-cost + stockout-penalty optimization. Aligns the model with the business's actual economic trade-offs.
5. **Integration with client's ERP** for automated order generation and closed-loop learning from order acceptance/rejection.
