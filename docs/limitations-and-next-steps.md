# Limitations, Assumptions & Next Steps

## Known Limitations

### Data

1. **Extreme sparsity**: ~91% of (Partner, SKU, Month) cells are zero. This dilutes aggregate WAPE and makes the model appear worse than it is on active demand.
2. **No true stock-out correction**: When demand is zero due to stockout (not lack of demand), we flag it but don't impute counterfactual demand. Production models need censored demand estimation.
3. **Returns as negative values**: Some rows have negative quantities (returns). Currently included as-is, which may distort lag features for affected SKU-months.
4. **Dual-agreement partners**: 8 partners have both Выкуп and Комиссионер agreements. We keep only one, which may misclassify some transactions.
5. **Price files had corrupt xlsx**: Required calamine engine fallback. Some price records may be missing if the corruption affected data rows.

### Modeling

1. **Single-step multi-horizon**: h=3 and h=6 forecasts are naive multiples of h=1, not recursive or direct multi-step models.
2. **No category encoding**: Бренд, Канал, Группа_товара, Сегмент_ABC are used as enrichment but not as LightGBM categoricals. Adding native categorical support should improve results.
3. **Rolling features approximate**: Used from available lag columns rather than true expanding windows, due to performance constraints on 2.8M rows.
4. **Small test set**: Only 8 months (Jul 2025 – Feb 2026). Results may not generalize. The wartime regime shift (Feb 2022) further complicates temporal stability.
5. **No external regressors**: No macro data (inflation, FX, GDP), no competitor data, no Google Trends.

### Infrastructure

1. **No MLOps**: No experiment tracking (MLflow), no model registry, no automated retraining.
2. **No data validation**: No Great Expectations or similar; relies on assertions in code.
3. **No CI/CD**: Manual pipeline execution only.

## Assumptions

- Monthly granularity is sufficient for procurement decisions (confirmed by client: 2-month lead time).
- Выкуп partners' demand = shipment quantity (they buy in bulk).
- Комиссионер partners' demand = retail sales quantity (commission-based).
- Forward-filling RRP prices is reasonable between price change events.
- Safety stock formula using quantile spread is appropriate for this demand profile.

## Recommended Next Steps (for production)

### Short term (if PoC approved)

1. **Filter to active SKUs only** before training — exclude (Partner, SKU) pairs with <3 non-zero months in last 12. This alone will dramatically improve WAPE.
2. **Add LightGBM categorical features** for Бренд, Канал, Группа_товара.
3. **Implement proper multi-step forecasting** (recursive or DirectMultiOutput).
4. **Censored demand estimation** for stockout periods.
5. **Expand Optuna budget** to 200+ trials.

### Medium term

1. **Add external features**: Ukrainian holidays calendar, wartime intensity proxy, FX rate (USD/UAH).
2. **Hierarchical reconciliation**: Ensure SKU-level forecasts sum to brand totals.
3. **MLflow experiment tracking** for reproducibility.
4. **Automated monthly retraining** pipeline.

### Long term

1. **Scale to all 20+ brands** (currently 3).
2. **Deep learning experiments** (N-BEATS, TFT) via Kaggle free GPU if tabular models plateau.
3. **Integration with client's ERP** for automated order generation.
