# ADR-002: V4 Ensemble Architecture

**Status:** Accepted
**Date:** 2026-04-22
**Context:** V3 (iterative feature engineering + tuning of the V2 two-stage model) plateaued around WAPE 0.509 / MAPE 0.537. Diminishing returns from additional features or Optuna budget motivated a search for architectural innovations.

## Decision

Ship V4 as a **weighted convex ensemble of three LightGBM-based models plus a lag baseline**, with weights optimized on validation WAPE via SciPy SLSQP.

```
V4_prediction = 0.431 · LogTarget
              + 0.338 · V3
              + 0.140 · PerChannel
              + 0.091 · MA_lag_avg
```

## Rationale

### Options considered

Six architectures were implemented and evaluated head-to-head (see `docs/v4-creative-approaches.md`):

| # | Approach | Verdict |
|---|----------|---------|
| 1 | Per-channel specialists | Kept as component |
| 2 | Log-target regressor | Kept as component |
| 3 | Hierarchical top-down reconciliation | Rejected (WAPE 0.654, worse than V3) |
| 4 | Segmented isotonic calibration | Rejected (overfits 2-month val) |
| 5 | GBDT meta-learner stacking | Rejected (overfits WAPE, destroys MAPE) |
| 6 | **Convex-blend ensemble (SLSQP)** | **Accepted** |

### Why convex-blend won

1. **Minimal parameters.** Only 4 weights constrained to the simplex (3 effective degrees of freedom). Hard to overfit even with 2 months of validation.
2. **Robust to validation noise.** Validation WAPE 0.474 → test WAPE 0.490 (gap +0.016). The learned alternatives (isotonic, GBDT meta) had gaps +0.057 and +0.047 respectively.
3. **Preserves interpretability.** Each weight has a clear interpretation: "trust LogTarget 43% of the time". A GBDT meta-learner is a black box over black boxes.
4. **Fast inference.** 4 booster evaluations + weighted sum; fits in <1 s for 34k rows.
5. **Easy to debug and retrain.** Swap any component, rerun SLSQP on fresh validation — single numeric solver call.

### Why the three chosen base models are complementary

| Model | Strength | Objective | When it wins |
|-------|----------|-----------|--------------|
| V3 (two-stage Tweedie) | General-purpose, well-calibrated | Tweedie log-likelihood | Medium-volume, regular demand |
| LogTarget (two-stage log1p) | Heavy-tail handling | MAE on log-space | High-volume SKUs |
| PerChannel (4 specialists) | Channel-specific dynamics | Tweedie per channel | Channels with unusual seasonality (ИМ, РС) |
| MA_lag_avg | Robustness / regularization | n/a (unparametrized) | SKUs with stable recent demand |

The ensemble leverages these non-overlapping strengths; individually each is slightly worse than the blend.

## Consequences

### Shipped
- `src/model_v4.py` contains production components
- `run_v4_final.py` is the canonical training entrypoint
- `output/model_v4_ensemble.joblib` contains all three LightGBM models + frozen weights + feature column list — single deployable artifact

### Not shipped (explored, archived)
- `src/model_v4_calibration.py` (isotonic + GBDT meta-learner) — kept for reference and potential future use when more validation data is available
- `HierarchicalReconciler` class — retained for potential reuse at coarser granularity

### Operational

- **Retraining cadence:** monthly (matches procurement cycle). Each retrain re-runs SLSQP to update weights — takes <5 s after base models retrain.
- **Drift monitoring:** track WAPE on the most recent month; if >10% degradation vs rolling 3-month baseline, trigger retrain.
- **Rollback:** each base model is independently trained and saved, so rolling back to V3-only requires zeroing the other weights — no retraining needed.

## Alternatives if Data Situation Changes

| If we get... | Consider... |
|--------------|-------------|
| 6+ months of validation data | Re-enable segmented isotonic calibration (Approach 4) |
| 12+ months of validation data | Re-enable GBDT meta-learner (Approach 5) |
| Data for all 20+ brands | Re-evaluate hierarchical reconciliation at brand level |
| External signals (macro, search) | Add as V3 features; blend weights will auto-adjust |

These alternatives are explicitly preserved in code so re-enabling is a configuration change, not new engineering work.
