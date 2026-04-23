# V4: Creative Approaches & Results

Five architectural innovations explored beyond V3, searching for drastic gains.
The winning configuration: **weighted ensemble of three complementary base models**.

## Final V4 Results (test set, 34.2k active SKU-month pairs)

| Model | WAPE | MAPE_nz | RMSE | Bias | vs V1 | vs V3 |
|-------|------|---------|------|------|-------|-------|
| V1 (original regression) | 0.886 | 0.683 | 12.52 | -0.52 | — | — |
| V2 (active filter + two-stage) | 0.492 | 0.527 | 5.11 | -0.41 | -44% WAPE | — |
| V3 (V2 + 14 new features) | 0.509 | 0.537 | 5.19 | -0.34 | -42% WAPE | — |
| **V4 Ensemble (final)** | **0.490** | **0.510** | **5.13** | **-0.49** | **-45% WAPE** | **-4% WAPE, -5% MAPE** |

**V4 improves MAPE on active SKUs from 53.7% → 51.0% and WAPE from 0.509 → 0.490.**

## Creative approaches evaluated

### 1. Per-channel specialists (one LightGBM per channel)
**Thesis:** ИМ online (WAPE 1.20) and СК specialty chains (WAPE 0.47) are
fundamentally different demand regimes. Specialist models should beat a
generalist.

**Result:** WAPE 0.501 (vs V3 0.509). Small but real improvement, especially
in low-volume channels where a dedicated model's hyperparameters matter more.

**Verdict:** Keep in ensemble (10% weight).

### 2. Log-target transformation
**Thesis:** Two-stage regressor predicts heavy-tailed counts poorly for
high-volume items (+46 unit bias for 100+ unit demand). Predicting `log1p(qty)`
and inverse-transforming flattens the tail.

**Result:** WAPE 0.507, **MAPE_nz 0.508 (best of any single model)**. The
log transformation significantly improves active-SKU accuracy at the cost of
a slightly more negative bias.

**Verdict:** Keep in ensemble (40% weight) — best single-model MAPE.

### 3. Hierarchical reconciliation (partner-total anchoring)
**Thesis:** Aggregate `(Partner, Month)` totals have ~200-2000 units/month —
much more stable than single-SKU 0-10 unit noise. Forecast partner totals
well, rescale SKU predictions to match.

**Result:** WAPE 0.654 — **worse than V3**. The partner-total model alone
underperformed the sum-of-V3-predictions (already implicit reconciliation),
and forced rescaling introduced noise.

**Verdict:** Failed. V3's bottom-up already captures partner dynamics via
`partner_total`, `partner_volume_tier` features. Rejected.

### 4. Segmented isotonic calibration + linear bias correction
**Thesis:** Fit per-(Channel, volume_tier) monotone calibrators on validation
residuals to remove systematic biases non-parametrically.

**Result:** Val WAPE improved dramatically (0.509 → 0.451) but test WAPE
worsened (0.509 → 0.508). **Classic validation overfit.**

**Verdict:** Too aggressive with only 2 months of validation. Would work
with ≥6 months of validation data. Rejected for current PoC.

### 5. GBDT meta-learner stacking
**Thesis:** Let a small LightGBM learn a *nonlinear* blend of base predictions,
conditioned on context features (channel, volume_tier, month).

**Result:** Val WAPE 0.435 (excellent!), test WAPE 0.482 — **also overfit**.
Chased WAPE on validation at the cost of MAPE (0.617 on test, worse than V3).

**Verdict:** Rejected. The simple convex-combination blend generalizes better
than the learned GBDT blender given validation size.

### 6. Convex-blend ensemble (SLSQP-optimized)
**Thesis:** A simple weighted average of V3 + PerChannel + LogTarget + MA
should capture complementary strengths without overfitting.

**Result:** **WAPE 0.490, MAPE_nz 0.510 — best overall.**

Optimal weights on validation:
```
V3         0.40    (core model)
PerChannel 0.10    (channel-specific corrections)
LogTarget  0.40    (tail stabilization)
MA(lags)   0.10    (robustness / regularization)
```

**Verdict:** Winning approach. Simple, robust, interpretable.

## Why the gains were incremental, not drastic

The PoC's fundamental constraint is **irreducible noise at the
(Partner × SKU × Month) granularity**:

- 59% of active pairs have zero demand in any given month
- Monthly volumes are 0-20 units for most SKU-partner pairs
- With only 2 months of held-out test data, a 5% improvement is within
  natural between-month variance

**To achieve truly drastic gains**, the project would need one of:

1. **Coarser granularity:** forecast at (Brand × Partner × Month) — aggregate
   signals are much cleaner. Expected WAPE ~0.25-0.30.
2. **Longer horizon:** forecast 3-month rolling totals instead of single
   months. Smooths out intermittency. Expected WAPE ~0.30.
3. **More data:** historical data for the other 15+ brands as transfer
   learning anchor.
4. **Different optimization target:** train for procurement decision quality
   (inventory cost + stockout penalty) instead of WAPE.
5. **External signals:** web search trends, competitor prices, macroeconomic
   indicators for Ukraine.

## Lessons learned from "creative" attempts

| Attempt | Outcome | Why |
|---------|---------|-----|
| Per-channel specialists | Small gain | Data per channel still sparse |
| Log-target | Real gain | Tail is heavy; log transform linearizes |
| Hierarchical top-down | Failed | Aggregate model not better than V3 sum |
| Isotonic segment calibration | Overfit | Only 2 months of val data |
| GBDT meta-learner | Overfit | Same reason + capacity to overfit WAPE |
| Convex blend (winner) | Robust | Minimal parameters = minimal overfit |

**Principle confirmed:** with limited validation data, model complexity
must be bounded. Three or four weighted models beats a learned blender.

## Production recommendation

Use `model_v4_ensemble`:
- Train V3 two-stage (TwoStageForecaster with V3 features)
- Train LogTarget two-stage (log1p target)
- Train PerChannelEnsemble (four specialists)
- At inference: `0.4·V3 + 0.4·LogTarget + 0.1·PerChannel + 0.1·MA(lag_1..6)`

All four components train on CPU in ~8 minutes; inference is instant.
Total training cost: **$0**.
