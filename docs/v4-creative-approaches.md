# V4: Creative Architectural Approaches & Results

Six architectural innovations explored beyond V3's iterative tuning, searching for drastic gains.
The winning configuration: **weighted convex ensemble of four complementary base models**, with weights optimized via SLSQP on validation WAPE.

## Final V4 Results (test set: Jul 2025 – Feb 2026, 34.2k active SKU-month pairs)

| Model | WAPE | MAPE on active SKUs | RMSE | Bias | vs V1 | vs V3 |
|-------|------|---------------------|------|------|-------|-------|
| V1 (original regression) | 0.886 | 0.683 | 12.52 | -0.52 | — | — |
| V2 (active filter + two-stage) | 0.492 | 0.527 | 5.11 | -0.41 | −44% WAPE | — |
| V3 (V2 + 14 new features) | 0.509 | 0.537 | 5.19 | -0.34 | −43% WAPE | — |
| **V4 Creative Ensemble (final)** | **0.490** | **0.509** | **5.13** | **-0.51** | **−45% WAPE, −25% MAPE** | **−3.8% WAPE, −5.3% MAPE** |

**Final ensemble weights (learned via SLSQP on validation WAPE):**

| Component | Weight |
|-----------|--------|
| LogTarget (two-stage log1p regressor) | **0.431** |
| V3 (two-stage Tweedie backbone) | **0.338** |
| PerChannel (specialist per channel) | **0.140** |
| MA_lag_avg (mean of lag_1, lag_2, lag_3, lag_6) | **0.091** |

## Six Creative Approaches Evaluated

### 1. Per-channel specialists

**Thesis:** ИМ online (V2 WAPE 1.20) and СК specialty chains (WAPE 0.47) are fundamentally different demand regimes. Channel-specific hyperparameters should beat a generalist model.

**Implementation:** Four independent `TwoStageForecaster` instances, one per channel (`src/model_v4.py :: PerChannelEnsemble`). Each trained on its channel's subset of train/val data with its own early-stopping.

**Result:** Test WAPE 0.501 (vs V3 0.509) — modest but real improvement. Best single-model WAPE of any candidate.

**Verdict:** ✅ Kept in ensemble (14% weight).

### 2. Log-target transformation

**Thesis:** The V3 two-stage regressor under-predicts heavy-tailed high-volume items (+46 unit bias for SKUs with 100+ unit demand). Predicting `log1p(qty)` and inverse-transforming with `expm1` linearizes the tail.

**Implementation:** `LogTargetForecaster` in `src/model_v4.py`. Classifier stage unchanged; regressor stage fits `log1p(target_qty)` on positive rows with L1/MAE objective instead of Tweedie.

**Result:** Test WAPE 0.507, **MAPE on active SKUs 0.508 — best single-model MAPE of any candidate**.

**Verdict:** ✅ Kept in ensemble (43% weight, largest contributor).

### 3. Hierarchical reconciliation (partner-total anchoring)

**Thesis:** Aggregate `(Partner, Month)` totals have 200–2000 units/month — an order of magnitude more stable than single-SKU 0–10 unit noise. A strong partner-total forecast combined with V3's SKU-level shape should beat V3 alone.

**Implementation:** `HierarchicalReconciler` in `src/model_v4.py`. Trains a separate LightGBM on aggregated partner-month data, then rescales V3's SKU predictions so each partner-month sum matches the anchor (capped at 0.3×–3× rescaling).

**Result:** Test WAPE **0.654 — substantially worse than V3**.

**Why it failed:** V3 already implicitly learns partner dynamics through `partner_total`, `partner_volume_tier`, and `brand_total` features. The separately-trained partner-total model was noisier than the sum of V3's bottom-up predictions. Reconciliation just added a layer of error.

**Verdict:** ❌ Rejected.

### 4. Segmented isotonic calibration

**Thesis:** Fit per-(Channel, volume_tier) monotone isotonic regressions on validation residuals. Non-parametrically corrects systematic bias without retraining the base models.

**Implementation:** `SegmentedIsotonicCalibrator` in `src/model_v4_calibration.py`. Fits one `sklearn.isotonic.IsotonicRegression` per segment (16 segments with ≥200 rows) plus a global fallback.

**Result:** Validation WAPE improved dramatically (0.509 → **0.451**), but test WAPE worsened (0.509 → 0.508). **Classic validation overfit.**

**Why it failed:** With only 2 months of validation data, the isotonic curves memorized val-specific idiosyncrasies that didn't generalize. Monotonic-per-segment mappings need many samples per segment to estimate robustly.

**Verdict:** ❌ Rejected. Would work with ≥6 months of validation data.

### 5. GBDT meta-learner stacking

**Thesis:** Let a small LightGBM learn a *nonlinear* blend of base predictions, conditioned on context features (channel, volume_tier, month, `rmean_6`). Should capture rules like "when volume_tier=high and channel=ИМ, trust LogTarget over V3".

**Implementation:** `GBDTMetaLearner` in `src/model_v4_calibration.py`. Input = base predictions + context features; trains on first 70% of validation, early-stops on last 30%.

**Result:** Validation WAPE 0.435 (excellent!), test WAPE 0.482 (best test WAPE of any model!) — **but** MAPE on active SKUs blew up to 0.617 (worse than V3) and bias worsened to −0.88.

**Why it failed:** The meta-learner chased WAPE on validation at the expense of MAPE. It predicted systematically low on positive-demand rows to reduce mean absolute error on the many small-value rows — a classic WAPE-MAPE tradeoff under limited data.

**Verdict:** ❌ Rejected. Shows WAPE can be gamed in ways that hurt business utility.

### 6. Convex-blend ensemble (SLSQP)

**Thesis:** A simple weighted average of V3 + LogTarget + PerChannel + MA-baseline captures complementary strengths (V3: general-purpose, LogTarget: tail handling, PerChannel: channel corrections, MA: robustness) without overfitting — only 4 free parameters.

**Implementation:** `optimize_ensemble_weights` in `src/model_v4.py`. Uses SciPy SLSQP with constraints (non-negative weights, sum to 1) and 5 multi-start Dirichlet seeds to escape local minima.

**Result:** **Test WAPE 0.490, MAPE on active SKUs 0.509 — best overall.**

**Verdict:** 🏆 Winner. Simple > learned.

## Cross-Experiment Summary Table

| Approach | Val WAPE | Test WAPE | Test MAPE_nz | Generalization Gap |
|----------|---------:|----------:|-------------:|-------------------:|
| V3 baseline | — | 0.509 | 0.537 | — |
| V4_PerChannel | — | 0.501 | 0.530 | small |
| V4_LogTarget | — | 0.507 | 0.508 | small |
| V4_Reconciled | — | 0.654 | 0.629 | catastrophic |
| V3 + Isotonic | **0.451** | 0.508 | 0.584 | **+0.057 (overfit)** |
| V4_MetaGBDT | **0.435** | 0.482 | 0.617 | **+0.047 (overfit)** |
| **V4 Ensemble** (SLSQP blend) | 0.474 | **0.490** | **0.509** | **+0.016 (robust)** |

**Observation:** The most complex methods had the largest validation-to-test gaps. The simplest blend generalized best.

## Why Gains Were Incremental, Not Drastic

The project's fundamental ceiling is **irreducible noise at the (Partner × SKU × Month) granularity**:

1. **59% of active pairs have zero demand in any given month** (even after aggressive filtering)
2. **Monthly volumes are 0–20 units** for most SKU-partner pairs — below the noise floor of most count models
3. **Only 8 months of test data** — a 5% observed WAPE improvement is within natural between-month variance
4. **Wartime regime shift** (Feb 2022 onward) means some pre-war training data has limited relevance

To achieve *drastic* improvements, the problem formulation itself needs to change:

| Direction | Expected WAPE | Rationale |
|-----------|--------------:|-----------|
| Coarser granularity (Brand × Partner × Month) | ~0.25–0.30 | Aggregate signals are 10× cleaner |
| 3-month rolling horizon instead of monthly points | ~0.30 | Smooths out intermittency |
| Decision-metric objective (inventory cost + stockout penalty) | N/A (different metric) | Aligns with actual business value |
| Transfer learning from the other 15+ brands in 1C | ~0.40 | More data per category |
| External signals (Ukr. search trends, competitor prices, macro) | ~0.42 | Captures exogenous shocks |

## Lessons Learned

1. **Simple > learned under limited validation data.** Two of the most sophisticated approaches (isotonic, GBDT meta-learner) overfit validation and regressed on test. The 4-parameter convex blend beat them both.

2. **WAPE is gameable.** The meta-learner achieved the best test WAPE (0.482) by systematically under-predicting, which destroyed MAPE (0.617). Always report multiple metrics and monitor bias.

3. **Architecture beats features at this stage.** The V3 → V4 jump came from *combining different modeling philosophies* (Tweedie, log-target, per-segment), not from more features. V3 already saturated the signal in the current feature set.

4. **Reconciliation assumes the aggregate model wins.** V3 was already a strong partner-total estimator (through bottom-up sums); a separate aggregate model wasn't strong enough to anchor it usefully.

5. **Channel heterogeneity matters.** Per-channel specialists beat the general model even with less training data per model, confirming the channels have genuinely different demand dynamics.

## Production Recommendation

Use `pipelines/run_v4_final.py` → `output/model_v4_ensemble.joblib`:

- **Training:** V3 two-stage (~30 s) + LogTarget two-stage (~50 s) + PerChannel 4× two-stage (~80 s) = **~3 min total** on a laptop CPU
- **Inference:** instant (4 booster predictions + weighted sum)
- **Artifacts:** single joblib containing all 3 model objects + frozen SLSQP weights + feature columns
- **Cost:** $0
