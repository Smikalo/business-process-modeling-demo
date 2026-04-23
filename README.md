# Demand Forecasting & Procurement Optimization — PoC

SKU-level demand forecasting and automated procurement recommendations for a Ukrainian toy distributor (Djeco, CubicFun, Infantino). Built as a zero-cost Proof of Concept — trains entirely on laptop CPU.

## Results (test set: Jul 2025 – Feb 2026, 34.2k active SKU-month pairs)

| Model | WAPE | MAPE on active SKUs | RMSE | Bias |
|-------|------|---------------------|------|------|
| Seasonal Naive (lag-12) | 0.759 | 0.985 | 8.28 | -0.53 |
| MA-lag baseline | 0.647 | 0.706 | 7.61 | -0.52 |
| V1 LightGBM (regression) | 0.886 | 0.683 | 12.52 | -0.52 |
| V2 Two-Stage (Tweedie + active filter) | 0.492 | 0.527 | 5.11 | -0.41 |
| V3 (V2 + 14 new features) | 0.509 | 0.537 | 5.19 | -0.34 |
| **V4 Creative Ensemble** | **0.490** | **0.509** | **5.13** | -0.51 |

**V1 → V4:** WAPE −45%, MAPE −25%, RMSE −59%
**V3 → V4:** WAPE −3.8%, MAPE −5.3%, RMSE −1.2%

### V4 creative approaches explored

Six architectural innovations were tested beyond iterative tuning:

| # | Approach | Test WAPE | Verdict |
|---|----------|-----------|---------|
| 1 | **Per-channel specialists** (one two-stage model per ИМ/СК/НКП/РС) | 0.501 | ✅ Kept (14% weight) |
| 2 | **Log-target regressor** (predict `log1p(qty)` to stabilize heavy tail) | 0.507 | ✅ Kept (43% weight, best single MAPE 0.508) |
| 3 | **Hierarchical reconciliation** (partner-total anchor × SKU share) | 0.654 | ❌ Failed — aggregate model worse than V3 sum |
| 4 | **Segmented isotonic calibration** (per channel × volume_tier monotone) | 0.508 | ❌ Overfit val (val 0.45 → test 0.51) |
| 5 | **GBDT meta-learner stacking** (nonlinear blend) | 0.482 / MAPE 0.617 | ❌ Overfit WAPE, destroyed MAPE |
| 6 | **Convex-blend ensemble (SLSQP)** | **0.490** | 🏆 Winner |

**Winning config:** SLSQP-optimized weights on validation WAPE:
```
0.34·V3 + 0.43·LogTarget + 0.14·PerChannel + 0.09·MA(lags)
```

**Key lesson:** with 2 months of validation data, simple convex blends beat learned blenders. Two of the most sophisticated approaches (isotonic, GBDT meta-learner) looked excellent on validation but regressed on test — a textbook overfitting cautionary tale.

See `docs/v4-creative-approaches.md` for full technical writeup and `docs/adr-002-ensemble-architecture.md` for the architecture decision.

### Compute & cost

| Stage | Time | Cost |
|-------|------|------|
| ABT build (ingest → features → active-pair filter) | ~4 min (first run), cached thereafter | $0 |
| V4 ensemble training (V3 + LogTarget + PerChannel) | ~3 min | $0 |
| V4 ensemble inference (34k rows) | <1 s | $0 |
| **Total V4 pipeline (first run)** | **~7 min** on a laptop CPU | **$0** |

## Project Structure

```
src/
  config.py           — file paths, period boundaries, split dates
  ingestion.py        — loaders for .txt and .xlsx (handles cp1251, calamine)
  aggregation.py      — monthly aggregation layer (clips negatives)
  master.py           — dense (Period, Partner, SKU) skeleton + master assembly
  enrichment.py       — nomenclature, partners, prices, promotions join
  features.py         — 41 core features (lags, rolling, calendar, stockout, lifecycle, hierarchical)
  evaluation.py       — WAPE/MAPE_nz/RMSE/Bias metrics + temporal split
  model.py            — V1 naive baselines + LightGBM regression
  model_v2.py         — active-pair filter, proper rolling, TwoStageForecaster (binary + Tweedie)
  model_v3.py         — +14 features (demand velocity, YoY, volume tiers, lag ranges, trends)
  model_v4.py         — PerChannelEnsemble, LogTargetForecaster, HierarchicalReconciler, SLSQP blender
  model_v4_calibration.py — isotonic + GBDT meta-learner (explored, not shipped)
  optimize.py         — Optuna hyperparameter search
  procurement.py      — multi-horizon forecasts + order recommendations (q50/q90 safety stock)

output/
  abt_v4_cached.parquet          — cached feature-engineered ABT (~10 MB)
  model_v4_ensemble.joblib       — final shippable V4 ensemble
  model_v{2,3}_*.joblib          — per-iteration checkpoints for comparison
  v4_final_metrics.csv           — final test metrics
  v4_final_config.json           — ensemble weights + reproducibility hash
  v4_experiment_results.csv      — Round-1 comparison (all base models)
  v4_round2_results.csv          — Round-2 calibration/meta-learner results
  feature_importance_v4.csv      — top features from V3 backbone
  order_recommendations.csv      — procurement recommendations with safety stock
  plot_*.png                     — diagnostic charts

docs/
  adr-001-training-architecture.md      — zero-cost CPU training decision
  adr-002-ensemble-architecture.md      — V4 convex-blend ensemble decision
  limitations-and-next-steps.md         — known issues + production roadmap
  v4-creative-approaches.md             — full writeup of creative experiments

data/                                   — raw client data (not committed)
```

### Top-level scripts

| Script | Purpose | Runtime (cached) |
|--------|---------|------------------|
| `run_pipeline.py` | Original V1 pipeline (ingest → V1 → recommendations + plots) | ~30 min |
| `run_v4_experiments.py` | Train all V4 base models (V3, PerChannel, LogTarget, Reconciled, baselines) | ~8 min |
| `run_v4_round2.py` | Post-hoc calibration + GBDT meta-learner experiments | ~1 min |
| `run_v4_final.py` | **Production V4 ensemble** (recommended) | ~3 min |

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Production V4 ensemble (recommended)
PYTHONPATH=. python3 run_v4_final.py

# Or: inspect every model side-by-side
PYTHONPATH=. python3 run_v4_experiments.py
```

On first run the ABT is built from raw data (~4 min) and cached to `output/abt_v4_cached.parquet`; subsequent runs load the cache and only retrain models.

## Key Design Decisions

- **LightGBM on CPU** — no GPU, no cloud, zero cost. Trains 2.8M rows in ~40s; V4 full ensemble in ~3 min. See `docs/adr-001`.
- **Active-pair filtering** (V2+): keeps only (Partner, SKU) pairs with ≥3 nonzero months in trailing 12. Removes 82% of rows, raises nonzero rate from 8% → 59%.
- **Two-stage forecasting** (V2+): classifier `P(demand > 0)` × Tweedie regressor `E[qty | demand > 0]`. Right-sized for zero-inflated count data.
- **Target clipping** (V2+): negative values (returns) are not forecastable demand — clipped to zero.
- **Выкуп vs Комиссионер**: different target definition per agreement type (shipment vs retail sale).
- **Convex-blend ensemble** (V4): 3–4 weighted models beat learned GBDT meta-learner under limited validation data. See `docs/adr-002`.
- **Quantile regression** (q50 + q90) for safety stock calculation in procurement module.
- **Optuna** (30 trials): used for V1 hyperparameter search; V4 models use hand-tuned configs.

## Current Limitations

- Only 8 months of held-out test data; 5% WAPE gaps are within natural between-month variance.
- Stock-out periods are flagged but demand is not counterfactually imputed (censored demand estimation not implemented).
- Three brands of interest (Djeco, CubicFun, Infantino); other 15+ brands in client's ERP not yet included.
- No external regressors (macro, Google Trends, competitor prices, wartime intensity).

See `docs/limitations-and-next-steps.md` for the full roadmap.
