# Demand Forecasting & Procurement Optimization — PoC

SKU-level demand forecasting and automated procurement recommendations for a Ukrainian toy distributor (Djeco, CubicFun, Infantino). Built as a zero-cost Proof of Concept.

## Results

| Model | WAPE | MAPE (active SKUs) | RMSE | Bias |
|-------|------|--------------------|------|------|
| MA-6 baseline | 0.647 | 0.706 | 7.61 | -0.52 |
| V1 LightGBM (regression) | 0.886 | 0.683 | 12.52 | -0.52 |
| V2 Two-Stage (Optuna-tuned) | 0.492 | 0.527 | 5.11 | -0.41 |
| V3 (V2 + 14 new features) | 0.509 | 0.537 | 5.19 | -0.34 |
| **V4 Creative Ensemble** | **0.490** | **0.510** | **5.13** | -0.49 |

**V4 improvement over V1: WAPE -45%, MAPE -25%, RMSE -59%**
**V4 improvement over V3: WAPE -3.8%, MAPE -5.0%, RMSE -1.2%**

### V4 creative approaches explored

Five architectural innovations were tested to push beyond iterative tuning:

1. **Per-channel specialists** (one model per channel ИМ/СК/НКП/РС) — WAPE 0.501 — kept in ensemble
2. **Log-target regressor** (predict log1p(qty) to stabilize heavy tail) — MAPE 0.508 (best single model) — kept in ensemble
3. **Hierarchical reconciliation** (partner-total anchor rescales SKU preds) — failed (WAPE 0.654)
4. **Segmented isotonic calibration** (per-segment monotone recalibration) — overfit val
5. **GBDT meta-learner stacking** (learned nonlinear blend) — overfit val WAPE

**Winner:** SLSQP-optimized convex blend: `0.4·V3 + 0.4·LogTarget + 0.1·PerChannel + 0.1·MA6`.
Simple > learned blender under limited validation data.

See `docs/v4-creative-approaches.md` for full writeup and lessons learned.

- Full V4 pipeline trains in **~8 minutes** on laptop CPU, zero cloud cost.

## Project Structure

```
src/
  config.py        — file paths, period boundaries, split dates
  ingestion.py     — loaders for all .txt and .xlsx sources
  aggregation.py   — monthly aggregation layer
  master.py        — dense skeleton + master DataFrame assembly
  enrichment.py    — nomenclature, partners, prices, promotions join
  features.py      — 41 engineered features (lags, rolling, calendar, stockout, lifecycle, hierarchical)
  evaluation.py    — WAPE/MAPE/RMSE/Bias metrics + temporal split
  model.py         — V1 baselines + LightGBM training
  model_v2.py      — active-pair filtering + TwoStageForecaster (classifier + Tweedie)
  model_v3.py      — +14 features (demand velocity, YoY, volume tiers, lag ranges)
  model_v4.py      — PerChannelEnsemble, LogTargetForecaster, HierarchicalReconciler, SLSQP blender
  model_v4_calibration.py — isotonic + GBDT meta-learner (explored, not in final)
  optimize.py      — Optuna hyperparameter search
  procurement.py   — multi-horizon forecasts + order recommendations

output/
  model_final.joblib              — trained LightGBM model
  optuna_best_params.json         — best hyperparameters
  order_recommendations.csv       — procurement recommendations
  plot_*.png                      — 6 diagnostic charts

docs/
  adr-001-training-architecture.md   — zero-cost design decision
  limitations-and-next-steps.md      — known issues + roadmap
  v4-creative-approaches.md          — creative architectural experiments + results

data/                             — raw client data (not committed)
```

### Top-level scripts

```
run_pipeline.py         — V1 end-to-end pipeline (ingest → V1 model → recommendations)
run_v4_experiments.py   — V4 base models (V3, PerChannel, LogTarget, Reconciled, baselines)
run_v4_round2.py        — post-hoc calibration + meta-learner experiments
run_v4_final.py         — final V4 ensemble production pipeline (recommended)
```

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# V4 final ensemble (recommended, ~8 min on laptop CPU)
PYTHONPATH=. python3 run_v4_final.py

# V1 baseline pipeline for comparison
PYTHONPATH=. python3 run_pipeline.py
```

## Key Design Decisions

- **LightGBM on CPU** — no GPU, no cloud, zero cost. 2.8M rows train in ~40s.
- **Active-pair filtering** (V2+): 82% of zero-noise rows removed, nonzero rate 8% → 59%.
- **Two-stage forecasting** (V2+): classifier (demand > 0?) × Tweedie regressor (how much?).
- **Target clipping** (V2+): negative values (returns) are not forecastable demand — clip to 0.
- **Выкуп vs Комиссионер**: different target definition per agreement type.
- **Convex-blend ensemble** (V4): 3-4 models beat learned GBDT meta-learner under limited val data.
- **Quantile regression** (q50 + q90) for safety stock calculation.
- **Optuna** (30 trials, 12 min): WAPE improved 0.80 → 0.78 on validation.
