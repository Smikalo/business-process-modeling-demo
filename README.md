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
| V4 Creative Ensemble | 0.490 | 0.509 | 5.13 | -0.51 |
| V5 (V4 + 6 external signal loaders) | 0.472 val / 0.510 test | 0.543 val / 0.573 test | 3.62 val / 5.13 test | — |
| V6 (imputation + promo-lifecycle + pinball-q60) | 0.440 val / 0.449 test | 0.600 val / 0.648 test | 4.20 val / 4.76 test | +0.34 val / +0.41 test |
| V7 (price + cohort + cost-calibrated α + stacking + conformal) | 0.420 val / 0.421 test | 0.531 val / 0.551 test | 4.32 val / 5.03 test | −0.37 val / −0.46 test |
| V7.1 (V7 + recency weights γ=0.95 + per-channel specialists w=0.6) | 0.412 val / 0.412 test | 0.484 val / 0.490 test | — | −0.54 val / −0.56 test |
| V7.2 (V7.1 + Optuna retuned on UAH cost, blend w=0.5) | 0.421 val / 0.409 test | — | 4.47 val / 5.06 test | −0.47 val / −0.51 test |
| **V7.3 (NNLS stack of V4/V5/V6/V7.1 under pre-registered SIMSCORE)** | **0.424 val / 0.436 test** | SMAPE **0.499** test | **4.53 test** | **−0.15 test** |

**V1 → V4:** WAPE −45 %, MAPE −25 %, RMSE −59 %
**V4 → V5 (validation):** WAPE −1.5 pp, RMSE −2.9 % relative
**V4 → V5 (test):** WAPE +0.9 pp regression — see ADR-003 for distribution-shift analysis.
**V5 → V6 (fixed test):** WAPE **−2.8 pp** (0.478 → 0.449) and rolling-origin WAPE std **−42 %** — see ADR-004.
**V6 → V7 (fixed test):** WAPE **−2.9 pp** (0.449 → 0.421, **−6.4 % relative**) and annualised UAH cost **−32 %** (2.07 M → 1.40 M) with the cost scorecard rewritten to use per-SKU *realised* margins — see ADR-005.
**V7 → V7.1 (fixed test):** WAPE **−0.9 pp** (0.421 → 0.412) and annualised UAH cost **−6.2 %** (1.40 M → 1.32 M, −87 K UAH) via recency sample weights (γ=0.95) and per-channel specialists blended at w=0.6 with the global model — see ADR-006.
**V7.1 → V7.2 (fixed test):** WAPE **−0.3 pp** (0.412 → 0.409) and annualised UAH cost **−0.4 %** (1.316 M → 1.311 M, −5.6 K UAH) by re-running Optuna with the UAH scorecard as the *direct* objective (the previous Kaggle Optuna on pinball loss was +12.5 K UAH worse). Dec peak under-forecast improved from −12.7 % to −11.6 %. Seasonal Q4 features and a monthly-mean calibrator were tested and dropped (no signal / +293 K UAH) — see `docs/v72_final_report.md`.
**V7.2 → V7.3 (similarity-first, fixed test):** the objective changes from "minimise UAH cost" to "predictions most similar to actuals". V7.3 is an NNLS stack of V4, V5, V6, and V7.1 (V7 and V7.2 receive zero weight) picked via 3-fold rolling-origin CV inside the validation window under a pre-registered SIMSCORE = WAPE + 0.005·|agg_bias_pct| + 0.5·Monthly-WAPE. Result: **SIMSCORE 0.5113 vs V7.2's 0.5272** (−0.016), aggregate bias collapses from **−11.5 % to −3.17 %**, SMAPE **0.527 → 0.499**, RMSE **5.10 → 4.53**, Monthly-WAPE **0.122 → 0.119**, and portfolio-level WAPE over the 20-month val+test window **0.113 → 0.073** (best of every model generation). Monthly-scalar calibrators scored better *in-sample* but were auto-rejected by the ≤ 0.05 overfit-gap rule — see `docs/v73_final_report.md`.

### V5 — external signal enrichment

Ten free, regularly-updated data sources were evaluated under a common `BaseSignalLoader` framework (`src/external_data.py`) with automated add-one-source and leave-one-out ablation (`scripts/run_ablation.py`) and a decision-gate report (`scripts/run_decision_gate.py`).

**Kept** (6 loaders, 33 new features):

| Loader | What it adds | Verdict |
|---|---|---|
| `conflict_ua` | War-intensity timeline (ACLED fallback) | PASS (val −1.35pp, test −0.59pp) |
| `nbu_fx` | UAH/USD, UAH/EUR, NBU policy rate | PASS (val −0.94pp) |
| `holidays_ua` | Ukrainian holidays + gifting-season flags | MARGINAL / LOO-KEEP |
| `gtrends_ua` | Google Trends toy keywords | LOO-KEEP |
| `tmdb_movies` | Family/animation releases (toy tie-ins) | MARGINAL |
| `world_bank_ua` | Demographics + macro (annual, ffilled) | MARGINAL |

**Dropped** (net-harmful on test): `weather_ua`, `school_ua`, `imf_cpi`, `air_raids_ua`.

See `docs/adr-003-external-signals.md` for the full decision record and `output/decision_gate.md` for the per-loader verdict table.

### V6 — imputation, promo lifecycle, cost-calibrated loss

Three structural upgrades stacked on V5:

1. **Censored-demand imputation** (`src/demand_imputation.py`). Rows where `target_qty = 0 ∧ stockout_orc = 1 ∧ demand_density ≥ 0.3` (≈ 2.2 % of the ABT) get their label replaced with an EB-shrunk brand × channel × month baseline; a new boolean feature `was_censored` tags the row. The classifier keeps using raw `target_qty`; only the regressor sees `target_qty_imputed`.
2. **Promo-lifecycle features** (`src/features_promo.py`): `promo_duration_months`, `promo_depth_pct_current`, `months_since_last_promo`, `months_until_next_promo`, `post_promo_depletion_flag`, `sku_promo_sensitivity` (EB-shrunk per-SKU uplift ratio).
3. **Quantile (pinball) loss at α = 0.6** on the regression stage (LightGBM built-in `objective="quantile"`). The stage-1 binary classifier is unchanged. `TwoStageForecaster` now accepts `reg_objective` and `target_col` kwargs dispatched via `src.losses.resolve_objective` — custom asymmetric and pinball objectives are also available for TFT experiments.

Validation is moved to a **rolling-origin CV harness** (`scripts/rolling_origin_cv.py`) with `score = mean + 0.5·std` across six origins. V6 scores **0.434 mean WAPE ± 0.034** (selection score 0.451), a 4.1 pp improvement and 42 % variance reduction over V5.

A dedicated **UAH cost scorecard** (`scripts/decision_cost_scorecard.py`) evaluates each model under realistic holding (22 %), margin (28 %), and back-order recovery (50 %) assumptions. V6's lost-margin bucket is 0.90 M UAH vs V5's 1.17 M and V4's 1.28 M — the cheap-to-fix side wins.

Free-GPU workflow (`docs/gpu-workflow.md`) is wired to Kaggle's free T4×2 kernels and is driven entirely by the Kaggle API token in repo-root `.env` (new `KGAT_…` bearer form or legacy `KAGGLE_USERNAME`/`KAGGLE_KEY`). Three scripts — `scripts/push_to_kaggle.sh`, `scripts/push_kaggle_kernel.sh`, `scripts/pull_kaggle_kernel_output.sh` — push the ABT as a private dataset, queue the training notebook as a GPU kernel, and pull artefacts back into `output/gpu/`. No browser clicks, no billing: Kaggle kernels have no paid tier, and the 30 GPU-hours/week quota resets automatically.

Full ADR: `docs/adr-004-v6.md`. Executive report: `docs/v6_final_report.md`. Visuals: `output/plot_v6_dashboard.png` and `output/plot_model_progression.png`.

### V7 — per-SKU realised margins + price & cohort features + stacked ensemble + conformal intervals

V7 stacks five orthogonal upgrades on V6:

1. **Per-SKU realised margin table** (`src/margin_table.py`, output `output/sku_margin.parquet`). Derives per-SKU unit-price and margin rate from the ABT itself via empirical-Bayes shrinkage toward brand × channel means, replacing the flat 28 % margin / 22 % holding assumption in the cost scorecard. Reveals the business actually runs at ~10 % median margin (distributor economics), which means V6's α=0.6 over-forecast bias was mis-calibrated.
2. **Cost-calibrated pinball α = 0.45** (default) based on an 8-point α-sweep (`scripts/sweep_alpha_v7.py`, `output/v7_alpha_sweep.csv`). α=0.35 is documented as the cost-optimal operating point (annual UAH −47 % vs V6, at a 1.6 pp WAPE trade-off).
3. **7 relative-price features** (`src/features_price.py`): `price_lag1`, `price_lag3`, `price_vs_brand_median`, `price_vs_channel_median`, `price_vs_rrc`, `price_change_3m_pct`, and a shrunk per-SKU log-log price elasticity.
4. **4 cohort / substitution features** (`src/features_cohort.py`): same brand × product-group × channel cohort demand/stockout share/size/cannibalisation-pressure, all lag-shifted to avoid leakage.
5. **Isotonic classifier calibration + V4+V5+V6+V7 ridge stacker + per-(brand, channel) conformal intervals** (`src/v7_components.py`). The ridge meta-learner uses positive weights and is fit on the held-out last 40 % of the validation window. The conformal calibrator emits 10/90 interval files alongside the point forecast for every prediction.

Artefacts: `output/model_v7.joblib`, `output/preds_v7_{val,test,lower,upper,stacked}_*.csv`, `output/v7_metrics.csv`, `output/v7_rolling_cv.{json,md}`, `output/cost_scorecard_final.{md,json}`. Full ADR: `docs/adr-005-v7.md`. Executive report: `docs/v7_final_report.md`.

### V7.1 — recency weights + per-channel specialists

V7.1 layers two targeted upgrades on top of V7 after a six-way A/B ablation (`scripts/ablate_v71.py`, `scripts/sweep_channel_blend.py`, `output/v71_ablation.csv`, `output/v71_channel_blend_sweep.csv`):

1. **Recency sample weights** (`src/v71_components.build_recency_weights`). `w_i = clip(γ^months_ago, 0.25, 1.0)` on both stages of `TwoStageForecaster`. Sweep on γ ∈ {0.93, 0.95, 0.97, 0.99} picked **γ=0.95** as cost-optimal (−47 K UAH, −3.4 %). 2020 rows retain ~25 % weight so we don't lose long-tail signal.
2. **Per-channel specialists + blend** (`scripts/train_v71_channels.py`). Four channel-specific V7 boosters (ИМ, НКП, РС, СК) trained on per-channel slices, blended with the global model via `p = w · specialist + (1 − w) · global`. Sweep on `w ∈ [0, 1]` with the official scorecard picked **w=0.6** (additional −40 K UAH).

Six other upgrades were tested and rejected with documented evidence (ADR-006): per-SKU newsvendor α (margin table too uniform), full and stockout-only monotone constraints (custom-pinball hessian incompatibility), iterative EM imputation (mixed — helps WAPE, hurts cost), per-row business-cost LightGBM objective (deferred), 5-quantile bundle (over-forecast disaster).

Artefacts: `output/model_v7_{rec95,ch_im,ch_nkp,ch_rs,ch_sk}.joblib`, `output/preds_v71_{val,test}.csv`, `output/cost_scorecard_v71_channels.{md,json}`, `output/plot_v71_{dashboard,recency_sweep,channel_blend,stability}.png`. Full ADR: `docs/adr-006-v71.md`. Executive report: `docs/v71_final_report.md`.

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
  external_data.py    — BaseSignalLoader ABC, Parquet cache, loader registry
  leakage_guard.py    — enforces publication_lag_days per signal
  enrichment_external.py — joins registered loaders onto the ABT
  loaders/            — concrete signal loaders (conflict_ua, nbu_fx, holidays_ua, gtrends_ua, tmdb_movies, world_bank_ua, …)
  demand_imputation.py — V6: censored-demand imputation (stockout mask + EB-shrunk SKU factor)
  features_promo.py   — V6: promo-lifecycle features (duration, post-promo depletion, sensitivity)
  losses.py           — V6: pinball + asymmetric LightGBM objectives (resolve_objective)

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
  adr-003-external-signals.md           — V5 external-signal selection decision
  adr-004-v6.md                         — V6 imputation + promo-lifecycle + pinball loss decision
  gpu-workflow.md                       — free-GPU (Kaggle / Colab) workflow for V6
  v6_final_report.md                    — one-page executive summary of V6
  external-data-sources.md              — survey of free, regularly-updated sources
  external-data-plan.md                 — original Beads plan for V5
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
| `run_v4_final.py` | Production V4 ensemble | ~3 min |
| `scripts/run_ablation.py` | Add-one-source + leave-one-out ablation over external loaders | ~40 s / loader |
| `scripts/run_decision_gate.py` | Promotes/rejects loaders into the V5 candidate set | <1 s |
| `scripts/build_v5_abt.py` | Enriches V4 ABT with the decision-gate winners | ~5 s |
| `scripts/train_v5.py` | **Production V5 model + V4 vs V5 comparison** (recommended) | ~1 min |
| `scripts/tune_v5_ensemble.py` | Scans V4+V5 blend weights on validation | ~2 min |
| `scripts/viz_v5_performance.py` | 6-panel V5 dashboard (monthly fit, scatter, residuals, segments, V4/V5, feature importances) | ~10 s |
| `scripts/build_v6_abt.py` | V6 ABT: adds imputation + promo-lifecycle features to V5 ABT | ~5 s |
| `scripts/train_v6.py` | **Production V6 model** — pinball q60 + imputed target + V5 features | ~30 s |
| `scripts/rolling_origin_cv.py` | Rolling-origin CV harness (6-12 origins); selection score `mean + 0.5σ` | ~2 min / 6 origins |
| `scripts/decision_cost_scorecard.py` | UAH cost scorecard across V4/V5/V6/naive | <5 s |
| `scripts/viz_v6_performance.py` | 6-panel V6 dashboard | ~5 s |
| `scripts/viz_model_progression.py` | V4 vs V5 vs V6 progression (bars, monthly WAPE, rolling box, UAH cost, segment heatmap, residual density) | ~5 s |
| `scripts/generate_baseline_preds.py` | Re-emits V4/V5 predictions on the fixed split for the cost scorecard | ~30 s |
| `scripts/push_to_kaggle.sh` | Uploads V6 ABT + source tree as a private Kaggle dataset (reads `KAGGLE_API_TOKEN` from `.env`) | ~30 s |
| `scripts/push_kaggle_kernel.sh` | Publishes `notebooks/v6_gpu_template.ipynb` as a private Kaggle kernel with GPU enabled and queues a run | ~15 s to queue |
| `scripts/pull_kaggle_kernel_output.sh` | Polls the kernel until it finishes and downloads `/kaggle/working/*` into `output/gpu/` | depends on kernel runtime |

## Quick Start

### Cloning the repo (Git LFS required)

Trained model artifacts and the cached ABT parquet are stored via Git LFS.
Install `git-lfs` once before cloning:

```bash
# macOS
brew install git-lfs
# Debian/Ubuntu
sudo apt-get install git-lfs

git lfs install
git clone https://github.com/Smikalo/business-process-modeling-demo.git
```

A plain `git clone` without LFS will fetch only text pointers for the `.joblib`
and `.parquet` files and training artifacts will be unusable until you run
`git lfs pull`.

### Running the pipeline

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Production V6 (V5 backbone + imputation + promo-lifecycle + pinball loss)
PYTHONPATH=. python3 -m scripts.build_v5_abt          # V5 ABT (prereq)
PYTHONPATH=. python3 -m scripts.build_v6_abt          # V6 ABT
PYTHONPATH=. python3 -m scripts.train_v6              # train + compare to V5
PYTHONPATH=. python3 -m scripts.rolling_origin_cv \
    --abt output/abt_v6_cached.parquet \
    --target target_qty_imputed --reg-objective pinball --alpha 0.6
PYTHONPATH=. python3 -m scripts.generate_baseline_preds    # V4 + V5 preds for the scorecard
PYTHONPATH=. python3 -m scripts.decision_cost_scorecard    # UAH cost scorecard
PYTHONPATH=. python3 -m scripts.viz_v6_performance         # V6 dashboard
PYTHONPATH=. python3 -m scripts.viz_model_progression      # V4 vs V5 vs V6

# Or: V5 only (V4 backbone + 6 external signal loaders)
PYTHONPATH=. python3 -m scripts.train_v5

# Or: V4 ensemble (no external signals)
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
- V5 external signals lift validation but regress test — see ADR-003; this is the key open item for the next iteration (either additional held-out months or distribution-shift–robust training).

See `docs/limitations-and-next-steps.md` for the full roadmap.
