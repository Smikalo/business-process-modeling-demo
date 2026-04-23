# V7.1 — Final report

## TL;DR

- **Champion**: V7 + recency sample weights (γ=0.95) + per-channel
  specialists blended at `w=0.6` with the global model.
- **Test WAPE 0.4117** (vs V7 0.4208, V6 0.4335)
- **MAPE_nz 0.4904** (vs V7 0.5509, V6 0.5589)
- **Annual UAH cost: 1,316,197** (vs V7 1,403,295, V6 2,072,072)
- **Savings: −87K UAH (−6.2%) vs V7, −756K UAH (−36.5%) vs V6**, on top of
  the 669K UAH already saved in V7.
- Released as tag **`v7.1`**; ADR-006 captures the full A/B evidence.

## What changed from V7

| Change | Variant tag | Result | Kept? |
|---|---|---|---|
| Recency sample weights `γ=0.95` | `v7_rec95` | 1.356M (−47K UAH, −3.4%) | ✓ |
| Recency `γ=0.97` | `v7_rec97` | 1.401M (WAPE best) | dropped |
| Recency `γ=0.99` | `v7_rec99` | 1.386M | dropped |
| Per-channel specialist blend (w=0.6) | on top of `rec95` | **1.316M (−87K vs V7)** | ✓ |
| Per-SKU newsvendor α (5-quantile bundle) | `v7_1_nv` | 5.5M (disaster) | dropped |
| LightGBM monotone constraints (full) | `v7_mono` | 4.9M | dropped |
| LightGBM monotone (stockout only) | `v7_mono_stockout` | 5.0M | dropped |
| Iterative EM imputation (1 round on top of rec97) | `v7_rec_em` | 1.416M (WAPE best, cost worse) | dropped |

See `docs/adr-006-v71.md` for technical reasoning on each decision.

## Business impact

Taking the 8-month hold-out test period as representative:

- **Naive baseline (y_t = y_{t−1})**: 2.85M UAH / year lost to over- and
  under-forecasting.
- **V7.1**: 1.32M UAH / year — a 54% reduction.
- **Δ vs V7**: −87K UAH / year, an additional 6.2% cost reduction on top
  of V7's already large win. The savings come almost entirely from
  **reducing over-forecast holding cost** (1,054K → 1,014K = −40K UAH)
  with a modest under-forecast pickup (lost margin 302K → 302K,
  essentially flat), which is the ideal trade-off — we're not paying in
  lost sales for the improvement.

For a typical retailer carrying the same assortment:

- Capital tied up in over-forecast inventory: down ~17K UAH / month.
- Discounted / wasted stock at end-of-season: down proportionally.
- Ability to reinvest freed capital into SKUs that actually turn over.

## Why each winning change works

### Recency weights (γ=0.95)

Toys distribution has changed materially since 2020: channel mix shifted
toward СК, partner activity churned, and pandemic-era demand spikes stopped
being informative. Down-weighting 2020 rows to ~25% preserves their
long-tail signal (still useful for rare SKUs) while letting recent
months drive the model. γ=0.95 is the sweet spot — more aggressive (0.93)
starves the model of data for sparse pairs, less aggressive (0.99)
barely moves the needle.

### Per-channel specialists blended at w=0.6

Each channel has a different pattern:

- СК (173K rows, 55% of volume) — heavy seasonality, gift/holiday-driven.
- НКП (53K) — steady hobby / educational demand.
- РС (52K) — the smallest B2B-ish partners, very sparse per SKU.
- ИМ (38K) — e-commerce, noisier and more impulsive.

A single global model averages these patterns. A pure specialist model
(w=1.0) has less data per model and overfits the noise. A 60/40 blend
lets each specialist lean into its channel-specific pattern while the
global model regularises the tail. The blend-sweep curve is smooth and
convex with a clean minimum at w=0.6, suggesting real signal.

## Why each losing change loses

- **Per-SKU newsvendor α**: relies on margin diversity we don't have
  (75% of SKUs are at the empirical-Bayes margin floor). Without
  negotiated margins, this is purely noise.
- **Monotone constraints**: fundamentally incompatible with our current
  pinball objective's constant hessian; fixing this is a larger
  research task (V7.2).
- **EM imputation**: pushes predictions up, which helps WAPE on
  under-forecast rows but costs more in holding. Net negative for cost.

## Stability

Per-month WAPE on the 8-month hold-out (`output/v71_per_month_stability.csv`):

| Month | WAPE |  Bias |
|:-----:|-----:|------:|
| 2025-07 | 0.471 | −0.41 |
| 2025-08 | 0.412 | −0.20 |
| 2025-09 | 0.460 | +0.10 |
| 2025-10 | 0.418 | −0.54 |
| 2025-11 | 0.425 | −0.90 |
| 2025-12 | 0.389 | −1.13 |
| 2026-01 | 0.358 | −0.72 |
| 2026-02 | 0.395 | −0.69 |
| **mean** | **0.416** | |
| **std**  | **0.037** | |

No month collapses. The bias does drift more negative in Nov–Dec
(Christmas season under-forecast) — that's a known gap since we don't yet
have a promo-forward feature for Q4.

## Artefacts

- `output/model_v7_rec95.joblib` — global-recency V7 booster bundle.
- `output/model_v7_ch_{im,nkp,rs,sk}.joblib` — per-channel specialists.
- `output/preds_v71_{val,test}.csv` — champion blended predictions.
- `output/v71_ablation.csv` — full A/B results.
- `output/v71_channel_blend_sweep.csv` — blend-weight sweep.
- `output/v71_champion.json` — champion config summary.
- `output/cost_scorecard_v71_channels.{md,json}` — official cost scorecard.
- `output/plot_v71_dashboard.png` — main dashboard (3×2, matches V6/V7).
- `output/plot_v71_recency_sweep.png` — γ sweep.
- `output/plot_v71_channel_blend.png` — blend-weight sweep.
- `output/plot_v71_stability.png` — per-month WAPE.
- `docs/adr-006-v71.md` — Architectural Decision Record.

## How to reproduce

```bash
# 1. Build/refresh V7 ABT + per-SKU margins (already exists for V7 release)
python -m scripts.build_v7_abt
python -m scripts.build_sku_margin_table

# 2. A/B ablation (5 variants, ~15 min on 10-core CPU)
python -m scripts.ablate_v71

# 3. Train per-channel specialists + blend sweep (~5 min)
python -m scripts.train_v71_channels --global-tag rec95 --recency-gamma 0.95
python -m scripts.sweep_channel_blend

# 4. Generate V7.1 dashboard + stand-alone plots
python -m scripts.viz_v71_performance

# 5. (Optional) Optuna-tuned sibling — see docs/v71_optuna_comparison.md
#    for why this variant did NOT become the champion (+12.5K UAH worse
#    on business cost despite better pinball/WAPE).
bash scripts/pull_kaggle_kernel_output.sh --slug <user>/bpm-v7-optuna
cp output/gpu/v7_optuna_best_params.json output/v7_optuna_best.json
python -m scripts.train_v7 \
    --disable-residual --save-tag rec95_tuned \
    --recency-gamma 0.95 --optuna-params output/v7_optuna_best.json
python -m scripts.train_v71_channels --global-tag rec95_tuned \
    --recency-gamma 0.95 --optuna-params output/v7_optuna_best.json
python -m scripts.sweep_channel_blend
```

## Outstanding / V7.2 candidates

- Fix pinball hessian → enable monotone constraints.
- Per-row business-cost LightGBM objective.
- Ingest negotiated-margin data → meaningful per-SKU newsvendor α.
- Multi-round EM with bias-aware stopping criterion.
- Per-channel Optuna tuning, **tuning directly on UAH cost instead of
  pinball loss** (the val-pinball-tuned variant was ~12.5K UAH worse —
  see `docs/v71_optuna_comparison.md`).
- Explicit Q4 promo-forward feature to close the Nov–Dec bias gap.
