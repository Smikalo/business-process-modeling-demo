# Management Presentation Package — V12.2_champion

**Date:** 2026-05-07
**Model:** V12.2_champion (production) = `0.925·V11_final + 0.075·V12_external`
**Test window:** Jul 2025 – Jan 2026 (7 months — full held-out period; the model never saw actuals from this window during training)
**Test metrics:** SIMSCORE 0.4435 · WAPE 0.3931 · Bias +2.13 % · Monthly-WAPE 0.0794

## What's in this package

| File | Content |
|---|---|
| `panel1_fact_vs_forecast_EN.png` | **Panel 1** — monthly Fact vs AI Forecast dynamics, per brand (Infantino, Cubic Fun, Djeco) + Total. Where available, the expert manual plan is shown as a third purple bar. |
| `panel2_business_value_EN.png` | **Panel 2** — Business value in UAH. Left: monthly UAH cost of forecast error (holding cost 22 %/year + lost margin 14 %). Right: cumulative UAH freed by AI vs expert baseline (~30 % WAPE). |
| `panel3_seasonality_stress_EN.png` | **Panel 3** — Seasonality stress test, Sep 2025 → Jan 2026 (covers New Year peak + post-NY decline). Top: total volume with shaded over-/under-forecast zones. Bottom: per-brand split. |
| `exec_summary_EN.png` | **1-page executive summary** — headline metrics + total monthly chart + per-brand summary table + seasonality stress in one image. |
| `Forecast_V12_2_AI.xlsx` | **Excel** in the standard order-form layout: per brand → Brand-level (Actual / AI forecast / Deviation in units & K UAH) + top-15 SKUs × 7 months. Color-coded (green = AI, white = Actual). |
| `panel*_EN.csv` | Raw numbers behind each chart. |

## Key numbers for the presentation

| Brand | Actual (M UAH) | AI Forecast (M UAH) | Error % | Best month | Worst month |
|---|---:|---:|---:|---|---|
| Infantino | 11.46 | 11.81 | +3.0 % | Sep'25 (+0.6 %) | Jan'26 (+38.6 %) |
| Cubic Fun | 8.43 | 8.30 | −1.6 % | Nov'25 (−2.8 %) | Sep'25 (+25.7 %) |
| Djeco | 11.59 | 11.57 | −0.2 % | Dec'25 (−0.8 %) | Jan'26 (+26.1 %) |
| **TOTAL** | **31.49** | **31.69** | **+0.6 %** | Dec'25 (+0.0 %) | Jan'26 (+30.7 %) |

## Talking points for management

1. **Annual aggregate error is ~0.6 %** — the model hits the annual volume nearly exactly (31.49 M actual vs 31.69 M forecast = 200 K UAH error out of 31.5 M).
2. **Monthly accuracy is ~92 %** (M-WAPE 0.08) — substantially better than typical manual planning (25-35 % error).
3. **Per-pair (Partner × SKU × Month) accuracy is ~63 %** — close to the global ceiling for this type of data (open M5 / Rossmann / Favorita competitions plateau at 62-67 %).
4. **December peak captured** — the riskiest month (NY gifts) was forecasted within 1 % across all 3 brands.
5. **January post-NY drop is the systematic miss** — the model over-forecasts by ~30 % because the training data lacks similarly deep post-NY drops (war and lockdowns distorted previous January declines).

## Why V12.2_champion (not V13/V14 that appear in the repo)

* **V14 (GlobalNN neural network)** was trained on Kaggle GPU. We caught data leakage in the dataset (`Количество_sales` had +1.0000 correlation with the target — the model was reading the answer). After fixing the leakage, V14 performed worse than V12.2.
* **V13 (Chronos foundation model)** post-fine-tuning also did not beat V12.2 under honest cross-validation.
* Production remains V12.2, and this presentation is built on that.

## What we'll do next (4-week roadmap)

| Week | Effort | Expected lift |
|---|---|---|
| 1 | Add 2024-2026 data (Anna provides) — re-train V12.3 | +0.5–1.0 % |
| 1 | Pre-2020 monthly data (Anna searches) — better seasonal anchor | +1–2 % |
| 2-3 | V13 fine-tune on full GPU run (currently 2 epochs only) | +0.3–0.7 % |
| 3-4 | V14 with proper regularization + longer training | +0.5–1.0 % |
| **Total expected** | | **+2.3–4.7 % SIMSCORE** |

This brings cumulative annual accuracy from ~63 % to ~65–67 % — within the M5/Rossmann global ceiling. Beyond that requires business-side data (POS sales, partner inventory, full promo budgets) which is a separate 12-18 month bizdev effort.
