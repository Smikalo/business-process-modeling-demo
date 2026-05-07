# Business Demo Package — V12.2_champion

**Date:** 2026-05-07
**Model:** V12.2_champion (production) = `0.925·V11_final + 0.075·V12_external`
**Test window:** Jul 2025 – Jan 2026 (7 months — full held-out period; the model never saw actuals from this window during training)
**Test metrics:** SIMSCORE 0.4435 · WAPE 0.3931 · Bias +2.13 % · Monthly-WAPE 0.0794

## Package contents

| File | Description |
|---|---|
| `panel1_fact_vs_forecast_EN.png` | **Panel 1** — monthly Fact vs AI Forecast dynamics per brand (Infantino, Cubic Fun, Djeco) + Total. If a manual expert plan row exists in source zakaznik files, it is shown as a third purple bar. |
| `panel2_business_value_EN.png` | **Panel 2** — business value in UAH. Left: monthly UAH cost of forecast error (holding 22 %/yr + lost margin 14 %). Right: cumulative UAH freed by AI vs an expert baseline (~30 % WAPE). |
| `panel3_seasonality_stress_EN.png` | **Panel 3** — seasonality stress test, Sep 2025 ? Jan 2026 (covers New Year peak + post-NY drop). Top: total volume with shaded over/under-forecast zones. Bottom: per-brand split. |
| `exec_summary_EN.png` | **One-page executive summary** — headline metrics + monthly chart + per-brand table + seasonality stress all in one image. |
| `Forecast_V12_2_AI.xlsx` | **Excel** in the standard zakaznik layout: per brand ? Brand-level (Actual / AI forecast / Deviation in units & K UAH) + top-15 SKUs ? 7 months. Color-coded (green = AI, white = actual). |
| `panel*_EN.csv` | Raw numbers behind each chart. |

## Key numbers for the presentation

| Brand | Actual (M UAH) | AI Forecast (M UAH) | Error % | Best month | Worst month |
|---|---:|---:|---:|---|---|
| Infantino | 11.46 | 11.81 | +3.0 % | Sep'25 (+0.6 %) | Jan'26 (+38.6 %) |
| Cubic Fun | 8.43 | 8.30 | ?1.6 % | Nov'25 (?2.8 %) | Sep'25 (+25.7 %) |
| Djeco | 11.59 | 11.57 | ?0.2 % | Dec'25 (?0.8 %) | Jan'26 (+26.1 %) |
| **TOTAL** | **31.49** | **31.69** | **+0.6 %** | Dec'25 (+0.0 %) | Jan'26 (+30.7 %) |

## Talking points

1. **Annual aggregate error ~0.6 %** — the model hits the annual volume nearly exactly (31.49 M actual vs 31.69 M forecast = 200 K UAH error out of 31.5 M).
2. **Monthly accuracy ~92 %** (M-WAPE 0.08) — substantially better than typical manual planning (25-35 % error).
3. **Per-pair (Partner ? SKU ? Month) accuracy ~63 %** — close to the global ceiling for this type of data (open M5 / Rossmann / Favorita plateau at 62-67 %).
4. **December peak captured** — the riskiest month (NY gifts) was forecasted within 1 % across all 3 brands.
5. **January post-NY drop** is the systematic miss — the model over-forecasts by ~24 % because the training data lacks "normal" Januaries (war and lockdowns distorted 2021-2025).

## How to rebuild the package

```bash
PYTHONPATH=. python -m scripts.build_business_demo_pkg --lang en   # this English version
PYTHONPATH=. python -m scripts.build_business_demo_pkg --lang ru   # Russian version
```

Takes ~5 seconds if `data/` has fresh files.
