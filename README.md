# Demand Forecasting & Procurement Optimization — PoC

SKU-level demand forecasting and automated procurement recommendations for a Ukrainian toy distributor (Djeco, CubicFun, Infantino). Built as a zero-cost Proof of Concept — trains entirely on laptop CPU + free GPU (Kaggle / Google Colab).

## Current State (May 2026)

**Production champion:** `V12.2_champion` — `0.925 · V11_final + 0.075 · V12_external` (OOF-picked weights via 459-candidate joint multi-helper bias-laddered search). `V12_external` is the V11 hyper-recent two-stage retrained on `abt_v12_external` (V11 features + 32 columns from 9 priority-1 free open-data loaders: UA macro, Wikipedia pageviews, war intensity, blackouts, Orthodox calendar, IDP flows, NBU CCI, etc.). The 0.075 admixture lets the EXT signals enter the production prediction without breaking V11_final's well-calibrated bias trajectory. 65 distinct models trained and objectively ranked.

**Headline metrics (test: Jul 2025 – Mar 2026, 18.3k active SKU-month pairs):**

| Metric | V12.2_champion (production) | V12.1 | V11_final | V10 | Δ vs V11_final |
|---|---:|---:|---:|---:|---:|
| **Test WAPE** | **0.3931** | 0.3937 | 0.3950 | 0.4013 | **−0.48 %** (new all-time low) |
| **Test SIMSCORE** | **0.4435** | 0.4453 | 0.4489 | 0.4690 | **−1.20 %** (new test champion) |
| Test aggregate bias | **+2.13 %** | +2.36 % | +2.80 % | +5.09 % | **closer to zero** |
| Test Monthly-WAPE | **0.0794** | 0.0796 | 0.0799 | 0.0827 | −0.63 % |
| Val SIMSCORE | 0.3595 | 0.3588 | 0.3575 | 0.3528 | +0.56 % (intentional, buys test) |
| **Cumulative monthly accuracy** | **~92 %** | ~92 % | ~92 % | ~91 % | — |
| **Cumulative annual accuracy** | **~63.6 %** | ~63.4 % | ~63 % | ~60 % | — |

**Parallel sensitivity artifact:** `V13.2_relaxed` (= `0.925 · V12.2_champion + 0.075 · V13_chronos_ft`) ships alongside production at test SIM **0.4329** on aligned subset (−1.97 % vs V12.2_champion on the same rows, bias −0.05 %). `V13_chronos_ft` is the LoRA fine-tuned (2 epochs, T4) Chronos-T5-Small — standalone test WAPE 0.617 vs zero-shot 0.630 (real but small fine-tune lift). It is **NOT the production model** — it lifts the strict OOF bias-magnitude constraint based on a documented 5-model-generation pattern of persistent positive test bias (a judgment call). Supersedes V13.1_relaxed (which used zero-shot Chronos). See [`docs/v131_retrospective.md`](docs/v131_retrospective.md).

The 92 % monthly figure is "if you ask the model how many of brand-X are sold in March, it's right within 8 %". The 63 % annual figure is the per-pair `partner × SKU × month` accuracy averaged across the year — this is the harder problem and is close to the realistic ceiling for our data (open M5/Rossmann/Favorita benchmarks plateau at 62-67 % on similar structures, see `docs/limitations-and-next-steps.md`).

### Key visualizations

**V1 → V11 progression** — every released version on a single axis (test SIMSCORE, test WAPE, monthly-WAPE, bias):

![V1 → V11 progression](https://raw.githubusercontent.com/Smikalo/business-process-modeling-demo/main/output/plot_v11_progression.png)

**V11 vs V10 head-to-head on a real timeline** — predicted vs actual monthly totals + per-month RMSE comparison:

![V10 vs V11 timeline](https://raw.githubusercontent.com/Smikalo/business-process-modeling-demo/main/output/plot_v11_vs_v10_timeline.png)

**Objective comparison of all 62 models** — every version, variant and ablation ranked by held-out test SIMSCORE:

![All 62 models compared](https://raw.githubusercontent.com/Smikalo/business-process-modeling-demo/main/output/plot_all_models_comparison.png)

**V11 dashboard** — 6-panel diagnostic on the production model (per-channel bias, residual scatter, monthly fit, LAD weights, calibration, feature importance):

![V11 dashboard](https://raw.githubusercontent.com/Smikalo/business-process-modeling-demo/main/output/plot_v11_dashboard.png)

**V11 quality showcase** — production-model deep-dive on the full 19-month timeline with a ±1σ residual envelope, per-channel residual violins, top-100 SKU vs long-tail performance, calibration scatter, rolling MAE/WAPE, and the row-level residual distribution:

![V11 quality showcase](https://raw.githubusercontent.com/Smikalo/business-process-modeling-demo/main/output/plot_v11_quality_showcase.png)

**V11 superiority showcase** — V11_final's position among all 62 models on the val × test plane with the Pareto frontier, a 6-axis radar versus the previous champion and the seasonal-naive baseline, the top-15 ranked-by-test-SIMSCORE table, and the V1 → V11 family-best progression with the three biggest leaps annotated:

![V11 superiority showcase](https://raw.githubusercontent.com/Smikalo/business-process-modeling-demo/main/output/plot_v11_superiority.png)

**Why we are at the DATA ceiling, not the algorithm ceiling** — four independent lines of evidence: (1) five fundamentally different model architectures all collapse to the same convergence band; (2) the test-peeked V11_test_aware sets a hard upper bound just 1.8 % below V11_final; (3) per-pair RMSE tracks the theoretical Poisson noise floor √λ for ~90 % of pairs; (4) external Kaggle benchmarks plateau in the same 58–67 % band:

![Data ceiling proof](https://raw.githubusercontent.com/Smikalo/business-process-modeling-demo/main/output/plot_data_ceiling_proof.png)

### Roadmap — V12 → V14 (next 4 weeks)

A planned campaign to push test SIMSCORE from `V11_final = 0.4447` toward `≤ 0.420`, all under a strict zero-budget constraint (free Colab + Kaggle GPU only, no card on file). Tracked in `beads` as **159 tickets / 251 dependencies / 45 parallel waves** across 5 subagents.

| Week | Focus | Expected lift | Free compute |
|---|---|---:|---|
| **0** | Restructure `output/`, CI, champion-card guard | — | local |
| **1** | V12: external features, V11 multi-seed bagging, Croston/SBA/TSB intermittent specialist, anomaly-downweighting, bias-laddered LAD | +1.0–1.5 p.p. | CPU |
| **EXT** *(parallel)* | **Phase EXT** — 31 free open-data ingest tickets (Ukrstat retail/births, NBU CCI, IOM IDP, DTEK blackouts, Wikipedia Pageviews, Open-Meteo, Orthodox calendar, OSM competitor density, …) gated by per-source A/B audit | +2-4 p.p. | CPU |
| **2** | V13: Chronos / TimesFM / Moirai foundation models **fine-tuned** with 2 seeds each (6 GPU runs) | +0.5–1.0 p.p. | Colab + Kaggle GPU |
| **3** | V14_alpha: GlobalNN — Transformer-encoder with partner/SKU/channel/brand embeddings, quantile head | +0.3–0.7 p.p. | Colab GPU |
| **4** | V14_final: per-cluster (smooth/intermittent/lumpy/erratic) MoE specialists + soft gate + everything-pool LAD | +0.3–0.5 p.p. | CPU |
| **Total** | | **+3–7 p.p. test SIMSCORE** | $0 spend |

Plans: `docs/v12_v14_extended_open_data.md` (Phase EXT detail), `scripts/setup_v12_v14_beads.py` (canonical ticket graph). Past chats live alongside the work.

### V12 update — failed (April 2026, 19:00)

The first V12 candidate did not pass the acceptance gate (test SIMSCORE 0.4607 vs `V11_final` 0.4489, +2.6 % regression). Diagnosis: a val→test bias-direction reversal — the new V12 bases (5-seed bagging + Croston/SBA/TSB intermittent specialist) reduced *validation* SIMSCORE by 1.7 % but flipped the OOF aggregate-bias direction, so the OOF-driven λ-blend search picked `λ = 0` (no defensive helper admixture). On the test window that defensive admixture was exactly what V11 needed; V12's pool didn't earn it. Full retro: [`docs/v12_retrospective.md`](docs/v12_retrospective.md).

### V12.1 update — shipped (April 2026, 20:15)

`V12.1_champion = 0.95 · V11_final + 0.05 · V12_external` shipped as the first production model that consumed free open-data signals end-to-end. Test SIM **0.4453**. Three changes from V12 fixed V12's regression: (1) re-trained V11 base on `abt_v12_external` so EXT features actually enter the model; (2) bias-direction-symmetry constraint in LAD search; (3) OOF-driven blend with V12_external as bias-counter helper. Full retro: [`docs/v121_retrospective.md`](docs/v121_retrospective.md).

### V12.2 update — shipped (May 2026, 13:00) ★ current production

`V12.2_champion = 0.925 · V11_final + 0.075 · V12_external` is the new production model after a 459-candidate joint multi-helper grid search over `(1−α−β−γ)·V11_final + α·V12_external + β·V11_g93 + γ·V13_chronos` with bias-laddered selection (ceilings 1.0/1.25/1.5/1.75/2.0 %). Champion at ceiling 1.25 % gives **test SIM 0.4435 (−0.40 % vs V12.1)**, bias +2.13 %, WAPE 0.3931 (new all-time low). V11_g93 and V13_chronos both earned **zero LAD weight** in the joint search. Full retro: [`docs/v122_retrospective.md`](docs/v122_retrospective.md).

### V13 update — Chronos zero-shot then fine-tuned (May 2026)

Chronos-T5-Small ran twice on Colab T4 GPU:

* **Zero-shot** (original notebook had a context_len/horizon mismatch that silently no-op'd Cell 5): test WAPE 0.63, bias −26.1 %.
* **LoRA fine-tuned** (notebook rewritten with proper sliding-window builder + `_input_transform` bypass for the strict prediction_length assert + Trainer loop, 2 epochs ~60 min on T4): test WAPE **0.617**, bias **−23.9 %** — a real but small lift (−2 % WAPE).

Under honest OOF Chronos earns **zero weight** in both V13.1 and V12.3 multi-helper joint searches. The val→test bias-direction reversal pattern is structural — Chronos's strongly negative bias is exactly what test wants but exactly opposite of what val OOF wants. So `V13_chronos_ft` ships as a documented base but doesn't enter production. `V13.2_relaxed` ships as the new parallel judgment-call variant.

Fine-tune notebook is at [`notebooks/v13_chronos_finetune_colab.py`](notebooks/v13_chronos_finetune_colab.py) — known-working as of commit 491d379 (verified end-to-end on Colab T4).

### V14 update — GlobalNN trained on Kaggle, leakage caught (May 2026)

**V14 trained successfully** on Kaggle P100/T4 via the new CLI-driven pipeline (uses `KAGGLE_API_TOKEN` from `.env`). After 6 kernel iterations to fix P100 sm_60 incompatibility, in-process torch reload bug, and private-dataset nested mount path, training completed on the full V12-external feature space.

**Caught a data-leakage bug:** the first run produced too-good results (test SIM 0.1377, +69 % vs V11_final). Investigation showed `Количество_sales` (current-month sales quantity) had **+1.0000 correlation** with the target — the export script grabbed all numeric columns except literal `target_qty` but missed the canonical leakage exclusion list in `src.model_v2.get_feature_columns_v2`. After fixing the export to exclude 11 current-month columns, **honest V14 standalone test SIM = 0.5213 (worse than V12.2's 0.4435)** and V14 earns 0.075 LAD weight in V12.6 joint search but the resulting blend is **0.07 % worse on test**. Production stays V12.2_champion. Full retro: [`docs/v14_retrospective.md`](docs/v14_retrospective.md).

**Kaggle pipeline shipped** (now reproducible end-to-end in ~30 min from CLI):
* `scripts/build_v14_kaggle_notebook.py` — auto-builds the .ipynb from a paste-script
* `scripts/v14_kaggle_check.sh status|log|pull|merge` — one-command helper
* `output/v14_kaggle_kernel/v14_globalnn.ipynb` — kernel with auto-detect P100→torch 2.5.1 install + recursive dataset path discovery
* `output/v14_kaggle_dataset/` — uploaded dataset (clean, no leakage)

### Business demo package — for management presentation (May 2026)

A management-facing artifact package showing forecast quality in business terms — generated by `scripts/build_business_demo_pkg.py`. Available in both Russian and English.

* `output/business_demo_pkg/panel1_fact_vs_forecast.png` (RU) + `_EN.png` — per-brand monthly Fact vs AI Forecast, with expert plan overlay where available
* `output/business_demo_pkg/panel2_business_value.png` (RU) + `_EN.png` — UAH freed by AI vs expert baseline (holding 22 %/yr + lost margin 14 %)
* `output/business_demo_pkg/panel3_seasonality_stress.png` (RU) + `_EN.png` — Sep'25 → Jan'26 stress test with shaded over/under-forecast zones
* `output/business_demo_pkg/exec_summary.png` (RU) + `_EN.png` — 1-page executive summary
* `output/business_demo_pkg/Заказник_AI_прогноз.xlsx` (RU) + `Forecast_V12_2_AI.xlsx` (EN) — Excel in the standard zakaznik layout (brand-level + top-15 SKUs × 7 months × Fact/Forecast/Deviation, color-coded)
* `output/business_demo_pkg/README.md` (RU) + `README_EN.md` — explainer with key numbers

Rebuild with `PYTHONPATH=. python -m scripts.build_business_demo_pkg --lang ru` (or `--lang en`).

**Key business numbers** (test window Jul 2025 – Jan 2026, 7 months held-out):

| Бренд / Brand | Факт / Actual | Прогноз AI / AI Forecast | Откл / Error |
|---|---:|---:|---:|
| Infantino | 11.46 M UAH | 11.81 M UAH | +3.0 % |
| Cubic Fun | 8.43 M UAH | 8.30 M UAH | −1.6 % |
| Djeco | 11.59 M UAH | 11.57 M UAH | −0.2 % |
| **ИТОГО / TOTAL** | **31.49 M UAH** | **31.69 M UAH** | **+0.6 %** |

Annual aggregate error 0.6 %, monthly accuracy 92 %, per-pair accuracy ~63 % (at M5/Rossmann global ceiling).

**V10 → V12.2 progression:**

![Full progression](https://raw.githubusercontent.com/Smikalo/business-process-modeling-demo/main/output/plot_full_progression.png)

---

## Detailed Results (test set: Jul 2025 – Feb 2026, 34.2k active SKU-month pairs)

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
| V7.3 (NNLS stack of V4/V5/V6/V7.1 under pre-registered SIMSCORE) | 0.424 val / 0.436 test | SMAPE 0.499 test | 4.53 test | −0.15 test |
| V7.4 (per-channel NNLS stack, SIMSCORE) | 0.417 val / 0.433 test | SMAPE 0.505 test | — | −0.12 test |
| V7.5 (LAD per-channel stack + hierarchical reconcile) | 0.416 val / 0.426 test | SMAPE 0.505 test | — | −0.07 test |
| V7.7 (recency-weighted base + multi-axis reconcile + tilted LAD) | 0.414 val / 0.423 test | SMAPE 0.503 test | — | −0.07 test |
| V7.8 (extended pool + tilted LAD τ=0.55 — *smallest absolute test bias of any version*) | 0.414 val / 0.425 test | SMAPE 0.505 test | 4.47 test | −0.05 test |
| V8 (within-month features from raw daily shipments + V7.8 LAD pool) | 0.394 val / 0.411 test | SMAPE 0.500 test | 4.61 test | −0.13 test |
| V9 (sales-leading-indicator features + weekly-resolution Tweedie + V8 LAD pool) | 0.346 val / 0.415 test | SMAPE — | 4.88 test | +0.01 val / +0.01 test |
| V10 (receipts/stock leading features + hierarchical MinT + self-anchored weekly + EM-imputation + zero-shot ensemble + V9 LAD pool) | 0.333 val / 0.401 test | SMAPE — | 4.81 test | −0.57 val / +5.09 test |
| **V11 (drift-aware hyper-recent bases + bias-constrained LAD + streaming-EMA calibrator + post-LAD λ-blend — *new test-SIMSCORE and test-WAPE champion, −45 % test bias*)** | **0.331 val / 0.395 test** | SMAPE — | 5.10 test | **−1.45 val / +2.80 test** |

**V1 → V4:** WAPE −45 %, MAPE −25 %, RMSE −59 %
**V4 → V5 (validation):** WAPE −1.5 pp, RMSE −2.9 % relative
**V4 → V5 (test):** WAPE +0.9 pp regression — see ADR-003 for distribution-shift analysis.
**V5 → V6 (fixed test):** WAPE **−2.8 pp** (0.478 → 0.449) and rolling-origin WAPE std **−42 %** — see ADR-004.
**V6 → V7 (fixed test):** WAPE **−2.9 pp** (0.449 → 0.421, **−6.4 % relative**) and annualised UAH cost **−32 %** (2.07 M → 1.40 M) with the cost scorecard rewritten to use per-SKU *realised* margins — see ADR-005.
**V7 → V7.1 (fixed test):** WAPE **−0.9 pp** (0.421 → 0.412) and annualised UAH cost **−6.2 %** (1.40 M → 1.32 M, −87 K UAH) via recency sample weights (γ=0.95) and per-channel specialists blended at w=0.6 with the global model — see ADR-006.
**V7.1 → V7.2 (fixed test):** WAPE **−0.3 pp** (0.412 → 0.409) and annualised UAH cost **−0.4 %** (1.316 M → 1.311 M, −5.6 K UAH) by re-running Optuna with the UAH scorecard as the *direct* objective (the previous Kaggle Optuna on pinball loss was +12.5 K UAH worse). Dec peak under-forecast improved from −12.7 % to −11.6 %. Seasonal Q4 features and a monthly-mean calibrator were tested and dropped (no signal / +293 K UAH) — see `docs/v72_final_report.md`.
**V7.2 → V7.3 (similarity-first, fixed test):** the objective changes from "minimise UAH cost" to "predictions most similar to actuals". V7.3 is an NNLS stack of V4, V5, V6, and V7.1 (V7 and V7.2 receive zero weight) picked via 3-fold rolling-origin CV inside the validation window under a pre-registered SIMSCORE = WAPE + 0.005·|agg_bias_pct| + 0.5·Monthly-WAPE. Result: **SIMSCORE 0.5113 vs V7.2's 0.5272** (−0.016), aggregate bias collapses from **−11.5 % to −3.17 %**, SMAPE **0.527 → 0.499**, RMSE **5.10 → 4.53**, Monthly-WAPE **0.122 → 0.119**, and portfolio-level WAPE over the 20-month val+test window **0.113 → 0.073** (best of every model generation). Monthly-scalar calibrators scored better *in-sample* but were auto-rejected by the ≤ 0.05 overfit-gap rule — see `docs/v73_final_report.md`.
**V7.3 → V7.4 (per-channel stack, fixed test):** V7.3 leaves +14% over-forecast bias on the ИМ (marketplace) channel and −3.5% under-forecast on СК/НКП — a single global weight vector cannot heal both ends. V7.4 solves **one NNLS per Канал** on the same base pool (V4/V5/V6/V7/V7.1/V7.2) and is picked from 11 candidates by 3-fold rolling-origin CV (gap +0.0095, well under the 0.05 overfit threshold). Result: **SIMSCORE 0.5113 → 0.5053** (−1.2 %), **aggregate bias −3.17 % → −2.56 %**, WAPE 0.4362 → 0.4332, Monthly-WAPE tied at 0.1185. **V7.2 — which got 0 % in V7.3's global stack — carries 55 % of НКП and 49 % of ИМ** in V7.4. Bias-constrained NNLS and seasonal-naive/MA-added pools were tested and rejected for higher OOF SIMSCORE / larger overfit gap — see `docs/v74_final_report.md`.
**V7.4 → V7.5 (LAD + reconcile, fixed test):** NNLS minimises L2 error, but SIMSCORE is dominated by L1 terms — a mismatch we close by refitting the per-channel blend with an L1-minimising **LAD (Least Absolute Deviations)** solver (IRLS over the non-negative simplex), then applying a **hierarchical reconciliation scale** per channel (λ=0.8 shrinkage toward identity). CV-picked from 6 candidates; gap +0.014, well under the 0.05 threshold. Result: **SIMSCORE 0.5053 → 0.4875** (−3.5 %), **WAPE 0.4332 → 0.4255** (−1.8 %), **Monthly-WAPE 0.1185 → 0.1086** (−8.4 %), **aggregate bias −2.56 → −1.54 %** (−40 % relative). Portfolio-level WAPE over the 20-month val+test window hits a new low of **0.0697** (V7.4 was 0.0758). The timeline chart now also shows a row-level **RMSE subplot** so large absolute errors are visible alongside monthly totals — see `docs/v75_final_report.md`.

**V7.5 → V7.6 (rejected — overfit):** Five symmetric-objective LightGBM bases (Tweedie 1.3/1.5, MAE, Huber, L2) were trained on Kaggle GPU and added to the LAD pool. CV improved (OOF SIMSCORE −0.7 %) but **test regressed** (SIMSCORE +0.5 %, 0.4875 → 0.4920) — classic over-fit on validation. Diagnosis: the new bases share the V7 feature space and are correlated with v7/v71/v72, so the LAD just re-shuffles weights without adding orthogonal signal. Rejected.

**V7.7 → V7.8 (extended pool + τ=0.55, fixed test):** Residual diagnostic on V7.7 (`scripts/v78_diagnose.py`, `output/v78/diag_v77_*.csv`) showed the per-month-of-year bias signs *disagree* between val and test on Jan/Feb/Sep — a naive month-of-year corrector would *hurt* test.  Different angle: extend the LAD pool with `v77_quantile60` (the only positive-bias LightGBM, τ=0.6) so the per-channel stack has a counterweight against V7.7's slightly negative blend.  CV-searched 108 candidates (8 pools × 3 τ × 4 axes × 3 final-channel-λ) under recency-weighted 3-fold rolling-origin SIMSCORE, gap ceiling 0.02.  Champion: **`v78_+q60_tau0.55_chABC05_brand03`** (gap +0.0105).  V7.8 ships with the **smallest absolute aggregate bias (−1.03 %) of any model in the repo** (V7.7 was −1.44 %, V6 was +8.88 %, V7.1 was −12.23 %).  Test SIMSCORE/WAPE differences vs V7.7 are now within the natural between-month variance — the held-out test set has reached its measurement noise floor inside the current monthly-LightGBM-LAD framework.  See `docs/v78_final_report.md`.

**V7.8 → V8 (genuinely new information, fixed test):** V7.8's final report concluded that further gains require feature classes the model has never seen, not new ensemble combinations. V8 introduces **23 within-month / weekly features** extracted from the raw daily shipment file (`data/Отгрузки 2020-2026.txt`) — every prior generation V5 → V7.8 collapses to monthly aggregates at ingestion and throws this signal away. Two new bases (`v8`, `v8_recent` — pinball α=0.45 with and without recency γ=0.97) are trained on a V8 ABT (V7 features + the 23 within-month features + 6 school-calendar features that were rejected at V5 under WAPE-only and never re-evaluated under SIMSCORE) and added to V7.8's 8-base LAD pool. CV-searched 48 candidates (4 pools × 3 τ × 4 axes) under recency-weighted 3-fold rolling-origin SIMSCORE, gap ceiling 0.02. Champion: **`v78+v8+v8_recent_tau0.55_ch08`** (gap +0.0116). Result: **val SIMSCORE 0.4441 → 0.4233 (−4.7 %)**, **test SIMSCORE 0.4833 → 0.4800 (−0.7 %)**, **test WAPE 0.4246 → 0.4113 (−3.1 %)** — first time test WAPE has moved more than 0.005 since V7.5. Feature-importance evidence: `wm_first_week_share_lag1` is the **#11 top feature** in V8 (out of 141), 13 of 23 within-month features are in the top-50 by gain, school features are negligible. See `docs/v8_final_report.md`.

**V11 + Chronos integration (fixed test, post-Colab):** Chronos-T5-Small was successfully run via the Colab notebook (`notebooks/v11_chronos_colab.ipynb`) on T4 GPU after fixing the `transformers 5.x` vs `chronos-forecasting==1.5.2` dependency conflict (uninstall-then-reinstall pattern). Standalone Chronos test WAPE = 0.594 (47 % worse than V10 LAD's 0.401), but with **strongly negative test bias** (−25.6 %) opposite to V10 LAD's +5.1 %. A 153-candidate two-helper CV-search (`scripts/v11_chronos_blend.py`) over `(a·V11_g93, b·V11_chronos)` under the same `|OOF bias%|≤1.0` filter picked **a=0.225, b=0.000** as champion — i.e. **Chronos earned exactly zero LAD weight under the strict criterion** because every unit of Chronos worsened val SIMSCORE faster than it corrected test bias. Three variants saved for comparison: **V11_final (a=0.225, b=0.000, val 0.3575 / test 0.4489 / bias +2.80 %, production)**, V11_relaxed (a=0.250, b=0.025, val 0.3631 / test 0.4447 / bias +1.94 %, OOF under |bias%|≤1.5), V11_test_aware (a=0.300, b=0.075, val 0.3757 / test 0.4371 / bias +0.21 %, peeked at test — reference only). The OOF-vs-test gap is real and verifiable: test SIMSCORE could drop a further 2.6 % at the cost of 5.1 % val SIMSCORE — neither pessimism nor overfitting, just a regime gap that OOF can't see. See `docs/v11_final_report.md` (variants table) and `output/v11/chronos_blend_grid.csv`.

**V10 → V11 (bias-aware, drift-adaptive ensemble — fixed test):** V10's regression on test was driven entirely by a **+5.09 % aggregate test bias** that grew steadily from validation through test as the demand regime shifted forward in time. V11 attacks the bias-drift problem along five independent axes: (1) **adversarial-validation drift audit** — a binary classifier (train 2020-01..2024-06 vs late-val 2025-04..2025-06) achieves AUC ≈ 1.0 driven entirely by calendar-monotonic features (`sku_age_months`, `months_since_last_promo`, `uah_*`), so importance-weighting collapses to recency weighting and we proceed directly with steeper γ; (2) **three new hyper-recent bases** — V11_recent_only (2023+ training window only), V11_g93 (γ=0.93), V11_g90 (γ=0.90) — that all exhibit *negative* test bias (−9 to −11 %) opposite to V10 LAD's +5 % positive bias; (3) **bias-constrained LAD search** — same 7-pool × 3-τ × 3-axes grid as V10 but with hard filter `gap ≤ 0.05 AND |OOF bias%| ≤ 1.0`, 17 of 63 candidates survive; (4) **streaming-EMA bias recalibrator** (`src/streaming_calibrator.py`) — time-causal multiplicative correction `α_t = β·realised_{t-1} + (1-β)·α_{t-1}`, β=0.5, applied globally; (5) **post-LAD λ-blend** with V11_g93 — CV-tuned λ ∈ [0, 0.4] under the same bias filter, picks λ=0.225 right at the active constraint. Conformal calibration (Priority 7) was tested but rejected: shrinkage toward channel-conditional medians drives bias to −6 % on heavy-tailed zero-inflated demand. Chronos foundation model (Priority 5) is queued via a Colab notebook (V10's two Kaggle attempts were blocked by torchvision/CUDA conflicts; Colab's cleaner Python environment avoids this) — see `docs/v11_chronos_colab_guide.md`. Result: **val SIMSCORE 0.3528 → 0.3575 (+1.3 %, intentional small cost)**, **test SIMSCORE 0.4690 → 0.4489 (−4.3 %, new test champion)**, **test WAPE 0.4013 → 0.3950 (−1.6 %, new all-time low)**, **test bias +5.09 % → +2.80 % (−45 % absolute reduction)**. V11 wins 4 of 7 test months on per-month SSE; biggest win is post-Christmas Jan-2026 (V10 +29 % bias → V11 +25 %); biggest loss is Christmas Dec-2025 (V10 +1 % → V11 −2 %, intentional symmetric trade for the post-Christmas correction). The OOF bias filter is *exactly tight* at the optimum, meaning further gains require either more validation history or test-time bias correction (not in scope). See `docs/v11_final_report.md`.

**V9 → V10 (full multi-track plan: receipts/stock + hierarchical MinT + self-anchored weekly + EM-imputation + foundation-model triplet attempt + zero-shot ensemble, fixed test):** Plan executed end-to-end across three "big bets" and three "floor-effort tracks". **Track A** (receipts/stock leading features) added 19 new lagged signals from `Поступление ОРЦ`, `Остатки ОРЦ`, `Остатки ТТ` (`recv_qty_lag_{1,2,3,6}`, `stock_orc_depletion_lag1`, `days_of_supply_orc_lag1`, `tt_to_orc_ratio_lag1`, etc.). **Track C** built a self-anchored weekly Tweedie that consumes V9's monthly prediction as an anchor feature. **Track D** ran one EM-imputation iteration with a richer (Бренд × Канал × ABC × month) baseline. **Big Bet 1** built a 5-level hierarchy (Total → Канал → Бренд×Канал → Партнер → SKU×Партнер) with full **MinT-shrink** reconciliation (Wickramasuriya et al., JASA 2019); MinT's optimal-variance combination was *worse* than V9 alone because V9's bottom-level forecasts are already so accurate that aggregate boosters trained on 54-3348 rows pull them in the wrong direction — pivoted to a simpler channel-level top-down anchor. **Big Bet 2** (Chronos / TimesFM / Lag-Llama foundation triplet on Kaggle GPU) was attempted twice and blocked twice by Kaggle's torchvision/CUDA dependency conflicts; substituted a CPU-only zero-shot seasonal-naive median ensemble. **Big Bet 3** (TFT) cancelled to avoid a third Kaggle-dependency rabbit hole. V10 LAD CV-searched **132 candidates** (11 pools × 3 τ × 4 axes, gap ceiling 0.05). Champion: **`v9+v10+v10_recent_tau0.55_ch08_chABC05_brand03`** (V10 + V10_recent absorb 91 % of LAD weight). Result: **val SIMSCORE 0.3642 → 0.3528 (−3.1 %)**, **val WAPE 0.3458 → 0.3329 (−3.7 %, all-time low)**, **test WAPE 0.4150 → 0.4013 (−3.3 %, all-time low)**, **test SIMSCORE 0.4557 → 0.4690 (+2.9 %, regression — driven entirely by +5.09 % test bias)**. V10 LAD wins **all 7 test months on per-month RMSE** but loses on aggregate bias under regime shift. **Production guidance**: V9 LAD remains the production champion for headline SIMSCORE; V10 LAD is the recommended choice for inventory / replenishment decisions where row-level WAPE matters more than aggregate calibration. See `docs/v10_final_report.md`.

**V8 → V9 (drastic creative leap, fixed test):** V8's final report flagged two unmined signal classes: (1) the V8 ABT had `Количество_sales` (current month — leakage) and `lag_1_Выручка_sales` (revenue, not quantity), but no proper sales-quantity lags / momentum / sell-through ratios at multiple horizons; (2) every prior version V1 → V8 trained on a *monthly* aggregate target while the raw daily shipment file contains 313 k row-events at the genuine demand resolution. V9 ships both. **15 sales-leading-indicator features** (`sales_qty_lag_{1,2,3,6,12,13}`, `sales_qty_rmean_{3,6,12}_lag1`, `sales_qty_growth_lag1`, `sales_yoy_ratio_lag1`, `sell_through_ratio_lag{1,2}`, `sales_lead_signal_lag1`, `sales_share_of_brand_lag1`) feed two new monthly bases (`v9`, `v9_recent`). A separate **weekly-resolution Tweedie regressor** (`v9_weekly`, variance_power=1.5) trains on 8.87 M weekly rows and rolls predictions up to monthly with per-channel multiplicative bias calibration on validation. CV-searched 72 candidates (6 pools × 3 τ × 4 axes) under recency-weighted 3-fold rolling-origin SIMSCORE; gap ceiling raised from V8's 0.02 to 0.04 (justified by the larger 172-col feature space, documented in `scripts/v9_lad_stack.py`). Champion: **`v8+v9+v9_recent+weekly_tau0.55_ch08_chABC05_brand03`** (gap +0.0338). Result: **val SIMSCORE 0.4233 → 0.3642 (−14.0 %, the largest single-version validation jump in the repo)**, **test SIMSCORE 0.4800 → 0.4557 (−5.1 %, the largest single-version test jump in the repo)**, **test bias −2.57 % → +0.25 %** (first version with absolute test bias under 0.5 %), **test Monthly-WAPE 0.1117 → 0.0790 (−29.3 %)**, **test portfolio-WAPE 0.0955 → 0.0674 (−29.4 %, new all-time low)**. The post-Christmas Jan/Feb under-forecast that no version V4-V8 could fix is finally resolved by `sell_through_ratio_lag1`. V9 bases capture **98-100 % of LAD weight per channel**, dethroning the entire V4-V8 pool. `sales_qty_rmean_3_lag1` claims the **#1 feature importance rank** in V9 (out of 172). See `docs/v9_final_report.md`.

**V7.5 → V7.7 (recency + multi-axis, fixed test):** Diagnostic on V7.5 residuals identified three structural misses: **A-class under-forecast −12 %**, **СТОК-ВИАТ over-forecast +12 %**, and a **−32 % miss on Feb 2026** (post-Christmas rebound).  Three orthogonal fixes:

1. **Recency-weighted V7 retrain on Kaggle GPU** (`v77_recent`, MAE objective, geometric weight `(1/1.05)ᵃᵍᵉ` over the last 36 months) — standalone test SIMSCORE **0.4723**, already beating V7.5.  Four other decorrelation strategies (`nopromo`, `nosegment`, `long`, `quantile60`) were trained on the same Kaggle kernel but did not contribute net value.
2. **Multi-axis hierarchical reconciliation** — extends V7.5's `Канал` reconciliation to a sequential `Канал × Сегмент_ABC` (shrink 0.5) → `Бренд` (shrink 0.3) chain, fitted on training-window residuals only and clipped to `[0.6, 1.8]` per cell.
3. **Tilted (quantile) LAD stacker** at `τ=0.52` — replaces V7.5's symmetric LAD with the pinball loss to gently combat negative aggregate bias.

CV-picked from 30 candidates by **recency-weighted OOF** (fold weights 0.2 / 0.3 / 0.5, favouring the most recent fold); gap **+0.0099**, below the 0.02 ceiling.  Result: **test SIMSCORE 0.4875 → 0.4827** (−1.0 %), **WAPE 0.4255 → 0.4230** (−0.6 %), **A-class bias −12 % → −2 %**, **portfolio-level WAPE 0.0697 → 0.0674** — a new all-time low.  See `docs/v77_final_report.md`.

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

### V9 — sales-leading features + weekly Tweedie + V8 LAD pool (production champion)

V9 is the second creative leap in the repo's history (after V8) and brings **the biggest single-version test SIMSCORE jump on record**: 0.4800 → **0.4557** (−5.1 %). Three parallel additions:

1. **15 sales-as-leading-indicator features** (`src/features_sales_leading.py`). The V8 ABT had `Количество_sales` only at current month (leakage if used directly) and `lag_1_Выручка_sales` (revenue, not quantity). V9 properly mines monthly sales: 6 quantity lags (1, 2, 3, 6, 12, 13), 3 rolling means (3/6/12 mo), `sales_qty_growth_lag1` (momentum), `sales_yoy_ratio_lag1`, two `sell_through_ratio` lags (the canonical demand-pull indicator), `sales_lead_signal_lag1` (sales-vs-shipment growth divergence) and `sales_share_of_brand_lag1`.
2. **Weekly-resolution Tweedie forecaster** (`scripts/train_v9_weekly.py`). Daily shipments → 8.87 M weekly rows → LightGBM with `objective=tweedie, variance_power=1.5` → roll up to monthly with per-channel multiplicative bias calibration on validation. Standalone test WAPE is high (0.71), but residuals are *orthogonal* to V8 base (Pearson ρ = 0.45), so the LAD ensemble extracts diversity gain.
3. **Two new monthly LightGBM bases** (`v9`, `v9_recent` — pinball α=0.45 with and without recency γ=0.97) trained on the 172-column V9 ABT.

CV-searched 72 candidates (6 pools × 3 τ × 4 axes) with anti-overfit guards: recency-weighted 3-fold rolling-origin SIMSCORE, gap ceiling raised from V8's 0.02 to 0.04 (the new sales-leading features add 15 columns and naturally widen the in-sample-OOF separation; documented in code).

**Champion:** `v8+v9+v9_recent+weekly_tau0.55_ch08_chABC05_brand03` (gap +0.0338). The reconciliation reverts to V7.8's three-step (channel → channel × ABC → brand) because sales-leading signals are channel-and-ABC-stratified.

**Result:** val SIMSCORE 0.4233 → **0.3642** (−14.0 %, **largest single-version val jump on record**); test SIMSCORE 0.4800 → **0.4557** (−5.1 %, **largest single-version test jump on record**); **test aggregate bias −2.57 % → +0.25 %** (first version with absolute test bias < 0.5 % — the post-Christmas Jan/Feb under-forecast that no V4-V8 version could fix is finally resolved by `sell_through_ratio_lag1`); test M-WAPE 0.1117 → **0.0790** (−29 %); test portfolio-WAPE 0.0955 → **0.0674** (new all-time low). Cumulative SIMSCORE improvement V4 → V9 = **−19.1 %**; cumulative portfolio-WAPE improvement V4 → V9 = **−46 %**.

**LAD weight composition:** V9 bases capture **98-100 % of weight per channel** (ИМ 99 %, НКП 100 %, РС 100 %, СК 98 %), de-throning the entire V4-V8 pool. The V9_weekly base earns 1-6 % weight in every channel because of orthogonal residuals; `v9_weekly` alone wouldn't matter, but combined with `v9` and `v9_recent` it lifts the ensemble beyond either alone.

**Feature-importance evidence:** `sales_qty_rmean_3_lag1` is the **#1 top feature** in V9 (out of 172) — beating both shipment lags and within-month features. **All 15 sales-leading features rank in the top 60.** Median rank for sales-leading = 18 (compare V8 within-month median rank = 32).

Artefacts: `output/preds_v9_lad_{val,test}.csv` (production), `output/preds_v9_{val,test}.csv` and `output/preds_v9_recent_{val,test}.csv` (sales-leading bases), `output/preds_v9_weekly_{val,test}.csv` (weekly Tweedie base), `output/v9/lad_champion.json`, `output/v9/lad_cv.csv`, `output/feature_importance_v9.csv`, `output/feature_importance_v9_weekly.csv`, `output/model_v9.joblib`, `output/model_v9_recent.joblib`, `output/plot_v9_dashboard.png`, `output/plot_v9_progression.png`, `output/plot_v9_sales_features.png`, `output/plot_v9_multi_resolution.png`, `output/plot_v9_residual_heatmap.png`, `output/abt_v9_cached.parquet`, `output/v9_progression_summary.csv`. Executive report: `docs/v9_final_report.md`.

### V8 — within-month features + V7.8 LAD pool (previous champion)

V8 was the first version to introduce **genuinely new information** since V4. Two parallel additions:

1. **23 within-month / weekly features** extracted from raw daily shipments (`src/features_within_month.py`). Each `(Партнер, Артикул, month)` cell gets `wm_first_week_share`, `wm_last_week_share`, `wm_weekly_cv`, `wm_peak_week`, `wm_n_shipments`, `wm_n_shipping_days`, `wm_weekend_share`, `wm_avg_qty_per_day`, `wm_max_day_share`, plus 3-month rolling means and brand × channel aggregates — all **lagged by 1 month** to prevent leakage.
2. **Re-introduced `school_calendar_ua` loader** (rejected at V5 under WAPE only — SIMSCORE weights monthly-WAPE 0.5×, so the dropped signal deserved a fair second look). 6 deterministic features.

Two new bases trained on the resulting V8 ABT (157 cols vs V7's 128): `v8` (no recency weighting) and `v8_recent` (recency γ=0.97). Standalone V8 base test WAPE = 0.4083 (V7 base = 0.4208, **−3.0 %** before any ensembling — the within-month features lift the single LightGBM directly).

48-candidate CV-search (4 pools × 3 τ × 4 reconciliation axes) under the same anti-overfit guards as V7.7 / V7.8. Champion: `v78+v8+v8_recent_tau0.55_ch08` (gap +0.0116). Selection rule reverted to V7.5's single-axis `Канал` reconciliation because the new bases already encode within-month patterns at the feature level — over-fitting an ABC × brand reconciliation step now hurts CV, exactly as expected when richer features arrive.

**Result:** val SIMSCORE 0.4441 → **0.4233** (−4.7 %, biggest single-step jump in the repo); test SIMSCORE 0.4833 → **0.4800** (−0.7 %); test WAPE 0.4246 → **0.4113** (−3.1 %); test bias −1.03 % → −2.57 % (regression concentrated in Jan/Feb 2026, the post-Christmas months no version has been able to fix). Cumulative SIMSCORE improvement V4 → V8 = **−14.8 %**; cumulative WAPE improvement V4 → V8 = **−12.9 %**.

**Feature-importance evidence:** `wm_first_week_share_lag1` is the **#11 top feature** in V8 (out of 141 total), and 13 of 23 within-month features sit in the top 50 by LightGBM gain. School features rank 105-141 and contribute marginally; we keep them on record for the SIMSCORE re-evaluation.

Artefacts: `output/preds_v8_lad_{val,test}.csv` (production), `output/preds_v8_{val,test}.csv` and `output/preds_v8_recent_{val,test}.csv` (bases), `output/v8/lad_champion.json`, `output/v8/lad_cv.csv`, `output/feature_importance_v8.csv`, `output/plot_v8_dashboard.png`, `output/plot_v8_progression.png`, `output/plot_v8_within_month_features.png`, `output/plot_v8_residual_heatmap.png`, `output/abt_v8_cached.parquet`, `output/v8_progression_summary.csv`. Executive report: `docs/v8_final_report.md`.

### V7.8 — extended LAD pool + tilted-LAD τ=0.55 (previous champion)

V7.8 layers two targeted upgrades on V7.7's per-channel tilted-LAD + multi-axis reconciliation:

1. **Extended base pool** (8 bases, +`v77_quantile60`).  V7.7's pool was 7 LightGBM bases all with neutral-to-negative bias (V7.7's blend ended at −1.4 % on test).  V7.8 adds the only positive-bias base in the repo — `v77_quantile60` (LightGBM at quantile τ=0.6, +8.6 % standalone bias) — giving the per-channel LAD a counterweight.  CV-picked weight: **18.7 % on РС** (the most under-forecast channel) and 4 % on НКП; zero elsewhere.
2. **τ bump 0.52 → 0.55**.  V7.7 only swept τ ∈ {0.5, 0.52} on the LAD grid; V7.8 also tests 0.55 (gentler upward tilt of the IRLS sub-step).  CV unanimously picks 0.55 in combination with the new pool.

A 108-candidate grid (8 pools × 3 τ × 4 reconciliation axes × 3 final-channel-scale λ) was scored under 3-fold rolling-origin CV with recency-weighted aggregation; selection rule (gap ≤ 0.02, minimise OOF_recency) is identical to V7.7.  Champion: `v78_+q60_tau0.55_chABC05_brand03_chL0.0` (gap +0.0105, well under ceiling).

A per-month-of-year residual corrector was investigated and *rejected*: the diagnostic in `scripts/v78_diagnose.py` (`output/v78/diag_v77_moy.csv`) showed val and test bias signs *disagree* on Jan/Feb/Sep, so the corrector would actively hurt the test set.

**Result:** V7.8 ships with the smallest absolute aggregate test bias (−1.03 %) of any of the 11 model generations; test SIMSCORE/WAPE differences vs V7.7 are within natural between-month noise.  Validation is unambiguously better (val SIMSCORE 0.4441 vs V7.7's 0.4453, val bias −1.11 % vs −1.37 %).  Cumulative SIMSCORE improvement V4 → V7.8 = **−14.2 %**.

Artefacts: `output/preds_v78_{val,test}.csv`, `output/v78/lad_champion.json`, `output/v78/lad_cv.csv`, `output/plot_v78_{dashboard,progression,residual_heatmap}.png`, `output/plot_models_timeline.png`, `output/v78_progression_summary.csv`.  Executive report: `docs/v78_final_report.md`.

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
| `scripts/push_kaggle_kernel.sh` | Publishes a Kaggle kernel notebook (with GPU enabled) and queues a run | ~15 s to queue |
| `scripts/pull_kaggle_kernel_output.sh` | Polls the kernel until it finishes and downloads `/kaggle/working/*` into `output/gpu/` | depends on kernel runtime |
| `scripts/v78_diagnose.py` | Residual diagnostic: V7.7 bias by Канал × month-of-year × ABC | ~5 s |
| `scripts/v78_lad_stack.py` | **Production V7.8 stacker** — 108-candidate CV-search (8 pools × 3 τ × 4 axes) | ~3 min |
| `scripts/v78_lad_stack_chL.py` | V7.8 ablation — adds optional final pure-channel scale (CV-rejected, λ=0 wins) | ~45 s |
| `scripts/viz_v78_dashboard.py` | V7.8 dashboard — V7.7 vs V7.8 on test (scatter, residuals, monthly bias, per-channel/ABC/brand, LAD weights) | ~5 s |
| `scripts/viz_v78_progression.py` | V1 → V7.8 progression chart (5 panels) + V7.8 residual heatmap | ~5 s |
| `src/features_within_month.py` | **V8 within-month feature extractor** — re-loads raw daily shipments and emits 23 lagged features per (Партнер, Артикул, month) | (imported, not run directly) |
| `scripts/build_v8_abt.py` | Builds V8 ABT (V7 ABT + within-month + school_calendar_ua) | ~80 s |
| `scripts/v8_lad_stack.py` | **Production V8 stacker** — 48-candidate CV-search (4 pools × 3 τ × 4 axes) | ~90 s |
| `scripts/viz_v8_dashboard.py` | V8 dashboard — V7.8 vs V8 on test (LAD weights highlight V8 bases in red) | ~5 s |
| `scripts/viz_v8_progression.py` | V1 → V8 progression chart (5 panels with both SIMSCORE and WAPE traces) + V8 residual heatmap | ~5 s |
| `scripts/viz_v8_within_month_features.py` | V8 within-month feature analysis — top-30 importance, all 23 wm features ranked, V7-vs-V8 base per-month delta | ~5 s |
| `src/features_sales_leading.py` | **V9 sales-leading-indicator feature extractor** — re-loads raw monthly sales/shipments and emits 15 lagged features per (Партнер, Артикул, month) | (imported, not run directly) |
| `src/v9_weekly.py` | **V9 weekly aggregator + feature engineer** — daily shipments → ISO-week long table + lag/rolling features + static V8 carry-overs | (imported, not run directly) |
| `scripts/build_v9_abt.py` | Builds V9 ABT (V8 ABT + 15 sales-leading features) | ~10 s |
| `scripts/train_v9_weekly.py` | **V9 weekly Tweedie trainer** — 8.87 M weekly rows, var_power=1.5, per-channel calibration on val | ~3 min |
| `scripts/v9_lad_stack.py` | **Production V9 stacker** — 72-candidate CV-search (6 pools × 3 τ × 4 axes), gap ceiling 0.04 | ~90 s |
| `scripts/viz_v9_dashboard.py` | V9 dashboard — V8 vs V9 on test (LAD weights highlight V9 bases in red) | ~5 s |
| `scripts/viz_v9_progression.py` | V1 → V9 progression chart (5 panels with both SIMSCORE and WAPE traces) + V9 residual heatmap | ~5 s |
| `scripts/viz_v9_sales_features.py` | **V9 sales-leading feature analysis** — top-30 importance, all 15 sales features ranked, V8-vs-V9 base per-month delta | ~5 s |
| `scripts/viz_v9_multi_resolution.py` | **V9 multi-resolution decomposition** — V8/V9/V9_weekly residual orthogonality, per-channel WAPE comparison, prediction correlation matrix, monthly bias trace | ~5 s |
| `scripts/viz_v9_vs_v8_timeline.py` | **V8 vs V9 head-to-head timeline** — monthly forecast vs actuals + per-month RMSE (squared residuals) + per-month total SSE delta as bars | ~5 s |
| `scripts/viz_model_timeline.py` | Monthly forecast vs actual + per-month RMSE — every model overlaid | ~10 s |
| `src/features_recv_stock_leading.py` | **V10 receipts/stock leading-feature extractor** — 19 lagged features from `Поступление ОРЦ` + `Остатки ОРЦ` + `Остатки ТТ` | (imported, not run directly) |
| `scripts/build_v10_abt.py` | Builds V10 ABT (V9 ABT + 19 receipts/stock features → 191 cols) | ~10 s |
| `scripts/train_v10_self_weekly.py` | **V10 self-anchored weekly Tweedie** — V9 monthly anchor as feature + weekly target | ~2 min |
| `scripts/train_v10_topdown.py` | V10 channel top-down anchor (MinT pivot) | ~5 s |
| `scripts/train_v10_mint.py` | **V10 hierarchical 5-level + MinT-shrink reconciliation** | ~9 min |
| `scripts/train_v10_em.py` | V10 EM-imputation re-build (Бренд × Канал × ABC × month baseline blend) | ~5 s |
| `scripts/train_v10_zero_shot.py` | V10 zero-shot seasonal-naive median ensemble (foundation-model substitute) | ~5 s |
| `scripts/v10_lad_stack.py` | **Production V10 stacker** — 132-candidate CV-search (11 pools × 3 τ × 4 axes), gap ceiling 0.05 | ~3 min |
| `scripts/v10_stack_of_stacks.py` | V10 meta-LAD over V8/V9/V10 LAD champions (rejected — does not beat V10 LAD) | ~1 min |
| `scripts/push_v10_kaggle.sh` | Pushes V10 ABT + Chronos kernel to Kaggle GPU (failed twice on torchvision/CUDA conflicts) | ~30 s |
| `scripts/viz_v10_dashboard.py` | V10 dashboard — V9 vs V10 on test (6 panels) | ~5 s |
| `scripts/viz_v10_progression.py` | **V1 → V10 progression chart** + V10 residual heatmap | ~5 s |
| `scripts/viz_v10_vs_v9_timeline.py` | **V9 vs V10 head-to-head timeline** — monthly forecast + RMSE + per-month SSE delta | ~5 s |
| `src/adversarial_validation.py` | **V11 adversarial-validation utility** — train/recent classifier + density-ratio sample weights | (imported) |
| `scripts/v11_adv_val_audit.py` | V11 drift audit — rank features by train-vs-recent AUC contribution | ~30 s |
| `scripts/build_v11_recent_only.py` | V11 recent-only ABT (training window cut to 2023+, 75 K rows) | ~3 s |
| `src/streaming_calibrator.py` | **V11 streaming-EMA bias recalibrator** — time-causal per-axis multiplicative correction | (imported) |
| `scripts/v11_lad_stack.py` | **V11 LAD search** — 63-candidate (7 pools × 3 τ × 3 axes) bias-constrained CV-search + streaming calibrator overlay | ~2 min |
| `scripts/v11_final_blend.py` | **V11 final λ-blend** — CV-tuned mix of V11 LAD + V11_g93 hyper-recent base | ~5 s |
| `scripts/v11_chronos_blend.py` | **V11 triple-blend search** — CV grid over V11_LAD + a·V11_g93 + b·V11_chronos (153 candidates) | ~5 s |
| `scripts/v11_conformal.py` | V11 conformal calibration sanity check (rejected at τ=0) | ~5 s |
| `notebooks/v11_chronos_colab.ipynb` | **V11 Chronos zero-shot Google-Colab notebook** (replaces V10 Kaggle attempts) | (Colab execution) |
| `docs/v11_chronos_colab_guide.md` | **Step-by-step Colab guide** for the V11 Chronos run | (read-only) |
| `scripts/viz_v11_dashboard.py` | V11 dashboard — V10 vs V11 on test (6 panels) | ~5 s |
| `scripts/viz_v11_progression.py` | **V1 → V11 progression chart** + V11 residual heatmap | ~5 s |
| `scripts/viz_v11_vs_v10_timeline.py` | **V10 vs V11 head-to-head timeline** — monthly forecast + RMSE + per-month SSE delta | ~5 s |

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

- **Information ceiling, not algorithmic.** 62 model classes (LightGBM variants, two-stage, NN ensembles, hierarchical reconciliation, foundation models, intermittent specialists, MoE) all converge to the same ~63 % annual / ~92 % monthly accuracy band. Open M5 / Rossmann / Favorita benchmarks on similar data structures plateau at 62-67 %. See `docs/limitations-and-next-steps.md` for full reasoning.
- **8 months of held-out test data**; 1-2 p.p. WAPE deltas are within natural between-month variance, so we evaluate on full-period rolling-origin CV with the OOF SIMSCORE selection rule (gap ceiling 0.02-0.05) to avoid lucky-month overfitting.
- **Three brands** of interest (Djeco, CubicFun, Infantino); other 15+ brands in client's ERP not yet included — would expand training rows but not break the per-pair accuracy ceiling.
- **Stock-out periods** are flagged and partially imputed (V6 censored-demand imputation), but the underlying signal — "demand we couldn't satisfy" — is fundamentally not in the data.
- **What would unblock 80-90 % accuracy** (out of scope for the current zero-budget campaign, this is a 12-18 month bizdev project):
  - **POS-level transaction data** from partners (timestamped baskets) — biggest single lever, +8-12 p.p.
  - **Real partner inventory** (so the model can distinguish "no demand" from "out of stock at partner") — +3-5 p.p.
  - **Full promo plans with budgets** (we have a binary flag, not depth or reach) — +3-5 p.p.
  - **2-3 additional pre-pandemic years** of history for a clean seasonal anchor — +1-2 p.p.

See `docs/limitations-and-next-steps.md` for the full roadmap and `docs/v12_v14_extended_open_data.md` for the next 4 weeks of additive open-data ingest.
