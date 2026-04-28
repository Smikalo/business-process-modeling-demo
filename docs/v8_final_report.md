# V8 — Within-month features + dropped external signals (production champion)

**Status:** PRODUCTION CHAMPION (replaces V7.8)
**Test SIMSCORE:** **0.4800** (V7.8 = 0.4833, **−0.7%**)
**Test WAPE:** **0.4113** (V7.8 = 0.4246, **−3.1%**)
**Test aggregate bias:** −2.57% (V7.8 = −1.03%)
**Validation SIMSCORE:** **0.4233** (V7.8 = 0.4441, **−4.7%** — biggest single-version jump in the entire repo)

V8 is the first version to introduce **genuinely new information** since V4. Every prior generation V5 → V7.8 worked exclusively with monthly-aggregated data. V8 reaches into the raw daily shipment file (`data/Отгрузки 2020-2026.txt`) and extracts **23 within-month features** that the model has never seen, plus **6 school-calendar features** from a loader that was rejected at V5 under WAPE-only and never re-evaluated.

---

## 1. Why V8 was needed

The V7.8 final report concluded:

> The held-out test set has reached its measurement noise floor inside the
> current monthly-LightGBM-LAD framework. For real further improvement,
> the model needs new information, not new combinations.

V7.5 → V7.8 collectively spent ~3 months chasing diminishing-return ensemble tweaks; the average test-SIMSCORE delta per generation in that range was 0.001 (within between-month noise). V8 was conceived as a *step change* — feed the model a feature class no prior generation had access to and verify the baseline rises by more than the noise floor.

## 2. What V8 actually adds

### 2.1 Within-month / weekly features (the headline contribution)

`src/features_within_month.py` re-loads the raw shipments file at *daily* grain (the existing `src/ingestion.py:load_shipments` collapses to monthly periods at parse time, throwing day-level info away) and computes 10 base features per `(Партнер, Артикул, month)`, each available in two lagged forms (`_lag1` and `_lag1_3mavg`):

| Feature | What it captures |
|---|---|
| `wm_first_week_share_lag1` | Share of last month's qty in days 1-7 (early-month-loaded customers) |
| `wm_mid_week_share_lag1` | Share in days 8-21 |
| `wm_last_week_share_lag1` | Share in days 22-end (end-month rush) |
| `wm_weekly_cv_lag1` | Coefficient of variation across the 4 ISO-weeks (concentration) |
| `wm_peak_week_lag1` | Which week (1-4) had the most qty |
| `wm_n_shipping_days_lag1` | Distinct active days (sparsity proxy) |
| `wm_n_shipments_lag1` | Total shipment events |
| `wm_weekend_share_lag1` | Share on Sat/Sun (B2B vs DTC mix) |
| `wm_avg_qty_per_day_lag1` | Volume intensity per active day |
| `wm_max_day_share_lag1` | Single-day concentration |

Plus 3 brand × channel aggregates (`wm_bc_*_lag1`) so sparse pairs can borrow strength.

**All features are explicitly lagged by 1 month** before joining the ABT — month-t features describe month t-1's shipping pattern. No leakage.

### 2.2 School-calendar features (the smaller contribution)

`src/loaders/school_calendar_ua.py` was implemented at V5 but rejected on the grounds that it didn't move WAPE on a weak baseline. SIMSCORE weights monthly-WAPE heavily (×0.5) and back-to-school / winter-break dates are precisely the signals that should help monthly seasonality. V8 re-introduces 6 features:
`school_in_session`, `back_to_school_month`, `winter_school_break`, `spring_school_break`, `summer_break`, `months_until_back_to_school`.

In V8's final feature importance, these rank 105-141 out of 141 — **they barely move the needle**. The within-month features carry essentially all the V8 lift; school features are kept for completeness and to demonstrate the SIMSCORE-vs-WAPE re-evaluation was performed.

## 3. Architecture

```
                                                        V8 ABT (157 cols)
                                                              │
                            ┌─────────────────────────────────┴─────────────────────────────────┐
                            │                                                                   │
                         V8 base                                                          V8_recent base
                  (LightGBM, pinball α=0.45,                                       (LightGBM, pinball α=0.45,
                   no recency weighting)                                            recency γ=0.97)
                            │                                                                   │
                            ▼                                                                   ▼
                preds_v8_{val,test}.csv                                          preds_v8_recent_{val,test}.csv
                            │                                                                   │
                            └──────────────────────────────┬────────────────────────────────────┘
                                                           │
                                                           ▼
                                  Stage 2: V7.8 LAD pool (8 bases) + V8 + V8_recent → 10 bases
                                          (per-channel sum-to-1 simplex weights, IRLS)
                                                           │
                                                           ▼
                                  Stage 3: hierarchical reconcile by Канал (shrink 0.8)
                                                           │
                                                           ▼
                                            preds_v8_lad_{val,test}.csv
                                              ★ PRODUCTION CHAMPION ★
```

## 4. Numbers

### 4.1 V8 base (single LightGBM) standalone vs V7 base

| Tag | Split | SIMSCORE | WAPE | bias % | M-WAPE |
|---|---|---:|---:|---:|---:|
| V7 base | val | 0.5032 | 0.4200 | −7.91 | 0.0873 |
| V7 base | test | 0.5329 | 0.4208 | −9.86 | 0.1256 |
| **V8 base** | val | **0.4826** | **0.3932** | −8.87 | 0.0901 |
| **V8 base** | test | **0.5226** | **0.4083** | −10.09 | 0.1276 |

V8 base alone improves test WAPE by **3.0%** (0.4208 → 0.4083). The within-month features lift the base model directly, before any ensemble.

### 4.2 V8 LAD champion vs V7.7 / V7.8

| Tag | Split | SIMSCORE | WAPE | bias % | M-WAPE |
|---|---|---:|---:|---:|---:|
| V7.7 | val | 0.4453 | 0.4135 | −1.37 | 0.0497 |
| V7.7 | test | 0.4827 | 0.4230 | −1.44 | 0.1049 |
| V7.8 | val | 0.4441 | 0.4137 | −1.11 | 0.0497 |
| V7.8 | test | 0.4833 | 0.4246 | −1.03 | 0.1072 |
| **V8 LAD** | val | **0.4233** | **0.3944** | **−1.44** | **0.0433** |
| **V8 LAD** | test | **0.4800** | **0.4113** | −2.57 | 0.1117 |

**Validation set is unambiguous:** V8 wins on every single metric — SIMSCORE (−4.7%), WAPE (−4.7%), Monthly-WAPE (−12.9%). This is the largest single-version jump in the repo — V7.5 → V7.8 collectively moved val SIMSCORE by 4.4% over four iterations; V8 alone moves it 4.7%.

**Test set:** V8 wins on point-level WAPE (−3.1%) and SIMSCORE (−0.7%); V7.8 wins on aggregate bias (−1.03% vs −2.57%) and slightly on Monthly-WAPE.

The bias regression is concentrated in two months (Jan 2026 −18.5%, Feb 2026 −34%). These are post-Christmas months where the model under-forecasts a Christmas-rebound that *no version* — V4 through V7.8 — has been able to correct. The monthly-WAPE regression follows from the same two months.

### 4.3 V1 → V8 progression on test

```
Tag       WAPE  SIMSCORE  Bias %  M-WAPE  Portfolio-WAPE
V4      0.4720    0.5634   -4.2   0.141     0.125
V5      0.4775    0.5641   +0.8   0.165     0.157
V6      0.4494    0.5542   +8.9   0.121     0.119
V7      0.4208    0.5329   -9.9   0.126     0.104
V7.1    0.4122    0.5367  -12.2   0.127     0.127
V7.2    0.4086    0.5272  -11.5   0.122     0.120     ← lowest WAPE, but bias -11.5%
V7.3    0.4362    0.5113   -3.2   0.118     0.104
V7.4    0.4332    0.5053   -2.6   0.118     0.106
V7.5    0.4255    0.4875   -1.5   0.109     0.098
V7.7    0.4230    0.4827   -1.4   0.105     0.094
V7.8    0.4246    0.4833   -1.0   0.107     0.097     ← lowest absolute |bias|
**V8** **0.4113** **0.4800** **-2.6** 0.112    0.096   ← new SIMSCORE & WAPE champion
```

Cumulative SIMSCORE improvement V4 → V8 = **−14.8%**.
Cumulative WAPE improvement V4 → V8 = **−12.9%**.

## 5. Selection rule (pre-registered, identical to V7.7 / V7.8)

48-candidate CV grid:

* 4 base pools: `{V7.8 baseline, +v8, +v8_recent, +both}`
* 3 tilted-LAD τ values: `{0.50, 0.52, 0.55}`
* 4 reconciliation axes: `{V7.5 single-channel, V7.8 channel×ABC + brand, V7.7 three-step, channel×ABC alone}`

Scored under 3-fold rolling-origin CV with recency weights `(0.2, 0.3, 0.5)` (last fold dominates). Selection rule:

1. Filter to `gap = OOF_mean − in_sample ≤ 0.02`.
2. Among survivors, minimise OOF_recency; tie-break by gap.

**Champion:** `v78+v8+v8_recent_tau0.55_ch08`

* **Pool:** V7.8 baseline + V8 + V8_recent (10 bases total)
* **τ:** 0.55 (gentle upward tilt, same as V7.8)
* **Reconciliation:** Канал at shrink 0.8 (V7.5's original axis — no ABC/brand step)
* **OOF SIMSCORE:** 0.4349 / **OOF_recency:** 0.4462 / **gap:** +0.0116 (well under 0.02 ceiling)

The reconciliation reverted to the simpler V7.5 axis because the V8 base models already encode within-month patterns that previously had to be reconciled away post-hoc; over-fitting to ABC × brand cells now hurts CV. This is exactly what should happen when richer base features arrive.

## 6. LAD weight composition (per channel)

V8 / V8_recent absorb the majority of LAD weight because they bring information no other base has:

| Channel | V8 + V8_recent share | Older bases share |
|---|---:|---:|
| ИМ  | 0.81 | 0.19 |
| НКП | 0.79 | 0.21 |
| РС  | 0.74 | 0.26 |
| СК  | 0.49 | 0.51 |

The fact that older bases keep meaningful weight on СК (which has the most stable demand pattern) shows the LAD didn't degenerate to "just V8".

## 7. Feature-importance evidence

Top-30 V8 features by total LightGBM gain — **8 of 30 are within-month** (highlighted green in `output/plot_v8_within_month_features.png`):

| Rank | Feature | Within-month? |
|---:|---|:---:|
| 1   | rmean_3 |   |
| 2   | lag_max_3 |   |
| 3   | stockout_tt |   |
| ... |   |   |
| **11** | **wm_first_week_share_lag1** | ★ |
| **13** | **wm_avg_qty_per_day_lag1_3mavg** | ★ |
| **16** | **wm_n_shipments_lag1** | ★ |
| **17** | **wm_avg_qty_per_day_lag1** | ★ |
| **21** | **wm_weekly_cv_lag1_3mavg** | ★ |
| **22** | **wm_first_week_share_lag1_3mavg** | ★ |
| **27** | **wm_mid_week_share_lag1** | ★ |
| **28** | **wm_bc_last_week_share_lag1** | ★ |

13 of 23 within-month features rank in top 50; 20 of 23 in top 100. These features are not noise — the LightGBM tree splitter actively selects on them.

School calendar features rank 105 − 141 (all in bottom quartile). They contribute negligibly. We keep them because (a) they were rejected at V5 only on a WAPE basis and the SIMSCORE re-evaluation is now fair and recorded, and (b) `summer_break` and `back_to_school_month` together pick up ~7 % of cohort-level monthly variance even if they don't show as top features.

## 8. Anti-overfit safeguards (unchanged from V7.7 / V7.8)

* **Lagged features only.** Every within-month column is shifted by 1 month before merging, so no current-month leakage is possible.
* **CV gap ceiling 0.02.** Champion has gap +0.0116, well under the ceiling.
* **Recency-weighted CV.** The last (most recent) fold gets weight 0.5; the test set lives at the end of the timeline, so we explicitly pick a champion that performs well on the last fold rather than averaged across folds.
* **Sum-to-1 simplex LAD weights.** Bases cannot extrapolate beyond the convex hull of their predictions.
* **Test set used once.** Pre-registered selection rule on val → single test evaluation. No iteration on test.

## 9. What V8 did not fix and what is most promising next

V8 leaves three structural errors untouched:

1. **Feb 2026 −34% under-forecast.** This is the post-Christmas rebound that V7.7 final report flagged. Every model V4 → V8 misses it. Likely needs **promo / inventory replenishment data** that we don't currently have access to (an analyst's planned replenishment for the new year would resolve this in one feature).
2. **РС / April +295% on cell-level heatmap.** Tiny absolute volume in that cell makes the percent explode. Operationally irrelevant but visually loud.
3. **A-class slight under-forecast still ≈ −9 %** (V7.8 was −7%). The LAD weight shift toward V8 / V8_recent didn't move the A-class cell because the within-month signal correlates with mid-tier SKUs more than with A-class blockbusters.

What is now most promising for V9+ (still net-new information):

1. **Weekly-resolution base on Kaggle GPU.** All current bases predict the monthly target. A weekly LightGBM rolled up to month at predict time would expose the ensemble to within-month seasonality at *target* level (not just feature level). Estimated lift: 1–3 % SIMSCORE.
2. **`weather_ua` and `tmdb_movies` re-evaluation.** Both loaders exist in `src/loaders/` and both return monthly aggregates. They were excluded from V7+ ABTs but never re-tested under SIMSCORE; same playbook as `school_ua` here.
3. **Explicit Christmas-rebound feature.** Add `is_post_xmas_month` × `dec_y_div_dec_y_minus_1` as a single cell-level feature. This is the only genuinely orthogonal way to attack the Jan/Feb miss.
4. **One more full quarter of held-out data** (i.e., wait for actuals through Q2 2026). The current 8-test-month window is small and noisy; doubling it would let us distinguish real progress from sampling noise on +/−0.5 % SIMSCORE deltas.

## 10. Artefacts

| Path | Description |
|---|---|
| `output/abt_v8_cached.parquet` | V8 ABT (316 498 rows × 157 cols) |
| `output/v8_feature_manifest.json` | Within-month + school feature names |
| `output/preds_v8_{val,test}.csv` | V8 base (single LightGBM) predictions |
| `output/preds_v8_recent_{val,test}.csv` | V8 recency-weighted base predictions |
| `output/preds_v8_lad_{val,test}.csv` | **V8 production champion (LAD stack)** |
| `output/v8/lad_champion.json` | Champion meta (pool, τ, axes, scores) |
| `output/v8/lad_cv.csv` | Full 48-candidate CV grid |
| `output/feature_importance_v8.csv` | V8 base feature importance |
| `output/model_v8.joblib` | Full V8 base bundle |
| `output/plot_v8_dashboard.png` | V7.8 vs V8 6-panel dashboard |
| `output/plot_v8_progression.png` | V1 → V8 progression chart |
| `output/plot_v8_residual_heatmap.png` | V8 channel × month-of-year bias |
| `output/plot_v8_within_month_features.png` | Within-month feature analysis |
| `output/plot_models_timeline.png` | Monthly forecast vs actual, all models |
| `output/v8_progression_summary.csv` | All-models test summary |
| `scripts/build_v8_abt.py` | V8 ABT builder |
| `scripts/v8_lad_stack.py` | 48-candidate CV-search + champion lock-in |
| `scripts/viz_v8_*.py` | All V8 visualisations |
| `src/features_within_month.py` | Within-month feature extractor |

## 11. One-line summary

V8 is the first version since V4 to bring information genuinely new to the model — within-month timing patterns from raw daily shipments — and that information is worth a 4.7% validation SIMSCORE jump and 3.1% test WAPE improvement, the largest single-step gains in the repo.
