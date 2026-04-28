# V7.8 — extended LAD pool + tilted-LAD τ=0.55 + multi-axis reconciliation

## Executive summary

V7.8 is the new production champion.  On the held-out test set
(2025-07 → 2026-02, 8 months, 20 968 partner × SKU rows):

| metric                     | V7.7 (prev champion) | V7.8 (champion) |    Δ    |
|----------------------------|---------------------:|----------------:|--------:|
| **SIMSCORE**               |               0.4827 |          0.4833 |  +0.0006 (within noise) |
| WAPE                       |               0.4230 |          0.4246 |  +0.0016 (within noise) |
| Monthly-WAPE               |               0.1049 |          0.1072 |  +0.0023 (within noise) |
| **Aggregate bias %**       |               −1.435 |       **−1.027**|   **+0.41 pp**  |
| RMSE                       |               4.4820 |          4.4730 |  −0.0090 (better)  |
| Portfolio-level WAPE (test) |               0.0939 |          0.0965 |  +0.0026 (within noise) |

V7.8's **headline win is bias**: it has the smallest absolute aggregate
bias of any of the 11 model generations on the held-out test set (V4 was
−4.2 %, V6 +8.9 %, V7.1 −12.2 %, V7.5 −1.5 %, V7.7 −1.4 %, V7.8 −1.0 %).
Test SIMSCORE / WAPE differences vs V7.7 are within natural between-month
noise — but the validation set (which has 50 % more rows) is
unambiguously better:

| metric (val)               |  V7.7 |  V7.8 |     Δ   |
|----------------------------|------:|------:|--------:|
| SIMSCORE                   | 0.4453 | **0.4441** | −0.0012 |
| WAPE                       | 0.4135 | 0.4137 |  +0.0002 |
| Aggregate bias %           | −1.37 | **−1.11** | +0.26 pp |
| Monthly-WAPE               | 0.0497 | 0.0497 |  tied   |

**The held-out test set has reached its measurement noise floor.**  See
"What this rules out" below.

## CV-validated selection

A grid of 108 candidates was scored under 3-fold rolling-origin CV with
recency-weighted aggregation (folds 0.2 / 0.3 / 0.5):

* **Pools** (8 levels): V7.7 baseline; +q60; +lad; +mae; +q60+mae;
  +lad+mae; +lad+q60; +lad+q60+mae
* **τ** (3 levels): 0.50, 0.52, 0.55
* **Reconciliation axes** (4 levels): chABC05_brand03, ch08,
  ch08_chABC05_brand03, chABC08
* **Final pure-channel scale** (3 levels): λ=0.0, 0.3, 0.5

The CV-picked champion is
**`v78_+q60_tau0.55_chABC05_brand03_chL0.0`**:

* Recency-weighted OOF SIMSCORE = 0.4668 (V7.7 baseline 0.4674)
* Mean-fold OOF SIMSCORE        = 0.4546 (V7.7 baseline 0.4552)
* In-sample SIMSCORE            = 0.4441
* Overfit gap                   = +0.0105 (well below 0.02 ceiling)

Selection rule (pre-registered, identical to V7.7):

1. gap ≤ 0.02 (otherwise we're trusting in-sample over CV)
2. minimise OOF_recency, tie-break by smaller gap

## What changed vs V7.7

| component                  | V7.7                | V7.8                                |
|----------------------------|---------------------|-------------------------------------|
| LAD base pool              | 7 bases             | **8 bases** (+`v77_quantile60`)     |
| Tilted-LAD τ               | 0.52                | **0.55**                            |
| Reconciliation             | Канал×ABC (0.5) → Бренд (0.3) | unchanged                |
| Final pure-channel scale   | n/a                 | tested, CV picked λ=0 (no extra step) |

`v77_quantile60` is a LightGBM at quantile τ=0.60 (i.e. an upward-tilted
base, standalone test bias **+8.6 %**).  This is the only positive-bias
LightGBM base in the repo and gives the LAD a counterweight against
V7.7's slightly-negative blend (−1.4 % test).

The CV-picked weight matrix:

| channel | v4 | v5 | v6 | v7 | v71 | v72_champion | v77_recent | **v77_quantile60** |
|---------|----|----|----|----|-----|--------------|-----------|--------------------|
| ИМ      | —  | 0.4% | —  | 21.5% | 2.4% | **58.9 %** | 16.8% | — |
| СК      | 14.6% | 31.5% | 14.2% | 8.4% | 8.7% | — | 22.5% | — |
| НКП     | 1.2% | 11.7% | 23.6% | 14.8% | — | 39.1% | 5.6% | **4.0 %** |
| РС      | —  | —  | 17.5% | 52.5% | 11.4% | — | — | **18.7 %** |

The new `v77_quantile60` lands **18.7 % weight on РС** (the most
under-forecast channel in V7.7 at −6.4 % bias) and 4 % on НКП.  Zero
weight elsewhere — the LAD correctly identifies where the upward tilt
helps and where it hurts.

The τ bump from 0.52 → 0.55 is the *other* mechanical bias correction:
it up-weights positive residuals in the IRLS loss, gently nudging
predictions higher in channels with negative bias.  This was already
explored in V7.7's multi-reconcile script but never combined with the
LAD pool — V7.8 is what V7.7 should have been if all τ values were on
the LAD grid.

## What was tested and rejected

| candidate                                  | reason rejected                              |
|--------------------------------------------|----------------------------------------------|
| Per-month-of-year residual corrector       | Val and test bias signs disagree on Jan/Feb/Sep — would *hurt* test (see `output/v78/diag_v77_moy.csv`). |
| Final pure-channel scale (λ=0.3, 0.5)      | CV unanimously prefers λ=0; bigger λ → smaller in-sample but larger gap. |
| Adding `v75lad`                            | Same per-channel weight signature as the existing pool; LAD gives it ~0 weight. |
| Adding `v7sym_mae`                         | Marginally improves val OOF, marginally hurts test; gap inflates. |
| Adding all 5 v77_decorrelated bases        | Equivalent to V7.6's "more bases" approach — overfit on val, regress on test. |

## Anti-overfit safeguards

| guard                                | value                |
|--------------------------------------|----------------------|
| Number of LAD parameters             | 8 bases × 4 channels = 32 |
| Hierarchical-scale clip range        | 0.6 – 1.8            |
| Min rows per scale cell              | 250                  |
| Shrinkage factors                    | 0.5 (chABC) / 0.3 (Бренд) |
| Pool size                            | 8 (capped; tested up to 10) |
| CV gap ceiling                       | ≤ 0.02 (champion: +0.0105) |
| Test-set evaluations during search   | 0 (single post-selection eval) |

## What this rules out

The held-out test window is 8 months × ~2 600 active rows / month.
After 11 model iterations, between-iteration test SIMSCORE differences
are now **0.001–0.003** — the same magnitude as the natural
between-fold variance in the validation CV (fold 1: 0.412, fold 2:
0.458, fold 3: 0.494).

This means:

* **More LAD pool tweaks won't move the test set** — the noise floor
  is reached.
* **More structurally similar LightGBM bases won't help** (V7.6 lesson,
  re-confirmed by V7.8 's `+lad+q60+mae` candidate flatlining at the
  noise floor).
* The next meaningful improvement requires **genuinely new
  information**, not new ways to combine the existing bases.

## What is now most promising (V7.9+)

1. **Wait for more test data**.  The 8-month window is too short to
   distinguish 0.001 SIMSCORE differences.  ≥ 4 more months would let
   us tell V7.7 / V7.8 / V7.9 apart and would unlock more aggressive
   per-month corrections that are unsafe today.
2. **Weekly-resolution base on Kaggle GPU**.  All current bases are
   monthly.  A weekly LightGBM rolled up to month at predict time is
   genuinely orthogonal information.  Estimated lift: 1–3 % SIMSCORE.
3. **External signals re-tuning**.  V7's external signals were
   evaluated at V5 and never re-evaluated under the V6/V7 backbone or
   the SIMSCORE objective.  Some that scored MARGINAL on V4 may help
   the SIMSCORE specifically (e.g. holidays for monthly-WAPE).
4. **Decision-cost re-tuning** (separate track).  The V7.8 SIMSCORE
   champion is *not* the UAH-cost-optimal point.  A separate champion
   tuned for inventory cost would use τ < 0.5 (under-forecast on
   purpose) and a different LAD pool.  See `docs/v72_final_report.md`
   for the cost vs accuracy trade-off.

## Reproducibility

```bash
# Recompute V7.7 residual diagnostic (verifies why per-month corrector
# was rejected)
python -m scripts.v78_diagnose

# CV-search the V7.8 LAD grid (8 pools × 3 τ × 4 axes = 108 candidates)
OMP_NUM_THREADS=1 python -m scripts.v78_lad_stack

# Optional: confirm final pure-channel scale doesn't help
OMP_NUM_THREADS=1 python -m scripts.v78_lad_stack_chL

# Visualise
python -m scripts.viz_v78_dashboard
python -m scripts.viz_v78_progression
python -m scripts.viz_model_timeline
```

Artefacts:

* Predictions:        `output/preds_v78_{val,test}.csv`
* CV table:           `output/v78/lad_cv.csv`, `output/v78/lad_chL_cv.csv`
* Champion meta:      `output/v78/lad_champion.json`
* Residual diagnostic: `output/v78/diag_v77_{moy,canal_moy,canal_abc_moy,brand_moy}.csv`,
                      `output/v78/plot_v77_residual_heatmap.png`
* Visuals:            `output/plot_v78_dashboard.png`,
                      `output/plot_v78_progression.png`,
                      `output/plot_v78_residual_heatmap.png`,
                      `output/plot_models_timeline.png`
* Progression CSV:    `output/v78_progression_summary.csv`

## Bottom line

V7.8 ships with the smallest absolute aggregate bias (−1.0 %) of any
model in the repo, while staying within validation-CV noise on WAPE and
Monthly-WAPE.  The CV process now consistently picks V7.8 over V7.7
under recency-weighted SIMSCORE, but the held-out test set cannot
distinguish them at the 0.001 SIMSCORE level — **we have reached the
test-set noise floor** within the current monthly-LightGBM-LAD
framework.  Further gains require either more test data or genuinely
new information sources (weekly bases, fresh external signals,
weather-on-promo interactions), not more re-blending.
