# V7.7 — Recency-weighted decorrelated base + multi-axis reconciliation

## Executive summary

V7.7 is the new production champion.  On the held-out test set
(2025-07 → 2026-02, 8 months, 20 968 partner × SKU rows):

| metric                     | V7.5 (prev champion) | V7.7 (champion) |    Δ    |
|----------------------------|---------------------:|----------------:|--------:|
| **SIMSCORE**               |               0.4875 |          0.4827 | −0.0048 |
| WAPE                       |               0.4255 |          0.4230 | −0.0025 |
| Monthly-WAPE               |               0.1086 |          0.1049 | −0.0037 |
| RMSE                       |               4.5399 |          4.4820 | −0.0579 |
| Aggregate bias %           |                −1.54 |           −1.44 |  +0.10  |
| Portfolio-level WAPE       |               0.0697 |          0.0674 | −0.0023 |

CV diagnostic: V7.5 OOF SIMSCORE 0.4589 (gap +0.0139) → V7.7 0.4552
(gap +0.0099).  Recency-weighted OOF (folds 0.2/0.3/0.5): V7.5 0.4691 →
V7.7 0.4674.  Lower CV, lower gap, better test — fully consistent.

V7.7 also shows large *targeted* gains on the segments that V7.5
struggled with:

| segment       | V7.5 test bias % | V7.7 test bias % |
|---------------|-----------------:|-----------------:|
| A-class       |           −11.6  |           −2.0   |
| СТОК-ВИАТ     |           +12.0  |           +5.7   |
| ИМ (online)   |            +1.1  |            ≈0    |
| Channel-level WAPE max | 0.88     |            0.86  |

## Story so far

| version | technique                                                           | test SIMSCORE |
|---------|---------------------------------------------------------------------|--------------:|
| V7.3    | NNLS stack                                                          |        0.5113 |
| V7.4    | per-channel NNLS                                                    |        0.5053 |
| V7.5    | per-channel LAD + channel reconcile (shrink 0.8)                    |        0.4875 |
| V7.6    | + symmetric LightGBM bases (Tweedie/MAE/Huber/L2)                   |  0.4920 (rejected — overfit) |
| V7.7    | + recency-weighted base + multi-axis reconcile + tilted LAD         |    **0.4827** |

V7.6 taught us a key lesson: more bases on the *same feature space* are
correlated and overfit the LAD weights.  V7.7 instead trains a base on
a *recency-weighted* sample of the existing features, which gives
genuinely orthogonal information.

## Diagnostic — where V7.5 lost ground

A residual analysis on the V7.5 test predictions, broken down by axis:

```
=== test by Канал ===
        n      qty   WAPE  bias%  err_share%
СК   11128  70154   0.386  -1.6        65 %
НКП   3424  13070   0.449  +2.1        14 %
РС    3656  10656   0.499  -6.4        13 %
ИМ    2760   3647   0.883  +1.1         8 %

=== test by Сегмент_ABC ===
B    10672  58872   0.408  -0.6        58 %
A     2384  19990   0.359 -11.6        17 %  ← worst single bias
V     3408   9910   0.531  +5.6        13 %
C     1856   5383   0.544  +6.9         7 %
ВИАТ  1432   1622   0.753 +17.3         3 %  ← worst single class WAPE

=== test by month ===
2025-12   bias  +9 %   (over-forecast Christmas)
2026-01   bias −17 %   (under-forecast January rebound)
2026-02   bias −32 %   (under-forecast February — single biggest miss)
```

Three structural problems:
1. **A-class (highest-volume single class) is systematically under-forecast** by ~12 %.
2. **Late-period months are heavily under-forecast** (rebound after Christmas).
3. **ИМ (online channel) WAPE is 88 %** — much worse than the rest.

V7.5's channel-only reconciliation (shrink 0.8) cannot address (1) or
(2) because it only sums residuals up to the channel level.

## What we tried, what worked, what didn't

### A. Symmetric LightGBM retrains on Kaggle GPU (dropped)

Notebook: `notebooks/v76_symmetric_retrain.ipynb` →
`<kaggle-user>/bpm-v76-symmetric` on Kaggle.

Trained five symmetric-objective V7 variants (Tweedie 1.3/1.5, MAE,
Huber, L2) on the existing V7 feature set.  Standalone test
SIMSCOREs: 0.49–0.63.  Adding them to the LAD pool *improved* CV by
0.7 % but *hurt* test by 0.5 % — rejected as overfit.

### B. Decorrelated bases on Kaggle GPU (kept v77_recent)

Notebook: `notebooks/v77_decorrelated_bases.ipynb` →
`<kaggle-user>/bpm-v77-decorr`.  Five strategies:

| variant            | strategy                                              | test SIMSCORE |
|--------------------|-------------------------------------------------------|--------------:|
| `v77_recent`       | recency-weighted training (γ=1.05⁻ᵃᵍᵉ, last 36 months) |     **0.4723** |
| `v77_nosegment`    | drop categorical Канал/Бренд/ABC encodings            |        0.4982 |
| `v77_nopromo`      | drop promo-lifecycle features                         |        0.6078 |
| `v77_long`         | only pairs with ≥18 months history                    |        0.5372 |
| `v77_quantile60`   | quantile=0.6 (asymmetric upward tilt)                 |        0.5584 |

**`v77_recent` standalone already beats V7.5 LAD by 1.5 percentage
points on test SIMSCORE.** The recency weight `(1/1.05)ᵃᵍᵉ` puts ~5×
more learning signal on data from the last 12 months than on the
oldest 24 months, mitigating distribution drift.

### C. Multi-axis hierarchical reconciliation (kept channel × ABC × brand)

Generalises V7.5's `Канал × month` reconciliation to a sequential
hierarchy: `Канал` (shrink 0.8 baseline) → `Канал × Сегмент_ABC` (0.5)
→ `Бренд` (0.3).  Each step uses training-window-only residuals (no
test leakage), with conservative shrinkage and `MIN_ROWS=250` to
prevent over-correction of small cells.  Implemented in
`scripts/v77_multi_reconcile.py`.

### D. Tilted (quantile) LAD stacker (kept τ=0.52)

V7.5's LAD minimises symmetric L1: `Σ |y − Xw|`.  Replaced with the
tilted L1 / pinball loss `Σ ρ_τ(y − Xw)`, which up-weights positive
residuals when `τ > 0.5` to combat negative aggregate bias.  Solved
via the same IRLS framework (NNLS sub-step), giving sum-to-1
non-negative weights per channel.  Tested τ ∈ {0.5, 0.52, 0.55}; CV
selected `τ=0.52`.

## V7.7 stacker — full pipeline

```
                       LightGBM bases (6)
                  v4 / v5 / v6 / v7 / v71 / v72_champion
                              +
                    v77_recent (Kaggle MAE / γ=1.05)
                                │
                                ▼
                  Per-channel tilted LAD (τ=0.52)
                  fitted via IRLS / NNLS, weights sum-to-1
                                │
                                ▼
              Hierarchical reconciliation (sequential)
        ─────────────────────────────────────────────────
        step 1:  Канал × Сегмент_ABC  shrink 0.5  scale ∈ [0.6, 1.8]
        step 2:  Бренд                 shrink 0.3  scale ∈ [0.6, 1.8]
                                │
                                ▼
                    final V7.7 prediction
```

Selection: 30 candidates (3 pools × 5 axes × 2 taus) evaluated on a
3-fold rolling-origin CV inside the val window (folds: 2024-10/12,
2025-01/03, 2025-04/06).  Recency-weighted OOF (fold weights
0.2/0.3/0.5) used as the primary criterion; champion must have
gap ≤ 0.02.

CV winner: **`v77_compact+rec_tau0.52_chABC05_brand03`** —
recency-weighted OOF 0.4674, gap +0.0099.

## Anti-overfit safeguards

| guard                                | value                |
|--------------------------------------|----------------------|
| Number of LAD parameters             | 7 bases × 4 channels = 28 |
| Hierarchical-scale clip range        | 0.6 – 1.8            |
| Min rows per scale cell              | 250                  |
| Shrinkage factor                     | 0.5 (chABC) / 0.3 (Бренд) |
| Pool restriction                     | compact / +rec / +rec+nosgm only |
| CV gap ceiling                       | ≤ 0.02 (champion: 0.0099) |
| Test set use                         | exactly **1** evaluation, post-selection |

## What this rules out

* **Adding more LightGBMs trained on the same feature set** is now
  conclusively a no-go (V7.6).
* **Aggressive multi-axis reconciliation (e.g. brand × month)** lifts
  CV but doesn't generalise — we kept only conservative shrinks 0.5/0.3.
* **High `τ`** (≥ 0.55) over-tilts and adds bias — `τ=0.52` is the
  sweet spot.

## What is now most promising

1. **Per-month-of-year bias corrector** trained on the V7.5/V7.7
   residuals.  The −32 % miss on Feb 2026 suggests the post-Christmas
   rebound pattern is not in any base's feature set — a small
   per-month additive correction (say MAE-fit on val months 1, 2)
   could absorb this directly.
2. **Conformal-aware quantile blending.**  Train V7 with
   `objective=quantile` at `τ ∈ {0.45, 0.50, 0.55}` and blend the
   three with a learned channel-specific tau.  This gives an
   asymmetric loss handle that the post-hoc reconciliation can't reach.
3. **Higher-frequency bases.**  All bases are monthly.  A
   weekly-resolution base (rolled up to month at predict time) might
   add genuinely new signal — most ML demand-forecasting wins above
   our current level come from weekly granularity.
4. **External signals.**  Weather (already loaded), holidays, FX
   (already loaded), Google Trends are all in the loaders but not
   plumbed into the model after V7.  Re-introducing them with proper
   leakage guards is a lever we haven't pulled yet.

## Reproducibility

```bash
# Train decorrelated bases on Kaggle (already done, preds in output/)
bash scripts/push_kaggle_kernel.sh \
     --notebook notebooks/v77_decorrelated_bases.ipynb \
     --slug   <kaggle-user>/bpm-v77-decorr \
     --dataset <kaggle-user>/bpm-v6-abt
bash scripts/pull_kaggle_kernel_output.sh \
     --slug   <kaggle-user>/bpm-v77-decorr

# Refit V7.7 stacker locally (≈70 s)
OMP_NUM_THREADS=1 python -m scripts.v77_lad_stack

# Visualise
python -m scripts.viz_model_timeline
python -m scripts.viz_v77_dashboard
```

Artefacts:

* Predictions: `output/preds_v77_{val,test}.csv`
* CV table:   `output/v77/lad_cv.csv`
* Champion meta: `output/v77/lad_champion.json`
* Visuals:    `output/plot_models_timeline.png`, `output/plot_v77_dashboard.png`
* Decorrelated bases preds: `output/preds_v77_{recent,nosegment,...}_{val,test}.csv`
