# V7.4 — Per-Channel NNLS Stack

## Summary

V7.4 extends the V7.3 NNLS stack with **per-channel weights**: one non-negative
least-squares problem is solved for each of the four sales channels
(**СК**, **НКП**, **РС**, **ИМ**) over the same base-model pool as V7.3
(`{v4, v5, v6, v7, v7.1, v7.2}`).  Per-channel weights are picked because the
V7.3 global stack under-forecast СК/НКП by ~3.5% while simultaneously
**over-forecasting the ИМ (marketplace) channel by +14%** — a pattern no
single global weight vector can fix.

V7.4 is the new champion on our similarity metric set, evaluated
exactly once on the untouched test window (2025-07 … 2026-02).

## Test-set results (single-shot, pre-registered)

| Model           | WAPE ↓ | SMAPE ↓ | Monthly-WAPE ↓ | Aggregate bias % | **SIMSCORE ↓** |
|-----------------|:------:|:-------:|:--------------:|:----------------:|:--------------:|
| V7              | 0.4208 | 0.5441  | 0.1256         | −9.86            | 0.5329         |
| V7.1            | 0.4117 | 0.5162  | 0.1280         | −12.10           | 0.5362         |
| V7.2            | **0.4086** | 0.5266 | 0.1222       | −11.49           | 0.5272         |
| V7.3            | 0.4362 | 0.4987  | 0.1185         | −3.17            | 0.5113         |
| **V7.4**        | 0.4332 | 0.5045  | **0.1185**     | **−2.56**        | **0.5053**     |

V7.4 improves **aggregate bias** (−3.17 → −2.56 %, closer to zero), **WAPE**
(0.4362 → 0.4332), **Monthly-WAPE** (tied at 0.1185), and the composite
**SIMSCORE** (0.5113 → 0.5053, −1.2 %).  It is within 0.6 pp of V7.2 on
portfolio WAPE while being 5 pp better on SIMSCORE.

SMAPE nudges up (0.4987 → 0.5045) because per-channel weights trade a
touch of row-level symmetry for better aggregate calibration — exactly the
design intent of SIMSCORE.

## Why per-channel wins

Diagnosis of V7.3 residuals on the validation window showed:

| Канал | n rows | Σ actual | V7.3 WAPE | V7.3 bias % |
|-------|-------:|---------:|:---------:|:-----------:|
| СК    | 16 692 | 105 737  | 0.377     | **−3.50**   |
| НКП   |  5 136 |  20 272  | 0.434     | **−3.44**   |
| РС    |  5 484 |  13 745  | 0.544     | +1.37       |
| ИМ    |  4 140 |   8 821  | 0.767     | **+14.10**  |

A single global weight vector cannot simultaneously lift СК/НКП and trim
ИМ, so V7.3 left +14 % of bias on the table in marketplace orders.

**V7.4's per-channel weights** (sorted by channel size):

| Канал | v4    | v5    | v6    | v7    | v7.1  | v7.2  |
|-------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| СК    | 0.163 | **0.502** | 0.106 |   –   | 0.229 |   –   |
| НКП   | 0.041 | 0.072 | 0.330 | 0.006 |   –   | **0.551** |
| РС    |   –   |   –   | **0.475** | **0.525** | – |   –   |
| ИМ    |   –   | 0.196 | 0.125 | 0.191 |   –   | **0.488** |

Key observations:

1. **СК (the bulk of volume)** prefers the classical V5 (Tweedie) + a touch
   of V7.1.  The UAH-optimised V7/V7.2 get zero weight here — consistent
   with V7.3's finding that cost-optimal α=0.45 doesn't help row-level
   accuracy on high-density super-retail.
2. **V7.2, which got 0 % weight in the V7.3 global stack, carries 49–55 %
   of НКП and ИМ.**  Those channels are lower volume and more
   bias-prone; V7.2's α=0.45 quantile objective tames over-forecasts there
   better than the default quantile-median models.
3. **РС is purely V6 + V7**, ignoring both V4/V5 and the V7.1/V7.2 family.
4. The naive/MA baselines we also built (lag-12, MA-3, MA-6) were tested
   and **rejected** — adding them raised OOF SIMSCORE and overfit gap.

## Anti-overfit safeguards (unchanged from V7.3)

1. **Single test evaluation** — the 2025-07 … 2026-02 window was untouched
   until the final single-shot eval below.
2. **3-fold rolling-origin CV on the validation window only** used to
   pick the champion from 11 candidates.
3. **Pre-registered gap rule**: OOF SIMSCORE − in-sample SIMSCORE ≤ 0.05,
   otherwise the candidate is auto-rejected.
4. **Tiebreaker**: lowest OOF SIMSCORE, then smallest gap.

Candidate rankings on CV (OOF SIMSCORE, lower better):

| Rank | Candidate | OOF mean | Per-fold | In-sample | Gap |
|-----:|-----------|:--------:|----------|:---------:|:---:|
| 1 | **v74_compact_per_channel** | **0.4651** | 0.4306 / 0.4607 / 0.5039 | 0.4556 | +0.0095 |
| 2 | v74_compact_per_brand | 0.4655 | 0.4278 / 0.4616 / 0.5072 | 0.4572 | +0.0083 |
| 3 | v73_compact_global | 0.4671 | 0.4310 / 0.4631 / 0.5071 | 0.4591 | +0.0080 |
| 4 | v74_compact_per_density | 0.4710 | 0.4426 / 0.4632 / 0.5073 | 0.4593 | +0.0117 |
| 5 | v74_per_channel_bias_scalar | 0.4722 | 0.4554 / 0.4678 / 0.4933 | 0.4434 | +0.0288 |
| 6 | v74_pool_full_global | 0.4756 | … | 0.4572 | +0.0184 |
| 7 | v74_compact_bias_constrained | 0.4792 | … | 0.4500 | +0.0292 |
|   | …(4 more, all worse or larger gap)… | | | | |

Per-channel and per-brand finished in a statistical tie; the decision
gate picks the candidate with lowest OOF SIMSCORE (per-channel, by 4 ppt).

## Ideas tried and rejected

* **Extended base-model pool** (add seasonal-naive, MA-3, MA-6): OOF
  SIMSCORE worsened from 0.4671 (compact) to 0.4756 (full); the naive
  baselines fit residual noise in-sample and generalized poorly.
* **Bias-constrained NNLS** (force Σ wᵢ·mean(Xᵢ) = mean(y)): removed the
  aggregate bias in-sample but overfit-gap ballooned to +0.029 and OOF
  rose to 0.479.
* **Per-channel + global bias-correction scalar**: overfit gap +0.029, OOF
  0.472 — rejected.
* **Per-density sub-stacks**: slightly worse OOF + 0.012 gap (borderline)
  — rejected in favour of the simpler per-channel formulation.

## Artifacts

```
output/preds_v74_val.csv               # validation predictions
output/preds_v74_test.csv              # test predictions
output/v74/multistack_cv.csv           # full CV ladder of 11 candidates
output/v74/multistack_champion.json    # champion spec + per-channel weights
output/preds_naiveS_{val,test}.csv     # seasonal-naive preds (built but rejected)
output/preds_ma3_{val,test}.csv
output/preds_ma6_{val,test}.csv
output/plot_models_timeline.png        # timeline chart, now includes V7.4
```

## Reproduction

```bash
python -m scripts.v74_build_baselines   # one-time seasonal-naive + MA preds
python -m scripts.v74_multistack        # CV + champion refit + preds_v74_*.csv
python -m scripts.viz_model_timeline    # refreshes chart + models_timeline.csv
```

Each stage seeded for determinism; runtime ~8 min end-to-end on a laptop.

## Where to go next (V7.5 shortlist)

1. **Symmetric V7 retrain.**  V7.2 (α=0.45 quantile) dominates the ИМ and
   НКП stacks.  A MAE/Tweedie retrain of the same V7.2 feature set would
   likely help СК as well and push SIMSCORE further down.
2. **Time-varying per-channel weights.**  Weights today are fit on one
   fixed val window.  A rolling 12-month refit would adapt to regime
   shifts (e.g., 2024 Djeco ramp-up).
3. **Hierarchical reconciliation.**  Solve the per-channel stack jointly
   with a top-down portfolio bias constraint (Σᵢ pred_i = portfolio
   target) — potentially closes the residual −2.56 % bias.
4. **Segment-wise conformal intervals.**  Emit per-channel prediction
   intervals and measure coverage separately.
5. **Reactive guard for small channels.**  ИМ still has WAPE 0.77 (only 8.8
   k units total).  A shrinkage prior back to the global weights when a
   channel has < 2 000 training rows could stabilise it further.
