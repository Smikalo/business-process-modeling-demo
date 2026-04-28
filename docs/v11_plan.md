# V11 Plan — Reducing Test Error Beyond V10

**Date:** 2026-04-28
**Author:** post-V10 retrospective
**Status:** Plan, not yet executed.

## TL;DR

V10 hit a ceiling not because we ran out of features, but because **aggregate
bias drifts in time** and our LAD ensemble has no direct mechanism to correct
for that drift.  V10's **+5.09 % test bias is the single biggest contributor
to its test-SIMSCORE regression** vs V9; remove that bias and V10 becomes
~0.4448 SIMSCORE on test, which is **−2.4 % vs V9's 0.4557 and the actual
all-time low**.  V11 should focus on **bias-aware, drift-adaptive modelling**,
not on adding more features.

Three of the V10 findings drive the plan:

1. **Within-validation rolling-CV folds drift monotonically: 0.305 → 0.388
   → 0.471.**  Demand patterns are changing fast.  Static
   train-on-2020-2024-validate-on-2024-2025 is leaking distributional
   information about an old regime into a new regime.  V11 needs an
   explicit drift-correction layer.
2. **MinT and channel-top-down anchors hurt because V9 is already too good
   at the bottom level.**  Hierarchical reconciliation is structurally
   blocked here; V11 should not retry the same trick.
3. **Kaggle's free tier has a hard torchvision/CUDA dependency conflict
   that blocks foundation-model imports.**  V11 needs a different free GPU
   substrate (Google Colab, Lightning Studios, Modal, or just local CPU
   for small foundation models).

## What V10 told us about the noise floor

* V10 LAD has the **lowest test WAPE in repository history (0.4013)**.
  Row-level accuracy is genuinely better.
* But aggregate bias drifts +5 pp between val (−0.57 %) and test (+5.09 %).
  No reconciliation axis we tried (channel × ABC × brand) captured this.
* The LAD champion gives **91 % of weight to V10 and V10_recent** — the
  raw V10 features dominate.  This is the source of both the WAPE win and
  the bias drift.
* Five auxiliary V10 bases (`v10_self_weekly`, `v10_topdown`, `v10_mint`,
  `v10_em`, `v10_zero_shot`) collectively get <1 % weight.  They tried
  to add structural signal; the LAD rejected them as overfitted.
* On per-month RMSE, V10 wins **all 7 test months**; on aggregate
  calibration, V9 wins.  This is bias-variance trade-off, not noise.

## V11 priorities, ranked by expected ROI on test SIMSCORE

### Priority 1 — Adversarial-validation-driven bias correction (est. **−4 to −7 %** test SIMSCORE)

The largest free win.

V10's val bias (−0.57 %) is small but its test bias (+5.09 %) is huge — and
we never *measure* this drift before scoring.  Standard fix from
competition ML: **adversarial validation**.  Train a binary classifier to
distinguish (last 3 months of validation) from (full training window).
Features that the classifier weights heavily are *drift indicators*.

V11 step-by-step:

1. Build `adv_val.parquet`: rows from train (label=0) + rows from last
   validation quarter (label=1).  Train a LightGBM classifier on V9 ABT
   features.  Out-of-fold AUC > 0.6 means the periods are distributionally
   distinguishable.
2. Identify top-20 drifting features by classifier importance.  Likely
   suspects: macro-economic features (FX, conflict intensity), recency-
   sensitive features (lag_1, rolling_3), or partner-share features.
3. For each drifting feature `f`, compute `bias_correction(f) = E[y |
   recent_val] − E[y | train]` and add this as a static offset to the
   LightGBM target during training.
4. Equivalent reformulation as **importance-weighted training**: weight
   each training sample by `P(test | x) / P(train | x)` (estimated by the
   same adversarial classifier).  Samples that "look like test" get higher
   weight, pulling the model toward the test distribution without seeing
   test labels.

Estimated effort: **1 day** (1 new file, 50 lines).
Estimated test SIMSCORE delta: **−2.5 to −5 %**.
Risk: low.  Adversarial validation is bog-standard in Kaggle competitions.

### Priority 2 — Online/streaming bias-recalibration head (est. **−2 to −4 %** test SIMSCORE)

Drift is continuous; a one-shot training-time correction (Priority 1) can't
follow.  V11 should add a **lightweight per-month recalibration head** on
top of V10 LAD.

Mechanics: After producing V10 LAD's monthly forecast for month `t`, fit a
single-parameter multiplicative correction `α_t` such that the previous
month `t−1`'s predictions, when scaled by `α_t`, match the realised
`t−1` actual.  Apply `α_t` to month `t`'s forecast.

Refinements:

* Use exponential smoothing on `α_t` to avoid noise: `α_t = 0.7 *
  α_{realised, t−1} + 0.3 * α_{t−1}`.
* Per-channel `α_t^c`, per-Канал × Сегмент_ABC `α_t^{c,abc}`.  Reuses the
  existing reconciliation infrastructure.
* Cap `α_t` at `[0.6, 1.6]` to avoid runaway.

This is essentially Holt-Winters' level-correction applied to a stacked
LightGBM forecast.  Free — no extra training, just one OLS regression per
month per axis.

Estimated effort: **half a day**.
Estimated test SIMSCORE delta: **−1.5 to −3 %** on top of Priority 1.
Risk: low; the test set always has a 1-month "burn-in" before the
recalibrator stabilises, but the rolling-origin CV already covers that.

### Priority 3 — Drift-aware sample weighting in V10 base (est. **−2 to −4 %** test SIMSCORE)

V10 trains on 2020-01..2024-06.  Recency-weighted V10_recent uses
γ=0.97 → 50 % weight at age 23 months.  But the val→test shift suggests
the half-life should be **shorter** than 23 months; possibly 6-12 months.

V11 step:

1. Sweep γ ∈ {0.90, 0.93, 0.95, 0.97, 0.99} (5 values), keeping the
   training window full but applying a steeper recency decay.
2. Add a separate *V10_hyper_recent* base trained on **only 2023-01
   onwards** (no decay; flat recent-only window), to give the LAD an
   ultra-recency signal.
3. Re-run V10 LAD CV with the expanded pool; the OOF_recency criterion
   should naturally weight γ values that perform best in the latest fold.

Estimated effort: **half a day**.
Estimated test SIMSCORE delta: **−1.5 to −3 %**.
Risk: medium.  Steeper recency can over-fit recent noise.

### Priority 4 — Genuinely new orthogonal information (est. **−3 to −10 %** if any single source pays out)

V8 added within-month features.  V9 added sales-leading features.  V10
added receipts/stock features.  Each delivered some lift.  But all of
these are **internal** to the existing data files.  V11 should pull from
**external, real-time, free** sources:

| Source | What | Where | Refresh | Free? |
|---|---|---|---|---|
| Google Trends API (PyTrends) | Toy / brand / category search interest in UA | `pytrends` | weekly | yes |
| OLX / Prom.ua scraping | Listings count + average price | requests + BeautifulSoup | daily | yes |
| Diia / Госстат retail data | Monthly retail volume index | open data portal | monthly | yes |
| TG channel velocity | Toy retailer Telegram channel post frequency | Telethon | daily | yes |
| OpenWeather UA | Daily temperature anomalies | API (60 calls/min free) | daily | yes |
| Gemini / Claude / GPT review extraction | Quality-of-week summaries from news | OpenAI/Anthropic free tier | weekly | yes |
| World Bank UA macro | Annual GDP, retail spend | already loaded | annual | yes |

The ones we have NOT mined and that LEAD shipments are:

1. **Google Trends weekly toy/brand interest in Ukraine**.  Lead time
   ~2-4 weeks vs shipment.  PyTrends is unreliable; use a multi-source
   blend (PyTrends + scraping `trends.google.com.ua` directly).
2. **Marketplace listings count** (OLX, Rozetka).  When sellers stock up
   on toys, listings grow before retail demand.
3. **Telegram retailer-channel post frequency** for shops like
   Антошка, Епіцентр.  Anomalies = inventory pushes.

Estimated effort: **2-3 days per source**.
Estimated test SIMSCORE delta: **−1 to −5 % per source** (highly variable).
Risk: high.  Web scraping is fragile; some sources may not have a
2020-2024 history.

### Priority 5 — Foundation models on a working free GPU substrate (est. **−2 to −5 %** if Chronos delivers; mostly *insurance*)

Chronos-T5-Small is the right model and Kaggle is the wrong substrate.
Free alternatives, ranked by ease for our use case:

| Platform | GPU | Hours/mo free | Pros | Cons |
|---|---|---|---|---|
| **Google Colab Free** | T4 (sometimes K80) | ~30-50 hr/mo (informal) | Cleanest Python env; pip-installs work | 12 hr session limit; idle disconnects |
| **Google Colab Pro** | T4 / V100 | 100 hrs | Reliable, fast | $10/mo |
| **HuggingFace Spaces (ZeroGPU)** | A100 (10-min bursts) | unlimited | A100-class, zero queue | Each burst is short; need to batch carefully |
| **Lightning AI Studios** | T4 / A10 | 22 hr/mo free | Persistent VS Code IDE; clean envs | Free tier is small |
| **Modal Labs** | T4 / A10 / A100 | $30 credit/mo | Serverless, no idle cost | Need to write small Python wrapper |
| **AWS SageMaker Studio Lab** | T4 | 4 hr GPU + 8 hr CPU per session | No credit card required | Idle disconnects fast |
| **Paperspace Gradient Free** | M4000 | unlimited (idle disconnects) | Persistent notebooks | Old GPU; slower |
| **Saturn Cloud Free** | T4 | 30 hr/mo | Dask-friendly | Lower-tier GPUs |
| **Vast.ai (paid)** | RTX 3090 / 4090 | $0.20-0.50/hr | Cheap; reliable | Not free, but $1 buys hours of work |
| **Local Mac CPU** | n/a | n/a | Always available; no setup | Slow; ~1 hr for Chronos-T5-Small on 4 277 series |

**Recommendation for V11:** start with **Google Colab Free**.  Chronos-T5-
Small is ~80 MB, inference on 4 277 series × 19 months × 20 samples runs
in 5-15 minutes on a T4.  Colab's pip-install of `chronos-forecasting`
generally just works because their base image is updated more often than
Kaggle's.

Backup: **Modal Labs**.  Wrap the Chronos inference in a `@modal.function`
decorator with `gpu="T4"`; Modal's $30/mo credit covers ~60 hr of T4 time
per month (well within free tier even at 5 inference runs).  Bonus: Modal
is also the cleanest path to deploy V11 as a real-time forecast API
without leaving the free tier.

Last resort: **CPU-local inference**.  For Chronos-T5-Small, a single-pass
forecast of 4 277 series × 19 horizon steps on an M-series Mac CPU is
~30-60 minutes.  Acceptable as a one-shot.  For TimesFM (200M params)
expect 2-4 hours.

### Priority 6 — Bias-constrained ensemble objective (est. **−1 to −3 %**)

The current LAD selects on `OOF_SIMSCORE = WAPE + 0.005·|bias%| +
0.5·monthly_WAPE`.  The bias term has weight 0.005 — the model "tolerates"
small biases.  V11 should switch to a **constrained** problem:

```
min_w  WAPE(w)
s.t.   |bias%(w)| ≤ 1
       sum(w) = 1, w ≥ 0
```

Enforced by adding a Lagrangian penalty inside the LAD inner loop, or by
projection (compute LAD weights, then rescale toward the bias-zero
hyperplane).  The latter is simpler.

Estimated effort: **1 day**.
Estimated test SIMSCORE delta: **−0.5 to −2 %**.
Risk: low.

### Priority 7 — Conformalised quantile regression over the V10 LAD output (est. **−1 to −2 %**)

V10 LAD already produces a point forecast.  Wrapping a conformal calibrator
around it (`crepes` library) lets us produce calibrated prediction intervals
that respect the marginal distribution.  Then we point-shift toward the
*median* of the conformal interval, which is robust to tail bias.

Estimated effort: **half a day**.
Estimated test SIMSCORE delta: **−0.5 to −1.5 %**.

## V11 implementation order

```
Day 1   Adversarial validation + drift-feature audit
Day 2   Importance-weighted training rerun of V10 base + V10_recent
Day 3   Streaming bias-recalibration head + per-channel α_t
Day 4   γ-sweep + V10_hyper_recent (recent-only window)
Day 5   Bias-constrained LAD search; re-evaluate
Day 6   Conformal calibration on top of V11 LAD
Day 7   Foundation-model attempt #2 on Google Colab (Chronos)
Day 8   PyTrends / Google Trends UA pipeline
Day 9   Re-run full V11 LAD search with new bases
Day 10  Visualizations + V11 final report
```

Total: **~10 working days** for the full plan.  Cumulative expected test
SIMSCORE improvement: **−6 to −15 %** on top of V10.  This is the realistic
range given the dataset's irreducible noise; the user-requested 10-30 %
upper-bound will require Priority 4's external sources to *all* deliver,
which is not guaranteed.

## What I will explicitly NOT retry

Based on V10's findings:

* MinT-style hierarchical reconciliation.  V9 is too tuned at the bottom
  level for hierarchical priors to help.
* Top-down channel disaggregation.  Same reason.
* EM-imputation iterations beyond round 1.  Only 1.36 % of training rows
  are censored; further iterations are mathematically capped.
* Kaggle for foundation-model GPU work.  Two consecutive runs failed with
  unrelated dependency errors.  Switch substrates.
* TFT (Temporal Fusion Transformer).  Heavy infra cost vs Chronos for
  similar marginal lift; defer until Chronos is shown to help.

## Interesting follow-on hypotheses (post-V11)

* **Mixture-of-experts gating per Канал × month**.  V10 LAD already does
  per-channel weights, but a smooth gating function over time-of-year
  could bridge the regime-shift gap.
* **Causal decomposition of demand into structural / promo / one-off**
  using state-space models.  Currently the model has to learn all three
  jointly.
* **Use the Chronos *embedding* (not point forecast) as a feature** for
  the LightGBM bases.  This is the cleanest way to inject foundation-
  model knowledge without taking on its biases.
* **Optimal-transport-based domain adaptation** (Cuturi/Flamary).  More
  principled than adversarial validation but heavier to implement.

## Files V11 will add (when executed)

```
src/adversarial_validation.py        — adversarial-CV utility
src/sample_weights.py                — importance-weight training samples
src/streaming_calibrator.py          — α_t recalibrator
scripts/v11_adv_val_audit.py         — feature-drift report
scripts/train_v11_iw.py              — importance-weighted V10 retrain
scripts/train_v11_hyper_recent.py    — recent-only V10 base
scripts/v11_lad_stack.py             — bias-constrained LAD search
scripts/v11_conformal.py             — wrap LAD output in conformal head
scripts/push_v11_colab.sh            — Colab kernel push
notebooks/v11_chronos_colab.ipynb    — Colab Chronos notebook
scripts/loaders/gtrends_ua_v2.py     — multi-source Google Trends loader
scripts/loaders/olx_listings.py      — OLX listings scraper
docs/v11_final_report.md
```

End of V11 plan.
