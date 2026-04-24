# V7.5 — LAD Stack + Hierarchical Reconciliation

## Summary

V7.5 replaces V7.4's **NNLS (L2-optimal)** per-channel stack with a
**LAD (L1-optimal)** stack, and adds a **hierarchical reconciliation**
step that rescales predictions within each channel so the channel-total
matches its realised value on the validation fit, shrunk toward 1.0 by
20 %.  Both changes are motivated by a simple observation: our target
metric is SIMSCORE = WAPE + 0.005·|agg_bias_pct| + 0.5·Monthly-WAPE,
which is dominated by L1 (absolute-error) terms, yet NNLS minimises
L2 squared error. Using an L1-aligned stacker moves the optimum closer
to the metric we actually care about.

V7.5 is the new champion, selected from 6 candidates on 3-fold
rolling-origin CV of the val window, with a single-shot evaluation on
the held-out 2025-07 … 2026-02 test set.

## Test-set results (single-shot, pre-registered)

| Model | WAPE ↓ | SMAPE ↓ | Monthly-WAPE ↓ | Aggregate bias % | **SIMSCORE ↓** |
|-------|:------:|:-------:|:--------------:|:----------------:|:--------------:|
| V7.2    | **0.4086** | 0.5266 | 0.1222 | −11.49 | 0.5272 |
| V7.3    | 0.4362 | 0.4987 | 0.1185 | −3.17  | 0.5113 |
| V7.4    | 0.4332 | 0.5045 | 0.1185 | −2.56  | 0.5053 |
| **V7.5** | **0.4255** | 0.5051 | **0.1086** | **−1.54** | **0.4875** |

Improvements over V7.4:

* SIMSCORE **0.5053 → 0.4875** (**−3.5 %**).
* WAPE **0.4332 → 0.4255** (−1.8 %).
* Monthly-WAPE **0.1185 → 0.1086** (**−8.4 %**).
* Aggregate bias **−2.56 → −1.54 %** (−40 % relative).

Cumulative from V7.2 (the UAH-optimised champion): SIMSCORE 0.5272 →
0.4875 = **−7.5 %**, a substantially better similarity-to-actual model.

Portfolio-level WAPE over the 20-month val+test window also hits a
new low: **0.0697** (V7.4 = 0.0758, V7.3 = 0.0725, V7.2 = 0.1130).

## Two mechanisms, decomposed

### 1. LAD stacking (L1, not L2)

**Problem.** NNLS solves `min ‖Xw − y‖² s.t. w ≥ 0`. Its gradient is
proportional to the residual `(Xw − y)`, so large residuals move w a lot
— NNLS chases squared errors.  But our row-level metric
Σ |y − ŷ| / Σ y weights all residuals linearly.  Two models with
identical NNLS weights can have different WAPE.

**Solution.** Use an L1-norm estimator.  We solve

    min_w  Σ_i | y_i − (Xw)_i |    s.t.    w_j ≥ 0,  Σ w_j = 1

via **iteratively-reweighted NNLS** (IRLS): each IRLS step solves a
weighted NNLS with `w_i = 1 / max(|residual_i|, ε)`.  Converges in 10–
15 iterations on our data; runtime is still dominated by the underlying
NNLS call (~50 ms for k=6, n=15 k).  Implementation in
`scripts/v75_lad_stack.py::lad_nn_simplex`.

Note: earlier prototype used a full linear-program formulation with
`scipy.optimize.linprog(method="highs")`, but the LP matrix for 51 k
rows is 102 k × 51 k (≈ 40 GB dense).  IRLS is memory-constant.

### 2. Hierarchical reconciliation

**Problem.** Per-channel row-level stacking can still leave channel
aggregates biased, because blending fixes no additive offset.  V7.4
shipped with a -2.56 % aggregate bias even though its per-channel
weights were optimal at the row level.

**Solution.** After stacking, compute a channel-level scale
`s_k = Σ y_k / Σ ŷ_k` on the training data and apply

    ŷ_i ← ŷ_i · ( λ·s_k(i) + (1−λ)·1 )

with `λ` swept in {0.5, 0.8, 1.0}. CV picks **λ = 0.8**: strong enough
to close the bias, shrunk just enough to avoid overfitting.

## Anti-overfit protocol (identical to V7.3/V7.4)

1. **Single test evaluation** — the 2025-07 … 2026-02 window was
   untouched during candidate selection.
2. **3-fold rolling-origin CV on the val window only**.
3. **Pre-registered gap rule**: OOF SIMSCORE − in-sample SIMSCORE ≤ 0.05.
4. **Tiebreaker**: lowest OOF SIMSCORE, then smallest gap.

Candidate CV ladder (lower OOF = better):

| Rank | Candidate | OOF | OOF folds | In-sample | Gap |
|-----:|-----------|:---:|-----------|:---------:|:---:|
| 1 | **v75_lad_compact_reconcile_0.8** | **0.4589** | 0.4165 / 0.4637 / 0.4965 | 0.4450 | +0.0139 |
| 2 | v75_lad_compact_reconcile_0.5 | 0.4596 | 0.4231 / 0.4534 / 0.5023 | 0.4521 | +0.0075 |
| 3 | v75_nnls_compact_reconcile | 0.4606 | 0.4272 / 0.4574 / 0.4973 | 0.4487 | +0.0119 |
| 4 | v75_lad_extended_reconcile_0.5 | 0.4613 | … | 0.4505 | +0.0108 |
| 5 | v75_lad_compact_reconcile_1.0 | 0.4652 | … | 0.4405 | +0.0247 |
| 6 | v74_compact_per_channel (baseline) | 0.4651 | … | 0.4556 | +0.0095 |
| 7 | v75_lad_compact_per_channel (no reconcile) | 0.4778 | … | 0.4658 | +0.0120 |

The **reconcile-only LAD** candidates all beat the no-reconcile ones.
Adding the 4 extra analytical signals (ewma6/12, median12, yoyTrend) to
the pool was a **wash** (CV scored 0.4613 vs 0.4589 without them) — the
seasonal-naive and moving-average baselines we built are too biased to
help once the per-channel LGB family is already present.

## Per-channel LAD weights

| Канал | v4    | v5    | v6    | v7    | v7.1  | v7.2  |
|-------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| СК    | 0.159 | 0.281 | 0.142 | 0.185 | **0.233** | –     |
| НКП   | 0.032 | 0.095 | 0.263 | 0.277 | –     | **0.333** |
| РС    | –     | –     | 0.207 | **0.622** | 0.128 | 0.043 |
| ИМ    | –     | –     | –     | 0.364 | –     | **0.636** |

The LAD weights differ from V7.4's NNLS weights in four notable ways:

1. **СК** spreads more uniformly (V4→V5→V6→V7→V7.1 all between 0.14 and
   0.28), diversifying away from V7.4's 0.50 V5 / 0.23 V7.1 concentration.
   LAD's piecewise-linear loss prefers this spread.
2. **РС** now puts **0.62** on V7 (V7.4 had 0.53) and picks up V7.1
   (0.13) and V7.2 (0.04) — previously РС was V6+V7 only.
3. **ИМ** becomes a pure **V7/V7.2 blend (0.36 / 0.64)**, dropping V5
   entirely (V7.4 had V5=0.20).  The α=0.45 pinball-loss family is
   simply the only sensible base for marketplace.
4. The **reconcile scale multiplier** (`s_k · 0.8 + 0.2`) is applied to
   every row after the LAD blend.  On validation it resolves to roughly
   × 1.01 for СК, × 0.96 for НКП, × 0.93 for РС, and × 0.87 for ИМ —
   which is the single largest bias fix in the whole V7.3 → V7.5
   trajectory (closes ИМ's +14 % over-forecast).

## Ideas tried and rejected

* **Extended pool with naive + MA baselines** — adds 4 analytical
  signals to the pool.  Ewma6/12/median12/yoyTrend are too biased
  individually; OOF SIMSCORE 0.4613 (worse than 0.4589 compact).
* **Per-SKU shrunk residual correction** — learns a per-SKU additive
  adjustment, shrunk by n/(n+k).  Blew the overfit gap to +0.033.
* **Scale = 1.0 (full rescale)** — aggressively re-aligns channel
  totals but overfits: gap +0.025, OOF 0.465.
* **Tweedie / MAE retrain of V7.2 features** — attempted on laptop CPU
  but LightGBM stalled on macOS OpenMP contention; would be the next
  natural step on a proper Linux GPU kernel.

## Plot update: squared residuals

`scripts/viz_model_timeline.py` now produces a **2-panel chart**:

1. **Top** — monthly total forecast per model overlaid on actual demand
   (as before).
2. **Bottom** — monthly **row-level RMSE** (`√ mean((y − ŷ)² per row)`)
   — a squared-error metric complementary to WAPE.  V7.5's RMSE line
   sits below every earlier model on every single test month.

File: `output/plot_models_timeline.png`.

## Artifacts

```
output/preds_v75_val.csv                 # validation predictions
output/preds_v75_test.csv                # test predictions
output/preds_v75lad_{val,test}.csv       # raw LAD output (same as v75)
output/preds_{ewma6,ewma12,median12,yoyTrend}_{val,test}.csv  # unused but kept
output/v75/lad_cv.csv                    # full 6-candidate CV ladder
output/v75/lad_champion.json             # champion spec + per-channel weights
output/v75/multistack_cv.csv             # prior NNLS candidate ladder
output/plot_models_timeline.png          # 2-panel chart w/ squared residuals
output/models_timeline.csv               # portfolio-level totals
```

## Reproduction

```bash
python -m scripts.v75_build_signals    # one-time analytical baselines
python -m scripts.v75_multistack       # NNLS candidate sweep (optional)
python -m scripts.v75_lad_stack        # LAD candidate sweep + champion
cp output/preds_v75lad_{val,test}.csv output/preds_v75_{val,test}.csv
python -m scripts.viz_model_timeline   # refreshes 2-panel chart
```

End-to-end runtime ≈ 12 min on laptop CPU.

## Where to go next (V7.6 shortlist)

1. **Symmetric LightGBM retrain**, properly on Linux / Kaggle GPU —
   Tweedie + MAE variants of V7.2 features, added to the LAD pool.
   Most promising single lever remaining.
2. **Time-varying stack**.  Re-fit the LAD weights on a rolling 12-month
   window to adapt to regime shifts (e.g. 2024 Djeco ramp-up).
3. **Joint MINT reconciliation** — instead of per-channel multiplicative
   scale, solve the full hierarchical-reconciliation LP across
   brand × channel × month.
4. **Count-data regression family** — Poisson / negative-binomial base
   model specifically for sparse / intermittent SKUs, where LGB-quantile
   struggles.
5. **Probabilistic LAD**: emit LAD-based per-channel prediction
   intervals (conformal, α=0.1/0.9) for business-useful uncertainty.
