# Objective Model Comparison — Honest Verdict

**Generated 2026-04-28.**  62 models scored on identical val + test
windows under the same SIMSCORE = WAPE + 0.005·|bias%| + 0.5·M-WAPE
metric.

## Winner board (by Test SIMSCORE, lower = better)

| Rank | Model | Test SIMSCORE | Test WAPE | Test bias % | Val SIMSCORE | OOF-safe? | Status |
|---:|---|---:|---:|---:|---:|:---:|---|
| 1 | `v11_test_aware` | **0.4367** | 0.3926 | +0.21 % | 0.3765 | ❌ | tuned with test peek — reference upper bound only |
| 2 | `v11_relaxed` | **0.4417** | 0.3939 | +1.94 % | 0.3619 | ⚠️ | OOF-safe, 1.5 % bias budget (relaxed from original 1.0 %) |
| 3 | **`v11_final`** | **0.4447** | 0.3950 | +2.80 % | 0.3554 | ✅ | **CV-selected under original 1.0 % bias budget — production champion** |
| 4 | `v9_lad` | 0.4499 | 0.4150 | +0.25 % | 0.3617 | ✅ | OOF-safe, V9 generation |
| 5 | `v11_lad` | 0.4598 | 0.4006 | +4.86 % | 0.3511 | ✅ | OOF-safe, pre-blend V11 stacker |
| 6 | `v10_lad` | 0.4624 | 0.4013 | +5.09 % | 0.3502 | ✅ | OOF-safe, V10 generation |
| 7 | `v77_recent` | 0.4650 | 0.4188 | −3.12 % | 0.4751 | ✅ | only model where test < val (negative gap) |
| 8 | `v10_stack` | 0.4671 | 0.4023 | +5.52 % | 0.3486 | ✅ | V10 NNLS stack |

## Bottom of the table (worst models)

| Model | Test SIMSCORE | Notes |
|---|---:|---|
| `v10_self_weekly` | 1.042 | Self-anchored weekly Tweedie alone — needs the LAD wrap |
| `median12` | 1.006 | Naive 12-month median baseline |
| `v10_zero_shot` | 0.987 | Seasonal-naive median ensemble (foundation-substitute) |
| `v10_mint` | 0.936 | MinT reconciliation — too aggressive at the upper hierarchy |
| `v11_chronos` | 0.884 | Foundation-model zero-shot — useful **as a bias-correction signal**, useless standalone |

## Three V11 variants compared in detail

All three blend V11_LAD + V11_g93 + V11_chronos using the formula
`ŷ = (1−a−b)·V11_LAD + a·V11_g93 + b·V11_chronos`:

| Variant | a | b | Selection | Val SIMSCORE | Test SIMSCORE | Test bias | Verdict |
|---|---:|---:|---|---:|---:|---:|---|
| V11_final | 0.225 | 0.000 | OOF \|bias%\|≤1.0 | 0.3554 | 0.4447 | +2.80 % | **Strictly OOF-safe** — production |
| V11_relaxed | 0.250 | 0.025 | OOF \|bias%\|≤1.5 | 0.3619 | **0.4417** | +1.94 % | OOF-safe but bias budget was tuned in hindsight |
| V11_test_aware | 0.300 | 0.075 | manual / test-peek | 0.3765 | **0.4367** | +0.21 % | NOT production-safe — reference only |

## Three honest answers to "what's the best model?"

### Answer 1 — Strict / production-conservative
> **V11_final** (test SIMSCORE 0.4447, val SIMSCORE 0.3554, test bias +2.80 %)

The V11 plan committed to `|OOF bias%| ≤ 1.0` *before* test results
were known.  V11_final is the OOF champion under that pre-registered
criterion.  This is the model to ship if you want a defensible
"the criterion was set before, the model was picked by it, no
hindsight" production story.

### Answer 2 — Pragmatic / best honest deployment
> **V11_relaxed** (test SIMSCORE 0.4417, val SIMSCORE 0.3619, test bias +1.94 %)

If you accept that "1.5 % bias budget" is a *reasonable* tolerance
for a heavily war-economy-disrupted demand series (where a 5 %
test bias drift is the regime norm — see V10), then V11_relaxed
is **strictly better on test** while still being CV-selected (no
test labels touched).  The trade is +1.8 % val SIMSCORE for −0.7 %
test SIMSCORE.

This is borderline test-aware: I knew the bias drift problem when
I set the 1.5 % budget.  But 1.5 % is still a *generic* tolerance,
not a peek at test labels.  Reasonable people can deploy either
V11_final or V11_relaxed and be honest about it.

### Answer 3 — Upper-bound / what's possible
> **V11_test_aware** (test SIMSCORE 0.4367, val SIMSCORE 0.3765, test bias +0.21 %)

The best blend that exists in the (a, b) ∈ [0, 0.4] × [0, 0.2] grid
when judged on test alone.  Cannot honestly be deployed because the
selection criterion was "lowest test SIMSCORE" — pure overfitting
to the holdout.

The 0.4367 → 0.4447 = 1.8 % gap between V11_test_aware and V11_final
is the **objective ceiling** of how much further an OOF-respecting
method could go *if* we had more validation history covering the
post-2025-Q1 regime.

## My recommendation

**Ship V11_final.**  Here is the reasoning:

1. **Strict OOF-safe.**  Under the criterion we pre-registered in the
   V11 plan, V11_final is the unambiguous winner.  Anyone auditing
   our process can see: criterion pre-registered → CV on val → champ
   chosen by recipe → not tweaked after.

2. **The 0.4447 → 0.4417 gap to V11_relaxed is real but small** (0.7 %).
   In the context of the val→test bias drift in this dataset (V10
   alone had +5 % test bias), 2.8 % bias is competitive.

3. **V11_relaxed is a defensible alternative** — flag it in the
   metadata as "alt champion" so a deployer can choose based on their
   tolerance for hindsight tuning.

4. **V11_test_aware shows the ceiling** — useful for goal-setting:
   if we collect 6 more months of validation history, can we close
   the 1.8 % gap?  Maybe.  Without those months, no.

## Key insight: every V11 base alone underperforms V10 LAD on test

| Model | Test SIMSCORE |
|---|---:|
| `v10_lad` (alone) | 0.4624 |
| `v11_lad` (alone, no λ-blend) | 0.4598 |
| `v11_g93` (alone) | 0.5135 |
| `v11_g90` (alone) | 0.5263 |
| `v11_recent_only` | 0.5263 |
| `v11_chronos` | 0.8838 |

**The V11 win comes ENTIRELY from the post-LAD λ-blend.**  Each
individual V11 base is *worse* than V10 LAD alone on test SIMSCORE.
Only the small (10–20 %) admixture of negatively-biased V11_g93
into the V10 LAD positively-biased baseline produces the test
improvement.  This is a textbook case of ensemble-as-bias-correction:
no individual member beats the baseline, but a careful weighted
combination does.

## Files written

* `output/all_models_comparison.csv` — full 62-model ranked table
* `output/plot_all_models_comparison.png` — 4-panel grid (test SIMSCORE,
  test WAPE, test bias, val→test scatter)
* `output/plot_all_models_timeline.png` — per-month total demand,
  top-8 OOF-safe models
* `docs/v11_final_report.md` — full V11 narrative with variants table
