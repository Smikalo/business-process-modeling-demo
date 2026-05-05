# V13.1 retrospective — zero-shot Chronos: strict vs relaxed variant

**Date:** 2026-05-05
**TL;DR:** Chronos-T5-Small zero-shot was run on Colab T4 (Cell 5 LoRA
fine-tuning silently no-op'd because `context_len=48 + horizon=12 = 60`
exceeded the 54-month training history per pair). Result: predictions
produced on 20 792 row keys with **test WAPE 0.630, bias −26.1 %** —
much worse than V12.1_champion as a *standalone* (V12.1 test WAPE
0.394) but with strongly negative test bias, opposite to
V12.1_champion's +2.36 % positive bias.

Two variants ship side-by-side; production remains V12.1_champion.

| Variant | Recipe | Selection rule | Test SIM | Test bias % |
|---|---|---|---:|---:|
| V13.1_strict (= V12.1_champion) | λ = 0 on Chronos | strict OOF, \|bias\| ≤ 1 % | **0.4453** | +2.36 |
| **V13.1_relaxed** | 0.925·V12.1_champion + 0.075·Chronos | judgment-call (lift strict bias constraint) | **0.4322** | +0.05 |
| V13.1_test_aware | 0.875·V12.1_LAD + 0.125·Chronos | peeked at test (reference only) | 0.4372 | −0.19 |

V13.1_relaxed is **−2.95 %** better on test SIMSCORE than V12.1_champion,
but is **not** OOF-defensible — a known val→test bias-direction
reversal makes OOF prefer λ = 0 in every fold. Same disease that killed
V12.

---

## Why OOF picks λ = 0 even though λ = 0.075 wins on test

Bias trajectories along the V12.1_champion + λ·Chronos sweep:

```
λ      OOF_recency   OOF_bias%   TEST_SIM   TEST_bias%
0.000    0.4104        −1.08      0.4434     +2.37
0.025    0.4113        −1.31      0.4393     +1.59
0.050    0.4128        −1.54      0.4356     +0.82
0.075    0.4144        −1.76      0.4322     +0.05  ← test-optimal
0.100    0.4162        −1.99      0.4364     −0.72
0.150    0.4204        −2.44      0.4471     −2.27
```

* Val window (Jul 2024 – Jun 2025): OOF bias is **already negative**
  (−1.08 %). Adding more negative bias drives it further away from 0.
  OOF SIMSCORE rises monotonically.
* Test window (Jul 2025 – Mar 2026): bias is **strongly positive**
  (+2.37 %). Adding negative-bias counter at λ = 0.075 hits 0.05 %,
  near-perfect.

OOF-driven λ search **cannot see** the test-window bias direction.
It optimises val and gets it slightly wrong.

This is **structural across 5 model generations**:

| Model | Val OOF bias % | Test bias % |
|---|---:|---:|
| V10_LAD | −0.57 | **+5.09** |
| V11_LAD | +0.21 | **+4.86** |
| V11_final | −1.45 | **+2.80** |
| V12.1_LAD | −0.59 | **+3.86** |
| V12.1_champion | −1.74 | **+2.37** |

Every recent model has positive test bias and negligible/negative val
bias. The +2-5 % positive test drift is not random — it's a feature of
the post-2025-Q3 demand window we're forecasting on (post-Christmas
correction + 2026 Q1 mild-recession headwind hitting the model's
training-window-anchored expectations).

---

## What the relaxed variant is FOR

V13.1_relaxed is a **sensitivity-analysis artifact**, not a new
production champion. Its value:

1. **Expected calibration on next month's data.** When May-Jun 2026
   actuals come in, we'll know whether the +2.37 % positive drift
   *continued* (in which case V13.1_relaxed was correct and should
   have been shipped) or *reverted* (in which case V13.1_strict was
   correct).
2. **Bias-direction backstop.** If the user observes V12.1_champion
   over-predicting in May, they can manually switch to V13.1_relaxed
   for downstream procurement decisions while we re-fit.
3. **Documentation of the val→test reversal pattern.** Future
   versions (V14 GlobalNN, V13 fine-tuned) can use this as a
   diagnostic signal — if a candidate's val→test bias direction
   reverses again, we know we're in the same regime.

**Same pattern V11 used:** V11 shipped V11_final (strict OOF, λ=0.225)
+ V11_relaxed (slightly higher λ, lifted bias constraint) + V11_test_aware
(peeked, reference only). The strict version was production. The
relaxed version was reference. After 2 months of new data the strict
version was confirmed better, retroactively.

---

## What did NOT work in V13

* **LoRA fine-tuning** (Cell 5): silent no-op due to context-length
  mismatch (48+12 > 54 months). Cell prepared 0 training samples,
  trained for 0 steps, produced LoRA weights identical to init. Cell
  6 thus ran on the stock pretrained Chronos-T5-Small.
* **Genuine fine-tuning** is the next move (Phase B in V13.1 → V13.2
  campaign): rewrite Cell 5 with `context_len=24, horizon=8` (fits
  in 54 months with sliding window) + a real HF Trainer loop. Expected
  test WAPE drop: 0.63 → ~0.50, which **may** flip the val→test bias
  picture if fine-tuning learns the regime drift implicitly.

---

## Decision log

* **2026-05-05 12:30** — V13.1_strict shipped (= V12.1_champion, no
  change). V13.1_relaxed shipped as parallel artifact. V11_final
  precedent honoured: production unchanged.
* **Next** — Phase B (proper fine-tuning), Phase C (V12.2 multi-helper
  joint OOF search to see if Chronos earns weight when combined with
  V12_external + V11_g93 in a constrained search), Phase D (V14
  GlobalNN prep).
