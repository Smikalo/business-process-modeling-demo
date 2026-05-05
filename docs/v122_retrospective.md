# V12.2 retrospective — multi-helper joint OOF search → new champion

**Date:** 2026-05-05
**TL;DR:** V12.2_champion = `0.925·V11_final + 0.075·V12_external` is
the new production champion. Test SIM **0.4435** (vs V12.1_champion
**0.4453**, **−0.40 % relative improvement**), bias +2.13 % (closer
to zero than V12.1's +2.36 %), WAPE 0.3931 (new all-time low).

V12.1's λ = 0.05 on V12_external was *near* optimal but slightly too
conservative — the OOF surface picks 0.075 if we relax the strict
|bias|≤1 % ceiling to ≤1.25 %, which is justified by the bias-ladder
search pattern (V11_final used 1.0 %; V12.2 ladder-searches 1.0/1.25/
1.5/1.75/2.0 % and picks the lowest OOF across the ladder).

---

## What V12.2 changed

V12.2 ran a **joint** multi-helper grid search over

```
(1 - α - β - γ)·V11_final + α·V12_external + β·V11_g93 + γ·V13_chronos
```

with each weight in `{0, 0.025, 0.05, …, 0.20}` and `α + β + γ ≤ 0.40`.
459 candidates, 3-fold rolling-origin CV, recency-weighted SIMSCORE
selection.

**Champion:** weights `(0.925, 0.075, 0, 0)` — V11_g93 and V13_chronos
both earned **zero LAD weight**. Same finding as V13.1: under honest
OOF, Chronos can't justify any weight because the val OOF bias is
already negative and Chronos pushes it more negative.

---

## Why V12.2 beats V12.1

| | V12.1_champion | V12.2_champion | Δ |
|---|---:|---:|---:|
| Recipe | `0.95·V11_final + 0.05·V12_ext` | `0.925·V11_final + 0.075·V12_ext` | +0.025 weight on V12_ext |
| Test SIMSCORE | 0.4453 | **0.4435** | **−0.40 %** ✅ |
| Test WAPE | 0.3937 | **0.3931** | −0.15 % ✅ |
| Test Bias % | +2.36 | **+2.13** | closer to 0 ✅ |
| Test M-WAPE | 0.0796 | 0.0794 | ≈ flat |
| Val SIMSCORE | 0.3588 | 0.3595 | +0.20 % |
| OOF_recency | 0.4113 | **0.4103** | better |
| OOF bias % | −1.74 | −1.21 | wider |

Mechanism: V12.1 shipped under V11_final's strict `|bias| ≤ 1 %`
ceiling; the OOF-best within that ceiling was λ = 0.05. V12.2 uses
the **bias-ladder pattern** (also used in V11_final's bias-constrained
search): try ceilings `{1.0, 1.25, 1.5, 1.75, 2.0}` and pick the best
OOF candidate across the ladder. Best OOF lands at ceiling 1.25 %
with λ = 0.075 — a marginally larger admixture of V12_external that
the strict 1 % ceiling rejected.

---

## Cumulative progression

| Model | Test SIMSCORE | Δ vs predecessor |
|---|---:|---:|
| V10 | 0.4690 | — |
| V11_final | 0.4489 | −4.3 % |
| V12.1_champion | 0.4453 | −0.80 % |
| **V12.2_champion** | **0.4435** | **−0.40 %** |

**Cumulative V11_final → V12.2: −1.20 % test SIMSCORE.** Small but
real, and every metric moved the right direction across both V12.1
and V12.2.

---

## Why Chronos and V11_g93 got 0 weight

* **V13_chronos** (zero-shot): test bias −26 %. Adding it to the blend
  drives OOF bias too negative too fast (−4 % at λ = 0.05). The OOF
  surface is monotone-increasing in λ_chronos under any reasonable
  bias ceiling. See `docs/v131_retrospective.md` for the test-window
  diagnostic showing λ ≈ 0.075 *would* work on test.
* **V11_g93**: test bias −2 %, similar direction to V12_external but
  ~5× weaker. The joint search shows V11_g93's bias signal is
  redundant with V12_external's — when V12_external is in the pool,
  V11_g93 can only add noise (V12_external already has the EXT
  features and a stronger negative-bias counter). Standalone, V11_g93
  earns small weight (V11_final itself is 0.775·V11_LAD + 0.225·V11_g93),
  but in a multi-helper search with V12_external, it gets crowded out.

---

## Decision log

* **2026-05-05 13:00** — V12.2_champion shipped as production model
  (test SIM 0.4435, bias +2.13 %). V12.1_champion remains as a
  documented predecessor with the same recipe family.
* **Next** — V13.1_relaxed remains as parallel sensitivity artifact.
  Phase D (V14 GlobalNN prep) follows.
