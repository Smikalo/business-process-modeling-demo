# V12 retrospective — what we tried, what worked, what didn't, what we learned

**Date:** 2026-04-29
**TL;DR:** V12 candidate **did not** beat V11 on the held-out test window
(test SIMSCORE 0.4607 vs V11_final 0.4489, **+2.63 % regression**).
**Action:** keep V11_final as production champion. V12 infrastructure
(EXT loaders, multi-seed bagging, intermittent specialist) ships as
groundwork for V12.1.

---

## What we tried in V12

| Component | Status | Effect |
|---|---|---|
| **Phase EXT — 9 priority-1 free open-data loaders** | ✅ all 9 loaders implemented | adds 32 EXT feature columns to ABT |
| `ukrstat_rti` — UA Retail Trade Index | ✅ implemented (synthetic fallback when scrape fails) | macro context |
| `ukrstat_births` — UA births by oblast×month | ✅ implemented (synthetic fallback) | toy demand cohort |
| `ukrstat_indprod` — UA Industrial Production Index | ✅ implemented | macro context |
| `nbu_cci` — NBU Consumer Confidence + Inflation Expectations | ✅ implemented (live API was 404; synthetic) | demand sentiment |
| `airraid_oblast` — alerts.in.ua oblast×day → nat-month | ✅ implemented (synthetic, token required for live) | war intensity |
| `blackout_dtek` — DTEK / Ukrenergo blackout schedule | ✅ implemented (synthetic, deferred live scrape to V12.1) | infra disruption |
| `iom_idp` — IOM Displacement Tracking Matrix | ✅ implemented (synthetic, PDF parse deferred) | demographic shift |
| `wiki_pageviews` — Wikipedia Pageviews top toy/franchise pages | ✅ implemented, **live API succeeded** | attention proxy |
| `orthodox_calendar` — UA Orthodox holiday calendar | ✅ implemented (deterministic) | religious gift-giving cycles |
| **Multi-seed bagging** — V11_recent_only × 5 seeds → averaged preds | ✅ trained, gives `preds_v12_multiseed` | reduces single-seed variance |
| **Croston/SBA/TSB intermittent specialist** | ✅ trained, gives `preds_v12_intermittent` | targets long-tail intermittent SKUs |
| **Bias-laddered LAD search** | ✅ ran 99 candidates over 4 bias ceilings | picks champion across the ladder |
| **Robust-objective final blend** | ✅ implemented | uses worst-case bias scenario |

---

## What actually happened on the test window

| metric | V11_final | V12_final | Δ |
|---|---:|---:|---:|
| Test SIMSCORE | 0.4489 | **0.4607** | **+2.63 %** ❌ |
| Test WAPE | 0.3950 | 0.3983 | +0.84 % |
| Test \|bias %\| | 2.80 % | **4.48 %** | **+1.68 pp** ❌ |
| Val SIMSCORE | 0.3575 | 0.3514 | −1.71 % ✅ |
| Val→Test gap | +0.0914 | **+0.1093** | **+0.0179** ❌ |

**The pattern: V12 looks better on validation (val SIMSCORE 0.3514 vs 0.3575) but worse on test.**

The reason is a **val→test bias-direction reversal**:

* On validation (and on all 3 CV-OOF folds), V12 has slightly *negative*
  aggregate bias (`-0.87 %` recency-weighted, `-1.47 %` on the most
  recent fold).
* On the held-out test window, V12 has strongly *positive* aggregate
  bias (`+4.48 %`).

This bias direction flip cannot be detected by OOF-driven
hyperparameter selection. Any λ-blend search that minimises OOF
SIMSCORE will pick `λ = 0` (no defensive blend) because adding the
negative-bias helper (`v11_g93`) makes OOF bias *more* negative,
which OOF SIMSCORE penalises. But it would help test bias.

V11_final didn't have this issue because V11_LAD itself had positive
bias on OOF (`+0.21 %` calibrated, originally `+5 %` raw), so adding
negative-bias `v11_g93` improved both OOF and test simultaneously. A
helper-direction match is a fragile assumption — V12 inadvertently
broke it.

---

## What we learned

1. **OOF SIMSCORE is a noisy estimator of test SIMSCORE.** When the
   improvement margin is small (<1 %), random val→test variation
   dominates the model improvement.

2. **Bias-direction stationarity is NOT a free assumption.** V11
   benefited from a coincidence: V11_LAD's OOF bias and test bias
   pointed the same way. We treated this as a feature of the LAD
   pipeline; it is actually a property of the *specific pool* V11
   chose. V12's pool (which differs by adding multi-seed + intermittent
   bases) reversed the OOF bias direction.

3. **Multi-seed bagging didn't help.** It reduced variance (each seed
   converged near the same point) but didn't address the dominant
   error source: bias drift. We expected ~0.5 % improvement; we
   measured noise.

4. **The Croston/SBA/TSB specialist didn't help either at the LAD
   merge level.** The intermittent specialist is *individually*
   competitive on intermittent pairs, but its predictions are
   strongly correlated with V10_recent on those same pairs, so LAD
   gave it minimal weight, and what weight it did receive worsened
   the bias profile.

5. **The EXT features did not enter the LAD search at all.** The LAD
   search only sees prediction CSVs, not features. To benefit from
   EXT, we would need to *re-train* a base on `abt_v12_external`
   (replacing V10's ABT with V12's). That re-training is what V12.1
   should do.

6. **Synthetic fallbacks worked exactly as designed.** Half of the EXT
   loaders fell back to synthetic data because live scraping failed.
   The pipeline didn't crash and produced 32 feature columns with
   100 % monthly coverage — confirming the resilience pattern is
   sound.

---

## What V12.1 should do

In priority order:

1. **Re-train at least one V11 base on `abt_v12_external`.** The EXT
   features can only help if a model actually learns from them. The
   cheapest test: take `build_v11_recent_only.py`'s training pipeline,
   point it at `abt_v12_external.parquet` instead of
   `abt_v10_cached.parquet`, and produce
   `preds_v12_recent_only_external_{val,test}.csv`. Add to LAD pool.
   *Expected lift if EXT signals carry information: 0.5–1.5 %.*

2. **Add bias-direction-symmetry constraint to LAD search.** Currently
   the LAD search constrains *magnitude* of OOF bias (`≤ 1.0 %`).
   Add a constraint that the OOF bias direction must *match* on the
   most recent CV fold — i.e., reject pools where the bias direction
   reverses across folds. This would have rejected V12's `v10+all_v12`
   pool because its most-recent-fold bias is `-1.47 %` while V11's is
   `+1.5 %` — opposing directions, signalling instability.

3. **Anomaly-downweighting base.** We deferred this from V12. Build
   it in V12.1 — a re-train that down-weights samples from war-shock
   months (Mar–May 2022) and the two big blackout windows
   (Oct 2022 – Mar 2023, Oct 2024 – Mar 2025). *Expected lift: 0.3 %.*

4. **Per-source A/B audit** of the EXT signals once a base actually
   uses them. The infrastructure is already in `output/v12_external_attribution.csv`;
   we need a model that reads them.

5. **Replace `streaming_calibrator` use in `v12_lad_bias_ladder.py`.**
   Currently it's run with `axes=["Канал"]` and made things worse on
   test (raw V12_LAD 0.4607 → after streaming 0.4670 → after λ-blend
   0.4607). A robust calibrator should *not* worsen scores; this one
   does because it over-corrects in low-volume channels. Investigate
   and either fix or remove.

6. **Honest test-set sanity panel in the LAD search.** Currently the
   LAD search reports OOF metrics only. Adding a separate "test peek"
   panel (used only for *gating*, not for *selection*) would have
   surfaced the OOF→test divergence immediately. We already have
   `audit_v12_oof.py`; just integrate its output into the LAD log.

---

## Decision log

* **2026-04-29** — V12 candidate fails acceptance gate
  (`v12_test_simscore_beats_v11`). V11_final remains production
  champion. V12 infrastructure is committed but `champion_card.json`
  is **not** bumped.

* **Next** — V12.1 plan above. Estimated 1 week of CPU work, no GPU
  required.

* **V13 / V14 status** — No change; the GPU-fine-tuned foundation
  models (V13) and the Transformer-encoder GlobalNN (V14) are
  *independent* of the bias-direction issue diagnosed here. Their
  scaffolding (`scripts/export_v13_fm_data.py`,
  `notebooks/v13_chronos_finetune_colab.py`,
  `scripts/v13_lad_stack.py`,
  `scripts/v14_lad_stack.py`,
  `src/models/global_nn.py`,
  `scripts/export_v14_globalnn_data.py`,
  `docs/v12_v14_human_action_guide.md`) is ready for the human to run
  on Colab/Kaggle whenever convenient.

---

## Things that *did* ship cleanly

* 9 new external open-data loaders, all with synthetic fallback.
* `output/abt_v12_external.parquet` — V11 ABT + 32 EXT columns,
  100 % monthly coverage. Ready for any V12.1 re-train.
* `output/preds_v12_multiseed_{val,test}.csv` — 5-seed bagged predictions.
* `output/preds_v12_intermittent_{val,test}.csv` — Croston/SBA/TSB
  specialist predictions.
* Full V13 GPU handoff package — Colab/Kaggle paste-and-run scripts +
  human-action guide.
* `src/models/global_nn.py` — Transformer-encoder skeleton ready
  for V14 fine-tuning.
* `scripts/v13_lad_stack.py`, `scripts/v14_lad_stack.py` — automatic
  LAD merges that activate as soon as the relevant prediction CSVs
  appear in `output/`.

The infrastructure is solid; the modeling-level decision (which
bases earn LAD weight) is what didn't pan out this round.
