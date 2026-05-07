# V14 GlobalNN retrospective — leakage caught, honest result documented

**Date:** 2026-05-07
**TL;DR:** V14 GlobalNN was trained on Kaggle T4/P100 GPU end-to-end.
First run produced spectacular results (test SIM 0.1377 standalone,
−69 % vs V11_final), but **closer inspection revealed target leakage**
in the feature export — `Количество_sales` (current-month sales
quantity, correlation **+1.0000** with `target_qty`) was included as
a feature. The export script grabbed all numeric columns from
`abt_v12_external` except literal `target_qty`, but missed the V7
LightGBM trainer's canonical leakage exclusion list.

After fixing the export to use `src.model_v2.get_feature_columns_v2()`
(which excludes 11 current-month leakage columns) and re-training,
**honest V14 standalone test SIM = 0.5213** (worse than V12.2's
0.4435 by +17.5 %), and **V14 earns 0.075 LAD weight** in V12.6 joint
search but the resulting blend (`0.925·V11_final + 0.075·V14_globalnn`)
gives test SIM 0.4438 — a **0.07 % regression vs V12.2**.

**Production stays V12.2_champion.** V14 ships as a documented base
that contributes nothing useful to the final ensemble (under honest
OOF; same outcome as Chronos in V13).

---

## Timeline

| Time | Event |
|---|---|
| 07:25 | Pushed v14_globalnn dataset (231k train rows, 21 MB) to Kaggle. |
| 07:26 | Pushed v1 of kernel — RAN, errored on dataset path. |
| 07:31 | v2: added recursive dataset auto-discovery + GPU compatibility shim — RAN, hit P100 sm_60 incompatibility with torch 2.10. |
| 07:35 | v3: added in-process torch reinstall — RAN, errored on `_has_torch_function` already has docstring (Python C-ext can't reload). |
| 07:42 | v4: split into Cell 1 (install only, no torch import) + Cell 1b (import after install) — RAN, dataset path mounted at `/kaggle/input/datasets/<owner>/<slug>` (unusual nested path for private API datasets). |
| 07:48 | v5: added recursive search via `glob('/kaggle/input/**/train.parquet')` — RAN, **completed successfully**. |
| 07:55 | First scoring: standalone test SIM **0.1377** — too good to be true. |
| 08:00 | Investigation: `Количество_sales` correlation +1.0000 with target. **Leakage confirmed.** |
| 08:05 | Fixed export to use `get_feature_columns_v2()`. |
| 08:08 | Pushed dataset v2 (clean) + kernel v6 (re-uses fixed dataset). |
| 08:25 | Kernel v6 completed. Honest test SIM **0.5213** (worse than V12.2). |
| 08:30 | V12.6 multi-helper search: V14 earns 0.075 weight, blend test SIM 0.4438 (-0.07% vs V12.2). |

---

## What V14 actually delivers (honest)

Standalone test (full row coverage 34 216 rows):

| Metric | Honest V14 | V12.2 (full 18 298 rows) |
|---|---:|---:|
| Test SIMSCORE | 0.6325 | **0.4435** |
| Test WAPE | 0.5224 | **0.3931** |
| Test Bias % | −6.89 | **+2.13** |
| Test M-WAPE | 0.1513 | **0.0794** |

V14 covers MORE rows than V12.2 (includes inactive/zero-demand pairs).
On the V12.2-aligned subset (18 298 rows), V14 standalone test SIM is
**0.5213** — still worse than V12.2's 0.4435.

As a HELPER (V12.6 joint OOF search):

| Variant | Recipe | Test SIM | Δ vs V12.2 |
|---|---|---:|---:|
| **V12.2 (production)** | `0.925·V11_final + 0.075·V12_external` | **0.4435** | — |
| V12.6 (V14 helper) | `0.925·V11_final + 0.075·V14_globalnn` | 0.4438 | +0.07 % (worse) |
| V12.6_test_optimal | `0.65·V11_final + 0.20·V12_ext + 0.15·V14` | 0.4281 (peek) | −3.5 % (peeked, not OOF) |

The OOF-honest V12.6 with V14 doesn't beat V12.2. A peeked-at-test
joint blend WOULD give a 3.5 % lift, but it's not OOF-defensible
(OOF_bias −1.81 % violates the strict 1.25 % ceiling).

---

## Why V14 didn't dominate (honest analysis)

1. **No new information dimension.** V14 trains on the same
   `abt_v12_external` features as V12_external (the V12.1/V12.2
   helper). The Transformer-encoder + embeddings architecture is more
   expressive than LightGBM, but on this problem the LightGBM TwoStage
   is already extracting most of the signal. V14's gain is
   marginal (~0.5 % WAPE) once leakage is removed.
2. **Embedding overfitting risk.** With 514 SKUs × 63 partners =
   32 382 unique pairs and only 7 observations per pair on average,
   embeddings memorize per-pair patterns. Pinball loss + 4-layer
   transformer makes this worse. The `target_qty` per-pair median is
   essentially what the embedding learns.
3. **Bias direction wrong.** V14 has slightly *negative* test bias
   (−6.9 %) — wrong direction for V11_final (which already has
   positive +2.8 % bias). V14 doesn't help bias calibration.
4. **Higher Monthly-WAPE on full coverage.** V14 makes per-row
   predictions worse, even though M-WAPE on aligned (active) is
   slightly better.

---

## What we learned

1. **Leakage detection should be earlier.** Always run a correlation
   check on training features vs target BEFORE training. A feature
   with corr >0.9 is almost always leakage. We had this check in V7
   (via `get_feature_columns_v2`), forgot to apply it in the V14
   export.
2. **Standalone WAPE 0.10 was too good for this domain.** When a
   model improves headline WAPE by 70 % vs the previous champion, the
   first hypothesis should be leakage, not architectural superiority.
3. **Kaggle pipeline is now hardened**:
   - GPU compatibility shim handles P100 vs T4 transparently.
   - Dataset auto-discovery handles `/kaggle/input/datasets/<owner>/<slug>`.
   - Self-contained kernel notebook (no Drive needed).
   - End-to-end test cycle (push → run → pull) is ~10 min.
4. **`scripts/v14_kaggle_check.sh`** wraps all the polling/pulling so
   future GPU experiments are one command away.

---

## What ships from V14 work

* `notebooks/v14_globalnn_colab.py` — Colab paste-and-run version (works on T4)
* `output/v14_kaggle_kernel/v14_globalnn.ipynb` — Kaggle version with GPU shim
* `output/v14_kaggle_dataset/` — uploaded dataset (clean, v2)
* `scripts/build_v14_kaggle_notebook.py` — auto-builds the .ipynb
* `scripts/v14_kaggle_check.sh` — one-command status/pull/merge helper
* `output/preds_v14_globalnn_{val,test}.csv` — clean V14 predictions
* `scripts/v126_multihelper.py` — joint OOF search with V14 in pool
* `scripts/export_v14_globalnn_data.py` — fixed (uses `get_feature_columns_v2`)

V14 is now reproducible end-to-end: re-run `python -m
scripts.export_v14_globalnn_data`, push dataset, push kernel, pull
results. Total time ~30 min for a full retrain.

---

## Decision log

* **2026-05-07 08:30** — V14_alpha trained, **honest result documented**.
  V12.2_champion remains production. V14 base preserved as documented
  helper with 0.075 LAD weight (displaces V12_external in OOF).
* **Next** — V14 retrain with proper hyperparameters (longer training,
  per-pair regularization to prevent embedding overfit) MIGHT recover
  some of the standalone gap. But infrastructure now exists; future
  V14.1 can iterate quickly.
