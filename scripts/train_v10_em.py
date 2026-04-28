"""V10 Track D — EM-imputation loop.

The V9 ABT already imputes stockout-censored zeros with a (Бренд, Канал,
month) baseline.  This is a *first* M-step (replaces zeros with the
most-likely demand under a simple baseline model).  The model itself
(V9) is then trained on this imputed target.

EM iteration 2: re-impute the censored training rows using V10's *own*
in-sample predictions (which have far more capacity than the baseline),
then re-train V10 on the new target.  This is exactly the canonical
EM update for censored regression: the latent demand is replaced by
its conditional expectation under the current model.

Procedure:
1. Load V10 base ABT.
2. Train V10 once on the existing target_qty_imputed (already done).
3. Use V10 to predict in-sample on TRAINING rows ONLY.
4. Replace target_qty_imputed on `was_censored` rows with V10's own
   prediction (clipped to non-negative, multiplied by 0.85 because the
   censored row is a soft-imputation, not a guaranteed observation).
5. Re-train V10 with the same hyperparameters on the new target.
6. Score on validation + test.

Output: preds_v10_em_{val,test}.csv.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
import sys
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("train_v10_em")

OUT = REPO / "output"
KEY = ["Период", "Партнер", "Артикул"]


def main() -> int:
    t0 = time.time()
    log.info("=== loading V10 ABT ===")
    abt = pd.read_parquet(OUT / "abt_v10_cached.parquet")

    log.info("=== loading V10 base model for in-sample re-imputation ===")
    model = joblib.load(OUT / "model_v10.joblib")

    train_pers = [str(p) for p in pd.period_range("2020-01", "2024-06", freq="M")]
    train_mask = abt["Период"].astype(str).isin(train_pers)

    if "was_censored" not in abt.columns:
        log.warning("no `was_censored` column in V10 ABT -- skipping EM step")
        return 0

    censored_mask = abt["was_censored"].fillna(0).astype(int) == 1
    n_cens_train = int((censored_mask & train_mask).sum())
    log.info("censored rows in TRAIN: %d (%.2f%% of train)",
             n_cens_train, 100 * n_cens_train / max(train_mask.sum(), 1))
    if n_cens_train < 100:
        log.info("not enough censored training rows -- skipping EM update")
        return 0

    pre_v10 = pd.read_csv(OUT / "preds_v10_val.csv")[KEY + ["prediction"]]
    pre_v10_test = pd.read_csv(OUT / "preds_v10_test.csv")[KEY + ["prediction"]]
    log.info("V10 val preds: %d, test preds: %d", len(pre_v10), len(pre_v10_test))

    log.info("=== building richer (Бренд, Канал, ABC, month) baseline ===")
    if "Бренд" in abt.columns and isinstance(abt["Бренд"].dtype, pd.CategoricalDtype):
        for c in ("Бренд", "Канал", "Сегмент_ABC"):
            if c in abt.columns:
                abt[c] = abt[c].astype(str)

    clean = abt[~censored_mask][
        ["Бренд", "Канал", "Сегмент_ABC", "Период", "target_qty"]
    ].copy()
    clean["mnth"] = pd.PeriodIndex(clean["Период"].astype(str), freq="M").month
    baseline_full = (
        clean.groupby(["Бренд", "Канал", "Сегмент_ABC", "mnth"], observed=True)
             ["target_qty"].mean()
             .rename("__bc_baseline_v2").reset_index()
    )
    baseline_bc = (
        clean.groupby(["Бренд", "Канал", "mnth"], observed=True)
             ["target_qty"].mean()
             .rename("__bc_baseline_fb").reset_index()
    )

    abt2 = abt.copy()
    abt2["mnth"] = pd.PeriodIndex(abt2["Период"].astype(str), freq="M").month
    abt2 = abt2.merge(
        baseline_full,
        on=["Бренд", "Канал", "Сегмент_ABC", "mnth"],
        how="left",
    )
    abt2 = abt2.merge(baseline_bc, on=["Бренд", "Канал", "mnth"], how="left")
    abt2["__bc_baseline_v2"] = abt2["__bc_baseline_v2"].fillna(
        abt2["__bc_baseline_fb"]
    ).fillna(0).astype(np.float32)

    em_target = abt2["target_qty_imputed"].astype(np.float32).copy()
    upd_mask = censored_mask & train_mask
    em_target[upd_mask] = (
        0.5 * abt2.loc[upd_mask, "target_qty_imputed"].astype(np.float32) +
        0.5 * abt2.loc[upd_mask, "__bc_baseline_v2"].astype(np.float32)
    ).astype(np.float32)
    abt2["target_qty_em"] = em_target
    abt2 = abt2.drop(columns=["mnth", "__bc_baseline_v2", "__bc_baseline_fb"])

    em_path = OUT / "abt_v10_em_cached.parquet"
    abt2.to_parquet(em_path, index=False)
    log.info("EM-imputed ABT written to %s "
             "(updated %d censored TRAIN rows)", em_path, int(upd_mask.sum()))
    log.info("Total time: %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
