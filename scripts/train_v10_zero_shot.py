"""V10 -- Robust zero-shot seasonal-naive baseline.

A purely classical, no-ML 'foundation-model-style' base: predict each
(Партнер, Артикул, target month) using a robust median-of-multiple-
naive-seasonal-estimators ensemble:

  pred = median(
    same-month-last-year (lag_12),
    same-month-two-years-ago (lag_24),
    last-3-months mean,
    last-6-months trimmed mean,
    last-12-months median,
  )

Pure series-level forecasting -- never sees structural features -- so
its residuals are GUARANTEED orthogonal to every LightGBM-driven base
in V1-V10.  Acts as a 'sanity-check' base in the LAD pool: when ML
overfits a quirk, the seasonal-naive median pulls the ensemble back
toward the long-run pattern.

Output
------
* preds_v10_zero_shot_val.csv
* preds_v10_zero_shot_test.csv
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("v10_zero_shot")

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
KEY = ["Период", "Партнер", "Артикул"]


def _trim_mean(s: np.ndarray) -> float:
    if len(s) == 0:
        return 0.0
    if len(s) <= 2:
        return float(np.mean(s))
    s_sorted = np.sort(s)
    trim = max(1, len(s_sorted) // 5)
    return float(np.mean(s_sorted[trim:-trim]))


def _median_estimators(ts: np.ndarray, idx: int) -> float:
    if idx < 1:
        return 0.0
    estimators = []
    if idx >= 12:
        estimators.append(ts[idx - 12])
    if idx >= 24:
        estimators.append(ts[idx - 24])
    last3 = ts[max(0, idx - 3):idx]
    if len(last3) > 0:
        estimators.append(float(np.mean(last3)))
    last6 = ts[max(0, idx - 6):idx]
    if len(last6) > 0:
        estimators.append(_trim_mean(last6))
    last12 = ts[max(0, idx - 12):idx]
    if len(last12) > 0:
        estimators.append(float(np.median(last12)))
    if not estimators:
        return 0.0
    return float(np.median(estimators))


def main() -> int:
    t0 = time.time()
    log.info("=== loading V9 ABT (for densification + active pairs) ===")
    abt = pd.read_parquet(OUT / "abt_v9_cached.parquet")
    abt["Период"] = abt["Период"].astype(str)

    val_pers = [str(p) for p in pd.period_range("2024-07", "2025-06", freq="M")]
    test_pers = [str(p) for p in pd.period_range("2025-07", "2026-01", freq="M")]
    target_pers = val_pers + test_pers

    pair_df = abt[abt["Период"].isin(target_pers)][["Партнер", "Артикул"]].drop_duplicates()
    log.info("active pairs in target horizon: %d", len(pair_df))

    months_all = sorted(abt["Период"].unique())
    month_to_ix = {m: i for i, m in enumerate(months_all)}
    log.info("history months: %d  (%s..%s)",
             len(months_all), months_all[0], months_all[-1])

    pivot = (
        abt.groupby(["Партнер", "Артикул", "Период"], observed=True)["target_qty"]
           .sum().reset_index()
           .pivot(index=["Партнер", "Артикул"], columns="Период", values="target_qty")
           .fillna(0).astype(np.float32)
           .reindex(columns=months_all, fill_value=0)
    )
    pivot = pivot.reindex(pd.MultiIndex.from_frame(pair_df))
    pivot = pivot.fillna(0)
    log.info("pivot shape: %s", pivot.shape)

    rows = []
    arr = pivot.values
    pairs = pivot.index.tolist()
    for tgt in target_pers:
        idx = month_to_ix[tgt]
        for i, (partner, sku) in enumerate(pairs):
            ts = arr[i]
            pred = _median_estimators(ts, idx)
            rows.append({
                "Период": tgt, "Партнер": partner, "Артикул": sku,
                "prediction": max(pred, 0.0),
            })
    out = pd.DataFrame(rows)
    log.info("predictions made: %d", len(out))

    target = abt[abt["Период"].isin(target_pers)][KEY + ["target_qty"]]
    out = out.merge(target, on=KEY, how="inner")
    out = out[KEY + ["target_qty", "prediction"]]

    val_out = out[out["Период"].isin(val_pers)]
    tst_out = out[out["Период"].isin(test_pers)]
    val_out.to_csv(OUT / "preds_v10_zero_shot_val.csv", index=False)
    tst_out.to_csv(OUT / "preds_v10_zero_shot_test.csv", index=False)

    from scripts.score_similarity import score_frame
    sv = score_frame(val_out)
    st = score_frame(tst_out)
    log.info("V10_zero_shot val   SIMSCORE=%.4f WAPE=%.4f bias%%=%+.2f M-WAPE=%.4f",
             sv["SIMSCORE"], sv["WAPE"], sv["Agg_Bias_pct"], sv["Monthly_WAPE"])
    log.info("V10_zero_shot test  SIMSCORE=%.4f WAPE=%.4f bias%%=%+.2f M-WAPE=%.4f",
             st["SIMSCORE"], st["WAPE"], st["Agg_Bias_pct"], st["Monthly_WAPE"])
    log.info("Total time: %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
