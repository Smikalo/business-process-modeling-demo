"""V10 -- Channel Top-Down anchor.

Pivot from full MinT: instead of trying to reconcile 4 360 series jointly
(under-determined with only 54 training months), train ONE high-quality
forecaster at the channel level (4 series × 74 months = 296 obs, ample)
and disaggregate to SKU level using V9's intra-channel monthly share.

Procedure
---------
1. Aggregate V9 ABT to (Канал, Period) -> total_qty.
2. Train a Tweedie LightGBM with channel as cat + lags + seasonality.
3. Predict per-channel monthly totals for val + test.
4. For each (val/test) period, compute V9's within-channel relative share
   per (Партнер, Артикул) as p_v9 / sum_in_channel(p_v9).
5. Reconcile: pred_v10_topdown = share_v9 * channel_total_pred.

This produces a SKU-level forecast that:
  * sums EXACTLY to the channel-level forecaster's prediction
    (a guaranteed top-down consistency)
  * preserves V9's relative SKU distribution within each channel

It is GENUINELY orthogonal to V9 because the channel forecaster sees the
clean aggregate (where seasonality is sharp and noise is averaged out)
while V9 sees a much noisier 60-row-per-pair signal.

Output
------
* preds_v10_topdown_val.csv
* preds_v10_topdown_test.csv
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("v10_topdown")

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
KEY = ["Период", "Партнер", "Артикул"]


def main() -> int:
    t0 = time.time()
    abt = pd.read_parquet(OUT / "abt_v9_cached.parquet")
    abt["Период"] = abt["Период"].astype(str)

    if "Канал" in abt.columns and isinstance(abt["Канал"].dtype, pd.CategoricalDtype):
        abt["Канал"] = abt["Канал"].astype(str)

    train_pers = [str(p) for p in pd.period_range("2020-01", "2024-06", freq="M")]
    val_pers = [str(p) for p in pd.period_range("2024-07", "2025-06", freq="M")]
    test_pers = [str(p) for p in pd.period_range("2025-07", "2026-01", freq="M")]

    log.info("=== aggregating to channel-level monthly totals ===")
    chan = (
        abt.groupby(["Канал", "Период"], observed=True)
           .agg(target_qty=("target_qty_imputed", "sum"))
           .reset_index()
    )
    chan["Период_p"] = pd.PeriodIndex(chan["Период"], freq="M")
    chan = chan.sort_values(["Канал", "Период_p"]).reset_index(drop=True)

    g = chan.groupby("Канал", observed=True)["target_qty"]
    for k in (1, 2, 3, 6, 12):
        chan[f"lag_{k}"] = g.shift(k).astype(np.float32)
    chan["__l1"] = g.shift(1)
    g_l1 = chan.groupby("Канал", observed=True)["__l1"]
    for w in (3, 6, 12):
        chan[f"rmean_{w}"] = (
            g_l1.transform(lambda s: s.rolling(w, min_periods=1).mean())
                .astype(np.float32)
        )
    chan = chan.drop(columns="__l1")
    chan["month"] = chan["Период_p"].apply(lambda p: p.month).astype(np.int8)
    chan["month_sin"] = np.sin(2 * np.pi * chan["month"] / 12).astype(np.float32)
    chan["month_cos"] = np.cos(2 * np.pi * chan["month"] / 12).astype(np.float32)
    chan["yoy_lag12"] = chan["lag_12"]
    chan["yoy_ratio"] = ((chan["lag_1"] + 1.0) /
                        (chan["lag_12"] + 1.0)).astype(np.float32)
    chan["months_since_invasion"] = (
        chan["Период_p"].apply(lambda p: max(0, (p - pd.Period("2022-02", "M")).n))
        .astype(np.int16)
    )
    chan["Канал"] = chan["Канал"].astype("category")

    feats = [c for c in chan.columns
             if c not in ["Период", "Период_p", "target_qty"]]
    train = chan[chan["Период"].isin(train_pers)]
    val = chan[chan["Период"].isin(val_pers)]
    test = chan[chan["Период"].isin(test_pers)]
    log.info("channel-level data: train=%d val=%d test=%d",
             len(train), len(val), len(test))

    train_ds = lgb.Dataset(train[feats], train["target_qty"].astype(np.float32),
                           categorical_feature=["Канал"])
    val_ds = lgb.Dataset(val[feats], val["target_qty"].astype(np.float32),
                         categorical_feature=["Канал"], reference=train_ds)
    booster = lgb.train(
        params={
            "objective": "tweedie", "tweedie_variance_power": 1.3,
            "metric": ["tweedie", "rmse"],
            "num_leaves": 31, "learning_rate": 0.04,
            "min_child_samples": 5, "feature_fraction": 0.85,
            "bagging_fraction": 0.85, "bagging_freq": 5,
            "lambda_l2": 1.0, "verbose": -1, "n_jobs": -1,
        },
        train_set=train_ds, valid_sets=[val_ds], num_boost_round=3000,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    log.info("channel booster trained: %d rounds", booster.best_iteration)

    chan_pred = pd.concat([train, val, test], ignore_index=True)
    chan_pred["chan_pred"] = np.clip(
        booster.predict(chan_pred[feats], num_iteration=booster.best_iteration),
        0, None,
    ).astype(np.float32)

    chan_pred_lookup = (
        chan_pred[["Канал", "Период", "chan_pred"]]
        .drop_duplicates(["Канал", "Период"])
    )
    chan_pred_lookup["Канал"] = chan_pred_lookup["Канал"].astype(str)

    log.info("=== loading V9 SKU-level preds for share weights ===")
    v9_val = pd.read_csv(OUT / "preds_v9_val.csv")
    v9_test = pd.read_csv(OUT / "preds_v9_test.csv")
    v9 = pd.concat([v9_val, v9_test], ignore_index=True)
    v9["Период"] = v9["Период"].astype(str)

    chan_lookup = abt[["Партнер", "Артикул", "Канал"]].drop_duplicates(
        subset=["Партнер", "Артикул"]
    )
    chan_lookup["Канал"] = chan_lookup["Канал"].astype(str)
    v9 = v9.merge(chan_lookup, on=["Партнер", "Артикул"], how="left")
    log.info("missing channel: %d / %d", v9["Канал"].isna().sum(), len(v9))

    sums_in_channel = (
        v9.groupby(["Канал", "Период"], observed=True)["prediction"]
          .sum().reset_index().rename(columns={"prediction": "chan_v9_total"})
    )
    sums_in_channel["chan_v9_total"] = sums_in_channel["chan_v9_total"].clip(lower=1e-3)
    v9 = v9.merge(sums_in_channel, on=["Канал", "Период"], how="left")
    v9["share"] = v9["prediction"] / v9["chan_v9_total"]

    v9 = v9.merge(chan_pred_lookup, on=["Канал", "Период"], how="left")
    v9["chan_pred"] = v9["chan_pred"].fillna(v9["chan_v9_total"])
    v9["v10_topdown_pred"] = (v9["share"] * v9["chan_pred"]).astype(np.float32)

    out = v9[KEY + ["target_qty", "v10_topdown_pred"]].rename(
        columns={"v10_topdown_pred": "prediction"}
    )
    val_out = out[out["Период"].isin(val_pers)]
    tst_out = out[out["Период"].isin(test_pers)]
    val_out.to_csv(OUT / "preds_v10_topdown_val.csv", index=False)
    tst_out.to_csv(OUT / "preds_v10_topdown_test.csv", index=False)

    from scripts.score_similarity import score_frame
    sv = score_frame(val_out)
    st = score_frame(tst_out)
    log.info("V10_topdown val   SIMSCORE=%.4f WAPE=%.4f bias%%=%+.2f M-WAPE=%.4f",
             sv["SIMSCORE"], sv["WAPE"], sv["Agg_Bias_pct"], sv["Monthly_WAPE"])
    log.info("V10_topdown test  SIMSCORE=%.4f WAPE=%.4f bias%%=%+.2f M-WAPE=%.4f",
             st["SIMSCORE"], st["WAPE"], st["Agg_Bias_pct"], st["Monthly_WAPE"])

    log.info("Total time: %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
