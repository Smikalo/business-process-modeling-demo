"""V10 Track C — self-anchored weekly forecaster.

The V9 weekly Tweedie model has to learn *both* the monthly-aggregate
level *and* the within-month timing pattern from scratch.  After
roll-up to monthly, the resulting predictions need a heavy per-channel
calibration step (factors 1.5-4×) to correct systematic under-bias.

Track C replaces this with a SELF-ANCHORED weekly model: the weekly
booster receives V9's MONTHLY prediction as a feature, divided by
(approximate) weeks-in-month, giving it a strong prior for the
weekly-level mean.  The model then only has to learn deviations from
this prior -- a much easier task that should:

* tighten weekly residuals
* eliminate the post-roll-up bias (V9 monthly is already calibrated)
* preserve V9's bias correction "for free"

We also add `v9_monthly_pred` as a static-per-month feature (same value
for all weeks in a given (pair, month)) so the weekly grain can pick
up monthly cyclical signal it would otherwise see only via lags.

Output: preds_v10_self_weekly_{val,test}.csv (monthly grain).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
import sys
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.features_within_month import load_shipments_daily  # noqa: E402
from src.v9_weekly import (  # noqa: E402
    add_weekly_features, attach_static_features,
    build_weekly_long, expand_to_dense,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("train_v10_self_weekly")

OUT = REPO / "output"
KEY = ["Период", "Партнер", "Артикул"]

CAT_COLS = ["Канал", "Бренд", "Сегмент_ABC", "Тип_соглашения", "Группа_товара"]


def _encode_cats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in CAT_COLS:
        if c in out.columns:
            out[c] = out[c].astype("category")
            if "__missing__" not in out[c].cat.categories:
                out[c] = out[c].cat.add_categories("__missing__")
            out[c] = out[c].fillna("__missing__")
    return out


def _split_weekly(weekly: pd.DataFrame):
    train_end = pd.Timestamp("2024-06-30")
    val_end = pd.Timestamp("2025-06-30")
    test_end = pd.Timestamp("2026-01-31")
    train = weekly[weekly["week_end"] <= train_end]
    val = weekly[(weekly["week_end"] > train_end) & (weekly["week_end"] <= val_end)]
    test = weekly[(weekly["week_end"] > val_end) & (weekly["week_end"] <= test_end)]
    return train, val, test


def _attach_v9_anchor(weekly: pd.DataFrame) -> pd.DataFrame:
    """Inject V9 monthly prediction as a per-(pair, month) anchor feature."""
    v9_val = pd.read_csv(OUT / "preds_v9_val.csv")
    v9_test = pd.read_csv(OUT / "preds_v9_test.csv")
    v9 = pd.concat([v9_val, v9_test], ignore_index=True)
    v9["Период"] = v9["Период"].astype(str)
    v9 = v9[KEY + ["prediction"]].rename(columns={"prediction": "v9_monthly_pred"})
    v9["mid_per"] = pd.PeriodIndex(v9["Период"], freq="M")

    weekly = weekly.copy()
    midweek = weekly["week_end"] - pd.Timedelta(days=3)
    weekly["mid_per"] = midweek.dt.to_period("M")
    weekly = weekly.merge(
        v9[["Партнер", "Артикул", "mid_per", "v9_monthly_pred"]],
        on=["Партнер", "Артикул", "mid_per"], how="left",
    )
    weekly["v9_monthly_pred"] = weekly["v9_monthly_pred"].fillna(0).astype(np.float32)
    weekly["v9_weekly_anchor"] = (weekly["v9_monthly_pred"] / 4.33).astype(np.float32)
    weekly = weekly.drop(columns="mid_per")
    return weekly


def main() -> int:
    t_all = time.time()
    log.info("Loading raw daily shipments…")
    daily = load_shipments_daily()
    weekly_long = build_weekly_long(daily)
    weekly = expand_to_dense(
        weekly_long,
        train_start=pd.Timestamp("2020-01-01"),
        train_end=pd.Timestamp("2024-06-30"),
    )
    weekly = add_weekly_features(weekly)
    v8_abt = pd.read_parquet(OUT / "abt_v7_cached.parquet")
    weekly = attach_static_features(weekly, v8_abt)
    weekly = _attach_v9_anchor(weekly)
    weekly = _encode_cats(weekly)
    log.info("weekly ABT shape: %s", weekly.shape)

    feats = [
        c for c in weekly.columns
        if c not in ("Партнер", "Артикул", "week_end", "qty",
                     "n_days", "n_ship", "v9_monthly_pred")
    ]
    log.info("feature count (with v9 anchor): %d", len(feats))

    train, val, test = _split_weekly(weekly)
    log.info("split: train=%d val=%d test=%d weekly rows",
             len(train), len(val), len(test))

    train_ds = lgb.Dataset(train[feats], train["qty"].astype(np.float32),
                           categorical_feature=CAT_COLS)
    val_ds = lgb.Dataset(val[feats], val["qty"].astype(np.float32),
                         categorical_feature=CAT_COLS, reference=train_ds)
    booster = lgb.train(
        params={
            "objective": "tweedie", "tweedie_variance_power": 1.5,
            "metric": ["tweedie", "rmse"],
            "num_leaves": 191, "learning_rate": 0.05,
            "min_child_samples": 60, "feature_fraction": 0.85,
            "bagging_fraction": 0.85, "bagging_freq": 5,
            "lambda_l2": 1.0, "verbose": -1, "n_jobs": -1,
        },
        train_set=train_ds, valid_sets=[val_ds], num_boost_round=2500,
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)],
    )
    log.info("self-anchored weekly trained: %d rounds", booster.best_iteration)

    val["pred"] = np.clip(
        booster.predict(val[feats], num_iteration=booster.best_iteration),
        0, None,
    ).astype(np.float32)
    test["pred"] = np.clip(
        booster.predict(test[feats], num_iteration=booster.best_iteration),
        0, None,
    ).astype(np.float32)

    def rollup(w):
        wc = w.copy()
        midweek = wc["week_end"] - pd.Timedelta(days=3)
        wc["per"] = midweek.dt.to_period("M")
        m = (wc.groupby(["per", "Партнер", "Артикул"], observed=True)["pred"]
             .sum().reset_index().rename(columns={"per": "Период",
                                                  "pred": "prediction"}))
        m["Период"] = m["Период"].astype(str)
        return m

    val_m = rollup(val[["Партнер", "Артикул", "week_end", "pred"]])
    tst_m = rollup(test[["Партнер", "Артикул", "week_end", "pred"]])

    target = pd.read_parquet(OUT / "abt_v7_cached.parquet")[
        KEY + ["target_qty"]
    ]
    target["Период"] = target["Период"].astype(str)
    val_out = target.merge(val_m, on=KEY, how="inner")
    tst_out = target.merge(tst_m, on=KEY, how="inner")
    val_out = val_out[KEY + ["target_qty", "prediction"]]
    tst_out = tst_out[KEY + ["target_qty", "prediction"]]

    val_out.to_csv(OUT / "preds_v10_self_weekly_val.csv", index=False)
    tst_out.to_csv(OUT / "preds_v10_self_weekly_test.csv", index=False)

    from scripts.score_similarity import score_frame
    sv = score_frame(val_out)
    st = score_frame(tst_out)
    log.info("V10_self_weekly val   SIMSCORE=%.4f WAPE=%.4f bias%%=%+.2f M-WAPE=%.4f",
             sv["SIMSCORE"], sv["WAPE"], sv["Agg_Bias_pct"], sv["Monthly_WAPE"])
    log.info("V10_self_weekly test  SIMSCORE=%.4f WAPE=%.4f bias%%=%+.2f M-WAPE=%.4f",
             st["SIMSCORE"], st["WAPE"], st["Agg_Bias_pct"], st["Monthly_WAPE"])
    log.info("Total time: %.1fs", time.time() - t_all)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
