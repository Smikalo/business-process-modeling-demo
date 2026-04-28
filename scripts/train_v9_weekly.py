"""V9 — train weekly LightGBM, predict, roll up to monthly grain.

End-to-end pipeline:

1.  Re-load raw daily shipments (re-uses the daily loader from
    `src/features_within_month.py`).
2.  Aggregate to weekly long table.
3.  Densify on the active (Партнер, Артикул) grid; fill missing weeks
    with qty=0.
4.  Add weekly lag/rolling time-series features.
5.  Attach static V8 ABT features (Канал, Бренд, Сегмент_ABC, etc.).
6.  Train two-stage LightGBM at weekly target.
7.  Predict weekly demand on val + test windows.
8.  Sum predicted weekly qty by (Партнер, Артикул, calendar-month) to
    produce monthly forecasts.
9.  Save as `output/preds_v9_weekly_{val,test}.csv` with the SAME schema
    as every other monthly base (Период, Партнер, Артикул, target_qty,
    prediction) so it drops straight into the V8 LAD pool.

Anti-leakage:
* Weekly features all use `groupby.shift(k)` where `k >= 1`.
* Train/val/test splits respect the existing monthly cuts (val starts
  Jul 2024, test starts Jul 2025) but expressed at weekly grain.
* Rolling means use `shift(1).rolling(...)`, not `rolling(...).shift(0)`.
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
log = logging.getLogger("train_v9_weekly")

OUT = REPO / "output"
KEY = ["Период", "Партнер", "Артикул"]


CAT_COLS = ["Канал", "Бренд", "Сегмент_ABC", "Тип_соглашения", "Группа_товара"]


def _encode_cats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in CAT_COLS:
        if c in out.columns:
            s = out[c]
            if isinstance(s.dtype, pd.CategoricalDtype):
                if "__missing__" not in s.cat.categories:
                    s = s.cat.add_categories(["__missing__"])
                s = s.fillna("__missing__")
            else:
                s = s.fillna("__missing__").astype("category")
            out[c] = s
    return out


def _split_weekly(weekly: pd.DataFrame) -> tuple:
    """Train: weeks ending on/before 2024-06-30.
       Val:   weeks 2024-07-01 .. 2025-06-30 (≈ 52 weeks).
       Test:  weeks 2025-07-01 .. 2026-02-28 (≈ 35 weeks).
    """
    we = weekly["week_end"]
    train = weekly[we < pd.Timestamp("2024-07-01")].copy()
    val = weekly[(we >= pd.Timestamp("2024-07-01")) &
                 (we < pd.Timestamp("2025-07-01"))].copy()
    test = weekly[(we >= pd.Timestamp("2025-07-01")) &
                  (we <= pd.Timestamp("2026-02-28"))].copy()
    return train, val, test


def _train_tweedie(train: pd.DataFrame, val: pd.DataFrame,
                   feats: list[str]) -> "lgb.Booster":
    """Tweedie regression for zero-inflated weekly qty.

    Tweedie with variance_power 1.5 is the canonical objective for
    compound-Poisson-Gamma data (cell either zero or a small positive
    drawn from a Gamma).  Two-stage degenerates at weekly grain
    (positives ≈ 2 % of cells); Poisson exploded numerically on this
    dataset due to extreme weekly qty outliers.  Tweedie 1.5 gave the
    smallest residual variance in pilot.
    """
    log.info("Training Tweedie regressor on %d weekly rows…", len(train))
    t0 = time.time()
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
    log.info("Tweedie fitted in %.1fs (%d rounds)", time.time() - t0,
             booster.best_iteration)
    return booster


def _predict(booster, df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    return np.clip(
        booster.predict(df[feats], num_iteration=booster.best_iteration),
        0, None,
    ).astype(np.float32)


def _per_channel_calibration(monthly_pred_with_chan: pd.DataFrame
                             ) -> dict[str, float]:
    """Per-channel multiplicative bias correction (validation only).

    Tweedie predictions are under-biased after roll-up (densification +
    l2 shrinks).  We compute the unbiased per-channel correction factor
        scale_c = sum_actual_c / sum_pred_c
    on the validation window only.  Applied to BOTH val and test before
    they enter the LAD pool, so the base arrives de-biased and LAD
    weights stay stable.  Capped at [0.5, 4.0] to prevent runaway.
    """
    by = monthly_pred_with_chan.groupby("Канал", observed=True).agg(
        actual=("target_qty", "sum"),
        predicted=("prediction", "sum"),
    )
    by["scale"] = (by["actual"] / by["predicted"].clip(lower=1e-3)).clip(0.5, 4.0)
    log.info("per-channel calibration scales:\n%s", by.to_string())
    return by["scale"].to_dict()


def _apply_calibration(monthly_pred_with_chan: pd.DataFrame,
                       scales: dict[str, float]) -> pd.DataFrame:
    df = monthly_pred_with_chan.copy()
    df["scale"] = (
        df["Канал"].astype(str).map({str(k): v for k, v in scales.items()})
                   .astype(float).fillna(1.0)
    )
    df["prediction"] = df["prediction"] * df["scale"]
    return df[KEY + ["target_qty", "prediction"]]


def _rollup_to_monthly(weekly_pred: pd.DataFrame) -> pd.DataFrame:
    """Sum weekly predictions by (Партнер, Артикул, calendar-month-end).

    A week is assigned to the month containing its mid-week (Wednesday)
    so weeks straddling month boundaries don't double-count.
    """
    w = weekly_pred.copy()
    midweek = w["week_end"] - pd.Timedelta(days=3)
    w["per"] = midweek.dt.to_period("M")
    monthly = (
        w.groupby(["per", "Партнер", "Артикул"], observed=True)["pred"]
         .sum()
         .reset_index()
         .rename(columns={"per": "Период", "pred": "prediction"})
    )
    monthly["Период"] = monthly["Период"].astype(str)
    return monthly


def main() -> int:
    t_all = time.time()

    log.info("Loading raw daily shipments…")
    daily = load_shipments_daily()
    log.info("daily rows: %d  (%s … %s)",
             len(daily), daily["Дата"].min(), daily["Дата"].max())

    log.info("Aggregating to weekly long table…")
    weekly_long = build_weekly_long(daily)
    log.info("weekly long rows: %d", len(weekly_long))

    log.info("Densifying weekly grid on active pairs (training window)…")
    weekly = expand_to_dense(
        weekly_long,
        train_start=pd.Timestamp("2020-01-01"),
        train_end=pd.Timestamp("2024-06-30"),
    )
    log.info("dense weekly rows: %d", len(weekly))

    log.info("Adding weekly time-series features…")
    weekly = add_weekly_features(weekly)

    log.info("Attaching static V8-ABT features…")
    v8_abt = pd.read_parquet(OUT / "abt_v7_cached.parquet")
    weekly = attach_static_features(weekly, v8_abt)
    weekly = _encode_cats(weekly)
    log.info("weekly ABT shape: %s", weekly.shape)

    feats = [
        c for c in weekly.columns
        if c not in ("Партнер", "Артикул", "week_end", "qty",
                     "n_days", "n_ship")
    ]
    log.info("feature count: %d", len(feats))

    train, val, test = _split_weekly(weekly)
    log.info("split: train=%d val=%d test=%d weekly rows",
             len(train), len(val), len(test))

    booster = _train_tweedie(train, val, feats)

    p_val = _predict(booster, val, feats)
    p_test = _predict(booster, test, feats)
    val["pred"] = p_val
    test["pred"] = p_test

    log.info("Rolling up weekly preds to monthly…")
    val_m = _rollup_to_monthly(val[["Партнер", "Артикул", "week_end", "pred"]])
    tst_m = _rollup_to_monthly(test[["Партнер", "Артикул", "week_end", "pred"]])

    target = pd.read_parquet(OUT / "abt_v7_cached.parquet")[
        KEY + ["target_qty", "Канал"]
    ]
    target["Период"] = target["Период"].astype(str)
    val_raw = target.merge(val_m, on=KEY, how="inner")
    tst_raw = target.merge(tst_m, on=KEY, how="inner")

    val_raw = val_raw[KEY + ["target_qty", "prediction", "Канал"]]
    tst_raw = tst_raw[KEY + ["target_qty", "prediction", "Канал"]]

    from scripts.score_similarity import score_frame
    sv_raw = score_frame(val_raw[KEY + ["target_qty", "prediction"]])
    st_raw = score_frame(tst_raw[KEY + ["target_qty", "prediction"]])
    log.info("V9_weekly  RAW  val   SIMSCORE=%.4f WAPE=%.4f bias%%=%+.2f M-WAPE=%.4f",
             sv_raw["SIMSCORE"], sv_raw["WAPE"], sv_raw["Agg_Bias_pct"],
             sv_raw["Monthly_WAPE"])
    log.info("V9_weekly  RAW  test  SIMSCORE=%.4f WAPE=%.4f bias%%=%+.2f M-WAPE=%.4f",
             st_raw["SIMSCORE"], st_raw["WAPE"], st_raw["Agg_Bias_pct"],
             st_raw["Monthly_WAPE"])

    log.info("Computing per-channel calibration on validation…")
    scales = _per_channel_calibration(val_raw)
    val_out = _apply_calibration(val_raw, scales)
    tst_out = _apply_calibration(tst_raw, scales)
    val_out.to_csv(OUT / "preds_v9_weekly_val.csv", index=False)
    tst_out.to_csv(OUT / "preds_v9_weekly_test.csv", index=False)

    sv = score_frame(val_out)
    st = score_frame(tst_out)
    log.info("V9_weekly  CAL  val   SIMSCORE=%.4f WAPE=%.4f bias%%=%+.2f M-WAPE=%.4f",
             sv["SIMSCORE"], sv["WAPE"], sv["Agg_Bias_pct"], sv["Monthly_WAPE"])
    log.info("V9_weekly  CAL  test  SIMSCORE=%.4f WAPE=%.4f bias%%=%+.2f M-WAPE=%.4f",
             st["SIMSCORE"], st["WAPE"], st["Agg_Bias_pct"], st["Monthly_WAPE"])

    fi = pd.DataFrame({
        "feature": booster.feature_name(),
        "gain_total": booster.feature_importance(importance_type="gain"),
    }).sort_values("gain_total", ascending=False)
    fi.to_csv(OUT / "feature_importance_v9_weekly.csv", index=False)

    log.info("Total time: %.1fs", time.time() - t_all)
    log.info("wrote preds_v9_weekly_{val,test}.csv "
             "(%d val + %d test monthly rows)",
             len(val_out), len(tst_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
