"""V9 — weekly-resolution forecaster, rolled up to monthly.

Every prior generation V1-V8 predicts at *monthly* grain.  V8 added
within-month *features* (week-shares, weekly CV) but the target is still
a monthly aggregate.  The V7.7 final report identified a weekly-target
forecaster as the highest-EV unexplored direction with estimated 1-3 %
SIMSCORE lift.  V9 ships it.

Architecture:

  raw daily shipments
        │
        ▼  aggregate per (Партнер, Артикул, ISO-week)
  weekly long table  (~ 750 k rows)
        │
        ▼  add weekly lag/rolling features + static V7 features
  weekly ABT
        │
        ▼  two-stage LightGBM (P_active * E[qty | active])
  weekly forecasts
        │
        ▼  sum weekly preds by (Партнер, Артикул, calendar-month-of-the-week-end)
  monthly forecast = preds_v9_weekly_{val,test}.csv
        │
        ▼  enter V8 LAD pool as a new base

Why this is "genuinely new information":
* Every prior model saw monthly totals as target.  A weekly model sees
  ~4× more training rows and is exposed to within-month timing as a
  *target*, not just as features.
* The roll-up from week to month is *not* a sum of independent
  predictions; the weekly model encodes cross-week dependencies (peak
  week, weekly autocorrelation) that the monthly model can't.
* Most importantly, the weekly forecast does NOT sit in the same Bayes-
  posterior as any monthly model -- by construction it brings
  orthogonal residuals to the LAD ensemble.

The output is `preds_v9_weekly_{val,test}.csv` with KEY = monthly grain
so it joins the V8 LAD pool seamlessly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"

log = logging.getLogger("v9_weekly")


def build_weekly_long(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily shipments to (Партнер, Артикул, week_end) grain."""
    d = daily.copy()
    d["week_end"] = d["Дата"] + pd.to_timedelta(
        (6 - d["Дата"].dt.dayofweek) % 7, unit="D",
    )
    weekly = (
        d.groupby(["Партнер", "Артикул", "week_end"], observed=True)
         .agg(qty=("Количество", "sum"),
              n_days=("Дата", "nunique"),
              n_ship=("Количество", "size"))
         .reset_index()
    )
    return weekly


def expand_to_dense(weekly: pd.DataFrame, train_start: pd.Timestamp,
                    train_end: pd.Timestamp) -> pd.DataFrame:
    """Make the weekly grid dense for active pairs.

    Active pair = a (Партнер, Артикул) that has at least one positive
    weekly shipment in the [train_start, train_end] window.  Filling with
    zeros is correct because no shipment that week genuinely means zero
    demand observed -- not missingness.
    """
    active = (
        weekly[(weekly["week_end"] >= train_start) &
               (weekly["week_end"] <= train_end) &
               (weekly["qty"] > 0)]
        [["Партнер", "Артикул"]].drop_duplicates()
    )
    log.info("active (Партнер, Артикул) pairs (weekly): %d", len(active))

    week_grid = pd.date_range(weekly["week_end"].min(),
                              weekly["week_end"].max(),
                              freq="W-SUN", name="week_end")
    grid = (
        active.assign(__k=1)
        .merge(pd.DataFrame({"week_end": week_grid, "__k": 1}), on="__k")
        .drop(columns="__k")
    )
    out = grid.merge(weekly, on=["Партнер", "Артикул", "week_end"], how="left")
    out["qty"] = out["qty"].fillna(0).astype(np.float32)
    out["n_days"] = out["n_days"].fillna(0).astype(np.int16)
    out["n_ship"] = out["n_ship"].fillna(0).astype(np.int16)
    return out


def add_weekly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time-series features at weekly grain. Strict 1-week lag minimum."""
    df = df.sort_values(["Партнер", "Артикул", "week_end"]).reset_index(drop=True)
    g = df.groupby(["Партнер", "Артикул"], observed=True)["qty"]

    for k in (1, 2, 4, 8, 13, 26, 52):
        df[f"qty_lag_{k}w"] = g.shift(k).astype(np.float32)

    df["__qty_lag1"] = g.shift(1)
    g_lag1 = df.groupby(["Партнер", "Артикул"], observed=True)["__qty_lag1"]
    for w in (4, 13, 26):
        df[f"qty_rmean_{w}w"] = (
            g_lag1.transform(lambda s: s.rolling(w, min_periods=1).mean())
                  .astype(np.float32)
        )
        df[f"qty_rstd_{w}w"] = (
            g_lag1.transform(lambda s: s.rolling(w, min_periods=1).std())
                  .fillna(0).astype(np.float32)
        )
        df[f"qty_rmax_{w}w"] = (
            g_lag1.transform(lambda s: s.rolling(w, min_periods=1).max())
                  .fillna(0).astype(np.float32)
        )
    df = df.drop(columns="__qty_lag1")

    def _zero_streak(s: pd.Series) -> pd.Series:
        s_prev = s.shift(1).fillna(1)
        zero_groups = (s_prev > 0).cumsum()
        return s_prev.eq(0).groupby(zero_groups).cumsum().astype(np.int16)
    df["qty_zero_streak"] = (
        df.groupby(["Партнер", "Артикул"], observed=True)["qty"]
          .transform(_zero_streak)
    )

    n_days_g = df.groupby(["Партнер", "Артикул"], observed=True)["n_days"]
    df["n_days_lag_1w"] = n_days_g.shift(1).fillna(0).astype(np.int16)
    df["__n_days_lag1"] = n_days_g.shift(1).fillna(0)
    df["n_days_rmean_4w"] = (
        df.groupby(["Партнер", "Артикул"], observed=True)["__n_days_lag1"]
          .transform(lambda s: s.rolling(4, min_periods=1).mean())
          .astype(np.float32)
    )
    df = df.drop(columns="__n_days_lag1")

    week_no = df["week_end"].dt.isocalendar().week.astype(int)
    df["woy"] = week_no
    df["woy_sin"] = np.sin(2 * np.pi * week_no / 52)
    df["woy_cos"] = np.cos(2 * np.pi * week_no / 52)
    df["wom"] = ((df["week_end"].dt.day - 1) // 7 + 1).astype(np.int8)
    df["dec_week"] = (df["week_end"].dt.month == 12).astype(np.int8)
    df["jan_week"] = (df["week_end"].dt.month == 1).astype(np.int8)
    df["pre_xmas"] = ((df["week_end"].dt.month == 12) &
                      (df["week_end"].dt.day <= 21)).astype(np.int8)
    df["q4_week"] = df["week_end"].dt.month.isin([10, 11, 12]).astype(np.int8)

    return df


def attach_static_features(weekly_abt: pd.DataFrame,
                           v8_abt: pd.DataFrame) -> pd.DataFrame:
    """Bring across SKU/partner/channel attributes from the V8 ABT.

    Use the latest non-null value per (Партнер, Артикул) -- these are
    structural attributes that don't change month-to-month (Канал,
    Бренд, Сегмент_ABC, partner_volume_tier, etc.)."""
    static_cols = [
        "Канал", "Бренд", "Сегмент_ABC", "Тип_соглашения",
        "Группа_товара", "partner_volume_tier", "demand_density",
        "demand_cv", "sku_age_months", "is_new_sku",
    ]
    static_cols = [c for c in static_cols if c in v8_abt.columns]
    static = (
        v8_abt[["Партнер", "Артикул"] + static_cols]
        .drop_duplicates(subset=["Партнер", "Артикул"], keep="last")
    )
    return weekly_abt.merge(static, on=["Партнер", "Артикул"], how="left")
