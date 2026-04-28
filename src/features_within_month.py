"""V8 — within-month / weekly features extracted from daily shipment data.

The raw shipments file (``data/Отгрузки 2020-2026.txt``) is at DAILY grain
but every ABT generation V1–V7.8 collapses to monthly totals at ingestion
time, throwing away the within-month timing signal entirely.  This module
re-extracts that signal so V8 has access to genuinely new information.

All features are computed per ``(Партнер, Артикул, Период)`` from raw
shipments and then **lagged by 1 month** before joining onto the ABT — so
month-t features describe month t-1's shipping pattern.  This avoids
leakage and matches the operational reality that the analyst observes
last month's daily activity before forecasting next month.

Features extracted (all suffixed ``_lag1`` after merge):

* ``wm_first_week_share``  — share of qty in calendar days 1-7
* ``wm_mid_week_share``    — share of qty in days 8-21
* ``wm_last_week_share``   — share of qty in days 22-end
* ``wm_weekly_cv``         — coefficient of variation across 4 ISO-weeks
* ``wm_peak_week``         — 1/2/3/4 = which week had highest qty
* ``wm_n_shipping_days``   — distinct days with shipments (sparsity proxy)
* ``wm_weekend_share``     — share of qty on Sat/Sun (B2B vs DTC mix proxy)
* ``wm_avg_qty_per_day``   — qty / n_shipping_days
* ``wm_max_day_share``     — max single-day share (concentration)
* ``wm_n_shipments``       — count of shipment-events

Also a *brand × channel* version aggregated up so sparse (Partner, SKU)
rows can borrow strength from peers (suffixed ``_brand_channel_lag1``).

Anti-leakage guards:
* Daily file is parsed once; period assigned via ``Дата.dt.to_period('M')``.
* All output columns are explicitly shifted by 1 month.
* Rows where lag-1 is missing get filled with the global median of that
  feature (computed on training-only data inside the model fit).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SHIPMENT_PATH
from src.ingestion import clean_numeric

log = logging.getLogger(__name__)


def load_shipments_daily(path: Path = SHIPMENT_PATH) -> pd.DataFrame:
    """Re-load shipments preserving the day-level Дата column."""
    df = pd.read_csv(
        path, sep="\t", skiprows=7, encoding="utf-8",
        names=["Партнер", "Артикул", "Дата", "Количество", "Выручка"],
        dtype={"Артикул": str, "Партнер": str, "Количество": str, "Выручка": str},
        on_bad_lines="warn",
    )
    df = df.dropna(subset=["Артикул"])
    df["Артикул"] = df["Артикул"].str.strip()
    df["Партнер"] = df["Партнер"].str.strip()
    df["Количество"] = clean_numeric(df["Количество"])
    df["Выручка"] = clean_numeric(df["Выручка"])
    df["Дата"] = pd.to_datetime(df["Дата"], format="%d.%m.%Y", errors="coerce")
    df = df.dropna(subset=["Дата"])
    df["Период"] = df["Дата"].dt.to_period("M")
    df["dom"] = df["Дата"].dt.day
    df["dow"] = df["Дата"].dt.dayofweek  # 0 = Mon, 6 = Sun
    df["iso_week_in_month"] = ((df["dom"] - 1) // 7).clip(upper=3) + 1
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["Количество"] = df["Количество"].clip(lower=0)
    return df[["Партнер", "Артикул", "Период", "Дата", "dom", "dow",
               "iso_week_in_month", "is_weekend", "Количество"]]


def _within_month_per_pair(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per (Партнер, Артикул, Период) → 10 within-month features."""
    g = daily.groupby(["Партнер", "Артикул", "Период"], observed=True)

    total = g["Количество"].sum().rename("__tot")
    n_ship = g.size().rename("wm_n_shipments")
    n_days = g["Дата"].nunique().rename("wm_n_shipping_days")
    weekend_qty = g.apply(
        lambda d: float(d.loc[d["is_weekend"] == 1, "Количество"].sum()),
        include_groups=False,
    ).rename("__wk_qty")
    max_day_qty = g.apply(
        lambda d: float(d.groupby("Дата")["Количество"].sum().max()),
        include_groups=False,
    ).rename("__max_day")

    # Week 1 (dom 1-7), 2 (8-14), 3 (15-21), 4 (22-end)
    daily = daily.copy()
    daily["__week"] = ((daily["dom"] - 1) // 7).clip(upper=3) + 1
    week_qty = (
        daily.pivot_table(
            index=["Партнер", "Артикул", "Период"],
            columns="__week", values="Количество", aggfunc="sum", fill_value=0,
        )
        .rename(columns={1: "wq1", 2: "wq2", 3: "wq3", 4: "wq4"})
    )
    # Ensure all four week columns exist
    for c in ("wq1", "wq2", "wq3", "wq4"):
        if c not in week_qty.columns:
            week_qty[c] = 0.0
    week_qty = week_qty[["wq1", "wq2", "wq3", "wq4"]]

    out = (
        pd.concat([total, n_ship, n_days, weekend_qty, max_day_qty], axis=1)
          .join(week_qty, how="left")
          .reset_index()
    )

    tot_safe = out["__tot"].clip(lower=1e-6)
    out["wm_first_week_share"]  = out["wq1"] / tot_safe
    out["wm_mid_week_share"]    = (out["wq2"] + out["wq3"]) / tot_safe
    out["wm_last_week_share"]   = out["wq4"] / tot_safe

    weeks = out[["wq1", "wq2", "wq3", "wq4"]].to_numpy()
    weeks_mean = weeks.mean(axis=1)
    weeks_std = weeks.std(axis=1)
    out["wm_weekly_cv"] = np.where(
        weeks_mean > 0, weeks_std / np.maximum(weeks_mean, 1e-6), 0.0
    )
    out["wm_peak_week"] = weeks.argmax(axis=1) + 1

    out["wm_weekend_share"] = out["__wk_qty"] / tot_safe
    out["wm_max_day_share"] = out["__max_day"] / tot_safe
    out["wm_avg_qty_per_day"] = out["__tot"] / out["wm_n_shipping_days"].clip(lower=1)

    feat_cols = [
        "wm_first_week_share", "wm_mid_week_share", "wm_last_week_share",
        "wm_weekly_cv", "wm_peak_week", "wm_n_shipments",
        "wm_n_shipping_days", "wm_weekend_share",
        "wm_avg_qty_per_day", "wm_max_day_share",
    ]
    return out[["Партнер", "Артикул", "Период"] + feat_cols]


def _brand_channel_aggregates(daily: pd.DataFrame, abt: pd.DataFrame) -> pd.DataFrame:
    """Brand × Канал aggregate: borrows strength for sparse pairs."""
    meta = abt[["Партнер", "Артикул", "Бренд", "Канал"]].drop_duplicates()
    d = daily.merge(meta, on=["Партнер", "Артикул"], how="inner")
    g = d.groupby(["Бренд", "Канал", "Период"], observed=True)
    week_qty = (
        d.pivot_table(
            index=["Бренд", "Канал", "Период"],
            columns="iso_week_in_month",
            values="Количество", aggfunc="sum", fill_value=0,
        )
        .rename(columns={1: "wq1", 2: "wq2", 3: "wq3", 4: "wq4"})
    )
    for c in ("wq1", "wq2", "wq3", "wq4"):
        if c not in week_qty.columns:
            week_qty[c] = 0.0
    week_qty = week_qty[["wq1", "wq2", "wq3", "wq4"]]
    week_qty["__tot"] = week_qty.sum(axis=1).clip(lower=1e-6)
    week_qty["wm_bc_first_week_share"] = week_qty["wq1"] / week_qty["__tot"]
    week_qty["wm_bc_last_week_share"] = week_qty["wq4"] / week_qty["__tot"]
    weeks = week_qty[["wq1", "wq2", "wq3", "wq4"]].to_numpy()
    week_qty["wm_bc_weekly_cv"] = np.where(
        weeks.mean(axis=1) > 0,
        weeks.std(axis=1) / np.maximum(weeks.mean(axis=1), 1e-6),
        0.0,
    )
    out = week_qty[
        ["wm_bc_first_week_share", "wm_bc_last_week_share", "wm_bc_weekly_cv"]
    ].reset_index()
    return out


def add_within_month_features(abt: pd.DataFrame) -> pd.DataFrame:
    """Attach within-month features to the V7 ABT.  Returns enriched ABT.

    All new columns end with ``_lag1`` (or ``_lag1_3mavg`` for rollups) so
    they describe the *previous* month's shipping pattern when used as
    predictors for the current month's demand.
    """
    log.info("loading raw daily shipments…")
    daily = load_shipments_daily()
    log.info("daily shipments: %d rows, %s – %s",
             len(daily), daily["Период"].min(), daily["Период"].max())

    pair_feats = _within_month_per_pair(daily)
    pair_feats = pair_feats.sort_values(["Партнер", "Артикул", "Период"])

    feat_cols = [c for c in pair_feats.columns
                 if c not in ("Партнер", "Артикул", "Период")]

    # 1-month lag (predictors describe last month, label is this month)
    pair_feats[[f"{c}_lag1" for c in feat_cols]] = (
        pair_feats.groupby(["Партнер", "Артикул"], observed=True)[feat_cols]
                  .shift(1)
    )

    # 3-month rolling mean of the lagged features (smoother trend signal)
    for c in feat_cols:
        pair_feats[f"{c}_lag1_3mavg"] = (
            pair_feats.groupby(["Партнер", "Артикул"], observed=True)[f"{c}_lag1"]
                      .rolling(window=3, min_periods=1).mean()
                      .reset_index(level=[0, 1], drop=True)
        )

    keep = ["Партнер", "Артикул", "Период"] + [
        f"{c}_lag1" for c in feat_cols
    ] + [f"{c}_lag1_3mavg" for c in feat_cols]
    pair_feats = pair_feats[keep]

    abt_p = abt.copy()
    abt_p["Период"] = abt_p["Период"].astype("period[M]")
    pair_feats["Период"] = pair_feats["Период"].astype("period[M]")
    out = abt_p.merge(
        pair_feats, on=["Партнер", "Артикул", "Период"], how="left",
    )

    # Brand × Канал backstop for sparse pairs
    bc_feats = _brand_channel_aggregates(daily, abt)
    bc_feats = bc_feats.sort_values(["Бренд", "Канал", "Период"])
    bc_cols = [c for c in bc_feats.columns
               if c.startswith("wm_bc_")]
    bc_feats[[f"{c}_lag1" for c in bc_cols]] = (
        bc_feats.groupby(["Бренд", "Канал"], observed=True)[bc_cols]
                .shift(1)
    )
    bc_feats = bc_feats[["Бренд", "Канал", "Период"] +
                        [f"{c}_lag1" for c in bc_cols]]
    bc_feats["Период"] = bc_feats["Период"].astype("period[M]")
    out = out.merge(bc_feats, on=["Бренд", "Канал", "Период"], how="left")

    n_new = sum(c.startswith("wm_") for c in out.columns)
    log.info("added %d within-month features (lagged); ABT now %d cols",
             n_new, out.shape[1])
    return out
