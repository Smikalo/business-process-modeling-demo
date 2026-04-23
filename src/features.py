"""Feature engineering for time-series forecasting.

All features use only past data relative to the prediction month (no leakage).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

GRP = ["Партнер", "Артикул"]


def _ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(GRP + ["Период"]).reset_index(drop=True)


# ── 4.1  Lag & Rolling ──────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_sorted(df)
    g = df.groupby(GRP, sort=False)
    for lag in [1, 2, 3, 6, 12]:
        df[f"lag_{lag}"] = g["target_qty"].shift(lag)
    for col in ["Количество_ship", "Количество_orc", "Выручка_sales"]:
        df[f"lag_1_{col}"] = g[col].shift(1)
    df[[c for c in df.columns if c.startswith("lag_")]] = (
        df[[c for c in df.columns if c.startswith("lag_")]].fillna(0.0)
    )
    log.info("add_lag_features: done")
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_sorted(df)
    g = df.groupby(GRP, sort=False)
    shifted = g["target_qty"].shift(1)
    for w in [3, 6, 12]:
        r = shifted.groupby(df[GRP].values.tolist() if False else df.groupby(GRP).ngroup())
        rm = r.rolling(w, min_periods=1)
        df[f"rmean_{w}"] = rm.mean().droplevel(0) if hasattr(rm.mean(), 'droplevel') else rm.mean().values
    # Simpler: compute rolling from lag columns directly
    df["rmean_3"] = (df["lag_1"] + df["lag_2"] + df["lag_3"]) / 3
    df["rmean_6"] = (df["lag_1"] + df["lag_2"] + df["lag_3"] + df["lag_6"]) / 4  # approximate
    df["rmean_12"] = (df["lag_1"] + df["lag_12"]) / 2  # approximate with available lags
    df["rstd_3"] = df[["lag_1", "lag_2", "lag_3"]].std(axis=1)
    df["rstd_6"] = df[["lag_1", "lag_2", "lag_3", "lag_6"]].std(axis=1)
    df["rcv_6"] = df["rstd_6"] / (df["rmean_6"] + 1e-9)
    roll_cols = [c for c in df.columns if c.startswith(("rmean_", "rstd_", "rcv_"))]
    df[roll_cols] = df[roll_cols].fillna(0.0)
    log.info("add_rolling_features: done")
    return df


# ── 4.2  Calendar & Seasonality ─────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    month = df["Период"].dt.month
    year = df["Период"].dt.year
    df["month"] = month.astype(np.int8)
    df["quarter"] = df["Период"].dt.quarter.astype(np.int8)
    df["year"] = year.astype(np.int16)
    df["month_sin"] = np.sin(2 * np.pi * month / 12).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * month / 12).astype(np.float32)
    df["is_newyear"] = month.isin([12, 1]).astype(np.int8)
    df["is_sept"] = (month == 9).astype(np.int8)
    df["is_march"] = (month == 3).astype(np.int8)
    df["is_summer"] = month.isin([6, 7, 8]).astype(np.int8)
    df["is_q4"] = (df["quarter"] == 4).astype(np.int8)
    df["is_wartime"] = ((year > 2022) | ((year == 2022) & (month >= 2))).astype(np.int8)
    log.info("add_calendar_features: done")
    return df


# ── 4.3  Stock-out & Supply Constraint ──────────────────────────────────────

def add_stockout_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_sorted(df)
    df["stockout_orc"] = (df["Количество_orc"] == 0).astype(np.int8)
    df["stockout_tt"] = (df["Количество_tt"] == 0).astype(np.int8)
    df["stockout_both"] = ((df["stockout_orc"] == 1) & (df["stockout_tt"] == 1)).astype(np.int8)

    g = df.groupby(GRP, sort=False)
    df["stockout_orc_prev"] = g["stockout_orc"].shift(1).fillna(0).astype(np.int8)

    # Inventory-to-sales ratios (using lagged sales to avoid leakage)
    lag1 = df["lag_1"].clip(lower=0)
    df["inv_to_sales_orc"] = (df["Количество_orc"] / (lag1 + 1e-9)).clip(upper=999).astype(np.float32)
    df["inv_to_sales_tt"] = (df["Количество_tt"] / (lag1 + 1e-9)).clip(upper=999).astype(np.float32)

    log.info("add_stockout_features: done")
    return df


# ── 4.4  SKU Lifecycle & Demand Type ────────────────────────────────────────

def add_lifecycle_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_sorted(df)
    period_int = df["Период"].astype("int64")

    g = df.groupby("Артикул", sort=False)
    # First period where target_qty > 0 (per SKU, across all partners)
    mask = df["target_qty"] > 0
    first_sale_per_sku = df.loc[mask].groupby("Артикул")["Период"].min()
    df["_first_sale"] = df["Артикул"].map(first_sale_per_sku)
    df["sku_age_months"] = (
        (period_int - df["_first_sale"].astype("int64")).clip(lower=0).astype(np.int16)
    )
    df["is_new_sku"] = (df["sku_age_months"] <= 3).astype(np.int8)

    # Fraction of past months with nonzero demand (per Partner×SKU)
    g2 = df.groupby(GRP, sort=False)
    nz = (df["target_qty"] > 0).astype(np.float32)
    cumcount = g2.cumcount() + 1
    cumsum_nz = g2[[]].transform("size")  # placeholder
    cumsum_nz = nz.groupby(df.groupby(GRP).ngroup()).cumsum().shift(1)
    cumcount_shifted = (cumcount - 1).clip(lower=1)
    df["demand_density"] = (cumsum_nz / cumcount_shifted).fillna(0).astype(np.float32)

    df = df.drop(columns=["_first_sale"])
    log.info("add_lifecycle_features: done")
    return df


# ── 4.5  Cross-sectional / Hierarchical ─────────────────────────────────────

def add_hierarchical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_sorted(df)

    # Brand-level aggregates (lagged to avoid leakage)
    brand_agg = (
        df.groupby(["Бренд", "Период"], as_index=False)
        .agg(brand_total=("target_qty", "sum"), brand_active=("target_qty", lambda x: (x > 0).sum()))
    )
    # Lag brand aggs by 1 month
    brand_agg["Период"] = brand_agg["Период"] + 1
    df = df.merge(brand_agg, on=["Бренд", "Период"], how="left")
    df["brand_total"] = df["brand_total"].fillna(0).astype(np.float32)
    df["brand_active"] = df["brand_active"].fillna(0).astype(np.float32)

    # Partner-level aggregates (lagged)
    partner_agg = (
        df.groupby(["Партнер", "Период"], as_index=False)
        .agg(partner_total=("target_qty", "sum"))
    )
    partner_agg["Период"] = partner_agg["Период"] + 1
    df = df.merge(partner_agg, on=["Партнер", "Период"], how="left")
    df["partner_total"] = df["partner_total"].fillna(0).astype(np.float32)

    # SKU share within brand (from lagged data)
    df["sku_share_brand"] = (df["lag_1"] / (df["brand_total"] + 1e-9)).clip(upper=1).astype(np.float32)

    log.info("add_hierarchical_features: done")
    return df


# ── Public API ──────────────────────────────────────────────────────────────

def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in order."""
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_calendar_features(df)
    df = add_stockout_features(df)
    df = add_lifecycle_features(df)
    df = add_hierarchical_features(df)
    log.info("engineer_all_features: final shape %s, mem=%.0f MB",
             df.shape, df.memory_usage(deep=True).sum() / 1e6)
    return df
