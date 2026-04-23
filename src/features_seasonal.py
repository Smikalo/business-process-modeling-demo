"""V7.2 — Q4 / seasonal-lift features.

The V7.1 champion systematically under-forecasts the Nov–Dec Christmas peak
(per-month bias −0.90 in Nov-2025, −1.13 in Dec-2025; portfolio-level
−13 to −14 % on Dec totals). These features expose the seasonal structure
the model currently has to infer implicitly.

All features are strictly backward-looking and computed from data available
at time t, so there is no leakage into `target_qty`:

- `is_xmas_window`, `month_of_year`, `months_to_xmas` are pure calendar
  features.
- `sku_dec_lift_lag1y` and `brand_channel_dec_lift` are computed from
  Dec rows strictly before ``train_cutoff`` (default: ABT_max_period −
  20 months ≈ V7 train cutoff), so validation / test Decembers never see
  their own target in the aggregation.
- `y_lag12` is the per-row target shifted 12 months — trivially safe.

Adds:
- `is_xmas_window`        : binary, 1 for Nov + Dec
- `month_of_year`         : int 1..12
- `months_to_xmas`        : int 0..11, distance to next December
- `sku_dec_lift_lag1y`    : per-SKU Dec uplift factor (EB-shrunk toward 1.0)
- `brand_channel_dec_lift`: (brand × channel) Dec uplift factor — fallback
                            for sparse / new SKUs
- `y_lag12`               : target from 12 months ago
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GROUP_COLS = ["Артикул", "Партнер"]


def _months_to_dec(month: pd.Series) -> pd.Series:
    """0 if month == December, 1 for November, ..., 11 for January."""
    return ((12 - month.astype("int16")) % 12).astype("int16")


def _dec_lift_by(abt_train: pd.DataFrame, keys: list[str]) -> pd.Series:
    """Compute (mean Dec volume) / (12-month median) per group, on training
    data only.  EB-shrunk toward 1.0 for groups with few Dec observations.

    Returns a Series indexed by the group keys with a single lift factor.
    """
    month = abt_train["Период"].dt.month
    dec_mask = month == 12
    dec_totals = (
        abt_train.loc[dec_mask]
        .groupby(keys, observed=True)["target_qty"]
        .mean()
        .rename("dec_mean")
    )
    med = (
        abt_train.groupby(keys, observed=True)["target_qty"]
        .median()
        .replace(0, np.nan)
        .rename("med")
    )
    dec_counts = (
        abt_train.loc[dec_mask]
        .groupby(keys, observed=True)
        .size()
        .rename("n")
    )
    df = pd.concat([dec_totals, med, dec_counts], axis=1)
    lift = (df["dec_mean"] / df["med"]).replace([np.inf, -np.inf], np.nan)
    n = df["n"].fillna(0)
    shrunk = (lift * n + 1.0 * 2) / (n + 2)
    return shrunk.fillna(1.0).clip(0.1, 10.0).astype("float32")


def add_seasonal_features(
    abt: pd.DataFrame,
    train_cutoff: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Add Q4 / seasonal-lift features.  Safe to call on the V7 ABT.

    Parameters
    ----------
    abt : the V7 analytical base table (with `target_qty`, `Период`, `Бренд`,
          `Канал`, `Артикул`, `Партнер`).
    train_cutoff : date; Dec-lift aggregates are computed ONLY from rows
          strictly earlier than this timestamp.  Defaults to ``max(Период) −
          20 months`` which matches the V7 train split (train ends ~2024-06,
          val 2024-07..2025-06, test 2025-07..2026-02).
    """
    out = abt.sort_values(GROUP_COLS + ["Период"]).copy()
    orig_dtype = out["Период"].dtype
    if isinstance(orig_dtype, pd.PeriodDtype):
        out["Период"] = out["Период"].dt.to_timestamp()
    elif not np.issubdtype(orig_dtype, np.datetime64):
        out["Период"] = pd.PeriodIndex(out["Период"].astype(str), freq="M").to_timestamp()

    if train_cutoff is None:
        train_cutoff = out["Период"].max() - pd.DateOffset(months=20)
    train_cutoff = pd.Timestamp(train_cutoff)

    month = out["Период"].dt.month
    out["month_of_year"] = month.astype("int16")
    out["is_xmas_window"] = month.isin([11, 12]).astype("int8")
    out["months_to_xmas"] = _months_to_dec(month)

    abt_train = out.loc[out["Период"] < train_cutoff]

    sku_lift = _dec_lift_by(abt_train, ["Артикул"])
    out["sku_dec_lift_lag1y"] = (
        out["Артикул"].map(sku_lift).fillna(1.0).astype("float32")
    )

    bc_lift = _dec_lift_by(abt_train, ["Бренд", "Канал"])
    bc_key = list(zip(out["Бренд"].astype(str), out["Канал"].astype(str)))
    bc_map = bc_lift.to_dict()
    out["brand_channel_dec_lift"] = np.array(
        [bc_map.get(k, 1.0) for k in bc_key], dtype="float32"
    )

    g = out.groupby(GROUP_COLS, observed=True)["target_qty"]
    out["y_lag12"] = g.shift(12).astype("float32").fillna(0.0)

    if isinstance(orig_dtype, pd.PeriodDtype):
        out["Период"] = out["Период"].dt.to_period("M")

    return out
