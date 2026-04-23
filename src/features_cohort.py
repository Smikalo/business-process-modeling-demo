"""Cohort / substitution features for V7.

For each (SKU, Partner, Period) we look at its "substitution cohort":
same Brand x Группа_товара x Channel in the same month, excluding the
target SKU itself. Captures cross-SKU substitution signal that a
per-series model otherwise misses.

All cohort aggregates are lag-shifted by one period so there is no
leakage from the current month's target.

Adds:
- `cohort_demand_lag1`       : mean target_qty across the cohort at t-1
- `cohort_stockout_share_lag1`: mean `stockout_orc` across the cohort at t-1
- `cohort_size`              : # active SKUs in the cohort at t-1
- `cannibalisation_pressure` : ratio of cohort fresh releases / cohort size
                               in the last 2 months.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


COHORT_KEYS = ["Бренд", "Группа_товара", "Канал", "Период"]


def add_cohort_features(abt: pd.DataFrame) -> pd.DataFrame:
    out = abt.copy()

    cohort_agg = (
        out.groupby(COHORT_KEYS, observed=True)
        .agg(
            cohort_total_qty=("target_qty", "sum"),
            cohort_n=("target_qty", "size"),
            cohort_stockouts=("stockout_orc", "sum"),
            cohort_new=("is_new_sku", "sum"),
        )
        .reset_index()
    )
    def _to_ts(s):
        if isinstance(s.dtype, pd.PeriodDtype):
            return s.dt.to_timestamp()
        return pd.to_datetime(s)

    cohort_agg["Период"] = _to_ts(cohort_agg["Период"])
    out["_per"] = _to_ts(out["Период"])
    cohort_agg = cohort_agg.sort_values(COHORT_KEYS[:-1] + ["Период"]).copy()
    cohort_agg_lag = cohort_agg.copy()
    cohort_agg_lag["_per_next"] = cohort_agg_lag["Период"] + pd.DateOffset(months=1)
    cohort_agg_lag = cohort_agg_lag.rename(
        columns={
            "cohort_total_qty": "cohort_total_qty_lag1",
            "cohort_n": "cohort_n_lag1",
            "cohort_stockouts": "cohort_stockouts_lag1",
            "cohort_new": "cohort_new_lag1",
        }
    )[
        ["Бренд", "Группа_товара", "Канал", "_per_next",
         "cohort_total_qty_lag1", "cohort_n_lag1",
         "cohort_stockouts_lag1", "cohort_new_lag1"]
    ]

    out = out.merge(
        cohort_agg_lag,
        left_on=["Бренд", "Группа_товара", "Канал", "_per"],
        right_on=["Бренд", "Группа_товара", "Канал", "_per_next"],
        how="left",
    )

    out["cohort_size"] = (out["cohort_n_lag1"].fillna(1) - 1).clip(lower=0).astype("int32")
    cohort_size_safe = out["cohort_size"].replace(0, 1)

    cohort_total_excl = (out["cohort_total_qty_lag1"].fillna(0) - out["lag_1"].fillna(0)).clip(lower=0)
    out["cohort_demand_lag1"] = (cohort_total_excl / cohort_size_safe).astype("float32")

    cohort_stockout_excl = (
        out["cohort_stockouts_lag1"].fillna(0) - out["stockout_orc_prev"].fillna(0)
    ).clip(lower=0)
    out["cohort_stockout_share_lag1"] = (cohort_stockout_excl / cohort_size_safe).astype("float32")

    cohort_new_excl = (out["cohort_new_lag1"].fillna(0) - out["is_new_sku"].fillna(0)).clip(lower=0)
    out["cannibalisation_pressure"] = (cohort_new_excl / cohort_size_safe).astype("float32")

    drop_cols = [
        "_per", "_per_next",
        "cohort_total_qty_lag1", "cohort_n_lag1",
        "cohort_stockouts_lag1", "cohort_new_lag1",
    ]
    return out.drop(columns=[c for c in drop_cols if c in out.columns])
