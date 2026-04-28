"""V9 — sales-as-leading-indicator features.

The V7/V8 ABT contains `Количество_sales` (current-month sales qty) and
`lag_1_Выручка_sales` (revenue one month back) but no proper sales-quantity
lag features, no multi-month sales rolling, no sell-through ratio at
multiple horizons, and no sales-momentum signals.  Sales is the
*downstream* signal in the supply chain (store→consumer) and it
LEADS shipments (supplier→distributor) by 1-3 weeks because retailers
replenish based on sell-through, not based on prior shipments.

This is a fundamentally different information class from anything used
in V1-V8: lagged sales is the *demand pull*, while shipment lags are
the *supply push*.  When the two diverge, divergence is the signal.

Features added (all monthly grain, all lagged ≥ 1 month):

* sales_qty_lag_{1,2,3,6}        — pure sales-volume lags
* sales_qty_rmean_{3,6,12}_lag1  — rolling sales averages
* sales_qty_growth_lag1          — log(sales_t-1 / sales_t-2) momentum
* sales_yoy_ratio_lag1           — sales_t-1 / sales_t-13 (yoy)
* sell_through_ratio_lag1        — sales_t-1 / shipments_t-1
* sell_through_ratio_lag2_lag1   — same, from t-2 (consumes 2-month gap)
* sales_lead_signal_lag1         — sign-aligned sales growth − shipment growth
* sales_share_of_brand_lag1      — partner share of brand-level monthly sales
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.ingestion import load_sales, load_shipments

log = logging.getLogger(__name__)


def build_sales_features(abt: pd.DataFrame) -> pd.DataFrame:
    """Attach 12 sales-leading-indicator features to the V8 ABT."""
    sales = load_sales()[["Партнер", "Артикул", "Период",
                          "Количество", "Выручка"]]
    sales = sales.rename(columns={"Количество": "sales_qty",
                                  "Выручка": "sales_rev"})
    sales["Период"] = sales["Период"].astype("period[M]")
    sales = (
        sales.groupby(["Партнер", "Артикул", "Период"], observed=True)
             .agg({"sales_qty": "sum", "sales_rev": "sum"})
             .reset_index()
    )

    ship = load_shipments()[["Партнер", "Артикул", "Период", "Количество"]]
    ship = ship.rename(columns={"Количество": "ship_qty"})
    ship["Период"] = ship["Период"].astype("period[M]")
    ship = (
        ship.groupby(["Партнер", "Артикул", "Период"], observed=True)
            .agg({"ship_qty": "sum"})
            .reset_index()
    )

    keys = abt[["Партнер", "Артикул"]].drop_duplicates()
    months = pd.period_range(
        abt["Период"].astype("period[M]").min(),
        abt["Период"].astype("period[M]").max(),
        freq="M",
    )
    grid = (
        keys.assign(__k=1)
            .merge(pd.DataFrame({"Период": months, "__k": 1}), on="__k")
            .drop(columns="__k")
    )
    grid["Период"] = grid["Период"].astype("period[M]")

    pair = (
        grid.merge(sales, on=["Партнер", "Артикул", "Период"], how="left")
            .merge(ship, on=["Партнер", "Артикул", "Период"], how="left")
    )
    pair["sales_qty"] = pair["sales_qty"].fillna(0).astype(np.float32)
    pair["sales_rev"] = pair["sales_rev"].fillna(0).astype(np.float32)
    pair["ship_qty"] = pair["ship_qty"].fillna(0).astype(np.float32)
    pair = pair.sort_values(["Партнер", "Артикул", "Период"])

    g = pair.groupby(["Партнер", "Артикул"], observed=True)["sales_qty"]
    for k in (1, 2, 3, 6, 12, 13):
        pair[f"sales_qty_lag_{k}"] = g.shift(k).astype(np.float32)

    pair["__sl1"] = g.shift(1)
    g_sl = pair.groupby(["Партнер", "Артикул"], observed=True)["__sl1"]
    for w in (3, 6, 12):
        pair[f"sales_qty_rmean_{w}_lag1"] = (
            g_sl.transform(lambda s: s.rolling(w, min_periods=1).mean())
                .astype(np.float32)
        )

    pair["sales_qty_growth_lag1"] = (
        np.log1p(pair["sales_qty_lag_1"].clip(lower=0)) -
        np.log1p(pair["sales_qty_lag_2"].clip(lower=0))
    ).astype(np.float32)
    pair["sales_yoy_ratio_lag1"] = (
        (pair["sales_qty_lag_1"] + 1.0) / (pair["sales_qty_lag_13"] + 1.0)
    ).astype(np.float32)

    g_sh = pair.groupby(["Партнер", "Артикул"], observed=True)["ship_qty"]
    pair["sell_through_ratio_lag1"] = (
        (pair["sales_qty_lag_1"] + 1.0) /
        (g_sh.shift(1).fillna(0) + 1.0)
    ).astype(np.float32)
    pair["sell_through_ratio_lag2"] = (
        (pair["sales_qty_lag_2"] + 1.0) /
        (g_sh.shift(2).fillna(0) + 1.0)
    ).astype(np.float32)

    ship_growth_lag1 = (
        np.log1p(g_sh.shift(1).clip(lower=0)) -
        np.log1p(g_sh.shift(2).clip(lower=0))
    ).astype(np.float32)
    pair["sales_lead_signal_lag1"] = (
        pair["sales_qty_growth_lag1"] - ship_growth_lag1
    ).astype(np.float32)

    if "Бренд" in abt.columns:
        meta = abt[["Партнер", "Артикул", "Бренд"]].drop_duplicates()
        pair = pair.merge(meta, on=["Партнер", "Артикул"], how="left")
        brand_sales = (
            pair.groupby(["Бренд", "Период"], observed=True)["sales_qty"]
                .sum()
                .reset_index()
                .rename(columns={"sales_qty": "brand_sales_total"})
        )
        pair = pair.merge(brand_sales, on=["Бренд", "Период"], how="left")
        pair["sales_share_of_brand_lag1"] = (
            (pair.groupby(["Партнер", "Артикул"], observed=True)["sales_qty"]
                  .shift(1) + 1.0) /
            (pair.groupby(["Бренд"], observed=True)["brand_sales_total"]
                  .shift(1) + 1.0)
        ).astype(np.float32)
        pair = pair.drop(columns=["brand_sales_total", "Бренд"])
    else:
        pair["sales_share_of_brand_lag1"] = np.float32(0.0)

    leak_cols = {"sales_qty", "sales_rev", "ship_qty", "__sl1"}
    new_cols = [
        c for c in pair.columns
        if (c.startswith("sales_") or c.startswith("sell_through"))
        and c not in leak_cols
    ]
    keep = ["Партнер", "Артикул", "Период"] + sorted(set(new_cols))
    out = pair[keep].copy()
    out["Период"] = out["Период"].astype("period[M]")
    out = out.drop_duplicates(subset=["Партнер", "Артикул", "Период"])

    abt2 = abt.copy()
    abt2["Период"] = abt2["Период"].astype("period[M]")
    merged = abt2.merge(out, on=["Партнер", "Артикул", "Период"], how="left")

    n_new = sum(c in merged.columns for c in keep[3:])
    log.info("added %d sales-leading features", n_new)
    return merged
