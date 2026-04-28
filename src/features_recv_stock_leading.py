"""V10 Track A — receipts & stock leading-indicator features.

Three signal classes are still **structurally unmined** by V9, with the
same playbook V9 used for sales:

1. **Central-warehouse receipts** (`Поступление ОРЦ`).  V9 only carries
   current-month `Количество_receipts`.  Receipts at the warehouse
   LEAD shipments by 2-4 weeks (you have to receive before you ship).

2. **Central-warehouse stock** (`Остатки ОРЦ`).  V9 only carries current
   `Количество_orc` and a `stockout_orc` flag.  Stock-depletion-rate
   and days-of-supply are direct supply-bottleneck signals.

3. **Retail-trade stock** (`Остатки ТТ`).  V9 only carries current
   `Количество_tt` and `stockout_tt`.  Lagged TT stock + TT-velocity
   measures the downstream demand pull.

All features are explicitly lagged ≥ 1 month before merging.  Mirrors
the feature-engineering pattern of `src/features_sales_leading.py`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.ingestion import (
    load_receipts_orc, load_rests_orc, load_rests_tt, load_shipments,
)

log = logging.getLogger(__name__)


def _ensure_period(s: pd.Series) -> pd.Series:
    return s.astype("period[M]")


def _build_dense_grid(abt: pd.DataFrame) -> pd.DataFrame:
    """Sku-level dense (Партнер, Артикул, Период) grid covering full history."""
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
    return grid


def build_receipts_stock_features(abt: pd.DataFrame) -> pd.DataFrame:
    """Attach receipts + ORC-stock + TT-stock leading features to V9 ABT."""
    log.info("loading raw receipts / orc / tt sources…")
    recv = load_receipts_orc()
    if "Партнер" not in recv.columns:
        recv["Партнер"] = "ALL"
    recv["Период"] = _ensure_period(recv["Период"])
    recv = (
        recv.groupby(["Артикул", "Период"], observed=True)["Количество"]
            .sum().reset_index().rename(columns={"Количество": "recv_qty"})
    )

    orc = load_rests_orc()
    orc["Период"] = _ensure_period(orc["Период"])
    orc = (
        orc.groupby(["Артикул", "Период"], observed=True)["Количество"]
           .sum().reset_index().rename(columns={"Количество": "stock_orc_qty"})
    )

    tt = load_rests_tt()
    tt["Период"] = _ensure_period(tt["Период"])
    tt = (
        tt.groupby(["Партнер", "Артикул", "Период"], observed=True)["Количество"]
          .sum().reset_index().rename(columns={"Количество": "stock_tt_qty"})
    )

    ship = load_shipments()[["Партнер", "Артикул", "Период", "Количество"]]
    ship["Период"] = _ensure_period(ship["Период"])
    ship = (
        ship.groupby(["Партнер", "Артикул", "Период"], observed=True)["Количество"]
            .sum().reset_index().rename(columns={"Количество": "ship_qty"})
    )

    grid = _build_dense_grid(abt)
    pair = (
        grid.merge(ship, on=["Партнер", "Артикул", "Период"], how="left")
            .merge(tt,   on=["Партнер", "Артикул", "Период"], how="left")
            .merge(recv, on=["Артикул", "Период"], how="left")
            .merge(orc,  on=["Артикул", "Период"], how="left")
    )
    for c in ("ship_qty", "stock_tt_qty", "recv_qty", "stock_orc_qty"):
        pair[c] = pair[c].fillna(0).astype(np.float32)

    pair = pair.sort_values(["Партнер", "Артикул", "Период"])

    # Receipts: SKU-level (no partner; receipts hit central warehouse first)
    g_recv = pair.groupby(["Партнер", "Артикул"], observed=True)["recv_qty"]
    for k in (1, 2, 3, 6):
        pair[f"recv_qty_lag_{k}"] = g_recv.shift(k).astype(np.float32)
    pair["__rl1"] = g_recv.shift(1)
    g_rl1 = pair.groupby(["Партнер", "Артикул"], observed=True)["__rl1"]
    for w in (3, 6):
        pair[f"recv_qty_rmean_{w}_lag1"] = (
            g_rl1.transform(lambda s: s.rolling(w, min_periods=1).mean())
                 .astype(np.float32)
        )
    pair["recv_qty_growth_lag1"] = (
        np.log1p(pair["recv_qty_lag_1"].clip(lower=0)) -
        np.log1p(pair["recv_qty_lag_2"].clip(lower=0))
    ).astype(np.float32)
    g_sh = pair.groupby(["Партнер", "Артикул"], observed=True)["ship_qty"]
    pair["recv_to_ship_ratio_lag1"] = (
        (pair["recv_qty_lag_1"] + 1.0) /
        (g_sh.shift(1).fillna(0) + 1.0)
    ).astype(np.float32)

    # ORC stock: depletion rate + days-of-supply
    g_orc = pair.groupby(["Партнер", "Артикул"], observed=True)["stock_orc_qty"]
    for k in (1, 2, 3):
        pair[f"stock_orc_lag_{k}"] = g_orc.shift(k).astype(np.float32)
    # Depletion = how much stock dropped between t-2 and t-1 (high = strong demand)
    pair["stock_orc_depletion_lag1"] = (
        pair["stock_orc_lag_2"] - pair["stock_orc_lag_1"]
    ).astype(np.float32)
    # Days-of-supply: stock_t-1 / mean shipping demand of t-1 (clip at 365 days)
    monthly_mean_ship = (
        g_sh.transform(lambda s: s.rolling(6, min_periods=1).mean()) + 1.0
    )
    pair["days_of_supply_orc_lag1"] = np.minimum(
        365.0,
        30.0 * pair["stock_orc_lag_1"] / monthly_mean_ship,
    ).astype(np.float32)
    pair["stock_orc_buildup_flag_lag1"] = (
        (pair["stock_orc_lag_1"] >
         pair.groupby(["Партнер", "Артикул"], observed=True)["stock_orc_lag_1"]
             .transform(lambda s: s.rolling(6, min_periods=1).mean()) * 1.5)
    ).astype(np.int8)

    # TT stock: same pattern + TT-velocity
    g_tt = pair.groupby(["Партнер", "Артикул"], observed=True)["stock_tt_qty"]
    for k in (1, 2, 3):
        pair[f"stock_tt_lag_{k}"] = g_tt.shift(k).astype(np.float32)
    pair["stock_tt_velocity_lag1"] = (
        pair["stock_tt_lag_2"] - pair["stock_tt_lag_1"]
    ).astype(np.float32)
    pair["tt_to_orc_ratio_lag1"] = (
        (pair["stock_tt_lag_1"] + 1.0) /
        (pair["stock_orc_lag_1"] + 1.0)
    ).astype(np.float32)

    new_cols = [c for c in pair.columns
                if c.startswith(("recv_", "stock_orc_", "stock_tt_",
                                 "days_of_supply_", "tt_to_orc_"))
                and c not in ("recv_qty", "stock_orc_qty", "stock_tt_qty",
                              "ship_qty", "__rl1")]
    keep = ["Партнер", "Артикул", "Период"] + sorted(set(new_cols))
    out = pair[keep].drop_duplicates(
        subset=["Партнер", "Артикул", "Период"]
    ).copy()
    out["Период"] = out["Период"].astype("period[M]")

    abt2 = abt.copy()
    abt2["Период"] = abt2["Период"].astype("period[M]")
    merged = abt2.merge(out, on=["Партнер", "Артикул", "Период"], how="left")
    log.info("added %d receipts/stock leading features", len(keep) - 3)
    return merged
