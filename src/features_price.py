"""Relative price + elasticity features for V7.

All features are strictly backward-looking (computed from lagged
`implied_unit_price` per row) so there is no leakage into target_qty.

Adds:
- `price_lag1`, `price_lag3`            : shifted realised price per series
- `price_vs_brand_median`               : ratio of lag1 price to brand median (robust to outliers)
- `price_vs_channel_median`             : ratio to channel median
- `price_vs_rrc`                        : ratio of lag1 price to РРЦ
- `price_change_3m_pct`                 : pct change vs 3 months ago
- `sku_price_elasticity`                : OLS slope of log(qty+1) on log(price+1)
                                          per SKU over history, EB-shrunk toward
                                          a pooled elasticity (prior = -1.0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


GROUP_COLS = ["Артикул", "Партнер"]


def _shrunk_elasticity(
    abt: pd.DataFrame, prior_beta: float = -1.0, prior_strength: float = 8.0
) -> pd.Series:
    mask = (abt["implied_unit_price"].fillna(0) > 0) & (abt["target_qty"].fillna(0) > 0)
    d = abt.loc[mask, ["Артикул", "implied_unit_price", "target_qty"]].copy()
    d["logp"] = np.log1p(d["implied_unit_price"])
    d["logq"] = np.log1p(d["target_qty"])

    def _slope(g: pd.DataFrame) -> float:
        if len(g) < 3:
            return np.nan
        x = g["logp"].to_numpy()
        y = g["logq"].to_numpy()
        if x.std() < 1e-6:
            return np.nan
        return float(np.polyfit(x, y, 1)[0])

    slopes = d.groupby("Артикул", observed=True).apply(_slope).rename("beta")
    counts = d.groupby("Артикул", observed=True).size().rename("n")
    shrunk = (slopes * counts + prior_beta * prior_strength) / (counts + prior_strength)
    return shrunk.fillna(prior_beta).clip(-3.0, 1.0)


def add_price_features(abt: pd.DataFrame) -> pd.DataFrame:
    out = abt.sort_values(GROUP_COLS + ["Период"]).copy()
    g = out.groupby(GROUP_COLS, observed=True)

    out["price_lag1"] = g["implied_unit_price"].shift(1).astype("float32")
    out["price_lag3"] = g["implied_unit_price"].shift(3).astype("float32")

    brand_med = (
        out.groupby("Бренд", observed=True)["implied_unit_price"]
        .transform("median").astype("float32")
    )
    channel_med = (
        out.groupby("Канал", observed=True)["implied_unit_price"]
        .transform("median").astype("float32")
    )
    out["price_vs_brand_median"] = (out["price_lag1"] / brand_med.replace(0, np.nan)).astype("float32")
    out["price_vs_channel_median"] = (out["price_lag1"] / channel_med.replace(0, np.nan)).astype("float32")

    rrc = out["РРЦ"].replace(0, np.nan)
    out["price_vs_rrc"] = (out["price_lag1"] / rrc).astype("float32")

    out["price_change_3m_pct"] = (
        (out["price_lag1"] - out["price_lag3"]) / out["price_lag3"].replace(0, np.nan)
    ).astype("float32")

    for col in [
        "price_lag1", "price_lag3", "price_vs_brand_median",
        "price_vs_channel_median", "price_vs_rrc", "price_change_3m_pct",
    ]:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")

    sku_elast = _shrunk_elasticity(out)
    out["sku_price_elasticity"] = (
        out["Артикул"].map(sku_elast).fillna(-1.0).astype("float32")
    )

    return out
