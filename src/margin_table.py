"""Per-SKU realised margin + unit-price table.

Replaces the flat `holding_rate=0.22 / margin_rate=0.28` assumption in the
cost scorecard with empirical per-SKU rates derived from the ABT's own
`Выручка_sales`, `Количество_sales`, `Стоимость_orc`, and `Количество_orc`.

Totals are aggregated over the whole history to avoid per-row mismatch
between bulk ORC inbound batches and retail sales events, then
empirical-Bayes shrunk toward the brand x channel mean using an
aggregate-quantity prior.

Output columns:
- `unit_price_uah`      : average realised revenue / qty sold across history
- `unit_cost_uah`       : average ORC inbound cost / qty received across history
- `margin_rate`         : clipped to [0.05, 0.80]
- `holding_rate_annual` : fixed 0.22 default; overridable by caller
"""

from __future__ import annotations

import pandas as pd


def build_margin_table(
    abt: pd.DataFrame,
    eb_prior: float = 50.0,
    default_holding: float = 0.22,
    margin_floor: float = 0.05,
    margin_ceiling: float = 0.80,
) -> pd.DataFrame:
    sku_sales = (
        abt.groupby("Артикул", observed=True)
        .agg(rev=("Выручка_sales", "sum"), qs=("Количество_sales", "sum"))
    )
    sku_cost = (
        abt.groupby("Артикул", observed=True)
        .agg(cost=("Стоимость_orc", "sum"), qo=("Количество_orc", "sum"))
    )
    sku = sku_sales.join(sku_cost, how="outer").fillna(0.0)

    first_bc = abt.drop_duplicates("Артикул").set_index("Артикул")[["Бренд", "Канал"]]
    sku = sku.join(first_bc, how="left")

    bc_sales = (
        abt.groupby(["Бренд", "Канал"], observed=True)
        .agg(bc_rev=("Выручка_sales", "sum"), bc_qs=("Количество_sales", "sum"))
    )
    bc_cost = (
        abt.groupby(["Бренд", "Канал"], observed=True)
        .agg(bc_cost=("Стоимость_orc", "sum"), bc_qo=("Количество_orc", "sum"))
    )
    bc = bc_sales.join(bc_cost, how="outer").fillna(0.0)
    bc["bc_price"] = bc["bc_rev"] / bc["bc_qs"].clip(lower=1)
    bc["bc_unit_cost"] = bc["bc_cost"] / bc["bc_qo"].clip(lower=1)

    sku = sku.join(bc[["bc_price", "bc_unit_cost"]], on=["Бренд", "Канал"])
    sku = sku.fillna({"bc_price": bc["bc_price"].mean(), "bc_unit_cost": bc["bc_unit_cost"].mean()})

    sku["unit_price_uah"] = (sku["rev"] + sku["bc_price"] * eb_prior) / (
        sku["qs"] + eb_prior
    )
    sku["unit_cost_uah"] = (sku["cost"] + sku["bc_unit_cost"] * eb_prior) / (
        sku["qo"] + eb_prior
    )

    sku["margin_rate"] = (
        (sku["unit_price_uah"] - sku["unit_cost_uah"]) / sku["unit_price_uah"].replace(0, 1.0)
    ).clip(margin_floor, margin_ceiling)
    sku["holding_rate_annual"] = default_holding

    return sku.reset_index()[
        [
            "Артикул",
            "Бренд",
            "Канал",
            "unit_price_uah",
            "unit_cost_uah",
            "margin_rate",
            "holding_rate_annual",
            "qs",
            "qo",
        ]
    ].rename(columns={"qs": "n_sales", "qo": "n_orc"})
