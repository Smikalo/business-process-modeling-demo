"""Build additional orthogonal signal streams for the V7.5 stacker pool.

Three new base predictors materialised to CSVs on val + test:

* ``ewma6``       — per-(partner, SKU) EWMA with half-life 6 months.
* ``ewma12``      — per-(partner, SKU) EWMA with half-life 12 months.
* ``median12``    — rolling-median over the last 12 months (robust, beats
                    mean on heavy-tailed SKUs).
* ``sku_channel_yoy`` — YoY (y[m-12]) lifted by channel × month trend ratio
                    (trend = sum_channel_m / sum_channel_m-12 across all SKUs).

Each row has keys (Период, Партнер, Артикул), target_qty (copied from ABT)
and prediction.  The CSVs have the same schema as the other preds_*.csv
files in output/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"

KEY = ["Период", "Партнер", "Артикул"]

VAL = [pd.Period(f"2024-{m:02d}", "M") for m in range(7, 13)] + \
      [pd.Period(f"2025-{m:02d}", "M") for m in range(1, 7)]
TEST = [pd.Period(f"2025-{m:02d}", "M") for m in range(7, 13)] + \
       [pd.Period(f"2026-{m:02d}", "M") for m in range(1, 3)]
ALL = VAL + TEST


def _load_abt() -> pd.DataFrame:
    abt = pd.read_parquet(OUT / "abt_v7_cached.parquet")
    abt["Период"] = abt["Период"].astype("period[M]")
    return abt[KEY + ["target_qty", "Канал"]].copy()


def ewma(hist: pd.DataFrame, target_periods: list[pd.Period],
         half_life: float) -> pd.DataFrame:
    """Per-pair EWMA using only past months."""
    alpha = 1.0 - np.exp(-np.log(2.0) / half_life)
    h = hist.sort_values(["Партнер", "Артикул", "Период"]).copy()
    g = h.groupby(["Партнер", "Артикул"], sort=False, group_keys=False)
    # Use ewm on the shifted series (strictly past)
    ema = g["target_qty"].apply(lambda s: s.shift(1).ewm(alpha=alpha, adjust=False).mean())
    h["prediction"] = ema.reset_index(drop=True).fillna(0.0).clip(lower=0).values
    mask = h["Период"].isin(target_periods)
    return h.loc[mask, KEY + ["target_qty", "prediction"]]


def rolling_median(hist: pd.DataFrame, target_periods: list[pd.Period],
                   window: int) -> pd.DataFrame:
    h = hist.sort_values(["Партнер", "Артикул", "Период"]).copy()
    g = h.groupby(["Партнер", "Артикул"], sort=False, group_keys=False)
    med = g["target_qty"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).median()
    )
    h["prediction"] = med.reset_index(drop=True).fillna(0.0).clip(lower=0).values
    mask = h["Период"].isin(target_periods)
    return h.loc[mask, KEY + ["target_qty", "prediction"]]


def sku_channel_yoy(hist: pd.DataFrame, target_periods: list[pd.Period]) -> pd.DataFrame:
    """ŷ[sku,p] = y[sku,p-12] * channel_m_trend[p].

    channel_m_trend[p] = Σ_{pairs∈channel} y[p-12] lifted by how much this
    channel-month grew in the *prior* period (p-12 over p-24) to avoid
    leakage.
    """
    h = hist.copy()
    # lag12
    h["lag12"] = h.groupby(["Партнер", "Артикул"], sort=False)["target_qty"].shift(12)
    # channel-month aggregates for trend
    h["_m"] = h["Период"].apply(lambda p: p.month)
    agg = h.groupby(["Канал", "Период"], observed=True).agg(
        q=("target_qty", "sum")
    ).reset_index()
    agg["q_yoy"] = agg.groupby("Канал")["q"].shift(12)
    agg["trend_prior"] = agg.groupby("Канал")["q"].shift(12) / \
                        agg.groupby("Канал")["q"].shift(24).replace(0, np.nan)
    agg["trend_prior"] = agg["trend_prior"].clip(0.5, 2.0).fillna(1.0)
    trend = agg[["Канал", "Период", "trend_prior"]]
    h = h.merge(trend, on=["Канал", "Период"], how="left")
    h["prediction"] = (h["lag12"].fillna(0.0) * h["trend_prior"].fillna(1.0)).clip(lower=0)
    mask = h["Период"].isin(target_periods)
    return h.loc[mask, KEY + ["target_qty", "prediction"]]


def _split_and_save(df: pd.DataFrame, tag: str) -> None:
    df = df.copy()
    df["Период"] = df["Период"].astype(str)
    vm = df["Период"].isin([str(p) for p in VAL])
    tm = df["Период"].isin([str(p) for p in TEST])
    dv = df[vm]
    dt = df[tm]
    dv.to_csv(OUT / f"preds_{tag}_val.csv", index=False)
    dt.to_csv(OUT / f"preds_{tag}_test.csv", index=False)
    print(f"  {tag:20s}  val rows={len(dv)}  test rows={len(dt)}")


def main() -> int:
    abt = _load_abt()

    print("EWMA half-life=6")
    _split_and_save(ewma(abt, ALL, half_life=6.0), "ewma6")

    print("EWMA half-life=12")
    _split_and_save(ewma(abt, ALL, half_life=12.0), "ewma12")

    print("Rolling median window=12")
    _split_and_save(rolling_median(abt, ALL, 12), "median12")

    print("SKU-channel YoY with prior-year trend")
    _split_and_save(sku_channel_yoy(abt, ALL), "yoyTrend")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
