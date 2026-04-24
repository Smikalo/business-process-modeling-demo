"""Materialize classical baseline predictions (seasonal-naive, MA-3, MA-6)
for the val and test windows so they can enter the V7.4 stacker pool.

These baselines are intentionally simple and have different error structure
from the LightGBM models.  They're most useful on sparse / low-density rows
where the LGB models over-smooth or over-forecast.

Training data = every month ≤ previous month for each (Партнер, Артикул) pair.

Output: ``output/preds_naiveS_{val,test}.csv``, ``preds_ma3_{val,test}.csv``,
``preds_ma6_{val,test}.csv``.

Each file has the same schema as our other ``preds_*.csv`` files:
    Период, Партнер, Артикул, target_qty, prediction
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"

KEY = ["Период", "Партнер", "Артикул"]
VAL_START = pd.Period("2024-07", "M")
VAL_END = pd.Period("2025-06", "M")
TEST_END = pd.Period("2026-02", "M")


def _hist(abt: pd.DataFrame) -> pd.DataFrame:
    """Long history frame of actual qty per (pair, period) up to test_end."""
    return abt[["Период", "Партнер", "Артикул", "target_qty"]].copy()


def seasonal_naive(hist: pd.DataFrame, target_periods: list[pd.Period]) -> pd.DataFrame:
    """ŷ[pair, p] = qty[pair, p - 12 months]."""
    hist = hist.copy()
    hist["Период"] = hist["Период"].astype("period[M]")
    rows = []
    for p in target_periods:
        src = p - 12
        lag = hist[hist["Период"] == src][["Партнер", "Артикул", "target_qty"]]
        lag = lag.rename(columns={"target_qty": "prediction"})
        lag["Период"] = p
        rows.append(lag[KEY + ["prediction"]])
    preds = pd.concat(rows, ignore_index=True)
    preds["prediction"] = preds["prediction"].fillna(0.0).clip(lower=0)
    return preds


def moving_avg(hist: pd.DataFrame, target_periods: list[pd.Period],
               window: int) -> pd.DataFrame:
    """ŷ[pair, p] = mean(qty[pair, p-1 .. p-window]) using groupby shifts."""
    h = hist.copy()
    h["Период"] = h["Период"].astype("period[M]")
    h = h.sort_values(["Партнер", "Артикул", "Период"])
    g = h.groupby(["Партнер", "Артикул"], sort=False)["target_qty"]
    ma = (
        g.shift(1).rolling(window, min_periods=1).mean()
         .rename("prediction")
    )
    out = pd.concat([h[["Период", "Партнер", "Артикул"]], ma], axis=1)
    mask = out["Период"].isin(target_periods)
    out = out[mask].copy()
    out["prediction"] = out["prediction"].fillna(0.0).clip(lower=0)
    return out[KEY + ["prediction"]]


def _attach_actual(preds: pd.DataFrame, abt: pd.DataFrame) -> pd.DataFrame:
    a = abt[KEY + ["target_qty"]].copy()
    a["Период"] = a["Период"].astype("period[M]")
    preds = preds.copy()
    preds["Период"] = preds["Период"].astype("period[M]")
    out = preds.merge(a, on=KEY, how="inner")
    out["Период"] = out["Период"].astype(str)
    return out[KEY + ["target_qty", "prediction"]]


def _save(df: pd.DataFrame, tag: str, periods: list[pd.Period], split: str) -> None:
    mask = df["Период"].astype(str).isin([str(p) for p in periods])
    d = df[mask]
    p = OUT / f"preds_{tag}_{split}.csv"
    d.to_csv(p, index=False)
    print(f"  wrote {p}  rows={len(d)}")


def main() -> int:
    abt = pd.read_parquet(OUT / "abt_v7_cached.parquet")
    abt["Период"] = abt["Период"].astype("period[M]")
    print(f"ABT: {len(abt)} rows, periods {abt['Период'].min()} .. {abt['Период'].max()}")

    hist = _hist(abt)

    val_periods = [pd.Period(f"{y}-{m:02d}", "M") for (y, m) in
                   [(2024, m) for m in range(7, 13)] +
                   [(2025, m) for m in range(1, 7)]]
    test_periods = [pd.Period(f"{y}-{m:02d}", "M") for (y, m) in
                    [(2025, m) for m in range(7, 13)] +
                    [(2026, 1), (2026, 2)]]
    all_periods = val_periods + test_periods

    print("\nseasonal-naive (lag-12)")
    sn = _attach_actual(seasonal_naive(hist, all_periods), abt)
    _save(sn, "naiveS", val_periods, "val")
    _save(sn, "naiveS", test_periods, "test")

    print("\nMA-3")
    ma3 = _attach_actual(moving_avg(hist, all_periods, 3), abt)
    _save(ma3, "ma3", val_periods, "val")
    _save(ma3, "ma3", test_periods, "test")

    print("\nMA-6")
    ma6 = _attach_actual(moving_avg(hist, all_periods, 6), abt)
    _save(ma6, "ma6", val_periods, "val")
    _save(ma6, "ma6", test_periods, "test")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
