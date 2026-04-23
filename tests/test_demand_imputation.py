"""Smoke tests for the censored-demand imputation utility."""

from __future__ import annotations

import pandas as pd
import pytest

from src.demand_imputation import impute_stockout_demand


def _synthetic_abt() -> pd.DataFrame:
    """Two active SKUs in the same brand-channel so the March baseline is
    estimable from non-censored rows (SKU3 provides the reference)."""
    periods = pd.period_range("2024-01", "2024-06", freq="M")
    rows = []
    # Partner A × SKU 1: regularly sells 10/mo, stockout in March (target=0 censored)
    for p in periods:
        rows.append(
            {
                "Период": p, "Партнер": "A", "Артикул": "SKU1",
                "Бренд": "Djeco", "Канал": "ИМ",
                "target_qty": 0 if p.month == 3 else 10,
                "stockout_orc": 1 if p.month == 3 else 0,
                "stockout_both": 1 if p.month == 3 else 0,
                "demand_density": 0.8,
            }
        )
    # Partner A × SKU 3 (reference SKU): sells 8/mo every month, no stockouts
    for p in periods:
        rows.append(
            {
                "Период": p, "Партнер": "A", "Артикул": "SKU3",
                "Бренд": "Djeco", "Канал": "ИМ",
                "target_qty": 8,
                "stockout_orc": 0, "stockout_both": 0,
                "demand_density": 0.9,
            }
        )
    # Partner B × SKU 2: inactive SKU, all zeros, no imputation should happen
    for p in periods:
        rows.append(
            {
                "Период": p, "Партнер": "B", "Артикул": "SKU2",
                "Бренд": "Djeco", "Канал": "ИМ",
                "target_qty": 0,
                "stockout_orc": 1, "stockout_both": 1,
                "demand_density": 0.05,
            }
        )
    df = pd.DataFrame(rows)
    df["Период"] = pd.PeriodIndex(df["Период"], freq="M")
    return df


def test_impute_flags_only_censored_high_density() -> None:
    df = _synthetic_abt()
    out, report = impute_stockout_demand(df)
    # Only the SKU1-March row should be censored
    assert report.n_censored == 1
    assert out.loc[(out["Артикул"] == "SKU1") & (out["Период"].dt.month == 3), "was_censored"].iloc[0] == 1
    # SKU2 is low-density, should NOT be censored
    assert (out.loc[out["Артикул"] == "SKU2", "was_censored"] == 0).all()


def test_imputed_value_positive_and_reasonable() -> None:
    df = _synthetic_abt()
    out, _ = impute_stockout_demand(df)
    imp_row = out[(out["Артикул"] == "SKU1") & (out["Период"].dt.month == 3)].iloc[0]
    # Imputed value should be meaningful (roughly the brand-channel March mean via SKU3)
    # and clearly above the 0.5 floor.
    assert imp_row["target_qty_imputed"] >= 1.0
    assert imp_row["target_qty_imputed"] < 30
    # Original target_qty untouched
    assert imp_row["target_qty"] == 0


def test_non_censored_rows_untouched() -> None:
    df = _synthetic_abt()
    out, _ = impute_stockout_demand(df)
    mask = out["was_censored"] == 0
    # For non-censored rows, imputed should equal original
    assert (out.loc[mask, "target_qty_imputed"] == out.loc[mask, "target_qty"]).all()


def test_raises_on_missing_columns() -> None:
    bad = pd.DataFrame({"target_qty": [1, 2]})
    with pytest.raises(ValueError):
        impute_stockout_demand(bad)
