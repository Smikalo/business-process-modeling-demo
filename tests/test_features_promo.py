"""Tests for promo-lifecycle features."""

from __future__ import annotations

import pandas as pd

from src.features_promo import add_promo_lifecycle


def _series() -> pd.DataFrame:
    """12 months for one SKU with promos in Mar, Apr, Sep."""
    periods = pd.period_range("2024-01", "2024-12", freq="M")
    is_promo = [1 if p.month in (3, 4, 9) else 0 for p in periods]
    depth = [0.3 if f else 0.0 for f in is_promo]
    return pd.DataFrame(
        {
            "Период": periods,
            "Партнер": "A",
            "Артикул": "SKU1",
            "is_promo": is_promo,
            "promo_discount_pct": depth,
            "target_qty": [10 if f else 5 for f in is_promo],
        }
    )


def test_promo_duration_matches_consecutive_prior_months() -> None:
    df = _series()
    out = add_promo_lifecycle(df).sort_values("Период").reset_index(drop=True)
    # March (month 3) is first promo — duration (prior consecutive) should be 0
    assert out.loc[out["Период"].dt.month == 3, "promo_duration_months"].iloc[0] == 0
    # April (month 4) follows March — duration should be 1
    assert out.loc[out["Период"].dt.month == 4, "promo_duration_months"].iloc[0] == 1
    # September (isolated) — duration 0
    assert out.loc[out["Период"].dt.month == 9, "promo_duration_months"].iloc[0] == 0


def test_post_promo_depletion_flag_fires_after_deep_promo() -> None:
    df = _series()
    out = add_promo_lifecycle(df).sort_values("Период").reset_index(drop=True)
    # May and June follow the Mar-Apr promo — flag should fire
    may = out.loc[out["Период"].dt.month == 5, "post_promo_depletion_flag"].iloc[0]
    jun = out.loc[out["Период"].dt.month == 6, "post_promo_depletion_flag"].iloc[0]
    assert may == 1
    assert jun == 1
    # Feb has no prior promo
    feb = out.loc[out["Период"].dt.month == 2, "post_promo_depletion_flag"].iloc[0]
    assert feb == 0


def test_months_until_next_promo_counts_down() -> None:
    df = _series()
    out = add_promo_lifecycle(df).sort_values("Период").reset_index(drop=True)
    # January is 2 months before March promo
    jan = out.loc[out["Период"].dt.month == 1, "months_until_next_promo"].iloc[0]
    assert jan == 2
    # March is 0 (current-month promo)
    mar = out.loc[out["Период"].dt.month == 3, "months_until_next_promo"].iloc[0]
    assert mar == 0


def test_sku_promo_sensitivity_above_one_when_uplift_positive() -> None:
    df = _series()
    out = add_promo_lifecycle(df)
    sens = out["sku_promo_sensitivity"].iloc[0]
    # Promo target = 10, non-promo = 5, so uplift ≈ 2×
    # Shrinkage pulls toward 1.0 but value should still be > 1.1
    assert sens > 1.1
