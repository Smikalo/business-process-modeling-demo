"""Unit tests for src.leakage_guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.leakage_guard import (
    _lag_in_months,
    apply_publication_lag,
    assert_no_future_leak,
)


def _synthetic_monthly(n: int = 24) -> pd.DataFrame:
    periods = pd.period_range("2022-01", periods=n, freq="M")
    return pd.DataFrame({"Период": periods, "x": np.arange(n, dtype=float)})


def test_lag_in_months_zero_for_zero_lag():
    assert _lag_in_months(0, forecast_horizon_months=1) == 0


def test_lag_in_months_rounds_up():
    # 14 days ceil / 30 = 1 month; horizon 1 -> shift 1
    assert _lag_in_months(14, forecast_horizon_months=1) == 1
    # 45 days ceil / 30 = 2 months
    assert _lag_in_months(45, forecast_horizon_months=1) == 2


def test_lag_in_months_accounts_for_horizon():
    assert _lag_in_months(0, forecast_horizon_months=3) == 2  # horizon-only shift
    assert _lag_in_months(14, forecast_horizon_months=3) == 3


def test_lag_in_months_rejects_negative_lag():
    with pytest.raises(ValueError):
        _lag_in_months(-1, 1)


def test_apply_zero_lag_is_passthrough():
    df = _synthetic_monthly()
    out = apply_publication_lag(df, publication_lag_days=0, signal_cols=["x"])
    pd.testing.assert_frame_equal(df, out)


def test_apply_lag_shifts_values_forward():
    df = _synthetic_monthly()
    out = apply_publication_lag(df, publication_lag_days=14, signal_cols=["x"])
    # First month now NaN (unobservable); rest of the column is the previous month's value.
    assert np.isnan(out.loc[0, "x"])
    assert out.loc[1, "x"] == df.loc[0, "x"]
    assert out.loc[10, "x"] == df.loc[9, "x"]


def test_apply_lag_per_group():
    df = pd.concat(
        [
            _synthetic_monthly().assign(g="a"),
            _synthetic_monthly().assign(g="b", x=lambda d: d["x"] + 100),
        ],
        ignore_index=True,
    )
    out = apply_publication_lag(
        df, publication_lag_days=30, signal_cols=["x"], group_cols=["g"]
    )
    # Each group's first row must be NaN (not contaminated by the other group).
    a = out[out.g == "a"].reset_index(drop=True)
    b = out[out.g == "b"].reset_index(drop=True)
    assert np.isnan(a.loc[0, "x"])
    assert np.isnan(b.loc[0, "x"])
    # Values inside each group are shifted from their own series.
    assert a.loc[5, "x"] == df[df.g == "a"].reset_index(drop=True).loc[4, "x"]
    assert b.loc[5, "x"] == df[df.g == "b"].reset_index(drop=True).loc[4, "x"]


def test_long_horizon_requires_more_shift():
    df = _synthetic_monthly()
    out1 = apply_publication_lag(df, 0, ["x"], forecast_horizon_months=1)
    out3 = apply_publication_lag(df, 0, ["x"], forecast_horizon_months=3)
    assert not out1["x"].isna().any()
    assert out3["x"].isna().sum() == 2


def test_assert_no_future_leak_passes_on_properly_lagged():
    df = _synthetic_monthly()
    shifted = apply_publication_lag(df, 14, ["x"])
    assert_no_future_leak(
        shifted, ["x"], publication_lag_days=14, max_training_period="2023-01"
    )


def test_assert_no_future_leak_catches_raw_unlagged_signal():
    df = _synthetic_monthly()  # untagged, raw
    with pytest.raises(AssertionError):
        assert_no_future_leak(
            df, ["x"], publication_lag_days=14, max_training_period="2023-01"
        )
