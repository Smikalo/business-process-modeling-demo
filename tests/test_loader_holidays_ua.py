from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.loaders.holidays_ua import UkrainianHolidaysLoader


def test_holidays_loader_has_expected_months(tmp_path: Path):
    loader = UkrainianHolidaysLoader(cache_dir=tmp_path)
    df = loader.load()
    assert str(df["Период"].dtype).startswith("period")
    # Coverage sanity: every month from 2019 through 2027 inclusive.
    assert df["Период"].min() == pd.Period("2019-01", freq="M")
    assert df["Период"].max() == pd.Period("2027-12", freq="M")


def test_december_flags_on(tmp_path: Path):
    loader = UkrainianHolidaysLoader(cache_dir=tmp_path)
    df = loader.load()
    dec = df[df["Период"] == pd.Period("2023-12", freq="M")].iloc[0]
    assert dec["is_dec"] == 1
    assert dec["preholiday_dec"] == 1
    assert dec["major_holiday_in_month"] == 1
    assert dec["days_to_ny"] >= 0


def test_days_to_counters_monotonic_within_gap(tmp_path: Path):
    """During a non-holiday month, days-to-next-NY decreases month over month
    until it resets."""
    loader = UkrainianHolidaysLoader(cache_dir=tmp_path)
    df = loader.load()
    jan = df[df["Период"] == pd.Period("2024-01", freq="M")].iloc[0]
    feb = df[df["Период"] == pd.Period("2024-02", freq="M")].iloc[0]
    assert feb["days_to_ny"] < jan["days_to_ny"] or jan["days_to_ny"] == 0


def test_signals_are_numeric(tmp_path: Path):
    loader = UkrainianHolidaysLoader(cache_dir=tmp_path)
    df = loader.load()
    for c in loader.signal_cols:
        assert df[c].dtype.kind in ("i", "u", "f")
