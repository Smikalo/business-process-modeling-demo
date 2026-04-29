"""Offline tests for the Orthodox calendar loader.

The loader is fully deterministic (no network), so all tests run
offline and exercise the full transform pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.loaders.orthodox_calendar import OrthodoxCalendarLoader


def test_imports_register_loader():
    from src.external_data import LOADER_REGISTRY
    from src.loaders import orthodox_calendar  # noqa: F401

    assert "orthodox_cal" in LOADER_REGISTRY


def test_orthodox_loader_full_range(tmp_path: Path):
    loader = OrthodoxCalendarLoader(cache_dir=tmp_path)
    df = loader.load()
    assert str(df["Период"].dtype).startswith("period")
    assert df["Период"].min() == pd.Period("2019-01", freq="M")
    assert df["Период"].max() == pd.Period("2027-12", freq="M")
    # 9 years × 12 months = 108 rows.
    assert len(df) == 108
    for col in loader.signal_cols:
        assert col in df.columns
        assert df[col].dtype.kind in ("i", "u", "f")


def test_christmas_and_st_nicholas_flags(tmp_path: Path):
    loader = OrthodoxCalendarLoader(cache_dir=tmp_path)
    df = loader.load()

    jan = df[df["Период"] == pd.Period("2024-01", freq="M")].iloc[0]
    assert jan["is_orthodox_christmas_month"] == 1
    assert jan["is_st_nicholas_month"] == 0

    dec = df[df["Период"] == pd.Period("2024-12", freq="M")].iloc[0]
    assert dec["is_orthodox_christmas_month"] == 0
    assert dec["is_st_nicholas_month"] == 1


def test_orthodox_easter_2024_in_may(tmp_path: Path):
    """Orthodox Easter 2024 fell on 2024-05-05, so May 2024 is the easter
    month and April 2024 is not."""
    loader = OrthodoxCalendarLoader(cache_dir=tmp_path)
    df = loader.load()

    may = df[df["Период"] == pd.Period("2024-05", freq="M")].iloc[0]
    apr = df[df["Период"] == pd.Period("2024-04", freq="M")].iloc[0]

    assert may["is_orthodox_easter_month"] == 1
    assert apr["is_orthodox_easter_month"] == 0


def test_days_to_orthodox_easter_decreases_into_easter(tmp_path: Path):
    """days_to_orthodox_easter should fall month-over-month leading up to
    Easter and then jump back up after."""
    loader = OrthodoxCalendarLoader(cache_dir=tmp_path)
    df = loader.load()

    feb = df[df["Период"] == pd.Period("2024-02", freq="M")].iloc[0]
    mar = df[df["Период"] == pd.Period("2024-03", freq="M")].iloc[0]
    apr = df[df["Период"] == pd.Period("2024-04", freq="M")].iloc[0]
    may = df[df["Период"] == pd.Period("2024-05", freq="M")].iloc[0]
    jun = df[df["Период"] == pd.Period("2024-06", freq="M")].iloc[0]

    assert feb["days_to_orthodox_easter"] > mar["days_to_orthodox_easter"]
    assert mar["days_to_orthodox_easter"] > apr["days_to_orthodox_easter"]
    assert apr["days_to_orthodox_easter"] >= 0
    # In May (the Easter month) the lookahead distance is small.
    assert may["days_to_orthodox_easter"] >= 0
    # June must look ahead to the *following* year's Easter, so
    # days_to_easter must be larger again than May's.
    assert jun["days_to_orthodox_easter"] > may["days_to_orthodox_easter"]


def test_lent_month_is_set_before_easter(tmp_path: Path):
    """The 40 days preceding Orthodox Easter 2024 (May 5) span late March
    through early May, so March, April and May 2024 must all have
    is_lent_month == 1."""
    loader = OrthodoxCalendarLoader(cache_dir=tmp_path)
    df = loader.load()

    for ym in ("2024-03", "2024-04", "2024-05"):
        row = df[df["Период"] == pd.Period(ym, freq="M")].iloc[0]
        assert row["is_lent_month"] == 1, f"{ym} should be a Lent month"

    # February 2024 starts >40d before Easter (May 5 → Lent starts Mar 26).
    feb = df[df["Период"] == pd.Period("2024-02", freq="M")].iloc[0]
    assert feb["is_lent_month"] == 0


def test_signals_are_numeric(tmp_path: Path):
    loader = OrthodoxCalendarLoader(cache_dir=tmp_path)
    df = loader.load()
    for c in loader.signal_cols:
        assert df[c].dtype.kind in ("i", "u", "f")
