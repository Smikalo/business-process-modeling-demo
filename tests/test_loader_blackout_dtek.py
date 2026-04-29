"""Offline tests for the blackout_dtek loader.

The loader is synthetic-only in V12.0 (live scraping deferred to V12.1) so
these tests fully exercise the calibrated profile without any network.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.loaders.blackout_dtek import BlackoutDTEKLoader


def test_blackout_dtek_covers_full_range(tmp_path: Path):
    df = BlackoutDTEKLoader(cache_dir=tmp_path).load()
    assert str(df["Период"].dtype).startswith("period")
    assert df["Период"].min() == pd.Period("2019-01", freq="M")
    assert df["Период"].max() == pd.Period("2027-12", freq="M")
    assert df["Период"].is_unique
    for col in BlackoutDTEKLoader.signal_cols:
        assert col in df.columns


def test_blackout_dtek_quiet_summer_2023(tmp_path: Path):
    """The spec says the first wave decays to ~4 hrs/day by Mar 2023, then
    drops to 0 through summer."""
    df = BlackoutDTEKLoader(cache_dir=tmp_path).load()
    summer = df[
        (df["Период"] >= pd.Period("2023-06", freq="M"))
        & (df["Период"] <= pd.Period("2023-09", freq="M"))
    ]
    assert (summer["blackout_avg_hours_per_day"] == 0.0).all()
    assert (summer["blackout_severity_index"] == 0.0).all()


def test_blackout_dtek_first_wave_peak_dec_2022(tmp_path: Path):
    df = BlackoutDTEKLoader(cache_dir=tmp_path).load()
    dec22 = df[df["Период"] == pd.Period("2022-12", freq="M")].iloc[0]
    # Spec: peak ~12 hrs/day in Dec 2022.
    assert 11.0 <= dec22["blackout_avg_hours_per_day"] <= 12.5
    assert dec22["blackout_severity_index"] >= 8.0
    assert dec22["blackout_pct_population_affected"] >= 50.0


def test_blackout_dtek_second_wave_peak_dec_2024(tmp_path: Path):
    df = BlackoutDTEKLoader(cache_dir=tmp_path).load()
    dec24 = df[df["Период"] == pd.Period("2024-12", freq="M")].iloc[0]
    # Spec: peak ~8 hrs/day.
    assert 7.0 <= dec24["blackout_avg_hours_per_day"] <= 9.0


def test_blackout_dtek_third_wave_peak_dec_2025(tmp_path: Path):
    df = BlackoutDTEKLoader(cache_dir=tmp_path).load()
    dec25 = df[df["Период"] == pd.Period("2025-12", freq="M")].iloc[0]
    # Spec: smaller wave Oct 2025 → Mar 2026 with peak ~5 hrs/day.
    assert 4.0 <= dec25["blackout_avg_hours_per_day"] <= 6.0


def test_blackout_dtek_severity_in_zero_to_ten(tmp_path: Path):
    df = BlackoutDTEKLoader(cache_dir=tmp_path).load()
    sev = df["blackout_severity_index"]
    assert sev.min() >= 0.0
    assert sev.max() <= 10.0


def test_blackout_dtek_zero_pre_crisis(tmp_path: Path):
    df = BlackoutDTEKLoader(cache_dir=tmp_path).load()
    pre = df[df["Период"] < pd.Period("2022-10", freq="M")]
    assert (pre["blackout_avg_hours_per_day"] == 0.0).all()
    assert (pre["blackout_severity_index"] == 0.0).all()


def test_blackout_dtek_signal_cols_numeric(tmp_path: Path):
    df = BlackoutDTEKLoader(cache_dir=tmp_path).load()
    for c in BlackoutDTEKLoader.signal_cols:
        assert df[c].dtype.kind in ("i", "u", "f"), (c, df[c].dtype)
