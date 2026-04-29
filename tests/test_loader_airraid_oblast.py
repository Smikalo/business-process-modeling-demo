"""Offline tests for the airraid_oblast loader.

The loader degrades gracefully to a calibrated synthetic when the
``alerts.in.ua`` API is unreachable or no token is configured, so these
tests run fully offline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.loaders.airraid_oblast import AirRaidOblastLoader


def test_airraid_oblast_loads_and_covers_full_range(tmp_path: Path):
    df = AirRaidOblastLoader(cache_dir=tmp_path).load()
    assert str(df["Период"].dtype).startswith("period")
    assert df["Период"].min() == pd.Period("2019-01", freq="M")
    assert df["Период"].max() == pd.Period("2027-12", freq="M")
    # No duplicates on the join key.
    assert df["Период"].is_unique
    # All declared signal cols present.
    for col in AirRaidOblastLoader.signal_cols:
        assert col in df.columns


def test_airraid_oblast_zero_before_invasion(tmp_path: Path):
    df = AirRaidOblastLoader(cache_dir=tmp_path).load()
    pre_war = df[df["Период"] < pd.Period("2022-02", freq="M")]
    assert (pre_war["airraid_total_hours_month"] == 0).all()
    assert (pre_war["airraid_alerts_count_month"] == 0).all()
    assert (pre_war["airraid_high_intensity_oblasts"] == 0).all()


def test_airraid_oblast_post_invasion_intensity(tmp_path: Path):
    df = AirRaidOblastLoader(cache_dir=tmp_path).load()
    # The synthetic peak month (2022-03) must be substantial.
    mar22 = df[df["Период"] == pd.Period("2022-03", freq="M")].iloc[0]
    assert mar22["airraid_total_hours_month"] > 500.0
    assert mar22["airraid_alerts_count_month"] > 100
    assert mar22["airraid_high_intensity_oblasts"] >= 5


def test_airraid_oblast_signal_cols_numeric(tmp_path: Path):
    df = AirRaidOblastLoader(cache_dir=tmp_path).load()
    for c in AirRaidOblastLoader.signal_cols:
        assert df[c].dtype.kind in ("i", "u", "f"), (c, df[c].dtype)


def test_airraid_oblast_never_raises_without_token(tmp_path: Path, monkeypatch):
    """No env var set, no network — the loader still produces a valid frame."""
    monkeypatch.delenv("ALERTS_IN_UA_TOKEN", raising=False)
    df = AirRaidOblastLoader(cache_dir=tmp_path).load()
    # Validation already runs inside .load(); explicit check for safety.
    assert len(df) == 12 * (2027 - 2019 + 1)
    # All values finite & non-negative.
    assert (df["airraid_total_hours_month"] >= 0).all()
    assert (df["airraid_alerts_count_month"] >= 0).all()
    assert (df["airraid_high_intensity_oblasts"] >= 0).all()


def test_airraid_oblast_cache_round_trip(tmp_path: Path):
    """First load writes cache; second load reads it back without raising."""
    loader = AirRaidOblastLoader(cache_dir=tmp_path)
    first = loader.load()
    second = loader.load()
    assert len(first) == len(second)
    pd.testing.assert_frame_equal(
        first.reset_index(drop=True), second.reset_index(drop=True)
    )
