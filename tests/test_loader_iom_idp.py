"""Offline tests for the iom_idp loader.

The loader is synthetic in V12.0 (PDF parsing deferred), so these tests
fully exercise the calibrated profile without any network.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.loaders.iom_idp import IOMDisplacementLoader


def test_iom_idp_covers_full_range(tmp_path: Path):
    df = IOMDisplacementLoader(cache_dir=tmp_path).load()
    assert str(df["Период"].dtype).startswith("period")
    assert df["Период"].min() == pd.Period("2019-01", freq="M")
    assert df["Период"].max() == pd.Period("2027-12", freq="M")
    assert df["Период"].is_unique
    for col in IOMDisplacementLoader.signal_cols:
        assert col in df.columns


def test_iom_idp_zero_pre_invasion(tmp_path: Path):
    df = IOMDisplacementLoader(cache_dir=tmp_path).load()
    pre = df[df["Период"] < pd.Period("2022-02", freq="M")]
    assert (pre["idp_total_count"] == 0).all()
    assert (pre["idp_returns_count_month"] == 0).all()
    assert (pre["idp_western_oblasts_pct"] == 0.0).all()


def test_iom_idp_calibration_anchors(tmp_path: Path):
    """Spec calibration anchors: ~6.5M Mar 2022, ~7.7M Jul 2022 peak, ~5.4M
    Dec 2022, ~3.7M Dec 2023, ~3.5M Dec 2024."""
    df = IOMDisplacementLoader(cache_dir=tmp_path).load()

    def total(p: str) -> int:
        return int(df[df["Период"] == pd.Period(p, freq="M")].iloc[0]["idp_total_count"])

    assert 6_300_000 <= total("2022-03") <= 6_700_000
    assert 7_500_000 <= total("2022-07") <= 7_900_000
    assert 5_200_000 <= total("2022-12") <= 5_600_000
    assert 3_500_000 <= total("2023-12") <= 3_900_000
    assert 3_300_000 <= total("2024-12") <= 3_700_000


def test_iom_idp_peak_in_summer_2022(tmp_path: Path):
    """Total IDP count should peak in summer 2022, before the long decline."""
    df = IOMDisplacementLoader(cache_dir=tmp_path).load()
    peak_period = df.loc[df["idp_total_count"].idxmax(), "Период"]
    assert pd.Period("2022-05", freq="M") <= peak_period <= pd.Period("2022-08", freq="M")


def test_iom_idp_western_pct_plausible(tmp_path: Path):
    df = IOMDisplacementLoader(cache_dir=tmp_path).load()
    active = df[df["idp_total_count"] > 0]["idp_western_oblasts_pct"]
    # All wartime-active months: share is between 20% and 45%.
    assert active.min() >= 20.0
    assert active.max() <= 45.0


def test_iom_idp_returns_nonneg(tmp_path: Path):
    df = IOMDisplacementLoader(cache_dir=tmp_path).load()
    assert (df["idp_returns_count_month"] >= 0).all()
    # At least one month with sizeable returns (the late-2022 pulse).
    assert df["idp_returns_count_month"].max() > 100_000


def test_iom_idp_signal_cols_numeric(tmp_path: Path):
    df = IOMDisplacementLoader(cache_dir=tmp_path).load()
    for c in IOMDisplacementLoader.signal_cols:
        assert df[c].dtype.kind in ("i", "u", "f"), (c, df[c].dtype)


def test_iom_idp_cache_round_trip(tmp_path: Path):
    loader = IOMDisplacementLoader(cache_dir=tmp_path)
    first = loader.load()
    second = loader.load()
    pd.testing.assert_frame_equal(
        first.reset_index(drop=True), second.reset_index(drop=True)
    )
