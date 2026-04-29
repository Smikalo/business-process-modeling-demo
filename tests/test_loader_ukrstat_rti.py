"""Smoke test for UkrstatRTILoader.

Network-independent: relies on the synthetic fallback baked into the
loader so the test passes offline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.loaders.ukrstat_rti import UkrstatRTILoader


def test_ukrstat_rti_loader_smoke(tmp_path: Path):
    loader = UkrstatRTILoader(cache_dir=tmp_path)
    df = loader.load(force_refresh=False)

    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 60, f"expected >= 60 rows, got {len(df)}"
    for col in loader.signal_cols:
        assert col in df.columns, f"missing signal col {col}"
    assert str(df["Период"].dtype).startswith("period")


def test_ukrstat_rti_covers_full_range(tmp_path: Path):
    loader = UkrstatRTILoader(cache_dir=tmp_path)
    df = loader.load(force_refresh=False)
    assert df["Период"].min() <= pd.Period("2019-01", freq="M")
    assert df["Период"].max() >= pd.Period("2027-12", freq="M")


def test_ukrstat_rti_war_shock_visible(tmp_path: Path):
    """When the synthetic fallback is used the Mar 2022 trough must be
    materially below the Feb 2022 level."""
    loader = UkrstatRTILoader(cache_dir=tmp_path)
    df = loader.load(force_refresh=False)
    feb22 = df[df["Период"] == pd.Period("2022-02", freq="M")]["retail_trade_idx"].iloc[0]
    mar22 = df[df["Период"] == pd.Period("2022-03", freq="M")]["retail_trade_idx"].iloc[0]
    assert mar22 < feb22, f"Expected Mar 2022 < Feb 2022 (got {mar22} vs {feb22})"
