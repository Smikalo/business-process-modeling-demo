"""Smoke test for UkrstatIndProdLoader.

Network-independent: relies on the synthetic fallback baked into the
loader so the test passes offline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.loaders.ukrstat_indprod import UkrstatIndProdLoader


def test_ukrstat_indprod_loader_smoke(tmp_path: Path):
    loader = UkrstatIndProdLoader(cache_dir=tmp_path)
    df = loader.load(force_refresh=False)

    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 60, f"expected >= 60 rows, got {len(df)}"
    for col in loader.signal_cols:
        assert col in df.columns, f"missing signal col {col}"
    assert str(df["Период"].dtype).startswith("period")


def test_ukrstat_indprod_full_period_coverage(tmp_path: Path):
    loader = UkrstatIndProdLoader(cache_dir=tmp_path)
    df = loader.load(force_refresh=False)
    assert df["Период"].min() <= pd.Period("2019-01", freq="M")
    assert df["Период"].max() >= pd.Period("2027-12", freq="M")


def test_ukrstat_indprod_war_shock(tmp_path: Path):
    """Mar 2022 industrial production must be sharply lower than Jan 2022."""
    loader = UkrstatIndProdLoader(cache_dir=tmp_path)
    df = loader.load(force_refresh=False)
    jan22 = df[df["Период"] == pd.Period("2022-01", freq="M")]["indprod_idx"].iloc[0]
    mar22 = df[df["Период"] == pd.Period("2022-03", freq="M")]["indprod_idx"].iloc[0]
    assert mar22 < 0.8 * jan22, (
        f"Expected Mar 2022 to drop > 20% vs Jan 2022 (got {mar22} vs {jan22})"
    )
