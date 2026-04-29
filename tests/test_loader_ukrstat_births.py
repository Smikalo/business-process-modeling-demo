"""Smoke test for UkrstatBirthsLoader.

Network-independent: relies on the synthetic fallback baked into the
loader so the test passes offline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.loaders.ukrstat_births import UkrstatBirthsLoader


def test_ukrstat_births_loader_smoke(tmp_path: Path):
    loader = UkrstatBirthsLoader(cache_dir=tmp_path)
    df = loader.load(force_refresh=False)

    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 60, f"expected >= 60 rows, got {len(df)}"
    for col in loader.signal_cols:
        assert col in df.columns, f"missing signal col {col}"
    assert str(df["Период"].dtype).startswith("period")


def test_ukrstat_births_full_period_coverage(tmp_path: Path):
    loader = UkrstatBirthsLoader(cache_dir=tmp_path)
    df = loader.load(force_refresh=False)
    assert df["Период"].min() <= pd.Period("2019-01", freq="M")
    assert df["Период"].max() >= pd.Period("2027-12", freq="M")


def test_ukrstat_births_postwar_lower_than_prewar(tmp_path: Path):
    """Births_total must be materially lower in 2024 than in 2019."""
    loader = UkrstatBirthsLoader(cache_dir=tmp_path)
    df = loader.load(force_refresh=False)
    pre = df[df["Период"].astype(str).str.startswith("2019")]["births_total"].mean()
    post = df[df["Период"].astype(str).str.startswith("2024")]["births_total"].mean()
    assert post < pre, f"Expected post-war < pre-war births (got {post} vs {pre})"
