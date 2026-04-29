"""Smoke test for NBUCCILoader.

Network-independent: relies on the synthetic fallback baked into the
loader so the test passes offline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.loaders.nbu_cci import NBUCCILoader


def test_nbu_cci_loader_smoke(tmp_path: Path):
    loader = NBUCCILoader(cache_dir=tmp_path)
    df = loader.load(force_refresh=False)

    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 60, f"expected >= 60 rows, got {len(df)}"
    for col in loader.signal_cols:
        assert col in df.columns, f"missing signal col {col}"
    assert str(df["Период"].dtype).startswith("period")


def test_nbu_cci_full_period_coverage(tmp_path: Path):
    loader = NBUCCILoader(cache_dir=tmp_path)
    df = loader.load(force_refresh=False)
    assert df["Период"].min() <= pd.Period("2019-01", freq="M")
    assert df["Период"].max() >= pd.Period("2027-12", freq="M")


def test_nbu_cci_war_trough_and_inflation_peak(tmp_path: Path):
    """CCI must drop in early 2022 and inflation expectations must spike."""
    loader = NBUCCILoader(cache_dir=tmp_path)
    df = loader.load(force_refresh=False)
    jan22 = df[df["Период"] == pd.Period("2022-01", freq="M")].iloc[0]
    mar22 = df[df["Период"] == pd.Period("2022-03", freq="M")].iloc[0]
    assert mar22["cci_overall"] < jan22["cci_overall"], (
        "expected CCI to drop into Mar 2022"
    )
    assert (
        mar22["inflation_expectations_12m_pct"]
        > jan22["inflation_expectations_12m_pct"]
    ), "expected inflation expectations to spike around invasion"
