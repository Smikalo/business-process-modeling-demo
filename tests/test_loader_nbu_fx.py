"""Smoke test for NBU FX loader — uses live cache if present, otherwise skips."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import OUTPUT_DIR
from src.loaders.nbu_fx import NBUFXLoader


@pytest.fixture(scope="module")
def cached_nbu_df() -> pd.DataFrame:
    cache = OUTPUT_DIR / "external" / "nbu_fx.parquet"
    if not cache.exists():
        pytest.skip("nbu_fx cache not warmed; run loader once with network access")
    return pd.read_parquet(cache)


def test_contains_expected_columns(cached_nbu_df: pd.DataFrame):
    for col in NBUFXLoader.signal_cols:
        assert col in cached_nbu_df.columns, f"missing {col}"


def test_usd_rate_plausible(cached_nbu_df: pd.DataFrame):
    rate = cached_nbu_df["uah_usd_eom"].dropna()
    # Post-2015 UAH/USD has been between ~23 and ~55.
    assert rate.min() > 20
    assert rate.max() < 100


def test_policy_rate_nonnegative(cached_nbu_df: pd.DataFrame):
    pr = cached_nbu_df["nbu_policy_rate_eom"].dropna()
    assert pr.min() >= 0
    assert pr.max() <= 100
