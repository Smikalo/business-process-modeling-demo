from __future__ import annotations

import pandas as pd
import pytest

from src.config import OUTPUT_DIR
from src.loaders.weather_ua import OpenMeteoWeatherLoader


@pytest.fixture(scope="module")
def cached_weather_df() -> pd.DataFrame:
    cache = OUTPUT_DIR / "external" / "weather_ua.parquet"
    if not cache.exists():
        pytest.skip("weather cache not warmed; run loader once with network access")
    return pd.read_parquet(cache)


def test_expected_columns(cached_weather_df: pd.DataFrame):
    for c in OpenMeteoWeatherLoader.signal_cols:
        assert c in cached_weather_df.columns


def test_temperature_range(cached_weather_df: pd.DataFrame):
    t = cached_weather_df["temp_mean_c"].dropna()
    assert t.min() > -30
    assert t.max() < 40


def test_cold_month_flag_winter(cached_weather_df: pd.DataFrame):
    # At least one January in history must be cold.
    jan = cached_weather_df[
        cached_weather_df["Период"].astype(str).str.endswith("-01")
    ]
    assert jan["cold_month_flag"].sum() > 0
