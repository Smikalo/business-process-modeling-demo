"""Open-Meteo historical monthly weather aggregates for major Ukrainian cities.

Open-Meteo's Archive API (https://archive-api.open-meteo.com) is free for
non-commercial use and requires no authentication.  We fetch daily weather
for a handful of population centers and aggregate to monthly means / sums,
then average across cities weighted by population so the resulting signal
represents a single "Ukrainian retail climate".

Feature hypothesis: cold winters and rainy spring months compress outdoor
play, shifting spend toward indoor toys and board games; hot summers
depress toy purchases as parents travel.
"""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd
import requests

from src.external_data import BaseSignalLoader, register_loader

log = logging.getLogger(__name__)

OPEN_METEO_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude={lat}&longitude={lon}"
    "&start_date={start}&end_date={end}"
    "&daily=temperature_2m_mean,precipitation_sum,snowfall_sum"
    "&timezone=Europe/Kyiv"
)

# Weighted by approximate population (millions)
CITIES: list[dict] = [
    {"name": "Kyiv",     "lat": 50.4501, "lon": 30.5234, "weight": 3.0},
    {"name": "Kharkiv",  "lat": 49.9935, "lon": 36.2304, "weight": 1.4},
    {"name": "Odesa",    "lat": 46.4825, "lon": 30.7233, "weight": 1.0},
    {"name": "Dnipro",   "lat": 48.4647, "lon": 35.0462, "weight": 1.0},
    {"name": "Lviv",     "lat": 49.8397, "lon": 24.0297, "weight": 0.7},
]


@register_loader
class OpenMeteoWeatherLoader(BaseSignalLoader):
    name = "weather_ua"
    signal_cols = [
        "temp_mean_c",
        "precip_mm",
        "snowfall_cm",
        "temp_anomaly_c",   # deviation from 2019-2024 monthly norm
        "cold_month_flag",  # temp_mean <= 0
        "wet_month_flag",   # precip > 75mm
    ]
    publication_lag_days = 7  # archive series have ~5-7 day settling lag
    upstream_url = "https://archive-api.open-meteo.com/v1/archive"
    cache_ttl_days = 14  # weather doesn't change retroactively

    def fetch_raw(self) -> pd.DataFrame:
        start = "2019-01-01"
        # Archive API only serves up to ~5 days ago.
        end = (pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=6)).strftime(
            "%Y-%m-%d"
        )
        all_daily: list[pd.DataFrame] = []
        for city in CITIES:
            url = OPEN_METEO_URL.format(
                lat=city["lat"], lon=city["lon"], start=start, end=end
            )
            try:
                r = requests.get(url, timeout=45)
                r.raise_for_status()
                payload = r.json()
                daily = pd.DataFrame(payload["daily"])
                daily["city"] = city["name"]
                daily["weight"] = city["weight"]
                all_daily.append(daily)
            except Exception as exc:  # noqa: BLE001
                log.warning("Open-Meteo fetch failed for %s: %s", city["name"], exc)
        if not all_daily:
            raise RuntimeError("All Open-Meteo city fetches failed")
        return pd.concat(all_daily, ignore_index=True)

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        raw = raw.copy()
        raw["date"] = pd.to_datetime(raw["time"])
        raw["Период"] = raw["date"].dt.to_period("M")
        raw["temperature_2m_mean"] = pd.to_numeric(raw["temperature_2m_mean"], errors="coerce")
        raw["precipitation_sum"] = pd.to_numeric(raw["precipitation_sum"], errors="coerce")
        raw["snowfall_sum"] = pd.to_numeric(raw["snowfall_sum"], errors="coerce")

        # Aggregate daily → monthly per city
        by_city_month = (
            raw.groupby(["Период", "city"], as_index=False)
            .agg(
                temp_mean_c=("temperature_2m_mean", "mean"),
                precip_mm=("precipitation_sum", "sum"),
                snowfall_cm=("snowfall_sum", "sum"),
                weight=("weight", "first"),
            )
        )

        def _weighted_mean(sub: pd.DataFrame, col: str) -> float:
            w = sub["weight"].values
            v = sub[col].values
            return float((v * w).sum() / w.sum()) if w.sum() else float("nan")

        monthly = by_city_month.groupby("Период", as_index=False).apply(
            lambda g: pd.Series(
                {
                    "temp_mean_c": _weighted_mean(g, "temp_mean_c"),
                    "precip_mm": _weighted_mean(g, "precip_mm"),
                    "snowfall_cm": _weighted_mean(g, "snowfall_cm") / 10,
                }
            ),
            include_groups=False,
        ).reset_index(drop=True)

        # Reattach Период (apply may drop it)
        if "Период" not in monthly.columns:
            monthly["Период"] = by_city_month.groupby("Период").size().index

        monthly["Период"] = monthly["Период"].astype("period[M]")
        monthly = monthly.sort_values("Период").reset_index(drop=True)

        # Temperature anomaly: deviation from 2019-2024 same-month mean
        baseline_range = (
            monthly["Период"] >= pd.Period("2019-01", freq="M")
        ) & (monthly["Период"] <= pd.Period("2024-12", freq="M"))
        month_norm = (
            monthly.loc[baseline_range]
            .assign(m=lambda d: d["Период"].apply(lambda p: p.month))
            .groupby("m")["temp_mean_c"]
            .mean()
            .to_dict()
        )
        monthly["temp_anomaly_c"] = monthly.apply(
            lambda row: row["temp_mean_c"] - month_norm.get(row["Период"].month, row["temp_mean_c"]),
            axis=1,
        )

        monthly["cold_month_flag"] = (monthly["temp_mean_c"] <= 0).astype(int)
        monthly["wet_month_flag"] = (monthly["precip_mm"] > 75).astype(int)

        for c in self.signal_cols:
            monthly[c] = pd.to_numeric(monthly[c], errors="coerce")

        return monthly[["Период"] + self.signal_cols]
