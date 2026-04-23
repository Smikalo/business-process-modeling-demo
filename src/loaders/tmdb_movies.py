"""TMDB (The Movie Database) family/animation release loader.

Aggregates monthly count of family-friendly and animation releases worldwide
as a proxy for toy-tie-in demand.  TMDB offers a free-tier API that still
requires signup; absent credentials we ship a hand-curated catalog of major
blockbuster releases (Frozen 2, Barbie, Super Mario Bros Movie, etc.) and
their release months so the signal is demonstrable out-of-the-box.
"""

from __future__ import annotations

import logging
import os
from datetime import date

import numpy as np
import pandas as pd
import requests

from src.external_data import BaseSignalLoader, register_loader

log = logging.getLogger(__name__)

TMDB_DISCOVER_URL = "https://api.themoviedb.org/3/discover/movie"


# Hand-curated: month of global theatrical release for major toy-tie-in films.
# Covers Disney/Pixar animation, LEGO-branded, and brand-driven tentpoles.
MAJOR_RELEASES: list[tuple[str, str, float]] = [
    # (YYYY-MM, title, influence_score 0-1)
    ("2019-03", "Captain Marvel", 0.6),
    ("2019-06", "Toy Story 4", 1.0),
    ("2019-07", "The Lion King (2019)", 0.9),
    ("2019-12", "Frozen 2", 1.0),
    ("2019-12", "Star Wars: Rise of Skywalker", 0.8),
    ("2021-07", "Space Jam 2", 0.5),
    ("2021-12", "Spider-Man: No Way Home", 0.8),
    ("2022-07", "Minions: The Rise of Gru", 0.8),
    ("2022-11", "Strange World (Disney)", 0.4),
    ("2023-04", "Super Mario Bros Movie", 0.9),
    ("2023-05", "Guardians Galaxy 3", 0.7),
    ("2023-06", "Elemental (Pixar)", 0.5),
    ("2023-07", "Barbie", 1.0),
    ("2023-11", "Trolls Band Together", 0.6),
    ("2023-11", "Wish (Disney)", 0.5),
    ("2024-03", "Kung Fu Panda 4", 0.7),
    ("2024-06", "Inside Out 2", 1.0),
    ("2024-07", "Despicable Me 4", 0.8),
    ("2024-11", "Moana 2", 0.9),
    ("2024-12", "Sonic 3", 0.7),
    ("2025-05", "Lilo & Stitch live action", 0.6),
    ("2025-06", "Elio (Pixar)", 0.5),
    ("2025-07", "Fantastic Four", 0.7),
    ("2025-12", "Zootopia 2", 0.8),
]


@register_loader
class TMDBMoviesLoader(BaseSignalLoader):
    name = "tmdb_movies"
    signal_cols = [
        "family_releases_count",
        "family_releases_influence",
        "release_next_month",
        "release_prior_month",
    ]
    publication_lag_days = 0  # film release schedules are known years ahead
    upstream_url = "https://api.themoviedb.org/3"
    cache_ttl_days = 90

    def fetch_raw(self) -> pd.DataFrame:
        key = os.getenv("TMDB_API_KEY")
        if not key:
            log.info("TMDB_API_KEY not set — using curated release catalog")
            return pd.DataFrame(
                [
                    {
                        "Период": pd.Period(p, freq="M"),
                        "title": title,
                        "influence": score,
                    }
                    for p, title, score in MAJOR_RELEASES
                ]
            )

        frames: list[pd.DataFrame] = []
        for year in range(2019, 2027):
            try:
                params = {
                    "api_key": key,
                    "primary_release_year": year,
                    "with_genres": "10751,16",  # Family, Animation
                    "vote_count.gte": 200,  # filter to notable releases
                    "page": 1,
                }
                r = requests.get(TMDB_DISCOVER_URL, params=params, timeout=30)
                r.raise_for_status()
                data = r.json()
                for item in data.get("results", []):
                    rd = item.get("release_date")
                    if not rd:
                        continue
                    frames.append(
                        pd.DataFrame(
                            [
                                {
                                    "Период": pd.Period(rd[:7], freq="M"),
                                    "title": item.get("title", ""),
                                    "influence": float(
                                        (item.get("vote_count", 0) or 0) / 5000
                                    ),
                                }
                            ]
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning("TMDB year %d failed: %s", year, exc)
        if not frames:
            return pd.DataFrame(
                [
                    {
                        "Период": pd.Period(p, freq="M"),
                        "title": title,
                        "influence": score,
                    }
                    for p, title, score in MAJOR_RELEASES
                ]
            )
        return pd.concat(frames, ignore_index=True)

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        raw = raw.copy()
        raw["Период"] = raw["Период"].astype("period[M]")
        agg = (
            raw.groupby("Период", as_index=False)
            .agg(
                family_releases_count=("title", "count"),
                family_releases_influence=("influence", "sum"),
            )
        )
        periods = pd.period_range("2019-01", "2027-12", freq="M")
        out = pd.DataFrame({"Период": periods})
        out = out.merge(agg, on="Период", how="left")
        out["family_releases_count"] = out["family_releases_count"].fillna(0)
        out["family_releases_influence"] = out["family_releases_influence"].fillna(0)

        # Signal leaks: a release in month t boosts sales in t-1 (pre-hype)
        # and t+1 (post-release merch).
        out["release_next_month"] = out["family_releases_influence"].shift(-1).fillna(0)
        out["release_prior_month"] = out["family_releases_influence"].shift(1).fillna(0)

        for c in self.signal_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
        out["Период"] = out["Период"].astype("period[M]")
        return out[["Период"] + self.signal_cols]
