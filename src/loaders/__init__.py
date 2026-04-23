"""External signal loaders. Importing this package registers every loader."""

from __future__ import annotations

# Loaders register themselves on import via @register_loader.
# Individual imports are added as each loader is implemented.

from src.loaders import air_raids_ua  # noqa: F401
from src.loaders import conflict_ua  # noqa: F401
from src.loaders import google_trends_ua  # noqa: F401
from src.loaders import holidays_ua  # noqa: F401
from src.loaders import imf_cpi  # noqa: F401
from src.loaders import nbu_fx  # noqa: F401
from src.loaders import school_calendar_ua  # noqa: F401
from src.loaders import tmdb_movies  # noqa: F401
from src.loaders import weather_ua  # noqa: F401
from src.loaders import world_bank_ua  # noqa: F401

__all__ = [
    "air_raids_ua",
    "conflict_ua",
    "google_trends_ua",
    "holidays_ua",
    "imf_cpi",
    "nbu_fx",
    "school_calendar_ua",
    "tmdb_movies",
    "weather_ua",
    "world_bank_ua",
]
