"""External signal loaders. Importing this package registers every loader."""

from __future__ import annotations

# Loaders register themselves on import via @register_loader.
# Individual imports are added as each loader is implemented.

from src.loaders import air_raids_ua  # noqa: F401
from src.loaders import airraid_oblast  # noqa: F401
from src.loaders import blackout_dtek  # noqa: F401
from src.loaders import conflict_ua  # noqa: F401
from src.loaders import google_trends_ua  # noqa: F401
from src.loaders import holidays_ua  # noqa: F401
from src.loaders import imf_cpi  # noqa: F401
from src.loaders import iom_idp  # noqa: F401
from src.loaders import nbu_cci  # noqa: F401
from src.loaders import nbu_fx  # noqa: F401
from src.loaders import orthodox_calendar  # noqa: F401
from src.loaders import school_calendar_ua  # noqa: F401
from src.loaders import tmdb_movies  # noqa: F401
from src.loaders import ukrstat_births  # noqa: F401
from src.loaders import ukrstat_indprod  # noqa: F401
from src.loaders import ukrstat_rti  # noqa: F401
from src.loaders import weather_ua  # noqa: F401
from src.loaders import wiki_pageviews  # noqa: F401
from src.loaders import world_bank_ua  # noqa: F401

__all__ = [
    "air_raids_ua",
    "airraid_oblast",
    "blackout_dtek",
    "conflict_ua",
    "google_trends_ua",
    "holidays_ua",
    "imf_cpi",
    "iom_idp",
    "nbu_cci",
    "nbu_fx",
    "orthodox_calendar",
    "school_calendar_ua",
    "tmdb_movies",
    "ukrstat_births",
    "ukrstat_indprod",
    "ukrstat_rti",
    "weather_ua",
    "wiki_pageviews",
    "world_bank_ua",
]
