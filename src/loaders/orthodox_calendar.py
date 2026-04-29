"""Orthodox religious calendar loader (deterministic, offline).

The Ukrainian Orthodox Church has officially shifted to the Revised Julian
(Gregorian-aligned) calendar since 2023, but a meaningful fraction of the
population still observes Old-Style Christmas (Jan 7) and Orthodox
(Julian-rule) Pascha — which falls on different dates than Western Easter
in most years.  These dates remain culturally salient anchors for
gift-giving (Christmas, St. Nicholas) and consumption-suppression (Lent),
both of which influence toy demand.

Signals exposed:
- ``is_orthodox_christmas_month`` — January (Old-Style Christmas Jan 7).
- ``is_orthodox_easter_month`` — month containing Orthodox Pascha.
- ``is_st_nicholas_month`` — December (St. Nicholas Day, Dec 19 OS).
- ``days_to_orthodox_easter`` — days from the 1st of the period month
  to the next Orthodox Easter (looks 12 months ahead).
- ``is_lent_month`` — month overlapping the 40-day Great Lent window
  preceding Orthodox Easter.

No network access.  Easter dates come from ``dateutil.easter`` with
``EASTER_ORTHODOX``; a hand-curated 2019–2030 fallback table is used when
``python-dateutil`` is unavailable.
"""

from __future__ import annotations

import calendar
import logging
from datetime import date, timedelta

import pandas as pd

from src.external_data import BaseSignalLoader, register_loader

log = logging.getLogger(__name__)


# Hand-curated fallback table of Orthodox (Julian-rule) Pascha dates.
# Source: publicly available Orthodox liturgical calendars.
_ORTHODOX_EASTER_FALLBACK: dict[int, date] = {
    2019: date(2019, 4, 28),
    2020: date(2020, 4, 19),
    2021: date(2021, 5, 2),
    2022: date(2022, 4, 24),
    2023: date(2023, 4, 16),
    2024: date(2024, 5, 5),
    2025: date(2025, 4, 20),
    2026: date(2026, 4, 12),
    2027: date(2027, 5, 2),
    2028: date(2028, 4, 16),
    2029: date(2029, 4, 8),
    2030: date(2030, 4, 28),
}


def _orthodox_easter(year: int) -> date | None:
    """Return Orthodox (Julian-rule) Easter for ``year``, or ``None`` if
    neither dateutil nor the fallback table covers the year."""
    try:
        from dateutil.easter import EASTER_ORTHODOX, easter

        return easter(year, EASTER_ORTHODOX)
    except Exception:  # noqa: BLE001
        return _ORTHODOX_EASTER_FALLBACK.get(year)


def _month_overlaps_window(
    year: int, month: int, win_start: date, win_end: date
) -> bool:
    last_day = calendar.monthrange(year, month)[1]
    m_start = date(year, month, 1)
    m_end = date(year, month, last_day)
    return not (m_end < win_start or m_start > win_end)


@register_loader
class OrthodoxCalendarLoader(BaseSignalLoader):
    name = "orthodox_cal"
    signal_cols = [
        "is_orthodox_christmas_month",
        "is_orthodox_easter_month",
        "is_st_nicholas_month",
        "days_to_orthodox_easter",
        "is_lent_month",
    ]
    publication_lag_days = 0
    upstream_url = (
        "computed (Computus + Julian-calendar feasts; offline)"
    )

    def fetch_raw(self) -> pd.DataFrame:
        periods = pd.period_range("2019-01", "2027-12", freq="M")
        return pd.DataFrame({"Период": periods})

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        try:
            df = raw.copy()
            df["Период"] = df["Период"].astype("period[M]")
        except Exception as exc:  # noqa: BLE001
            log.warning("orthodox_cal: bad input frame (%s) — rebuilding", exc)
            periods = pd.period_range("2019-01", "2027-12", freq="M")
            df = pd.DataFrame({"Период": periods})

        rows: list[dict] = []
        for p in df["Период"]:
            y = int(p.year)
            m = int(p.month)

            e_this = _orthodox_easter(y)
            e_next = _orthodox_easter(y + 1)

            anchor = date(y, m, 1)
            if e_this is not None and e_this >= anchor:
                next_easter = e_this
            elif e_next is not None:
                next_easter = e_next
            else:
                # Sentinel — no Easter date available for the lookup window.
                next_easter = anchor + timedelta(days=365)
            days_to_easter = (next_easter - anchor).days

            is_easter_month = int(
                e_this is not None
                and e_this.year == y
                and e_this.month == m
            )

            is_lent = 0
            if e_this is not None:
                lent_end = e_this - timedelta(days=1)
                lent_start = e_this - timedelta(days=40)
                if _month_overlaps_window(y, m, lent_start, lent_end):
                    is_lent = 1

            rows.append(
                {
                    "Период": p,
                    "is_orthodox_christmas_month": int(m == 1),
                    "is_orthodox_easter_month": is_easter_month,
                    "is_st_nicholas_month": int(m == 12),
                    "days_to_orthodox_easter": int(days_to_easter),
                    "is_lent_month": int(is_lent),
                }
            )

        out = pd.DataFrame(rows)
        out["Период"] = out["Период"].astype("period[M]")
        for c in self.signal_cols:
            out[c] = pd.to_numeric(out[c], downcast="integer")
        return out
