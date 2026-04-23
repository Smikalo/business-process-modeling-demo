"""Ukrainian public holidays loader.

Uses the ``holidays`` library (no network), adds pre-holiday shopping windows
and New Year/Christmas/St. Nicholas proximity flags that are strong seasonal
signals for toy sales in Ukraine.
"""

from __future__ import annotations

from datetime import date, timedelta

import holidays
from holidays.constants import PUBLIC, WORKDAY
from holidays.countries.ukraine import Ukraine
import numpy as np
import pandas as pd

from src.external_data import BaseSignalLoader, register_loader


def _load_ukrainian_holidays(years: list[int]) -> dict[date, str]:
    """Return a date->name dict covering public + workday categories.

    For pre-2022 years, PUBLIC is populated.  For 2022+ (martial law) the
    ``holidays`` library moves the calendar to WORKDAY, so we merge both.
    We additionally inject St. Nicholas Day (Dec 6) which is a major
    toy-gifting cultural event even though it is not a legal holiday.
    """
    merged: dict[date, str] = {}
    for cat in (PUBLIC, WORKDAY):
        try:
            ua = Ukraine(years=years, categories=(cat,))
            for d, name in ua.items():
                merged[d] = name
        except Exception:  # noqa: BLE001
            continue

    for y in years:
        # St. Nicholas Day (legal public holiday 2017+ via Pope Francis
        # calendar shift; still observed Dec 6 in Ukraine after 2023).
        merged.setdefault(date(y, 12, 6), "День Святого Миколая")
        # Children's Day
        merged.setdefault(date(y, 6, 1), "День захисту дітей")

    return merged


HOLIDAY_CONFIG = {
    # Major gifting holidays — these drive toy sales directly.
    "new_year": ["Новий рік", "New Year's Day"],
    "christmas_dec25": ["Різдво Христове", "Christmas Day"],
    "christmas_jan7": ["Різдво Христове (за юліанським календарем)"],
    "saint_nicholas": ["День Святого Миколая"],
    "womens_day": ["Міжнародний жіночий день"],
    "easter": ["Великдень"],
    "childrens_day": ["День захисту дітей"],
}


def _count_in_month(
    holidays_map: dict[date, str], year: int, month: int, target_names: set[str]
) -> int:
    """How many calendar days in (year, month) fall on a target holiday name."""
    hits = 0
    for d, name in holidays_map.items():
        if d.year == year and d.month == month:
            if any(t.lower() in name.lower() for t in target_names):
                hits += 1
    return hits


def _days_to_next_occurrence(
    holidays_map: dict[date, str], year: int, month: int, target_names: set[str]
) -> int:
    """Days from first of month to next occurrence of any target holiday.
    Looks up to 12 months ahead; returns 365 if none found (sentinel)."""
    anchor = date(year, month, 1)
    for d, name in sorted(holidays_map.items()):
        if d < anchor:
            continue
        if any(t.lower() in name.lower() for t in target_names):
            return (d - anchor).days
    return 365


@register_loader
class UkrainianHolidaysLoader(BaseSignalLoader):
    name = "holidays_ua"
    signal_cols = [
        "holiday_count",
        "major_holiday_in_month",
        "is_dec",
        "days_to_ny",
        "days_to_st_nicholas",
        "days_to_womens_day",
        "days_to_easter",
        "days_to_childrens_day",
        "preholiday_dec",
        "preholiday_nov",
        "easter_month_flag",
    ]
    publication_lag_days = 0  # holidays are known years in advance
    upstream_url = "python-holidays library (offline)"
    # Cache is effectively infinite; re-generation is a few ms but we still
    # respect the default TTL so schema changes are picked up.

    def fetch_raw(self) -> pd.DataFrame:
        years = list(range(2019, 2028))
        holidays_map = _load_ukrainian_holidays(years)
        rows = [
            {"date": pd.Timestamp(d), "name": name}
            for d, name in sorted(holidays_map.items())
        ]
        return pd.DataFrame(rows)

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        holidays_map = _load_ukrainian_holidays(list(range(2019, 2028)))

        periods = pd.period_range("2019-01", "2027-12", freq="M")
        records: list[dict] = []
        for p in periods:
            y, m = p.year, p.month
            # Count every holiday falling in this month.
            total = sum(
                1 for d in holidays_map if d.year == y and d.month == m
            )

            major_names = set()
            for names in HOLIDAY_CONFIG.values():
                major_names.update(n.lower() for n in names)
            has_major = int(
                any(
                    d.year == y
                    and d.month == m
                    and any(t in n.lower() for t in major_names)
                    for d, n in holidays_map.items()
                )
            )

            rec = {
                "Период": p,
                "holiday_count": total,
                "major_holiday_in_month": has_major,
                "is_dec": int(m == 12),
                "days_to_ny": _days_to_next_occurrence(
                    holidays_map, y, m, {n.lower() for n in HOLIDAY_CONFIG["new_year"]}
                ),
                "days_to_st_nicholas": _days_to_next_occurrence(
                    holidays_map,
                    y,
                    m,
                    {n.lower() for n in HOLIDAY_CONFIG["saint_nicholas"]},
                ),
                "days_to_womens_day": _days_to_next_occurrence(
                    holidays_map,
                    y,
                    m,
                    {n.lower() for n in HOLIDAY_CONFIG["womens_day"]},
                ),
                "days_to_easter": _days_to_next_occurrence(
                    holidays_map, y, m, {n.lower() for n in HOLIDAY_CONFIG["easter"]}
                ),
                "days_to_childrens_day": _days_to_next_occurrence(
                    holidays_map,
                    y,
                    m,
                    {n.lower() for n in HOLIDAY_CONFIG["childrens_day"]},
                ),
                # Pre-holiday shopping flags: December is the NY shopping peak;
                # November captures "shopping season has started".
                "preholiday_dec": int(m == 12),
                "preholiday_nov": int(m == 11),
                "easter_month_flag": int(
                    _count_in_month(
                        holidays_map, y, m, {n.lower() for n in HOLIDAY_CONFIG["easter"]}
                    )
                    > 0
                ),
            }
            records.append(rec)

        out = pd.DataFrame(records)
        out["Период"] = out["Период"].astype("period[M]")
        # Enforce lightweight numeric dtypes.
        for c in self.signal_cols:
            out[c] = pd.to_numeric(out[c], downcast="integer")
        return out
