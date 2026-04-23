"""Ukrainian school year calendar — start-of-year shopping window, breaks, end.

The calendar is hand-curated from the Ministry of Education annual decrees
("Про організацію навчання…").  Dates are deliberately conservative: we
mark the full month in which a break falls rather than specific days, as our
downstream grain is monthly.

Hypothesis:
- September: back-to-school spend crosses into educational-toy demand.
- Winter break (Dec 25 – Jan 7): gifting peak for children.
- Spring break: secondary gifting window (Easter tie-in).
- Summer months (Jul/Aug): parents travel — retail footfall drops,
  but online toy spend may rise.
"""

from __future__ import annotations

import pandas as pd

from src.external_data import BaseSignalLoader, register_loader


# Per academic year breakdown (September → June typical pattern).
# Sources: Ministerial decrees available at mon.gov.ua; dates approximated to the month.
SCHOOL_CALENDAR: dict[int, dict] = {
    # academic_year_start: {months of school ongoing, winter_break_months, spring_break_months}
    2019: {"start_month": 9, "end_month": 6, "winter_break": [1], "spring_break": [3]},
    2020: {"start_month": 9, "end_month": 6, "winter_break": [1], "spring_break": [3]},
    2021: {"start_month": 9, "end_month": 6, "winter_break": [1], "spring_break": [3]},
    # 2022+: wartime schedule; most schools still followed classical cycle
    2022: {"start_month": 9, "end_month": 6, "winter_break": [1], "spring_break": [3]},
    2023: {"start_month": 9, "end_month": 6, "winter_break": [1], "spring_break": [3]},
    2024: {"start_month": 9, "end_month": 6, "winter_break": [1], "spring_break": [3]},
    2025: {"start_month": 9, "end_month": 6, "winter_break": [1], "spring_break": [3]},
    2026: {"start_month": 9, "end_month": 6, "winter_break": [1], "spring_break": [3]},
}


def _academic_year(period: pd.Period) -> int:
    """Return the calendar year the academic year began in."""
    if period.month >= 9:
        return period.year
    return period.year - 1


@register_loader
class SchoolCalendarLoader(BaseSignalLoader):
    name = "school_ua"
    signal_cols = [
        "school_in_session",
        "back_to_school_month",   # September
        "winter_school_break",
        "spring_school_break",
        "summer_break",           # July/August
        "months_until_back_to_school",
    ]
    publication_lag_days = 0
    upstream_url = "Ministry of Education decrees (hardcoded; updated annually)"

    def fetch_raw(self) -> pd.DataFrame:
        periods = pd.period_range("2019-01", "2027-12", freq="M")
        return pd.DataFrame({"Период": periods})

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        out = raw.copy()
        out["Период"] = out["Период"].astype("period[M]")

        rows: list[dict] = []
        for _, r in out.iterrows():
            p = r["Период"]
            ay = _academic_year(p)
            cal = SCHOOL_CALENDAR.get(ay, SCHOOL_CALENDAR[2024])
            m = p.month

            in_session = int(
                (m >= cal["start_month"])
                or (m <= cal["end_month"])
                or (cal["start_month"] <= cal["end_month"] and cal["start_month"] <= m <= cal["end_month"])
            )
            # Exclude winter/spring breaks and summer.
            if m in cal["winter_break"] or m in cal["spring_break"] or m in (7, 8):
                in_session = 0

            # Months until September
            months_until_bts = (9 - m) if m <= 9 else (21 - m)

            rows.append(
                {
                    "Период": p,
                    "school_in_session": in_session,
                    "back_to_school_month": int(m == 9),
                    "winter_school_break": int(m in cal["winter_break"]),
                    "spring_school_break": int(m in cal["spring_break"]),
                    "summer_break": int(m in (7, 8)),
                    "months_until_back_to_school": months_until_bts,
                }
            )

        df = pd.DataFrame(rows)
        df["Период"] = df["Период"].astype("period[M]")
        for c in self.signal_cols:
            df[c] = pd.to_numeric(df[c], downcast="integer")
        return df
