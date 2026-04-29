"""IOM Displacement Tracking Matrix (DTM) — internally displaced persons.

Source intent: scrape the quarterly DTM PDF reports published at
https://dtm.iom.int/ukraine.  PDF parsing across changing layouts is too
brittle for a one-week PoC, so V12.0 ships a calibrated synthetic series
derived from IOM's published headline numbers.  Replacing this with a
live PDF parser later only requires changing :meth:`fetch_raw`.

Calibration anchors (IOM DTM published headline IDP totals):
* Mar 2022:  ~6.5M (initial flight from east/south)
* Jul 2022:  ~7.7M peak
* Dec 2022:  ~5.4M (early returns; some areas re-occupied)
* Dec 2023:  ~3.7M
* Dec 2024:  ~3.5M
* End 2026:  ~3.0M (gentle decline as the front stabilises)

Pre-Feb-2022 the figure is 0 (the 1.5M Donbas IDP backlog from 2014-21 is
captured elsewhere; this loader specifically tracks **post-2022 invasion
displacement** as DTM defines it).

Returns / movement and the western-oblast share are derived from the same
DTM tables (Mobility Tracking and Area Baseline rounds).  Returns peak in
late-2022 and again Apr 2023 - Apr 2024.  Western-oblast share starts at
~38% in Mar 2022, drops as some IDPs spill to central oblasts then settles
around 28-30% by 2024.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from src.external_data import BaseSignalLoader, register_loader

log = logging.getLogger(__name__)

# Calibrated monthly anchors for total IDP population (millions of people).
# Linear-interpolated between anchors below; pre-Feb-2022 stays at 0 and
# post-Dec-2026 we extrapolate gently downward.
TOTAL_ANCHORS_M: list[tuple[str, float]] = [
    ("2022-02", 0.0),
    ("2022-03", 6.5),
    ("2022-04", 7.1),
    ("2022-05", 7.4),
    ("2022-06", 7.6),
    ("2022-07", 7.7),
    ("2022-08", 7.0),
    ("2022-09", 6.5),
    ("2022-10", 6.0),
    ("2022-11", 5.7),
    ("2022-12", 5.4),
    ("2023-03", 5.1),
    ("2023-06", 4.7),
    ("2023-09", 4.0),
    ("2023-12", 3.7),
    ("2024-03", 3.6),
    ("2024-06", 3.5),
    ("2024-09", 3.5),
    ("2024-12", 3.5),
    ("2025-06", 3.4),
    ("2025-12", 3.3),
    ("2026-06", 3.1),
    ("2026-12", 3.0),
    ("2027-12", 2.8),
]


# Returns (people moving back to their pre-displacement oblast) per month,
# in millions.  Two main pulses: late-2022 after Kharkiv/Kherson liberations
# and steady late-2023 → mid-2024 returns to safer rear oblasts.
RETURNS_ANCHORS_M: list[tuple[str, float]] = [
    ("2022-04", 0.0),
    ("2022-06", 0.05),
    ("2022-09", 0.30),
    ("2022-10", 0.45),
    ("2022-11", 0.40),
    ("2022-12", 0.30),
    ("2023-03", 0.18),
    ("2023-06", 0.15),
    ("2023-09", 0.20),
    ("2023-12", 0.18),
    ("2024-04", 0.15),
    ("2024-09", 0.10),
    ("2024-12", 0.08),
    ("2025-06", 0.06),
    ("2025-12", 0.05),
    ("2026-12", 0.04),
    ("2027-12", 0.03),
]


# Share of IDPs hosted in the seven western oblasts (Lviv, IF, Zakarpattia,
# Volyn, Rivne, Ternopil, Chernivtsi) — proxy for where displaced demand
# concentrates.  Based on IOM Area Baseline rounds.
WESTERN_PCT_ANCHORS: list[tuple[str, float]] = [
    ("2022-03", 38.0),
    ("2022-06", 35.0),
    ("2022-12", 32.0),
    ("2023-06", 30.0),
    ("2023-12", 29.0),
    ("2024-12", 28.0),
    ("2025-12", 28.0),
    ("2027-12", 28.0),
]


def _piecewise(periods: list[pd.Period], anchors: list[tuple[str, float]]) -> dict:
    """Linear-interpolate ``anchors`` over the requested periods.  Periods
    before the first anchor or after the last get the boundary value."""
    if not anchors:
        return {p: 0.0 for p in periods}
    pa = sorted(((pd.Period(s, freq="M"), v) for s, v in anchors), key=lambda x: x[0])
    out: dict[pd.Period, float] = {}
    first_p, first_v = pa[0]
    last_p, last_v = pa[-1]
    for p in periods:
        if p <= first_p:
            out[p] = first_v
            continue
        if p >= last_p:
            out[p] = last_v
            continue
        for (ap, av), (bp, bv) in zip(pa[:-1], pa[1:]):
            if ap <= p <= bp:
                if ap == bp:
                    out[p] = av
                else:
                    span = (bp - ap).n
                    pos = (p - ap).n
                    out[p] = av + (bv - av) * (pos / span)
                break
    return out


@register_loader
class IOMDisplacementLoader(BaseSignalLoader):
    """Calibrated synthetic IOM DTM IDP indicators."""

    name = "iom_idp"
    signal_cols = [
        "idp_total_count",
        "idp_returns_count_month",
        "idp_western_oblasts_pct",
    ]
    publication_lag_days = 30
    upstream_url = (
        "https://dtm.iom.int/ukraine "
        "(synthetic calibration — PDF parsing deferred to V12.1)"
    )
    cache_ttl_days = 60

    def _build_synthetic(self) -> pd.DataFrame:
        periods = list(pd.period_range("2019-01", "2027-12", freq="M"))
        invasion = pd.Period("2022-02", freq="M")

        total_m = _piecewise(periods, TOTAL_ANCHORS_M)
        returns_m = _piecewise(periods, RETURNS_ANCHORS_M)
        western_pct = _piecewise(periods, WESTERN_PCT_ANCHORS)

        rows: list[dict] = []
        for p in periods:
            if p < invasion:
                rows.append(
                    {
                        "Период": p,
                        "idp_total_count": 0,
                        "idp_returns_count_month": 0,
                        "idp_western_oblasts_pct": 0.0,
                    }
                )
                continue
            tot = max(0.0, total_m.get(p, 0.0)) * 1_000_000.0
            rets = max(0.0, returns_m.get(p, 0.0)) * 1_000_000.0
            # Western-oblast share is undefined when there are no IDPs at all;
            # report 0 in that case for downstream-friendly numerics.
            west = (
                max(0.0, min(100.0, western_pct.get(p, 0.0)))
                if tot > 0
                else 0.0
            )
            rows.append(
                {
                    "Период": p,
                    "idp_total_count": int(round(tot)),
                    "idp_returns_count_month": int(round(rets)),
                    "idp_western_oblasts_pct": float(round(west, 2)),
                }
            )
        return pd.DataFrame(rows)

    def fetch_raw(self) -> pd.DataFrame:
        log.info(
            "iom_idp: PDF parsing deferred to V12.1 — using calibrated synthetic",
        )
        try:
            return self._build_synthetic()
        except Exception as exc:  # noqa: BLE001
            log.warning("iom_idp synthetic build failed (%s); empty frame", exc)
            periods = pd.period_range("2019-01", "2027-12", freq="M")
            empty = pd.DataFrame({"Период": periods})
            for c in self.signal_cols:
                empty[c] = 0
            return empty

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        full = pd.period_range("2019-01", "2027-12", freq="M")
        out = pd.DataFrame({"Период": full})
        out = out.merge(raw, on="Период", how="left")

        out["idp_total_count"] = (
            pd.to_numeric(out.get("idp_total_count"), errors="coerce")
            .fillna(0)
            .astype(np.int64)
        )
        out["idp_returns_count_month"] = (
            pd.to_numeric(out.get("idp_returns_count_month"), errors="coerce")
            .fillna(0)
            .astype(np.int64)
        )
        out["idp_western_oblasts_pct"] = (
            pd.to_numeric(out.get("idp_western_oblasts_pct"), errors="coerce")
            .fillna(0.0)
            .astype(float)
        )

        out["Период"] = out["Период"].astype("period[M]")
        return out[["Период"] + self.signal_cols]
