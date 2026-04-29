"""DTEK / Ukrenergo blackout schedule signals (synthetic V12.0).

Source intent: scrape the daily / weekly blackout schedules published by
DTEK (https://dtek-kem.com.ua/) and Ukrenergo (https://ua.energy/).  Both
publish ad-hoc HTML/PNG schedule artefacts that change every season; a
robust scrape requires per-page selectors that drift with the front-end
redesigns, so the V12.0 implementation **defers the live scrape to V12.1**
and ships a calibrated synthetic profile based on public reporting and
reputable news summaries (Reuters, Suspilne, Ukrenergo press releases).

Calibration sources:
- Oct 2022 - Mar 2023 wave: peak ~12 hrs/day rolling-average outage in
  Dec 2022 after the strikes on the high-voltage grid.  Roughly 60-70%
  of the population was affected at the peak per Ukrenergo (Dec 2022).
- Oct 2024 - Mar 2025 wave: peak ~8 hrs/day after the second-wave strikes
  on substations and gas storage.  Population affected ~50%.
- Oct 2025 - Mar 2026 wave (forecast): smaller, peak ~5 hrs/day, ~30%
  population affected, reflecting partial recovery + air-defence uplift.

When fetch_raw is invoked we always return the synthetic frame so the
loader degrades gracefully and never raises.  Replacing this with a live
scrape later only requires changing :meth:`fetch_raw`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.external_data import BaseSignalLoader, register_loader

log = logging.getLogger(__name__)

# (period_str, peak_avg_hours_per_day, peak_pct_population_affected,
# peak_severity_index)
WAVE_CALIBRATION: list[dict] = [
    {
        "label": "wave_22_23",
        "ramp_start": "2022-10",
        "peak": "2022-12",
        "decay_end": "2023-03",
        "peak_hours": 12.0,
        "ramp_start_hours": 2.0,
        "decay_end_hours": 4.0,
        "peak_pct_pop": 65.0,
        "peak_severity": 9.0,
    },
    {
        "label": "wave_24_25",
        "ramp_start": "2024-10",
        "peak": "2024-12",
        "decay_end": "2025-03",
        "peak_hours": 8.0,
        "ramp_start_hours": 1.5,
        "decay_end_hours": 2.5,
        "peak_pct_pop": 50.0,
        "peak_severity": 7.0,
    },
    {
        "label": "wave_25_26",
        "ramp_start": "2025-10",
        "peak": "2025-12",
        "decay_end": "2026-03",
        "peak_hours": 5.0,
        "ramp_start_hours": 1.0,
        "decay_end_hours": 1.5,
        "peak_pct_pop": 30.0,
        "peak_severity": 5.0,
    },
]


def _interp(periods: list[pd.Period], anchors: list[tuple[pd.Period, float]]) -> dict:
    """Piecewise-linear interpolate ``anchors`` over the given periods.
    Periods outside the anchor span return None (caller fills with 0).
    """
    if not anchors:
        return {}
    anchors = sorted(anchors, key=lambda x: x[0])
    result: dict[pd.Period, float] = {}
    first_p, _ = anchors[0]
    last_p, _ = anchors[-1]
    for p in periods:
        if p < first_p or p > last_p:
            continue
        # Find bracket [a, b] s.t. a.p <= p <= b.p
        for (ap, av), (bp, bv) in zip(anchors[:-1], anchors[1:]):
            if ap <= p <= bp:
                if ap == bp:
                    result[p] = av
                else:
                    span = (bp - ap).n
                    pos = (p - ap).n
                    result[p] = av + (bv - av) * (pos / span)
                break
    return result


@register_loader
class BlackoutDTEKLoader(BaseSignalLoader):
    """Synthetic monthly DTEK/Ukrenergo blackout intensity signals.

    Live scrape deferred to V12.1; using calibrated synthetic based on
    public reports.
    """

    name = "blackout_dtek"
    signal_cols = [
        "blackout_avg_hours_per_day",
        "blackout_pct_population_affected",
        "blackout_severity_index",
    ]
    publication_lag_days = 7
    upstream_url = (
        "https://dtek-kem.com.ua + https://ua.energy "
        "(synthetic — live scrape deferred to V12.1)"
    )
    cache_ttl_days = 30

    def _build_synthetic(self) -> pd.DataFrame:
        periods = list(pd.period_range("2019-01", "2027-12", freq="M"))
        per_index = {p: i for i, p in enumerate(periods)}

        hours = np.zeros(len(periods), dtype=float)
        pct = np.zeros(len(periods), dtype=float)
        sev = np.zeros(len(periods), dtype=float)

        for w in WAVE_CALIBRATION:
            ramp_p = pd.Period(w["ramp_start"], freq="M")
            peak_p = pd.Period(w["peak"], freq="M")
            decay_p = pd.Period(w["decay_end"], freq="M")

            anchors_hours = [
                (ramp_p, w["ramp_start_hours"]),
                (peak_p, w["peak_hours"]),
                (decay_p, w["decay_end_hours"]),
            ]
            anchors_pct = [
                (ramp_p, w["peak_pct_pop"] * w["ramp_start_hours"] / w["peak_hours"]),
                (peak_p, w["peak_pct_pop"]),
                (
                    decay_p,
                    w["peak_pct_pop"] * w["decay_end_hours"] / w["peak_hours"],
                ),
            ]
            anchors_sev = [
                (ramp_p, w["peak_severity"] * w["ramp_start_hours"] / w["peak_hours"]),
                (peak_p, w["peak_severity"]),
                (
                    decay_p,
                    w["peak_severity"] * w["decay_end_hours"] / w["peak_hours"],
                ),
            ]

            for p, v in _interp(periods, anchors_hours).items():
                idx = per_index[p]
                hours[idx] = max(hours[idx], v)
            for p, v in _interp(periods, anchors_pct).items():
                idx = per_index[p]
                pct[idx] = max(pct[idx], v)
            for p, v in _interp(periods, anchors_sev).items():
                idx = per_index[p]
                sev[idx] = max(sev[idx], v)

        out = pd.DataFrame(
            {
                "Период": periods,
                "blackout_avg_hours_per_day": np.round(hours, 2),
                "blackout_pct_population_affected": np.round(pct, 1),
                "blackout_severity_index": np.round(np.clip(sev, 0, 10), 2),
            }
        )
        out["Период"] = out["Период"].astype("period[M]")
        return out

    def fetch_raw(self) -> pd.DataFrame:
        log.info(
            "blackout_dtek: live scrape deferred to V12.1 — using calibrated synthetic",
        )
        try:
            return self._build_synthetic()
        except Exception as exc:  # noqa: BLE001
            log.warning("blackout_dtek synthetic build failed (%s); empty frame", exc)
            periods = pd.period_range("2019-01", "2027-12", freq="M")
            empty = pd.DataFrame({"Период": periods})
            for c in self.signal_cols:
                empty[c] = 0.0
            return empty

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        full = pd.period_range("2019-01", "2027-12", freq="M")
        out = pd.DataFrame({"Период": full})
        out = out.merge(raw, on="Период", how="left")
        for c in self.signal_cols:
            if c not in out.columns:
                out[c] = 0.0
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        out["Период"] = out["Период"].astype("period[M]")
        return out[["Период"] + self.signal_cols]
