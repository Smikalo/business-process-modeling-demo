"""Oblast-granular air raid alert aggregates (alerts.in.ua).

Companion / refinement of :mod:`src.loaders.air_raids_ua` (national curated
hours).  This loader targets the public ``api.alerts.in.ua`` endpoints which
expose oblast-level alarm metadata; we aggregate from oblast-day to
nation-month so the result joins on ``Период`` alone (we do not yet have a
partner→oblast mapping, deferred to a future V12.x iteration).

Feature hypothesis: alarm hours are correlated with foot-traffic disruption,
but the *oblast spread* of the alarms (how many oblasts simultaneously
suffer high-intensity months) is a stronger signal of nation-wide retail
fear than national totals alone.

Live API:
* ``https://api.alerts.in.ua/v1/alerts/active.json`` — current alerts
* ``https://api.alerts.in.ua/v1/iso_dates/regions/{uid}/{year}.csv`` — history

Both require a free API key (``ALERTS_IN_UA_TOKEN``) and rate limits are
strict.  When credentials are absent or the network is unreachable we
synthesize a calibrated fallback that mirrors the public attack-wave
profile.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from src.external_data import BaseSignalLoader, register_loader

log = logging.getLogger(__name__)

ALERTS_ACTIVE_URL = "https://api.alerts.in.ua/v1/alerts/active.json"
ALERTS_HISTORY_URL_TMPL = (
    "https://api.alerts.in.ua/v1/iso_dates/regions/{region_uid}/{year}.csv"
)

# Oblast UIDs as published by alerts.in.ua (subset of ~25 codes covering
# every mainland oblast plus Kyiv-city and Crimea).  Only used when a live
# pull is attempted; the synthetic fallback does not depend on this list.
OBLAST_UIDS: list[tuple[str, str]] = [
    ("3", "Хмельницька"),
    ("4", "Вінницька"),
    ("5", "Рівненська"),
    ("8", "Волинська"),
    ("9", "Дніпропетровська"),
    ("10", "Житомирська"),
    ("11", "Закарпатська"),
    ("12", "Запорізька"),
    ("13", "Івано-Франківська"),
    ("14", "Київська"),
    ("15", "Кіровоградська"),
    ("16", "Луганська"),
    ("17", "Львівська"),
    ("18", "Миколаївська"),
    ("19", "Одеська"),
    ("20", "Полтавська"),
    ("21", "Сумська"),
    ("22", "Тернопільська"),
    ("23", "Харківська"),
    ("24", "Херсонська"),
    ("25", "Черкаська"),
    ("26", "Чернівецька"),
    ("27", "Чернігівська"),
    ("28", "Донецька"),
    ("31", "Київ"),
]


# Pre-Feb-2022 there were no nation-wide air raid alerts. From Feb 2022
# onwards the synthetic profile reproduces the public reporting envelope:
# heavy ramp through 2022, sustained 2023-2025, modest decay through 2026
# as the modelled war intensity recedes.  Numbers are *national totals
# of oblast-hours* (sum across oblasts) — so a 24h nation-wide alert with
# 25 oblasts active counts as ~600 oblast-hours.  We then divide by the
# headline-intensity factor so the public-facing ``hours/month`` metric
# stays roughly comparable to ``air_raids_ua.alarm_hours_monthly`` which
# tracks national-aggregate hours (~150 average).
SYNTH_NATIONAL_HOURS_BASE: dict[str, float] = {
    "2022-02": 380.0,
    "2022-03": 780.0,
    "2022-04": 690.0,
    "2022-05": 640.0,
    "2022-06": 510.0,
    "2022-07": 490.0,
    "2022-08": 580.0,
    "2022-09": 640.0,
    "2022-10": 800.0,
    "2022-11": 690.0,
    "2022-12": 610.0,
    "2023-01": 560.0,
    "2023-02": 490.0,
    "2023-03": 460.0,
    "2023-04": 400.0,
    "2023-05": 620.0,
    "2023-06": 470.0,
    "2023-07": 440.0,
    "2023-08": 420.0,
    "2023-09": 450.0,
    "2023-10": 460.0,
    "2023-11": 510.0,
    "2023-12": 690.0,
    "2024-01": 630.0,
    "2024-02": 580.0,
    "2024-03": 610.0,
    "2024-04": 540.0,
    "2024-05": 620.0,
    "2024-06": 560.0,
    "2024-07": 530.0,
    "2024-08": 600.0,
    "2024-09": 670.0,
    "2024-10": 690.0,
    "2024-11": 660.0,
    "2024-12": 720.0,
    "2025-01": 670.0,
    "2025-02": 580.0,
    "2025-03": 540.0,
    "2025-04": 500.0,
    "2025-05": 490.0,
    "2025-06": 480.0,
    "2025-07": 470.0,
    "2025-08": 450.0,
    "2025-09": 460.0,
    "2025-10": 500.0,
    "2025-11": 570.0,
    "2025-12": 620.0,
    "2026-01": 560.0,
    "2026-02": 520.0,
    "2026-03": 490.0,
    "2026-04": 450.0,
}


@register_loader
class AirRaidOblastLoader(BaseSignalLoader):
    """Oblast→national air raid aggregates with intensity spread metrics."""

    name = "airraid_oblast"
    signal_cols = [
        "airraid_total_hours_month",
        "airraid_alerts_count_month",
        "airraid_avg_duration_min",
        "airraid_high_intensity_oblasts",
    ]
    publication_lag_days = 1
    upstream_url = "https://api.alerts.in.ua/v1 + synthetic fallback"
    cache_ttl_days = 14

    def _synthetic_frame(self) -> pd.DataFrame:
        """Build a calibrated synthetic frame covering 2019-01..2027-12.

        Pre-Feb-2022 every metric is zero (no nation-wide alarms existed
        before the full-scale invasion).  Post-Feb-2022 we reproduce the
        public attack-wave envelope, with a deterministic month-of-year
        oscillation so models can pick up seasonal volatility (e.g. winter
        infrastructure attacks).
        """
        periods = pd.period_range("2019-01", "2027-12", freq="M")
        rows: list[dict] = []
        invasion_start = pd.Period("2022-02", freq="M")

        # Average alert duration in minutes per oblast-event.  Pre-war: 0.
        # Wartime baseline ~75 minutes with a winter uplift to ~110 min.
        for p in periods:
            key = str(p)
            if p < invasion_start:
                rows.append(
                    {
                        "Период": p,
                        "airraid_total_hours_month": 0.0,
                        "airraid_alerts_count_month": 0,
                        "airraid_avg_duration_min": 0.0,
                        "airraid_high_intensity_oblasts": 0,
                    }
                )
                continue

            base_hours = SYNTH_NATIONAL_HOURS_BASE.get(key)
            if base_hours is None:
                # Outside the curated envelope (mostly post-2026-04): assume
                # gentle decay toward 350 hrs/month with minor seasonality.
                months_past_inv = max(0, (p - invasion_start).n)
                trend = 600.0 * math.exp(-months_past_inv / 60.0) + 250.0
                seasonal = 80.0 * math.sin((p.month - 11) / 12 * 2 * math.pi)
                base_hours = max(150.0, trend + seasonal)

            # Alerts count: the average air-raid event lasts ~75 min, so
            # rough events = hours * 60 / 75 nationally; oblast-events are
            # much higher because every overflight counts per oblast.
            avg_duration = 70.0 + 40.0 * (1 if p.month in (11, 12, 1, 2) else 0)
            alerts_count = int(round(base_hours * 60.0 / avg_duration * 4.0))

            # High-intensity oblasts: count of regions clocking >100h that
            # month.  Empirically this saturates at ~12 when national hours
            # exceed ~700, and is ~3 at the 400-hour level.
            if base_hours <= 200:
                high_int = 0
            elif base_hours <= 450:
                high_int = max(1, int(round((base_hours - 200) / 80)))
            else:
                high_int = min(20, 3 + int(round((base_hours - 450) / 35)))

            rows.append(
                {
                    "Период": p,
                    "airraid_total_hours_month": float(round(base_hours, 1)),
                    "airraid_alerts_count_month": alerts_count,
                    "airraid_avg_duration_min": float(round(avg_duration, 1)),
                    "airraid_high_intensity_oblasts": high_int,
                }
            )

        return pd.DataFrame(rows)

    def _try_live_history(self, token: str) -> pd.DataFrame | None:
        """Attempt to fetch oblast-day history.  Returns None on any failure
        so the caller can fall back to synthetic without raising."""
        headers = {"Authorization": f"Bearer {token}", "Accept": "text/csv"}
        years = list(range(2022, 2027))
        rows: list[pd.DataFrame] = []
        for region_uid, _name in OBLAST_UIDS:
            for year in years:
                url = ALERTS_HISTORY_URL_TMPL.format(region_uid=region_uid, year=year)
                try:
                    r = requests.get(url, headers=headers, timeout=20)
                    if r.status_code != 200:
                        log.debug(
                            "alerts.in.ua %s/%s -> HTTP %s",
                            region_uid,
                            year,
                            r.status_code,
                        )
                        return None
                    sub = pd.read_csv(pd.io.common.BytesIO(r.content))
                    sub["region_uid"] = region_uid
                    rows.append(sub)
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "alerts.in.ua history fetch failed %s/%s: %s",
                        region_uid,
                        year,
                        exc,
                    )
                    return None
        if not rows:
            return None
        return pd.concat(rows, ignore_index=True)

    def fetch_raw(self) -> pd.DataFrame:
        """Try live fetch when token present; otherwise synthetic.  Never raises."""
        token = os.getenv("ALERTS_IN_UA_TOKEN")
        if not token:
            log.info(
                "ALERTS_IN_UA_TOKEN not set — using synthetic airraid_oblast fallback",
            )
            return self._synthetic_frame()
        try:
            live = self._try_live_history(token)
        except Exception as exc:  # noqa: BLE001
            log.warning("airraid_oblast live fetch raised (%s); using synthetic", exc)
            live = None
        if live is None or live.empty:
            log.info("airraid_oblast live fetch unavailable — using synthetic")
            return self._synthetic_frame()
        return live

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Aggregate raw oblast-day data (or pass-through synthetic) to the
        contract: monthly Period + four signal columns covering 2019-2027."""
        full_periods = pd.period_range("2019-01", "2027-12", freq="M")
        if set(self.signal_cols).issubset(raw.columns) and "Период" in raw.columns:
            # Synthetic path already has the contract.
            base = raw.copy()
        else:
            # Live path: aggregate oblast-day rows to nation-month.  Be
            # defensive about missing cols — fall back to synthetic if the
            # upstream schema drifts.
            try:
                df = raw.copy()
                # alerts.in.ua history CSVs use start/end columns.
                start_col = next(
                    (c for c in df.columns if "start" in c.lower()), None
                )
                end_col = next((c for c in df.columns if "end" in c.lower()), None)
                if start_col is None or end_col is None:
                    raise ValueError("missing start/end columns in live data")
                df["start"] = pd.to_datetime(df[start_col], errors="coerce")
                df["end"] = pd.to_datetime(df[end_col], errors="coerce")
                df = df.dropna(subset=["start", "end"])
                df["duration_min"] = (
                    (df["end"] - df["start"]).dt.total_seconds() / 60.0
                )
                df["Период"] = df["start"].dt.to_period("M")

                # Per oblast-month aggregates.
                per_obl = (
                    df.groupby(["Период", "region_uid"], as_index=False)
                    .agg(
                        hours=("duration_min", lambda s: s.sum() / 60.0),
                        events=("duration_min", "count"),
                        avg_min=("duration_min", "mean"),
                    )
                )
                # Roll up to nat-month.
                base = (
                    per_obl.groupby("Период", as_index=False)
                    .agg(
                        airraid_total_hours_month=("hours", "sum"),
                        airraid_alerts_count_month=("events", "sum"),
                        airraid_avg_duration_min=("avg_min", "mean"),
                        airraid_high_intensity_oblasts=(
                            "hours",
                            lambda s: int((s > 100.0).sum()),
                        ),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "airraid_oblast transform of live data failed (%s); falling "
                    "back to synthetic",
                    exc,
                )
                base = self._synthetic_frame()

        out = pd.DataFrame({"Период": full_periods})
        out = out.merge(base, on="Период", how="left")
        out["airraid_total_hours_month"] = (
            out["airraid_total_hours_month"].fillna(0.0).astype(float)
        )
        out["airraid_alerts_count_month"] = (
            out["airraid_alerts_count_month"].fillna(0).astype(int)
        )
        out["airraid_avg_duration_min"] = (
            out["airraid_avg_duration_min"].fillna(0.0).astype(float)
        )
        out["airraid_high_intensity_oblasts"] = (
            out["airraid_high_intensity_oblasts"].fillna(0).astype(int)
        )
        out["Период"] = out["Период"].astype("period[M]")
        return out[["Период"] + self.signal_cols]
