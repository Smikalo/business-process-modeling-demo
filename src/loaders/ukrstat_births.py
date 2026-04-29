"""Ukrstat live-births loader (`EXT_UKRSTAT_BIRTHS`).

Source: Ukrainian State Statistics Service — monthly live births by oblast
(`https://ukrstat.gov.ua/operativ/operativ2024/ds/kn/...`).

**Simplification (deliberate, documented):** the upstream feed is
oblast × month, but this V12 wave does not yet have a partner→oblast
mapping wired through the join keys.  We therefore aggregate to the
**national monthly** level (`Период` only) so the signal is at least
joinable to the existing ABT.  When the partner-oblast mapping lands we
will revisit and emit oblast-level rows by adding `Регіон` to
``join_keys``.

The ukrstat HTML / Excel publication cadence is brittle and, post-2022,
many tables are gated behind martial-law access controls.  We therefore
ship a curated synthetic series anchored on publicly-reported UA birth
statistics:

* 2019: ~25 k births / month nationally (~308 k / yr).
* Pre-war declining ~5 % YoY.
* Mar 2022 onset: drop to ~18 k / month, accelerating decline through
  2023 (war, displacement, delayed family formation).
* By 2025: ~15 k / month, slowly stabilising.

`births_per_1000` annualises the monthly count against the population
estimate (~41 M pre-war declining to ~33 M post-war reflecting both
emigration and territorial loss).

The fallback is tagged ``synthetic: true`` in the cache meta JSON.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from src.external_data import BaseSignalLoader, LoaderMetadata, register_loader

log = logging.getLogger(__name__)

UKRSTAT_BIRTHS_URL = (
    "https://ukrstat.gov.ua/operativ/operativ2024/ds/kn/kn_u/kn_u_2024.html"
)

PERIOD_START = "2019-01"
PERIOD_END = "2027-12"

# Light seasonality: more births in summer-autumn, fewer in deep winter.
# Multiplicative around the annual mean.
_SEASONAL = {
    1: 0.95, 2: 0.93, 3: 1.00, 4: 1.00, 5: 1.02, 6: 1.04,
    7: 1.07, 8: 1.07, 9: 1.06, 10: 1.02, 11: 0.95, 12: 0.92,
}


def _annual_mean_births(year: int) -> float:
    """Best public estimate of monthly mean births nationally."""
    if year <= 2019:
        return 25_700.0  # 308k / 12
    if year == 2020:
        return 24_400.0  # ~293k
    if year == 2021:
        return 22_500.0  # ~270k (continuing decline + COVID)
    if year == 2022:
        return 18_300.0  # invasion year, partial-year shock
    if year == 2023:
        return 15_500.0
    if year == 2024:
        return 14_500.0
    if year == 2025:
        return 14_000.0
    # Forward-project: gentle decline asymptote.
    return max(13_000.0, 14_000.0 * (0.98 ** (year - 2025)))


def _annual_pop_estimate(year: int) -> float:
    """Approx Ukraine resident population (ex-occupied territory)."""
    if year <= 2019:
        return 41_900_000.0
    if year == 2020:
        return 41_500_000.0
    if year == 2021:
        return 41_200_000.0
    if year == 2022:
        return 36_500_000.0  # mass emigration
    if year == 2023:
        return 33_500_000.0
    if year == 2024:
        return 33_200_000.0
    if year == 2025:
        return 33_000_000.0
    return 33_000_000.0  # projected stable


def _synthetic_frame() -> pd.DataFrame:
    periods = pd.period_range(PERIOD_START, PERIOD_END, freq="M")
    rows = []
    for p in periods:
        annual_mean = _annual_mean_births(p.year)
        births = annual_mean * _SEASONAL[p.month]
        # Add a small smooth trend within the year so YoY changes look natural.
        sub_year_drift = 1.0 - 0.005 * (p.month - 6)  # ±2.5 % across year
        births *= sub_year_drift
        rows.append({
            "Период": p,
            "births_total": float(np.round(births, 0)),
            "_pop": _annual_pop_estimate(p.year),
        })
    return pd.DataFrame(rows)


@register_loader
class UkrstatBirthsLoader(BaseSignalLoader):
    name = "ukrstat_births"
    signal_cols = [
        "births_total",
        "births_yoy_pct",
        "births_per_1000",
    ]
    # Aggregated to national; partner-oblast mapping deferred to a later wave.
    join_keys = ["Период"]
    publication_lag_days = 60
    upstream_url = UKRSTAT_BIRTHS_URL
    cache_ttl_days = 30

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._used_synthetic = False

    def fetch_raw(self) -> pd.DataFrame:
        try:
            resp = requests.get(UKRSTAT_BIRTHS_URL, timeout=10)
            if resp.status_code != 200 or len(resp.content) < 1000:
                raise RuntimeError(f"ukrstat births HTTP {resp.status_code}")
            # Even when the page returns 200, post-2022 it is frequently a
            # login-walled stub.  We do not trust the parser to extract a
            # clean monthly oblast×month table; bail to synthetic.
            raise RuntimeError(
                "ukrstat_births: post-2022 page schema not stable; using synthetic"
            )
        except Exception as exc:  # noqa: BLE001
            log.info(
                "ukrstat_births scrape skipped (%s) — using synthetic frame", exc
            )
            self._used_synthetic = True
            return _synthetic_frame()

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        if "Период" not in df.columns or "births_total" not in df.columns:
            df = _synthetic_frame()
            self._used_synthetic = True

        full = pd.period_range(PERIOD_START, PERIOD_END, freq="M")
        out = pd.DataFrame({"Период": full}).merge(df, on="Период", how="left")
        out["births_total"] = pd.to_numeric(out["births_total"], errors="coerce")
        out["births_total"] = out["births_total"].ffill().bfill()

        # Population: rebuild from annual lookup if missing.
        if "_pop" not in out.columns or out["_pop"].isna().all():
            out["_pop"] = out["Период"].apply(lambda p: _annual_pop_estimate(p.year))
        out["_pop"] = pd.to_numeric(out["_pop"], errors="coerce").ffill().bfill()

        out = out.sort_values("Период").reset_index(drop=True)
        out["births_yoy_pct"] = out["births_total"].pct_change(12) * 100.0
        # Annualised crude birth rate per 1 000 inhabitants.
        out["births_per_1000"] = out["births_total"] * 12.0 / out["_pop"] * 1000.0

        out["Период"] = out["Период"].astype("period[M]")
        for c in self.signal_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out[["Период"] + self.signal_cols]

    def _write_cache(self, df: pd.DataFrame) -> None:
        df.to_parquet(self.cache_path, index=False)
        meta = LoaderMetadata(
            source_name=self.name,
            fetch_timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            upstream_url=self.upstream_url,
            row_count=len(df),
            date_range_min=str(df["Период"].min()),
            date_range_max=str(df["Период"].max()),
            schema_hash=self._schema_hash(),
            signal_cols=list(self.signal_cols),
            publication_lag_days=self.publication_lag_days,
        ).to_dict()
        meta["synthetic"] = bool(self._used_synthetic)
        self.meta_path.write_text(json.dumps(meta, indent=2))
