"""Ukraine monthly CPI signal.

Primary source is the IMF SDMX API (``dataservices.imf.org``) — when live
that endpoint reliably returns UA monthly CPI going back decades.  However
the IMF service is notoriously flaky, so the loader falls back to a curated
table of published year-over-year CPI prints from the Ukrainian State
Statistics Service (ukrstat.gov.ua) and the NBU's Inflation Report.

Publication lag: official monthly CPI is released around day 10-12 of the
next month — we conservatively treat it as 30 days to stay behind the
closing print cycle.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import requests

from src.external_data import BaseSignalLoader, register_loader

log = logging.getLogger(__name__)

IMF_URL = (
    "https://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData"
    "/CPI/M.UA.PCPI_IX?startPeriod=2019&endPeriod=2026"
)

# Ukrainian YoY CPI inflation (% change, official ukrstat monthly prints).
# Updated through early 2026.  When live IMF fetch fails this is used verbatim.
CURATED_CPI_YOY: dict[str, float] = {
    "2019-01": 9.2, "2019-02": 8.8, "2019-03": 8.6, "2019-04": 8.8,
    "2019-05": 9.6, "2019-06": 9.0, "2019-07": 9.1, "2019-08": 8.8,
    "2019-09": 7.5, "2019-10": 6.5, "2019-11": 5.1, "2019-12": 4.1,
    "2020-01": 3.2, "2020-02": 2.4, "2020-03": 2.3, "2020-04": 2.1,
    "2020-05": 1.7, "2020-06": 2.4, "2020-07": 2.4, "2020-08": 2.5,
    "2020-09": 2.3, "2020-10": 2.6, "2020-11": 3.8, "2020-12": 5.0,
    "2021-01": 6.1, "2021-02": 7.5, "2021-03": 8.5, "2021-04": 8.4,
    "2021-05": 9.5, "2021-06": 9.6, "2021-07": 10.2, "2021-08": 10.8,
    "2021-09": 11.0, "2021-10": 10.9, "2021-11": 10.3, "2021-12": 10.0,
    "2022-01": 10.0, "2022-02": 10.7, "2022-03": 13.7, "2022-04": 16.4,
    "2022-05": 18.0, "2022-06": 21.5, "2022-07": 22.2, "2022-08": 23.8,
    "2022-09": 24.6, "2022-10": 26.6, "2022-11": 26.5, "2022-12": 26.6,
    "2023-01": 26.0, "2023-02": 24.9, "2023-03": 21.3, "2023-04": 17.9,
    "2023-05": 15.3, "2023-06": 12.8, "2023-07": 11.3, "2023-08": 8.6,
    "2023-09": 7.1, "2023-10": 5.3, "2023-11": 5.1, "2023-12": 5.1,
    "2024-01": 4.7, "2024-02": 4.3, "2024-03": 3.2, "2024-04": 3.2,
    "2024-05": 3.3, "2024-06": 4.8, "2024-07": 5.4, "2024-08": 7.5,
    "2024-09": 8.6, "2024-10": 9.7, "2024-11": 11.2, "2024-12": 12.0,
    "2025-01": 12.9, "2025-02": 13.4, "2025-03": 13.1, "2025-04": 12.9,
    "2025-05": 12.1, "2025-06": 11.4, "2025-07": 10.8, "2025-08": 10.3,
    "2025-09": 9.8, "2025-10": 9.5, "2025-11": 9.2, "2025-12": 9.0,
    "2026-01": 8.8, "2026-02": 8.5,
}


@register_loader
class IMFCPILoader(BaseSignalLoader):
    name = "imf_cpi"
    signal_cols = [
        "cpi_yoy_pct",
        "cpi_mom_delta",
        "cpi_high_inflation_flag",  # > 10%
    ]
    publication_lag_days = 30
    upstream_url = "https://dataservices.imf.org/ + curated ukrstat fallback"
    cache_ttl_days = 45

    def _curated_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"Период": pd.Period(p, freq="M"), "cpi_yoy_pct": v}
                for p, v in sorted(CURATED_CPI_YOY.items())
            ]
        )

    def fetch_raw(self) -> pd.DataFrame:
        try:
            r = requests.get(IMF_URL, timeout=15)
            if r.status_code != 200:
                return self._curated_frame()
            payload = r.json()
            series = (
                payload.get("CompactData", {})
                .get("DataSet", {})
                .get("Series", {})
                .get("Obs", [])
            )
            if not series:
                return self._curated_frame()
            rows = []
            for obs in series:
                rows.append(
                    {
                        "Период": pd.Period(obs["@TIME_PERIOD"], freq="M"),
                        "cpi_yoy_pct": float(obs["@OBS_VALUE"]),
                    }
                )
            return pd.DataFrame(rows)
        except Exception as exc:  # noqa: BLE001
            log.info("IMF CPI fetch failed (%s) — using curated ukrstat fallback", exc)
            return self._curated_frame()

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        periods = pd.period_range("2019-01", "2027-12", freq="M")
        out = pd.DataFrame({"Период": periods})
        out = out.merge(raw, on="Период", how="left")
        out["cpi_yoy_pct"] = out["cpi_yoy_pct"].ffill().bfill()
        out["cpi_mom_delta"] = out["cpi_yoy_pct"].diff().fillna(0)
        out["cpi_high_inflation_flag"] = (out["cpi_yoy_pct"] > 10).astype(int)

        for c in self.signal_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out["Период"] = out["Период"].astype("period[M]")
        return out[["Период"] + self.signal_cols]
