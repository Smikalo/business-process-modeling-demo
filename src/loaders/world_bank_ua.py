"""World Bank annual indicators for Ukraine (forward-filled to monthly grain).

Free public API, no auth, JSON format.  Indicators fetched:
- SP.POP.0014.TO: population ages 0-14 (target buyer universe)
- SP.DYN.TFRT.IN: total fertility rate (leading birth-cohort indicator)
- NY.GDP.MKTP.KD.ZG: real GDP growth (macro)
- FP.CPI.TOTL.ZG: annual CPI inflation

Annual release cadence → publication_lag_days = 365 for conservative safety.
We forward-fill the last observed year across 12 months, so the April 2024
forecast sees 2023 data at the earliest.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import requests

from src.external_data import BaseSignalLoader, register_loader

log = logging.getLogger(__name__)

WB_URL_FMT = (
    "https://api.worldbank.org/v2/country/UA/indicator/{ind}"
    "?format=json&date=2015:2026&per_page=60"
)

INDICATORS = {
    "wb_pop_0_14": "SP.POP.0014.TO",
    "wb_fertility": "SP.DYN.TFRT.IN",
    "wb_gdp_growth": "NY.GDP.MKTP.KD.ZG",
    "wb_cpi_inflation": "FP.CPI.TOTL.ZG",
}


@register_loader
class WorldBankUALoader(BaseSignalLoader):
    name = "world_bank_ua"
    signal_cols = list(INDICATORS.keys())
    publication_lag_days = 365
    upstream_url = "https://api.worldbank.org/v2"
    cache_ttl_days = 180

    def fetch_raw(self) -> pd.DataFrame:
        rows: list[dict] = []
        for col, ind in INDICATORS.items():
            try:
                r = requests.get(WB_URL_FMT.format(ind=ind), timeout=30)
                r.raise_for_status()
                payload = r.json()
                if not payload or len(payload) < 2:
                    log.warning("World Bank empty response for %s", ind)
                    continue
                data = payload[1] or []
                for rec in data:
                    if rec.get("value") is None:
                        continue
                    rows.append(
                        {
                            "indicator": col,
                            "year": int(rec["date"]),
                            "value": float(rec["value"]),
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning("World Bank fetch failed for %s: %s", ind, exc)
        if not rows:
            raise RuntimeError("World Bank returned no data")
        return pd.DataFrame(rows)

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        pivot = raw.pivot(index="year", columns="indicator", values="value")
        pivot = pivot.sort_index()
        pivot.index = pivot.index.astype(int)

        # Forward-fill to monthly grain.
        periods = pd.period_range("2019-01", "2027-12", freq="M")
        out = pd.DataFrame({"Период": periods})
        out["year"] = out["Период"].apply(lambda p: p.year)
        out = out.merge(pivot.reset_index(), on="year", how="left")
        out = out.drop(columns=["year"])
        # Forward-fill across missing years (current year has no entry yet).
        out[self.signal_cols] = out[self.signal_cols].ffill()

        for c in self.signal_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out["Период"] = out["Период"].astype("period[M]")
        return out[["Период"] + self.signal_cols]
