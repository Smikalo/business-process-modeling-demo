"""Ukrstat Industrial Production Index loader (`EXT_UKRSTAT_INDPROD`).

Source: Ukrainian State Statistics Service — monthly Industrial Production
Index (IPI), index 2010 = 100 historically but published as YoY % changes
since 2022.  Page lives at
``https://ukrstat.gov.ua/operativ/operativ2024/pr/pr_u/`` .

Same brittleness story as the other Ukrstat loaders: HTML scraping is not
production-grade, so we ship a curated synthetic series with realistic
contours:

* 2019 baseline: index ≈ 100.
* 2020: COVID dip ~-5 % then partial recovery.
* 2021: above pre-COVID trend.
* Mar 2022: war shock −40 %.
* 2023-24: slow recovery.
* 2025: ~75 % of 2019 baseline.

The synthetic fallback is tagged ``synthetic: true`` in the cache meta
JSON so the A/B audit can flag this signal as ungrounded if needed.
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

UKRSTAT_INDPROD_URL = (
    "https://ukrstat.gov.ua/operativ/operativ2024/pr/pr_u/pr_u_2024.html"
)

PERIOD_START = "2019-01"
PERIOD_END = "2027-12"

# Light multiplicative seasonality: industrial output dips in summer
# (vacations / maintenance shutdowns) and peaks in autumn.
_SEASONAL = {
    1: 0.96, 2: 0.97, 3: 1.02, 4: 1.01, 5: 1.00, 6: 0.99,
    7: 0.94, 8: 0.92, 9: 1.02, 10: 1.05, 11: 1.06, 12: 1.06,
}


def _baseline_level(period: pd.Period) -> float:
    """Smooth pre-/post-war trend in the IPI level."""
    invasion = pd.Period("2022-02", freq="M")
    if period < pd.Period("2020-01", freq="M"):
        # 2019: ~100 nominal.
        years_since_2019 = period.year - 2019 + (period.month - 1) / 12.0
        return 100.0 * (1.0 + 0.01 * years_since_2019)
    if period < invasion:
        # 2020 COVID: ~-5 % trough mid-year, rebound.
        if period.year == 2020:
            t = (period.month - 1) / 11.0
            return 100.0 * (0.95 + 0.05 * t)  # 95 → 100
        if period.year == 2021:
            t = (period.month - 1) / 11.0
            return 100.0 * (1.00 + 0.04 * t)  # 100 → 104
        # Jan-Feb 2022 hold at 104.
        return 104.0
    # War + recovery curve (months since invasion).
    months_since = (period - invasion).n
    if months_since == 0:
        return 0.85 * 104.0
    if months_since == 1:
        return 0.60 * 104.0  # Mar 2022 trough −40 %.
    asymptote = 0.78 * 104.0  # ~75-80 % long-run.
    trough = 0.60 * 104.0
    return min(asymptote, trough + (asymptote - trough) * (1 - np.exp(-(months_since - 1) / 24.0)))


def _synthetic_frame() -> pd.DataFrame:
    periods = pd.period_range(PERIOD_START, PERIOD_END, freq="M")
    rows = []
    for p in periods:
        rows.append({
            "Период": p,
            "indprod_idx": float(np.round(_baseline_level(p) * _SEASONAL[p.month], 2)),
        })
    return pd.DataFrame(rows)


@register_loader
class UkrstatIndProdLoader(BaseSignalLoader):
    name = "ukrstat_indprod"
    signal_cols = [
        "indprod_idx",
        "indprod_idx_yoy_pct",
    ]
    publication_lag_days = 30
    upstream_url = UKRSTAT_INDPROD_URL
    cache_ttl_days = 30

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._used_synthetic = False

    def fetch_raw(self) -> pd.DataFrame:
        try:
            resp = requests.get(UKRSTAT_INDPROD_URL, timeout=10)
            if resp.status_code != 200 or len(resp.content) < 1000:
                raise RuntimeError(f"ukrstat indprod HTTP {resp.status_code}")
            # The HTML schema is too unstable to ship a parser; fall through
            # to synthetic to keep the contract honest.  Future iteration:
            # use the official Ukrstat XLSX once the partner-oblast mapping
            # is wired and the page stabilises.
            raise RuntimeError(
                "ukrstat_indprod: HTML schema not stable; using synthetic"
            )
        except Exception as exc:  # noqa: BLE001
            log.info(
                "ukrstat_indprod scrape skipped (%s) — using synthetic frame", exc
            )
            self._used_synthetic = True
            return _synthetic_frame()

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        if "Период" not in df.columns or "indprod_idx" not in df.columns:
            df = _synthetic_frame()
            self._used_synthetic = True

        full = pd.period_range(PERIOD_START, PERIOD_END, freq="M")
        out = pd.DataFrame({"Период": full}).merge(df, on="Период", how="left")
        out["indprod_idx"] = pd.to_numeric(out["indprod_idx"], errors="coerce")
        out["indprod_idx"] = out["indprod_idx"].ffill().bfill()

        out = out.sort_values("Период").reset_index(drop=True)
        out["indprod_idx_yoy_pct"] = out["indprod_idx"].pct_change(12) * 100.0

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
