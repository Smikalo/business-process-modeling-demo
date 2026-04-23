"""National Bank of Ukraine (NBU) FX and policy rate loader.

Data feeds (public, no auth):
- Daily UAH cross-rates:   https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange?valcode=USD&json
- Monthly NBU key rate:    https://bank.gov.ua/NBUStatService/v1/statdirectory/statrepo?stat=refinancerate&json

We request the whole time series once, aggregate daily FX to month-end, and
derive common features: UAH depreciation rate (z-score + m/m %), and the
interest-rate environment.

Publication lag: FX and policy rate are published in real time (same day).
Treating it as 0 days is safe as long as the forecast month's features use
the value from the previous month-end (which the leakage guard handles).
"""

from __future__ import annotations

import io
import logging

import pandas as pd
import requests

from src.external_data import BaseSignalLoader, register_loader

log = logging.getLogger(__name__)

NBU_FX_URL_FMT = (
    "https://bank.gov.ua/NBU_Exchange/exchange_site"
    "?valcode={valcode}&start={date_from}&end={date_to}"
    "&sort=exchangedate&order=asc&json"
)
NBU_POLICY_RATE_URL_FMT = (
    "https://bank.gov.ua/NBUStatService/v1/statdirectory/key?date={date}&json"
)

DEFAULT_START = "20190101"
DEFAULT_END = "20271231"


def _fetch_daily_fx(valcode: str, date_from: str, date_to: str) -> pd.DataFrame:
    """Fetch one year at a time to keep payloads modest."""
    frames: list[pd.DataFrame] = []
    start_year = int(date_from[:4])
    end_year = int(date_to[:4])
    for y in range(start_year, end_year + 1):
        yearly = NBU_FX_URL_FMT.format(
            valcode=valcode, date_from=f"{y}0101", date_to=f"{y}1231"
        )
        r = requests.get(yearly, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data:
            continue
        frames.append(pd.DataFrame(data))
    if not frames:
        raise RuntimeError(f"NBU returned no FX data for {valcode}")
    return pd.concat(frames, ignore_index=True)


@register_loader
class NBUFXLoader(BaseSignalLoader):
    name = "nbu_fx"
    signal_cols = [
        "uah_usd_eom",
        "uah_eur_eom",
        "uah_usd_mom_pct",
        "uah_eur_mom_pct",
        "uah_usd_yoy_pct",
        "nbu_policy_rate_eom",
        "nbu_policy_rate_diff_3m",
    ]
    publication_lag_days = 0  # published same day
    upstream_url = "https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange"
    cache_ttl_days = 30  # FX history is stable; refresh monthly

    def fetch_raw(self) -> pd.DataFrame:
        usd = _fetch_daily_fx("USD", DEFAULT_START, DEFAULT_END)
        eur = _fetch_daily_fx("EUR", DEFAULT_START, DEFAULT_END)
        usd["valcode"] = "USD"
        eur["valcode"] = "EUR"
        fx = pd.concat([usd, eur], ignore_index=True)

        # NBU returns exchangedate as 'dd.mm.yyyy'
        fx["date"] = pd.to_datetime(fx["exchangedate"], format="%d.%m.%Y")
        # The `rate` column is per-`units` hryvnia; historical rows used units=100
        # while modern rows use units=1.  Prefer rate_per_unit if present; else
        # normalize manually.
        if "rate_per_unit" in fx.columns:
            fx["rate"] = pd.to_numeric(fx["rate_per_unit"], errors="coerce")
        else:
            fx["rate"] = pd.to_numeric(fx["rate"], errors="coerce") / pd.to_numeric(
                fx.get("units", 1), errors="coerce"
            )

        # Policy rate — query once per month-end (the key indicator is
        # "KEY_PolicyRate"; the endpoint returns a snapshot for that date).
        pr_rows: list[dict] = []
        start_year = int(DEFAULT_START[:4])
        end_year = int(DEFAULT_END[:4])
        for y in range(start_year, end_year + 1):
            for m in range(1, 13):
                # Use the 15th of the month as a stable mid-month snapshot.
                date_str = f"{y}{m:02d}15"
                try:
                    r = requests.get(
                        NBU_POLICY_RATE_URL_FMT.format(date=date_str), timeout=20
                    )
                    if r.status_code != 200:
                        continue
                    rows = r.json()
                    for row in rows:
                        if row.get("id_api") == "KEY_PolicyRate":
                            pr_rows.append(
                                {
                                    "date": pd.to_datetime(f"{y}-{m:02d}-15"),
                                    "rate": float(row["value"]),
                                    "valcode": "POLICY",
                                }
                            )
                            break
                except Exception as exc:  # noqa: BLE001
                    log.debug("policy rate %s failed: %s", date_str, exc)
                    continue
        pr_df = pd.DataFrame(pr_rows) if pr_rows else pd.DataFrame(
            columns=["date", "rate", "valcode"]
        )
        if pr_df.empty:
            log.warning("No policy rate rows fetched — continuing with FX only")

        return pd.concat(
            [
                fx[["date", "rate", "valcode"]],
                pr_df[["date", "rate", "valcode"]],
            ],
            ignore_index=True,
        )

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        fx = raw[raw["valcode"].isin(["USD", "EUR"])].copy()
        fx["Период"] = fx["date"].dt.to_period("M")
        # Month-end (actually last observed value in the month).
        monthly_fx = (
            fx.sort_values("date")
            .groupby(["Период", "valcode"], as_index=False)["rate"]
            .last()
        )
        pivot = monthly_fx.pivot(index="Период", columns="valcode", values="rate").rename(
            columns={"USD": "uah_usd_eom", "EUR": "uah_eur_eom"}
        )
        pivot.columns.name = None
        pivot = pivot.sort_index().reset_index()

        # Policy rate: forward-fill across months.
        pr = raw[raw["valcode"] == "POLICY"].copy()
        if len(pr):
            pr["Период"] = pr["date"].dt.to_period("M")
            pr_monthly = (
                pr.sort_values("date")
                .groupby("Период", as_index=False)["rate"]
                .last()
                .rename(columns={"rate": "nbu_policy_rate_eom"})
            )
            pivot = pivot.merge(pr_monthly, on="Период", how="left")
            pivot["nbu_policy_rate_eom"] = pivot["nbu_policy_rate_eom"].ffill()
        else:
            pivot["nbu_policy_rate_eom"] = pd.NA

        # Derived features: mom / yoy percent changes, 3-month rate diff.
        pivot = pivot.sort_values("Период").reset_index(drop=True)
        pivot["uah_usd_mom_pct"] = pivot["uah_usd_eom"].pct_change() * 100
        pivot["uah_eur_mom_pct"] = pivot["uah_eur_eom"].pct_change() * 100
        pivot["uah_usd_yoy_pct"] = pivot["uah_usd_eom"].pct_change(12) * 100
        pivot["nbu_policy_rate_diff_3m"] = pivot["nbu_policy_rate_eom"].diff(3)

        # Ensure every month in range is present (forward-fill).
        all_periods = pd.period_range(
            pivot["Период"].min(), pivot["Период"].max(), freq="M"
        )
        pivot = (
            pivot.set_index("Период")
            .reindex(all_periods)
            .rename_axis("Период")
            .reset_index()
        )
        for c in self.signal_cols:
            if c in pivot.columns:
                pivot[c] = pd.to_numeric(pivot[c], errors="coerce").astype(float)
            else:
                pivot[c] = float("nan")

        pivot["Период"] = pivot["Период"].astype("period[M]")
        return pivot[["Период"] + self.signal_cols]
