"""Wikipedia Pageviews loader (en + uk + ru).

Tracks attention to a curated catalog of toy / brand / franchise pages
across three language editions of Wikipedia.  The Wikimedia REST API is
free and unauthenticated; per-article monthly counts are pulled and then
aggregated into three category totals plus a YoY change ratio.

Hypothesis: monthly pageviews on Lego, Barbie, Frozen, Pokemon, Bluey, etc.
lead Ukrainian toy-retail sales by 1-3 months when a film/show drives
brand awareness.

Failure model:
- A single article that 404s or rate-limits is skipped (logged, no raise).
- If *every* article fails, transform yields an all-NaN stub covering the
  full 2019-01..2027-12 window so downstream joins remain unbroken.
"""

from __future__ import annotations

import logging
import time
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from src.external_data import BaseSignalLoader, register_loader

log = logging.getLogger(__name__)

WIKI_PV_BASE = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "{project}/all-access/all-agents/{article}/monthly/{start}/{end}"
)
USER_AGENT = (
    "v12-v14-demand-forecast-research/1.0 (academic; smikalo@gmail.com)"
)

# (project, article, category)
# Article titles are the canonical (capitalized) Wikipedia URL slugs;
# the pageviews API is case-sensitive.
ARTICLES: list[tuple[str, str, str]] = [
    # English Wikipedia — global brand / franchise attention.
    ("en.wikipedia", "Lego", "brands"),
    ("en.wikipedia", "Hasbro", "brands"),
    ("en.wikipedia", "Mattel", "brands"),
    ("en.wikipedia", "Barbie", "franchises"),
    ("en.wikipedia", "Funko", "brands"),
    ("en.wikipedia", "Pokémon_(franchise)", "franchises"),
    ("en.wikipedia", "Spider-Man", "franchises"),
    ("en.wikipedia", "Frozen_(film)", "franchises"),
    ("en.wikipedia", "Bluey_(2018_TV_series)", "franchises"),
    ("en.wikipedia", "Paw_Patrol", "franchises"),
    ("en.wikipedia", "Hot_Wheels", "toys"),
    ("en.wikipedia", "Disney_Princess", "franchises"),
    ("en.wikipedia", "Mickey_Mouse", "franchises"),
    ("en.wikipedia", "Star_Wars", "franchises"),
    # Ukrainian Wikipedia — Ukraine-specific attention.
    ("uk.wikipedia", "Іграшка", "toys"),
    ("uk.wikipedia", "Lego", "brands"),
    ("uk.wikipedia", "Лялька", "toys"),
    ("uk.wikipedia", "Конструктор", "toys"),
    ("uk.wikipedia", "Disney", "brands"),
    # Russian Wikipedia — adjacent regional attention.
    ("ru.wikipedia", "Игрушка", "toys"),
    ("ru.wikipedia", "Lego", "brands"),
    ("ru.wikipedia", "Барби", "franchises"),
]

CATEGORIES = ("toys", "brands", "franchises")


@register_loader
class WikiPageviewsLoader(BaseSignalLoader):
    name = "wiki_pageviews"
    signal_cols = [
        "wiki_pv_toys_total",
        "wiki_pv_brands_total",
        "wiki_pv_franchises_total",
        "wiki_pv_yoy_pct",
    ]
    publication_lag_days = 1
    upstream_url = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
    )
    cache_ttl_days = 30

    START_YYYYMM = "201901"
    END_YYYYMM = "202712"
    REQUEST_GAP_SECONDS = 0.2

    def _empty_frame(self) -> pd.DataFrame:
        periods = pd.period_range("2019-01", "2027-12", freq="M")
        out = pd.DataFrame({"Период": periods})
        for col in self.signal_cols:
            out[col] = np.nan
        out["Период"] = out["Период"].astype("period[M]")
        return out

    def _fetch_one(
        self,
        project: str,
        article: str,
        start_yyyymm: str,
        end_yyyymm: str,
    ) -> list[dict]:
        """Single-article pageviews call.  Raises on HTTP error so that the
        outer loop in :meth:`fetch_raw` can decide to skip the article."""
        headers = {"User-Agent": USER_AGENT}
        url = WIKI_PV_BASE.format(
            project=project,
            article=quote(article, safe=""),
            start=f"{start_yyyymm}01",
            end=f"{end_yyyymm}28",
        )
        r = requests.get(url, timeout=15, headers=headers)
        r.raise_for_status()
        return r.json().get("items", []) or []

    def fetch_raw(self) -> pd.DataFrame:
        rows: list[dict] = []
        successes = 0
        for project, article, category in ARTICLES:
            try:
                items = self._fetch_one(
                    project, article, self.START_YYYYMM, self.END_YYYYMM
                )
                for it in items:
                    ts = str(it.get("timestamp", ""))
                    views_raw = it.get("views", 0)
                    try:
                        views = int(views_raw or 0)
                    except (TypeError, ValueError):
                        views = 0
                    rows.append(
                        {
                            "project": project,
                            "article": article,
                            "category": category,
                            "timestamp": ts,
                            "views": views,
                        }
                    )
                successes += 1
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "wiki_pageviews skip %s/%s: %s", project, article, exc
                )
            finally:
                time.sleep(self.REQUEST_GAP_SECONDS)

        if successes == 0 or not rows:
            log.warning(
                "wiki_pageviews: no articles succeeded — returning empty raw"
            )
            return pd.DataFrame(
                columns=["project", "article", "category", "timestamp", "views"]
            )
        return pd.DataFrame(rows)

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        if raw is None or raw.empty:
            return self._empty_frame()
        if not {"timestamp", "category", "views"}.issubset(raw.columns):
            return self._empty_frame()

        df = raw.copy()
        ts_str = df["timestamp"].astype(str)
        good = ts_str.str.len() >= 6
        df = df.loc[good].copy()
        if df.empty:
            return self._empty_frame()

        df["Период"] = ts_str.loc[good].apply(
            lambda s: pd.Period(f"{s[:4]}-{s[4:6]}", freq="M")
        )

        agg = (
            df.groupby(["Период", "category"], as_index=False)["views"].sum()
        )

        wide = (
            agg.pivot(index="Период", columns="category", values="views")
            .reset_index()
        )
        wide.columns.name = None
        for cat in CATEGORIES:
            if cat not in wide.columns:
                wide[cat] = np.nan
        wide = wide.rename(
            columns={
                "toys": "wiki_pv_toys_total",
                "brands": "wiki_pv_brands_total",
                "franchises": "wiki_pv_franchises_total",
            }
        )

        full = pd.period_range("2019-01", "2027-12", freq="M")
        out = pd.DataFrame({"Период": full})
        out = out.merge(wide, on="Период", how="left")

        cat_cols = [
            "wiki_pv_toys_total",
            "wiki_pv_brands_total",
            "wiki_pv_franchises_total",
        ]
        for col in cat_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)

        # Total across all categories; min_count=1 keeps NaN when *every*
        # category is missing for that month (vs. an empirical zero).
        total = out[cat_cols].sum(axis=1, min_count=1)
        prior_year = total.shift(12)
        with np.errstate(divide="ignore", invalid="ignore"):
            yoy = np.where(
                (prior_year.notna()) & (prior_year > 0) & (total.notna()),
                (total - prior_year) / prior_year * 100.0,
                np.nan,
            )
        out["wiki_pv_yoy_pct"] = yoy.astype(float)

        out["Период"] = out["Период"].astype("period[M]")
        return out[["Период"] + self.signal_cols]
