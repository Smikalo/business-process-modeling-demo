"""Ukrainian Google Trends loader (via ``pytrends``).

Tracks search interest for toy-adjacent Ukrainian keywords.  Google Trends is
aggressive about rate-limiting anonymous requests, so this loader degrades
gracefully: if Google returns 429, we emit a stub DataFrame covering the
expected month range with NaN values (the downstream ablation then ignores
the signal).  The cache still records the attempt so re-runs don't hammer
the API.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from src.external_data import BaseSignalLoader, register_loader

log = logging.getLogger(__name__)

# Keyword groups.  Each inner list is sent to Google as a single payload
# (max 5 terms per request).  Sums/means are pooled across pairs.
KEYWORD_GROUPS: list[list[str]] = [
    ["іграшки", "дитячі іграшки", "конструктор", "ляльки"],
    ["Lego", "Hasbro", "Playmobil", "Barbie"],
    ["настільна гра", "пазл", "розвиваючі іграшки"],
]
ALL_KEYWORDS = [k for group in KEYWORD_GROUPS for k in group]


@register_loader
class GoogleTrendsUALoader(BaseSignalLoader):
    name = "gtrends_ua"
    signal_cols = [
        "trends_toys_general",
        "trends_toy_brands",
        "trends_board_games_puzzles",
    ]
    # Trends are finalized ~2 weeks after month end.
    publication_lag_days = 14
    upstream_url = "https://trends.google.com (via pytrends)"
    cache_ttl_days = 30  # keep trends data a month before re-pulling

    def _empty_frame(self) -> pd.DataFrame:
        periods = pd.period_range("2019-01", "2027-12", freq="M")
        out = pd.DataFrame({"Период": periods})
        for col in self.signal_cols:
            out[col] = np.nan
        return out

    def fetch_raw(self) -> pd.DataFrame:
        try:
            from pytrends.request import TrendReq
        except ImportError:
            log.warning("pytrends not installed — returning stub frame")
            return self._empty_frame()

        pt = TrendReq(hl="uk-UA", tz=120, timeout=(10, 30))

        frames: list[pd.DataFrame] = []
        for group in KEYWORD_GROUPS:
            try:
                pt.build_payload(
                    kw_list=group[:5],
                    timeframe="2019-01-01 2027-12-31",
                    geo="UA",
                )
                df = pt.interest_over_time()
                if df.empty:
                    log.warning("Google Trends returned empty for group %s", group)
                    continue
                if "isPartial" in df.columns:
                    df = df.drop(columns=["isPartial"])
                df = df.reset_index().rename(columns={"date": "date"})
                df["group_id"] = "|".join(group)
                frames.append(df)
                time.sleep(2)  # polite spacing; 429 may still occur
            except Exception as exc:  # noqa: BLE001
                log.warning("Google Trends failed for %s: %s", group, exc)
                continue

        if not frames:
            return self._empty_frame()
        return pd.concat(frames, ignore_index=True)

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        if "date" not in raw.columns:
            return raw.reset_index(drop=True)

        raw = raw.copy()
        raw["Период"] = pd.to_datetime(raw["date"]).dt.to_period("M")

        # Aggregate: each group's average across its keywords, then monthly mean.
        def _avg_of_kws(sub: pd.DataFrame) -> float:
            cols = [c for c in sub.columns if c not in {"date", "Период", "group_id"}]
            return float(sub[cols].mean(axis=1).mean())

        agg = (
            raw.groupby(["Период", "group_id"])
            .apply(_avg_of_kws, include_groups=False)
            .unstack("group_id")
            .reset_index()
        )

        mapping = {
            "|".join(KEYWORD_GROUPS[0]): "trends_toys_general",
            "|".join(KEYWORD_GROUPS[1]): "trends_toy_brands",
            "|".join(KEYWORD_GROUPS[2]): "trends_board_games_puzzles",
        }
        agg = agg.rename(columns=mapping)

        # Backfill missing months with a full range.
        full = pd.period_range("2019-01", "2027-12", freq="M")
        out = pd.DataFrame({"Период": full})
        out = out.merge(agg, on="Период", how="left")

        for col in self.signal_cols:
            if col not in out.columns:
                out[col] = np.nan
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out["Период"] = out["Период"].astype("period[M]")
        return out[["Период"] + self.signal_cols]
