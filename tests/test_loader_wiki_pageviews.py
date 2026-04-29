"""Offline tests for the Wikipedia Pageviews loader.

Network calls are monkeypatched so the suite passes without internet
access.  We exercise three paths:

1. Synthetic-data path: every article succeeds with deterministic
   monthly views; verify aggregation, schema, YoY computation.
2. All-fail path: every article raises; verify the loader degrades to
   an all-NaN stub frame covering the full 2019–2027 window.
3. Per-article failure path: a single article fails while others
   succeed; verify the loader keeps going and still produces a frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests


def _stub_items(per_month: int = 100) -> list[dict]:
    """Return synthetic monthly items spanning 2023-01 .. 2024-12.

    Two full calendar years gives us a non-trivial YoY denominator.
    """
    months = pd.period_range("2023-01", "2024-12", freq="M")
    return [
        {"timestamp": f"{p.year:04d}{p.month:02d}0100", "views": per_month}
        for p in months
    ]


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Make the polite gap a no-op for tests."""
    monkeypatch.setattr("time.sleep", lambda *_args, **_kw: None)


def test_imports_register_loader():
    """The module must register itself in the global LOADER_REGISTRY."""
    from src.external_data import LOADER_REGISTRY
    from src.loaders import wiki_pageviews  # noqa: F401

    assert "wiki_pageviews" in LOADER_REGISTRY


def test_wiki_pageviews_synthetic_full_pipeline(tmp_path: Path, monkeypatch):
    from src.loaders.wiki_pageviews import WikiPageviewsLoader

    def fake_fetch_one(self, project, article, start, end):
        return _stub_items(per_month=100)

    monkeypatch.setattr(WikiPageviewsLoader, "_fetch_one", fake_fetch_one)
    loader = WikiPageviewsLoader(cache_dir=tmp_path)
    df = loader.load()

    assert str(df["Период"].dtype).startswith("period")
    assert df["Период"].min() == pd.Period("2019-01", freq="M")
    assert df["Период"].max() == pd.Period("2027-12", freq="M")
    for col in WikiPageviewsLoader.signal_cols:
        assert col in df.columns

    # No duplicate periods.
    assert df["Период"].is_unique

    # 2024-01 must have non-zero totals across every category since the
    # synthetic stub gives 100 views per article per month.
    jan = df[df["Период"] == pd.Period("2024-01", freq="M")].iloc[0]
    assert jan["wiki_pv_toys_total"] > 0
    assert jan["wiki_pv_brands_total"] > 0
    assert jan["wiki_pv_franchises_total"] > 0

    # YoY between 2024-01 and 2023-01 should be ~0 (constant 100 per month).
    assert abs(float(jan["wiki_pv_yoy_pct"])) < 1e-6

    # Months outside the synthetic window (e.g. 2019-06) should be NaN.
    early = df[df["Период"] == pd.Period("2019-06", freq="M")].iloc[0]
    assert pd.isna(early["wiki_pv_toys_total"])
    assert pd.isna(early["wiki_pv_yoy_pct"])


def test_wiki_pageviews_all_fail_falls_back(tmp_path: Path, monkeypatch):
    from src.loaders.wiki_pageviews import WikiPageviewsLoader

    def raising_fetch(self, project, article, start, end):
        raise requests.HTTPError("simulated 404")

    monkeypatch.setattr(WikiPageviewsLoader, "_fetch_one", raising_fetch)
    loader = WikiPageviewsLoader(cache_dir=tmp_path)
    df = loader.load()

    assert df["Период"].min() == pd.Period("2019-01", freq="M")
    assert df["Период"].max() == pd.Period("2027-12", freq="M")
    for col in WikiPageviewsLoader.signal_cols:
        assert col in df.columns
        assert df[col].isna().all(), f"{col} should be all-NaN on full failure"


def test_wiki_pageviews_partial_failure_skips_articles(
    tmp_path: Path, monkeypatch
):
    """If a single article 404s the loader keeps going."""
    from src.loaders.wiki_pageviews import WikiPageviewsLoader

    call_count = {"n": 0}

    def flaky_fetch(self, project, article, start, end):
        call_count["n"] += 1
        # First article raises, rest succeed.
        if call_count["n"] == 1:
            raise requests.HTTPError("simulated 404 for one article")
        return _stub_items(per_month=50)

    monkeypatch.setattr(WikiPageviewsLoader, "_fetch_one", flaky_fetch)
    loader = WikiPageviewsLoader(cache_dir=tmp_path)
    df = loader.load()

    # Aggregates must still be > 0 for synthetic months.
    jan = df[df["Период"] == pd.Period("2024-01", freq="M")].iloc[0]
    cat_total = (
        float(jan["wiki_pv_toys_total"] or 0)
        + float(jan["wiki_pv_brands_total"] or 0)
        + float(jan["wiki_pv_franchises_total"] or 0)
    )
    assert cat_total > 0


def test_wiki_pageviews_yoy_growth(tmp_path: Path, monkeypatch):
    """When 2024 traffic is 2x 2023, YoY should be ~+100%."""
    from src.loaders.wiki_pageviews import WikiPageviewsLoader

    def growing_fetch(self, project, article, start, end):
        items = []
        for p in pd.period_range("2023-01", "2024-12", freq="M"):
            v = 200 if p.year == 2024 else 100
            items.append(
                {"timestamp": f"{p.year:04d}{p.month:02d}0100", "views": v}
            )
        return items

    monkeypatch.setattr(WikiPageviewsLoader, "_fetch_one", growing_fetch)
    loader = WikiPageviewsLoader(cache_dir=tmp_path)
    df = loader.load()

    jan_2024 = df[df["Период"] == pd.Period("2024-01", freq="M")].iloc[0]
    assert np.isclose(float(jan_2024["wiki_pv_yoy_pct"]), 100.0, atol=1e-6)
