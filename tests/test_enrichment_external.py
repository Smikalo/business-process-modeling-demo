"""Tests for src.enrichment_external."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.enrichment_external import enrich_with_external
from src.external_data import BaseSignalLoader


class _GlobalLoader(BaseSignalLoader):
    name = "_test_global"
    signal_cols = ["z"]
    publication_lag_days = 0
    upstream_url = "test://global"

    def fetch_raw(self) -> pd.DataFrame:
        periods = pd.period_range("2022-01", "2022-12", freq="M")
        return pd.DataFrame({"Период": periods, "z": np.arange(12, dtype=float)})

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        out = raw.copy()
        out["Период"] = out["Период"].astype("period[M]")
        return out


class _LaggedLoader(_GlobalLoader):
    name = "_test_lagged"
    publication_lag_days = 45  # -> shift 2 months
    signal_cols = ["z_lagged"]

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        out = raw.copy()
        out["Период"] = out["Період"] = out["Период"].astype("period[M]")
        out["z_lagged"] = out["z"].astype(float)
        return out[["Период", "z_lagged"]]


def _build_abt() -> pd.DataFrame:
    periods = pd.period_range("2022-01", "2022-12", freq="M")
    rows = []
    for p in periods:
        for partner in ["A", "B", "C"]:
            rows.append({"Период": p, "Партнер": partner, "y": 1.0})
    return pd.DataFrame(rows)


def test_join_preserves_row_count(tmp_path: Path):
    abt = _build_abt()
    loader = _GlobalLoader(cache_dir=tmp_path)
    out = enrich_with_external(abt, [loader])
    assert len(out) == len(abt)
    assert "z" in out.columns


def test_lag_introduces_nans_in_early_months(tmp_path: Path):
    abt = _build_abt()
    loader = _LaggedLoader(cache_dir=tmp_path)
    out = enrich_with_external(abt, [loader])
    jan = out[out["Период"] == pd.Period("2022-01", freq="M")]
    assert jan["z_lagged"].isna().all(), "2022-01 should be NaN after 2-month lag"
    mar = out[out["Період"] == pd.Period("2022-03", freq="M")] if False else out[out["Период"] == pd.Period("2022-03", freq="M")]
    assert (mar["z_lagged"] == 0.0).all(), "2022-03 should carry 2022-01 value"


def test_double_join_idempotent(tmp_path: Path):
    abt = _build_abt()
    loader = _GlobalLoader(cache_dir=tmp_path)
    first = enrich_with_external(abt, [loader])
    second = enrich_with_external(first, [loader])
    assert first.shape == second.shape


def test_summary_recorded(tmp_path: Path):
    abt = _build_abt()
    loader = _GlobalLoader(cache_dir=tmp_path)
    out = enrich_with_external(abt, [loader])
    summary = out.attrs["external_signal_summary"]
    assert len(summary) == 1
    assert summary[0]["loader"] == "_test_global"
    assert summary[0]["cols_added"] == 1
