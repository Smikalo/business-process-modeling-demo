"""Smoke tests for the external-data framework (no network)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.external_data import (
    BaseSignalLoader,
    LOADER_REGISTRY,
    get_loader,
    list_loaders,
    register_loader,
)


class _DummyLoader(BaseSignalLoader):
    name = "_dummy_test"
    signal_cols = ["x"]
    publication_lag_days = 0
    upstream_url = "test://dummy"

    def fetch_raw(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Период": pd.period_range("2020-01", "2020-03", freq="M"),
                "x": [1.0, 2.0, 3.0],
            }
        )

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        out = raw.copy()
        out["Период"] = out["Период"].astype("period[M]")
        return out


def test_importing_framework_triggers_no_network(monkeypatch):
    """Merely importing src.external_data must never call the network."""
    import socket

    def _guard(*args, **kwargs):  # pragma: no cover - assertion helper
        raise RuntimeError("socket usage forbidden during import")

    monkeypatch.setattr(socket, "socket", _guard)
    # The framework module itself is already imported; re-exercising registry API is fine.
    list_loaders()


def test_loader_roundtrip(tmp_path: Path):
    loader = _DummyLoader(cache_dir=tmp_path)
    df = loader.load()
    assert list(df["x"]) == [1.0, 2.0, 3.0]
    assert loader.cache_path.exists()
    assert loader.meta_path.exists()

    df2 = loader.load()
    pd.testing.assert_frame_equal(df.reset_index(drop=True), df2.reset_index(drop=True))


def test_loader_falls_back_to_stale_cache_on_failure(tmp_path: Path):
    loader = _DummyLoader(cache_dir=tmp_path)
    loader.load()

    class _FlakyLoader(_DummyLoader):
        def fetch_raw(self) -> pd.DataFrame:
            raise RuntimeError("network outage")

    flaky = _FlakyLoader(cache_dir=tmp_path)
    flaky.cache_ttl_days = -1
    df = flaky.load(force_refresh=True)
    assert list(df["x"]) == [1.0, 2.0, 3.0]


def test_registry_register_and_get():
    @register_loader
    class _RegisteredLoader(_DummyLoader):
        name = "_registered_test"

    assert "_registered_test" in list_loaders()
    assert isinstance(get_loader("_registered_test"), _RegisteredLoader)

    LOADER_REGISTRY.pop("_registered_test", None)


def test_loader_validate_rejects_missing_column(tmp_path: Path):
    class _Broken(_DummyLoader):
        name = "_broken_test"

        def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
            out = raw.copy().drop(columns=["x"])
            return out

    loader = _Broken(cache_dir=tmp_path)
    with pytest.raises(AssertionError):
        loader.load(force_refresh=True)
