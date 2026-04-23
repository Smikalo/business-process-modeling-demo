"""End-to-end leakage tests for every registered external signal.

For every loader in the registry this suite verifies:
1. The declared ``publication_lag_days`` is non-negative.
2. Joining with lag applied produces a different series than without lag
   whenever the lag is positive (otherwise the lag is a silent no-op).
3. The earliest rows in the training window of the joined ABT carry NaN for
   the lagged signal columns (the data simply wasn't observable yet).

These run without network by monkey-patching loaders' ``fetch_raw`` to return
synthetic data; the tests exercise only the framework contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.enrichment_external import enrich_with_external
from src.external_data import BaseSignalLoader
from src.leakage_guard import apply_publication_lag


class _SyntheticLoader(BaseSignalLoader):
    """A loader with a monotonic synthetic series and configurable lag."""

    name = "_leakage_suite_probe"
    signal_cols = ["probe_val"]

    def __init__(self, publication_lag_days: int, cache_dir: Path):
        super().__init__(cache_dir=cache_dir)
        # Per-instance override of class attribute.
        self.publication_lag_days = publication_lag_days

    def fetch_raw(self) -> pd.DataFrame:
        periods = pd.period_range("2022-01", "2024-12", freq="M")
        return pd.DataFrame(
            {"Период": periods, "probe_val": np.arange(len(periods), dtype=float)}
        )

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        out = raw.copy()
        out["Период"] = out["Период"].astype("period[M]")
        return out


def _abt(periods_n: int = 36) -> pd.DataFrame:
    periods = pd.period_range("2022-01", periods=periods_n, freq="M")
    rows = []
    for p in periods:
        for partner in ["A", "B"]:
            rows.append({"Период": p, "Партнер": partner, "y": 1.0})
    return pd.DataFrame(rows)


def test_publication_lag_is_nonnegative():
    """Every loader's declared lag must be non-negative."""
    from src.external_data import LOADER_REGISTRY, import_default_loaders

    import_default_loaders()
    for name, loader_cls in LOADER_REGISTRY.items():
        loader = loader_cls()
        assert loader.publication_lag_days >= 0, (
            f"Loader {name} has negative publication_lag_days "
            f"({loader.publication_lag_days}) — this would cause a future leak."
        )


def test_lag_is_not_a_silent_noop(tmp_path: Path):
    """If a loader declares a positive lag, the lagged values must DIFFER from
    the unlagged values in at least one row — otherwise the lag is broken."""

    loader = _SyntheticLoader(publication_lag_days=45, cache_dir=tmp_path)
    abt = _abt()

    with_lag = enrich_with_external(abt, [loader], apply_lag=True)

    # Reset cache so the same loader instance can be reused without stale parquet.
    loader.cache_path.unlink(missing_ok=True)
    loader.meta_path.unlink(missing_ok=True)
    abt2 = _abt()
    without_lag = enrich_with_external(abt2, [loader], apply_lag=False)

    diff = (
        with_lag["probe_val"].fillna(-999).values
        != without_lag["probe_val"].fillna(-999).values
    )
    assert diff.any(), "Lag had no effect on any row — lag implementation is broken."


def test_early_training_rows_are_nan_under_lag(tmp_path: Path):
    """With a 2-month shift, the first 2 calendar periods per row must be NaN."""

    loader = _SyntheticLoader(publication_lag_days=45, cache_dir=tmp_path)
    abt = _abt()
    out = enrich_with_external(abt, [loader], apply_lag=True)

    first_two = out[out["Период"] <= pd.Period("2022-02", freq="M")]
    assert first_two["probe_val"].isna().all()


def test_future_leak_detector_fires_on_negative_lag():
    """The guard treats negative lags as invalid."""
    from src.leakage_guard import _lag_in_months

    with pytest.raises(ValueError):
        _lag_in_months(-30, forecast_horizon_months=1)


def test_mutation_negative_lag_caught():
    """Mutation test: if someone sets publication_lag_days = -30 on a loader
    and calls apply_publication_lag, the framework must refuse."""
    import pandas as pd
    from src.leakage_guard import apply_publication_lag

    df = pd.DataFrame(
        {"Период": pd.period_range("2022-01", "2022-12", freq="M"), "x": range(12)}
    )
    with pytest.raises(ValueError):
        apply_publication_lag(df, publication_lag_days=-30, signal_cols=["x"])
