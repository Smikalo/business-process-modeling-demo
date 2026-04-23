"""Join external signal loaders onto the master ABT.

This module is intentionally separate from :mod:`src.enrichment` (which handles
the internal reference-data joins) so that external-signal failures cannot
break the core pipeline.
"""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from src.external_data import BaseSignalLoader, get_loader
from src.leakage_guard import apply_publication_lag

logger = logging.getLogger(__name__)


def _resolve_loader(name_or_instance: str | BaseSignalLoader) -> BaseSignalLoader:
    if isinstance(name_or_instance, BaseSignalLoader):
        return name_or_instance
    return get_loader(name_or_instance)


def enrich_with_external(
    abt: pd.DataFrame,
    loaders: Iterable[str | BaseSignalLoader],
    forecast_horizon_months: int = 1,
    force_refresh: bool = False,
    apply_lag: bool = True,
) -> pd.DataFrame:
    """Left-join each external signal onto the ABT after applying its lag.

    The caller must guarantee that ``abt`` has a ``Период`` column typed as
    ``period[M]``.  Row count is preserved; joining the same loader twice is
    idempotent (second join sees no new columns).
    """

    assert "Период" in abt.columns, "ABT is missing Период"
    assert str(abt["Период"].dtype).startswith("period"), (
        f"ABT.Период must be period[M], got {abt['Период'].dtype}"
    )

    original_rows = len(abt)
    out = abt
    seen_cols: set[str] = set()
    summary: list[dict] = []

    for raw in loaders:
        loader = _resolve_loader(raw)
        df = loader.load(force_refresh=force_refresh)
        if apply_lag and loader.publication_lag_days > 0:
            df = apply_publication_lag(
                df,
                publication_lag_days=loader.publication_lag_days,
                signal_cols=loader.signal_cols,
                group_cols=[k for k in loader.join_keys if k != "Период"],
                forecast_horizon_months=forecast_horizon_months,
            )

        new_cols = [c for c in loader.signal_cols if c not in seen_cols and c not in out.columns]
        if not new_cols:
            logger.info("Loader %s added 0 new columns (idempotent)", loader.name)
            continue
        df = df[loader.join_keys + new_cols]

        before_rows = len(out)
        out = out.merge(df, how="left", on=loader.join_keys, validate="many_to_one")
        assert len(out) == before_rows, (
            f"Row count changed after joining {loader.name}: {before_rows} -> {len(out)}"
        )
        seen_cols.update(new_cols)

        nan_rate = out[new_cols].isna().mean().mean()
        summary.append(
            {
                "loader": loader.name,
                "cols_added": len(new_cols),
                "signal_cols": new_cols,
                "nan_rate": float(nan_rate),
                "publication_lag_days": loader.publication_lag_days,
            }
        )
        logger.info(
            "Joined %s: +%d cols, NaN rate %.1f%%",
            loader.name,
            len(new_cols),
            nan_rate * 100,
        )

    assert len(out) == original_rows, "ABT row count changed overall — enrichment failure"
    out.attrs["external_signal_summary"] = summary
    return out
