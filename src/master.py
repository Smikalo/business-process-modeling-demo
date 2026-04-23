"""Dense skeleton construction and master DataFrame assembly."""

from __future__ import annotations

import logging

import pandas as pd

from src.aggregation import build_all_aggregates
from src.config import PERIOD_END, PERIOD_START

log = logging.getLogger(__name__)


def build_skeleton(
    aggs: dict[str, pd.DataFrame],
    period_start: str = PERIOD_START,
    period_end: str = PERIOD_END,
) -> pd.DataFrame:
    """Cross-product of all unique (Партнер, Артикул) pairs × full period range."""
    partner_level = ["sales", "shipments", "rests_tt"]
    pairs = pd.concat(
        [aggs[k][["Партнер", "Артикул"]] for k in partner_level],
        ignore_index=True,
    ).drop_duplicates().reset_index(drop=True)

    all_periods = pd.period_range(start=period_start, end=period_end, freq="M")

    idx = pd.MultiIndex.from_product(
        [all_periods, pairs.index], names=["Период", "_pair_idx"]
    )
    skeleton = (
        pd.DataFrame(index=idx)
        .reset_index()
        .merge(pairs, left_on="_pair_idx", right_index=True)
        .drop(columns=["_pair_idx"])
    )
    log.info(
        "skeleton: %d rows (%d periods × %d pairs)",
        len(skeleton), len(all_periods), len(pairs),
    )
    return skeleton


def assemble_master(
    aggs: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build the full dense master ABT from raw aggregated sources."""
    if aggs is None:
        aggs = build_all_aggregates()

    skeleton = build_skeleton(aggs)

    df = (
        skeleton
        .merge(aggs["sales"],        on=["Период", "Партнер", "Артикул"], how="left")
        .merge(aggs["shipments"],    on=["Период", "Партнер", "Артикул"], how="left")
        .merge(aggs["rests_tt"],     on=["Период", "Партнер", "Артикул"], how="left")
        .merge(aggs["rests_orc"],    on=["Период", "Артикул"],            how="left")
        .merge(aggs["receipts_orc"], on=["Период", "Артикул"],            how="left")
    )

    metric_cols = [c for c in df.columns if c not in ("Период", "Партнер", "Артикул")]
    df[metric_cols] = df[metric_cols].fillna(0.0)

    log.info("master: %d rows × %d cols, mem=%.1f MB", len(df), len(df.columns), df.memory_usage(deep=True).sum() / 1e6)
    return df
