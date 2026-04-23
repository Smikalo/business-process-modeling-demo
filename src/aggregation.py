"""Monthly aggregation of raw ingested DataFrames."""

from __future__ import annotations

import logging

import pandas as pd

from src.ingestion import (
    load_receipts_orc,
    load_rests_orc,
    load_rests_tt,
    load_sales,
    load_shipments,
)

log = logging.getLogger(__name__)


def aggregate_sales(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = load_sales()
    agg = (
        df.groupby(["Период", "Партнер", "Артикул"], as_index=False)
        .agg({"Количество": "sum", "Выручка": "sum"})
        .rename(columns={"Количество": "Количество_sales", "Выручка": "Выручка_sales"})
    )
    log.info("aggregate_sales: %d rows", len(agg))
    return agg


def aggregate_shipments(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = load_shipments()
    agg = (
        df.groupby(["Период", "Партнер", "Артикул"], as_index=False)
        .agg({"Количество": "sum", "Выручка": "sum"})
        .rename(columns={"Количество": "Количество_ship", "Выручка": "Выручка_ship"})
    )
    log.info("aggregate_shipments: %d rows", len(agg))
    return agg


def aggregate_rests_tt(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = load_rests_tt()
    agg = (
        df.groupby(["Период", "Партнер", "Артикул"], as_index=False)
        .agg({"Количество": "sum", "Стоимость": "sum"})
    )
    agg["Количество"] = agg["Количество"].clip(lower=0)
    agg = agg.rename(columns={"Количество": "Количество_tt", "Стоимость": "Стоимость_tt"})
    log.info("aggregate_rests_tt: %d rows", len(agg))
    return agg


def aggregate_rests_orc(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = load_rests_orc()
    agg = (
        df.groupby(["Период", "Артикул"], as_index=False)
        .agg({"Количество": "sum", "Стоимость": "sum"})
    )
    agg["Количество"] = agg["Количество"].clip(lower=0)
    agg = agg.rename(columns={"Количество": "Количество_orc", "Стоимость": "Стоимость_orc"})
    log.info("aggregate_rests_orc: %d rows", len(agg))
    return agg


def aggregate_receipts_orc(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = load_receipts_orc()
    agg_dict = {"Количество": "sum"}
    if "ЦенаВВалюте" in df.columns:
        agg_dict["ЦенаВВалюте"] = "mean"
    agg = (
        df.groupby(["Период", "Артикул"], as_index=False)
        .agg(agg_dict)
        .rename(columns={"Количество": "Количество_receipts"})
    )
    log.info("aggregate_receipts_orc: %d rows", len(agg))
    return agg


def build_all_aggregates() -> dict[str, pd.DataFrame]:
    """Load and aggregate all 5 transactional sources. Returns a dict keyed by name."""
    return {
        "sales": aggregate_sales(),
        "shipments": aggregate_shipments(),
        "rests_tt": aggregate_rests_tt(),
        "rests_orc": aggregate_rests_orc(),
        "receipts_orc": aggregate_receipts_orc(),
    }
