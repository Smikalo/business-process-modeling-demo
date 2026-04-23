"""Enrich the master DataFrame with reference data: nomenclature, partners, prices, promos."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.ingestion import load_nomenclature, load_partners, load_prices, load_promotions

log = logging.getLogger(__name__)


def enrich_nomenclature(df: pd.DataFrame) -> pd.DataFrame:
    """Join SKU reference (brand, product group, ABC segment) onto master."""
    nom = load_nomenclature()
    n_before = len(df)
    df = df.merge(
        nom[["Артикул", "Бренд", "Группа_товара", "Сегмент_ABC"]],
        on="Артикул", how="left",
    )
    assert len(df) == n_before, "Row duplication after nomenclature join"
    coverage = df["Бренд"].notna().mean()
    df["Бренд"] = df["Бренд"].fillna("unknown")
    df["Группа_товара"] = df["Группа_товара"].fillna("unknown")
    df["Сегмент_ABC"] = df["Сегмент_ABC"].fillna("unknown")
    log.info("enrich_nomenclature: coverage=%.1f%%, brands=%s", coverage * 100, df["Бренд"].nunique())
    return df


def enrich_partners(df: pd.DataFrame) -> pd.DataFrame:
    """Join partner reference (channel, agreement type) and build target_qty."""
    part = load_partners()
    part = part.rename(columns={"Направление": "Канал", "Соглашение": "Тип_соглашения"})
    # Some partners have dual agreements; keep one row per partner, prefer Комиссионер
    part = part.sort_values("Тип_соглашения")  # Выкуп < Комиссионер alphabetically
    part = part.drop_duplicates(subset=["Партнер"], keep="last")
    n_before = len(df)
    df = df.merge(
        part[["Партнер", "Канал", "Тип_соглашения"]],
        on="Партнер", how="left",
    )
    assert len(df) == n_before, "Row duplication after partner join"
    df["Канал"] = df["Канал"].fillna("unknown")
    df["Тип_соглашения"] = df["Тип_соглашения"].fillna("unknown")

    # Target variable: for Выкуп, shipment IS the sale; for Комиссионер, use consumer sales
    df["target_qty"] = np.where(
        df["Тип_соглашения"] == "Выкуп",
        df["Количество_ship"],
        df["Количество_sales"],
    )
    log.info(
        "enrich_partners: Комиссионер=%d, Выкуп=%d, unknown=%d",
        (df["Тип_соглашения"] == "Комиссионер").sum(),
        (df["Тип_соглашения"] == "Выкуп").sum(),
        (df["Тип_соглашения"] == "unknown").sum(),
    )
    return df


def _build_price_timeline(period_range: pd.PeriodIndex) -> pd.DataFrame:
    """Build per-(Артикул, Период) RRP using forward-fill from price change events."""
    prices = load_prices()
    latest = (
        prices.sort_values("Период")
        .groupby(["Артикул", "Период"], as_index=False)
        .last()
    )
    # Pivot to wide (Артикул rows × Период columns), then ffill across time
    pivoted = latest.pivot_table(index="Артикул", columns="Период", values="РРЦ")
    pivoted = pivoted.reindex(columns=period_range).ffill(axis=1)
    melted = pivoted.reset_index().melt(
        id_vars="Артикул", var_name="Период", value_name="РРЦ"
    )
    melted = melted.dropna(subset=["РРЦ"])
    melted["Период"] = melted["Период"].astype("period[M]")
    return melted


def enrich_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Merge forward-filled RRP timeline + compute price-change flag and implied unit price."""
    period_range = df["Период"].unique()
    period_range = pd.PeriodIndex(sorted(period_range))
    timeline = _build_price_timeline(period_range)

    n_before = len(df)
    df = df.merge(timeline, on=["Артикул", "Период"], how="left")
    assert len(df) == n_before, "Row duplication after price join"

    # Implied unit price from actual transactions
    df["implied_unit_price"] = np.where(
        df["Количество_sales"] > 0,
        df["Выручка_sales"] / df["Количество_sales"],
        np.nan,
    )
    # Price change flag (per SKU, month-over-month)
    df = df.sort_values(["Партнер", "Артикул", "Период"])
    prev_price = df.groupby(["Партнер", "Артикул"])["РРЦ"].shift(1)
    df["price_change_flag"] = ((df["РРЦ"] != prev_price) & prev_price.notna()).astype(np.int8)

    coverage = df["РРЦ"].notna().mean()
    log.info("enrich_prices: RRP coverage=%.1f%%", coverage * 100)
    return df


def enrich_promotions(df: pd.DataFrame) -> pd.DataFrame:
    """Merge national promotion data onto master — binary flag + discount %."""
    promo = load_promotions()
    # Aggregate to (Артикул, Период) level: max discount, presence flag
    promo_agg = (
        promo.groupby(["Артикул", "Период"], as_index=False)
        .agg({"Скидка_pct": "max"})
        .rename(columns={"Скидка_pct": "promo_discount_pct"})
    )
    promo_agg["is_promo"] = 1

    n_before = len(df)
    df = df.merge(promo_agg, on=["Артикул", "Период"], how="left")
    assert len(df) == n_before, "Row duplication after promo join"

    df["is_promo"] = df["is_promo"].fillna(0).astype(np.int8)
    df["promo_discount_pct"] = df["promo_discount_pct"].fillna(0.0)
    log.info("enrich_promotions: %d SKU-months with promos", (df["is_promo"] == 1).sum())
    return df


def enrich_all(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all enrichment steps in sequence."""
    df = enrich_nomenclature(df)
    df = enrich_partners(df)
    df = enrich_prices(df)
    df = enrich_promotions(df)
    return df
