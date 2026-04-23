"""Data ingestion for all raw sources (txt + xlsx)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    NOMENCLATURE_PATH,
    PARTNERS_PATH,
    PRICE_CUBICFUN_PATH,
    PRICE_DJECO_PATH,
    PRICE_INFANTINO_PATH,
    PROMOTIONS_PATH,
    RECEIPTS_ORC_PATH,
    RESTS_ORC_PATH,
    RESTS_TT_PATH,
    SALES_PATH,
    SHIPMENT_PATH,
)

log = logging.getLogger(__name__)

# ── helpers ──────────────────────────────────────────────────────────────────

def clean_numeric(series: pd.Series) -> pd.Series:
    """Parse 1C numeric format: '1 233,60' → 1233.60"""
    return pd.to_numeric(
        series.astype(str)
        .str.replace(r"\s+", "", regex=True)
        .str.replace(",", ".", regex=False),
        errors="coerce",
    ).fillna(0.0)


def _to_period(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%d.%m.%Y", errors="coerce").dt.to_period("M")


# ── .txt loaders ─────────────────────────────────────────────────────────────

def load_sales(path: Path = SALES_PATH) -> pd.DataFrame:
    """Sales (Продажи): utf-8, 6-row header, monthly grain."""
    df = pd.read_csv(
        path, sep="\t", skiprows=6, encoding="utf-8",
        names=["Партнер", "Артикул", "Дата", "Количество", "Выручка"],
        dtype={"Артикул": str, "Партнер": str, "Количество": str, "Выручка": str},
        on_bad_lines="warn",
    )
    df = df.dropna(subset=["Артикул"])
    df["Артикул"] = df["Артикул"].str.strip()
    df["Партнер"] = df["Партнер"].str.strip()
    df["Количество"] = clean_numeric(df["Количество"])
    df["Выручка"] = clean_numeric(df["Выручка"])
    df["Период"] = _to_period(df["Дата"])
    df = df.drop(columns=["Дата"]).dropna(subset=["Период"])
    log.info("load_sales: %d rows, %s – %s", len(df), df["Период"].min(), df["Период"].max())
    return df


def load_shipments(path: Path = SHIPMENT_PATH) -> pd.DataFrame:
    """Shipments (Отгрузки): utf-8, 7-row header, DAILY grain."""
    df = pd.read_csv(
        path, sep="\t", skiprows=7, encoding="utf-8",
        names=["Партнер", "Артикул", "Дата", "Количество", "Выручка"],
        dtype={"Артикул": str, "Партнер": str, "Количество": str, "Выручка": str},
        on_bad_lines="warn",
    )
    df = df.dropna(subset=["Артикул"])
    df["Артикул"] = df["Артикул"].str.strip()
    df["Партнер"] = df["Партнер"].str.strip()
    df["Количество"] = clean_numeric(df["Количество"])
    df["Выручка"] = clean_numeric(df["Выручка"])
    df["Период"] = _to_period(df["Дата"])
    df = df.drop(columns=["Дата"]).dropna(subset=["Период"])
    log.info("load_shipments: %d rows, %s – %s", len(df), df["Период"].min(), df["Период"].max())
    return df


def load_rests_orc(path: Path = RESTS_ORC_PATH) -> pd.DataFrame:
    """Warehouse inventory (Остатки ОРЦ): cp1251, 1-row header, monthly snapshots."""
    df = pd.read_csv(
        path, sep="\t", skiprows=1, encoding="cp1251", header=None,
        names=["Дата", "Артикул", "Количество", "Стоимость"],
        dtype={"Артикул": str, "Количество": str, "Стоимость": str},
        on_bad_lines="warn",
    )
    df = df.dropna(subset=["Артикул"])
    df["Артикул"] = df["Артикул"].str.strip()
    df["Количество"] = clean_numeric(df["Количество"])
    df["Стоимость"] = clean_numeric(df["Стоимость"])
    df["Период"] = _to_period(df["Дата"])
    df = df.drop(columns=["Дата"]).dropna(subset=["Период"])
    log.info("load_rests_orc: %d rows, %s – %s", len(df), df["Период"].min(), df["Период"].max())
    return df


def load_rests_tt(path: Path = RESTS_TT_PATH) -> pd.DataFrame:
    """Retail partner inventory (Остатки ТТ): cp1251, 1-row header, monthly snapshots."""
    df = pd.read_csv(
        path, sep="\t", skiprows=1, encoding="cp1251", header=None,
        names=["Дата", "Партнер", "Артикул", "Количество", "Стоимость"],
        dtype={"Артикул": str, "Партнер": str, "Количество": str, "Стоимость": str},
        on_bad_lines="warn",
    )
    df = df.dropna(subset=["Артикул"])
    df["Артикул"] = df["Артикул"].str.strip()
    df["Партнер"] = df["Партнер"].str.strip()
    df["Количество"] = clean_numeric(df["Количество"])
    df["Стоимость"] = clean_numeric(df["Стоимость"])
    df["Период"] = _to_period(df["Дата"])
    df = df.drop(columns=["Дата"]).dropna(subset=["Период"])
    log.info("load_rests_tt: %d rows, %s – %s", len(df), df["Период"].min(), df["Период"].max())
    return df


# ── .xlsx loaders ────────────────────────────────────────────────────────────

def load_receipts_orc(path: Path = RECEIPTS_ORC_PATH) -> pd.DataFrame:
    """Warehouse receipts (Поступление ОРЦ): xlsx, per-shipment dates."""
    df = pd.read_excel(path)
    df["Артикул"] = df["Артикул"].astype(str).str.strip()
    df["Период"] = df["Дата"].dt.to_period("M")
    df["Количество"] = df["Количество"].astype("float64")
    if "ЦенаВВалюте" in df.columns:
        df["ЦенаВВалюте"] = df["ЦенаВВалюте"].astype("float64")
    df = df.drop(columns=["Дата"])
    log.info("load_receipts_orc: %d rows", len(df))
    return df


def load_nomenclature(path: Path = NOMENCLATURE_PATH) -> pd.DataFrame:
    """SKU reference table (Справочник номенклатуры).

    The file has a merged-cell layout: header in row 2, data starts row 6.
    Relevant columns by index: 0=Артикул, 4=Номенклатура, 13=Бренд, 14=Група, 15=Сегмент.
    """
    raw = pd.read_excel(path, header=None)
    col_map = {0: "Артикул", 4: "Номенклатура", 13: "Бренд", 14: "Группа_товара", 15: "Сегмент_ABC"}
    df = raw.iloc[6:, list(col_map.keys())].copy()
    df.columns = list(col_map.values())
    df = df.dropna(subset=["Артикул"])
    df["Артикул"] = df["Артикул"].astype(str).str.strip()
    df = df.reset_index(drop=True)
    log.info("load_nomenclature: %d rows, cols=%s", len(df), list(df.columns))
    return df


def load_partners(path: Path = PARTNERS_PATH) -> pd.DataFrame:
    """Partner reference table (Справочник партнеров)."""
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    if "Партнер" in df.columns:
        df["Партнер"] = df["Партнер"].str.strip()
    log.info("load_partners: %d rows, cols=%s", len(df), list(df.columns))
    return df


def load_prices(brand: str | None = None) -> pd.DataFrame:
    """Price lists (Прайс *.xlsx). Uses calamine engine (openpyxl can't open these).

    The files have merged cells producing 7 raw columns; useful data is at
    indices 0 (date str), 3 (Артикул), 6 (РРЦ).
    """
    brand_map = {
        "Djeco": PRICE_DJECO_PATH,
        "CubicFun": PRICE_CUBICFUN_PATH,
        "Infantino": PRICE_INFANTINO_PATH,
    }
    paths = {brand: brand_map[brand]} if brand else brand_map
    frames = []
    for b, p in paths.items():
        raw = pd.read_excel(p, engine="calamine", header=None, skiprows=7)
        df = pd.DataFrame({
            "Дата_str": raw.iloc[:, 0],
            "Артикул": raw.iloc[:, 3],
            "РРЦ": raw.iloc[:, 6] if raw.shape[1] > 6 else raw.iloc[:, -1],
        })
        df = df.dropna(subset=["Артикул"])
        df["Артикул"] = df["Артикул"].astype(str).str.strip()
        df["РРЦ"] = pd.to_numeric(df["РРЦ"], errors="coerce")
        df["Период"] = pd.to_datetime(df["Дата_str"], format="%d.%m.%Y", errors="coerce").dt.to_period("M")
        df = df.drop(columns=["Дата_str"]).dropna(subset=["Период", "РРЦ"])
        df["Бренд"] = b
        frames.append(df)
    result = pd.concat(frames, ignore_index=True)
    log.info("load_prices: %d rows (%s)", len(result), list(paths.keys()))
    return result


def load_promotions(path: Path = PROMOTIONS_PATH) -> pd.DataFrame:
    """National promotions (Нац. акции 2024).

    Header row has garbled col names for cols 5+.
    Real schema: Дата, Бренд, Артикул, Скидка(%), Сегмент, РРЦ_со_скидкой, Кол_шт.
    """
    df = pd.read_excel(path, header=None, skiprows=1,
                        names=["Дата", "Бренд", "Артикул", "Скидка_pct", "Сегмент",
                               "РРЦ_со_скидкой", "Кол_шт", "_7", "_8"])
    df = df.drop(columns=["_7", "_8"], errors="ignore")
    df = df.dropna(subset=["Артикул"])
    df["Артикул"] = df["Артикул"].astype(str).str.strip()
    df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce")
    df["Период"] = df["Дата"].dt.to_period("M")
    df = df.drop(columns=["Дата"]).dropna(subset=["Период"])
    log.info("load_promotions: %d rows, date range %s – %s", len(df), df["Период"].min(), df["Период"].max())
    return df
