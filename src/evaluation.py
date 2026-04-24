"""Evaluation metrics and train/test splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PERIOD_END, PERIOD_START, TRAIN_END, VAL_END


# ── Metrics ──────────────────────────────────────────────────────────────────

def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Weighted Absolute Percentage Error — avoids division-by-zero on sparse demand."""
    total = np.abs(actual).sum()
    return np.abs(actual - predicted).sum() / total if total > 0 else 0.0


def mape_nonzero(actual: np.ndarray, predicted: np.ndarray) -> float:
    """MAPE computed only on observations where actual > 0."""
    mask = actual > 0
    if mask.sum() == 0:
        return 0.0
    return np.abs((actual[mask] - predicted[mask]) / actual[mask]).mean()


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return np.sqrt(np.mean((actual - predicted) ** 2))


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(predicted - actual))


def compute_all_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "WAPE": round(wape(actual, predicted), 4),
        "MAPE_nz": round(mape_nonzero(actual, predicted), 4),
        "RMSE": round(rmse(actual, predicted), 4),
        "Bias": round(bias(actual, predicted), 4),
    }


# ── Splitting ────────────────────────────────────────────────────────────────

def split_train_val_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Temporal split: train ≤ TRAIN_END, val ≤ VAL_END, test = rest."""
    train_end = pd.Period(TRAIN_END, freq="M")
    val_end = pd.Period(VAL_END, freq="M")
    df_train = df[df["Период"] <= train_end].copy()
    df_val = df[(df["Период"] > train_end) & (df["Период"] <= val_end)].copy()
    df_test = df[df["Период"] > val_end].copy()
    return df_train, df_val, df_test


def rolling_cv_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    horizon_months: int = 3,
    final_train_end: str = "2024-12",
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Walk-forward CV splits that never touch the held-out test window.

    The last training month of fold i is ``final_train_end`` − (n_folds − i)·horizon.
    Each fold validates on the next ``horizon_months`` months.

    With defaults (n_folds=5, horizon=3, final_train_end=2024-12) the folds are:

    - fold 1: train ≤ 2023-12, val 2024-01..2024-03
    - fold 2: train ≤ 2024-03, val 2024-04..2024-06
    - fold 3: train ≤ 2024-06, val 2024-07..2024-09
    - fold 4: train ≤ 2024-09, val 2024-10..2024-12
    - fold 5: train ≤ 2024-12, val 2025-01..2025-03

    This leaves 2025-04..2025-06 as a silent inner buffer and 2025-07..2026-02
    (the true test window) strictly untouched.
    """
    final_train = pd.Period(final_train_end, freq="M")
    out: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for i in range(n_folds):
        shift = (n_folds - 1 - i) * horizon_months
        train_end = final_train - shift
        val_start = train_end + 1
        val_end = train_end + horizon_months
        df_train = df[df["Период"] <= train_end].copy()
        df_val = df[(df["Период"] >= val_start) & (df["Период"] <= val_end)].copy()
        out.append((df_train, df_val))
    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns (exclude IDs, target, and text columns)."""
    exclude = {
        "Период", "Партнер", "Артикул", "Бренд", "Группа_товара",
        "Сегмент_ABC", "Канал", "Тип_соглашения", "target_qty",
        "Количество_sales", "Выручка_sales", "Количество_ship", "Выручка_ship",
        "Количество_tt", "Стоимость_tt", "Количество_orc", "Стоимость_orc",
        "Количество_receipts", "ЦенаВВалюте", "implied_unit_price", "Номенклатура",
    }
    return [c for c in df.columns if c not in exclude and df[c].dtype.kind in ("f", "i", "u")]
