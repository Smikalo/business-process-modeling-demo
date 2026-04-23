"""Baseline forecasters and LightGBM training (CPU-only, zero cost)."""

from __future__ import annotations

import logging
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluation import compute_all_metrics, get_feature_columns

log = logging.getLogger(__name__)


# ── Baselines ────────────────────────────────────────────────────────────────

def baseline_naive(df: pd.DataFrame) -> np.ndarray:
    """Last month's value."""
    return df["lag_1"].values.clip(min=0)


def baseline_seasonal(df: pd.DataFrame) -> np.ndarray:
    """Same month last year."""
    return df["lag_12"].values.clip(min=0)


def baseline_ma3(df: pd.DataFrame) -> np.ndarray:
    """3-month moving average."""
    return df["rmean_3"].values.clip(min=0)


def baseline_ma6(df: pd.DataFrame) -> np.ndarray:
    """6-month moving average."""
    return df["rmean_6"].values.clip(min=0)


def baseline_drift(df: pd.DataFrame) -> np.ndarray:
    """Linear extrapolation from last two months."""
    return (df["lag_1"] + (df["lag_1"] - df["lag_2"])).values.clip(min=0)


BASELINES = {
    "Naive (lag-1)": baseline_naive,
    "Seasonal Naive": baseline_seasonal,
    "MA-3": baseline_ma3,
    "MA-6": baseline_ma6,
    "Drift": baseline_drift,
}


def evaluate_baselines(df_eval: pd.DataFrame) -> pd.DataFrame:
    """Run all baselines on a dataset and return metrics table."""
    actual = df_eval["target_qty"].values
    rows = []
    for name, fn in BASELINES.items():
        preds = fn(df_eval)
        metrics = compute_all_metrics(actual, preds)
        metrics["Model"] = name
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("Model")


# ── LightGBM ────────────────────────────────────────────────────────────────

DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "n_jobs": -1,
    "device": "cpu",
    "verbose": -1,
}


def train_lightgbm(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    params: dict | None = None,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50,
) -> lgb.Booster:
    """Train LightGBM on CPU. Returns the Booster."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    feature_cols = get_feature_columns(df_train)

    train_set = lgb.Dataset(df_train[feature_cols], label=df_train["target_qty"])
    val_set = lgb.Dataset(df_val[feature_cols], label=df_val["target_qty"])

    t0 = time.time()
    model = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(200),
        ],
    )
    elapsed = time.time() - t0
    log.info("train_lightgbm: %d rounds in %.1fs, best_iter=%d",
             model.current_iteration(), elapsed, model.best_iteration)
    return model


def predict_lightgbm(model: lgb.Booster, df: pd.DataFrame) -> np.ndarray:
    feature_cols = get_feature_columns(df)
    return model.predict(df[feature_cols]).clip(min=0)


def feature_importance(model: lgb.Booster, df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = get_feature_columns(df)
    return (
        pd.DataFrame({
            "feature": feature_cols,
            "gain": model.feature_importance("gain"),
            "split": model.feature_importance("split"),
        })
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
