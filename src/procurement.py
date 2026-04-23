"""Multi-horizon forecasting and procurement recommendation engine."""

from __future__ import annotations

import logging

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluation import get_feature_columns

log = logging.getLogger(__name__)


def _train_quantile_model(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    alpha: float,
    base_params: dict,
) -> lgb.Booster:
    feature_cols = get_feature_columns(df_train)
    params = {
        **base_params,
        "objective": "quantile",
        "alpha": alpha,
        "verbose": -1,
    }
    ts = lgb.Dataset(df_train[feature_cols], label=df_train["target_qty"])
    vs = lgb.Dataset(df_val[feature_cols], label=df_val["target_qty"])
    model = lgb.train(
        params, ts, num_boost_round=300,
        valid_sets=[vs],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    return model


def build_quantile_models(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    base_params: dict,
) -> dict[str, lgb.Booster]:
    """Train point forecast + upper quantile (for safety stock)."""
    models = {}
    for alpha in [0.5, 0.9]:
        log.info("Training quantile model alpha=%.1f...", alpha)
        models[f"q{int(alpha*100)}"] = _train_quantile_model(
            df_train, df_val, alpha, base_params
        )
    return models


def forecast_multihorizon(
    model: lgb.Booster,
    df_latest: pd.DataFrame,
    horizons: list[int] = [1, 3, 6],
) -> pd.DataFrame:
    """Generate point forecasts at multiple horizons.

    For h=1, uses actual features. For h>1, this is an approximation:
    we reuse h=1 features (iterative multi-step is out of scope for PoC).
    """
    feature_cols = get_feature_columns(df_latest)
    preds = model.predict(df_latest[feature_cols]).clip(min=0)

    result = df_latest[["Партнер", "Артикул", "Период"]].copy()
    for h in horizons:
        result[f"forecast_h{h}"] = preds * h
    return result


def compute_order_recommendations(
    point_model: lgb.Booster,
    upper_model: lgb.Booster,
    df_latest: pd.DataFrame,
    lead_time_months: int = 2,
    service_level_z: float = 1.28,
) -> pd.DataFrame:
    """Compute SKU-level order recommendations.

    Logic:
    - Demand over lead time = point_forecast × lead_time_months
    - Safety stock = (upper_q90 - point_q50) × sqrt(lead_time)
    - Reorder point = lead_time_demand + safety_stock
    - Order qty = max(0, reorder_point - current_stock)
    """
    feature_cols = get_feature_columns(df_latest)

    pred_point = point_model.predict(df_latest[feature_cols]).clip(min=0)
    pred_upper = upper_model.predict(df_latest[feature_cols]).clip(min=0)

    lt = lead_time_months
    demand_lt = pred_point * lt
    safety_stock = (pred_upper - pred_point).clip(min=0) * np.sqrt(lt)
    reorder_point = demand_lt + safety_stock

    current_stock = df_latest["Количество_orc"].values.clip(min=0)
    order_qty = (reorder_point - current_stock).clip(min=0)

    result = df_latest[["Партнер", "Артикул", "Период", "Бренд"]].copy()
    result["forecast_monthly"] = np.round(pred_point, 1)
    result["demand_lead_time"] = np.round(demand_lt, 1)
    result["safety_stock"] = np.round(safety_stock, 1)
    result["reorder_point"] = np.round(reorder_point, 1)
    result["current_stock_orc"] = current_stock
    result["recommended_order"] = np.round(order_qty, 0).astype(int)

    # Filter to actionable recommendations (order > 0)
    actionable = result[result["recommended_order"] > 0].copy()
    actionable = actionable.sort_values("recommended_order", ascending=False)
    log.info(
        "Order recommendations: %d/%d SKU-partner pairs need reorder, total units=%d",
        len(actionable), len(result), actionable["recommended_order"].sum(),
    )
    return actionable
