"""V4: Creative architectural rethinks to break through the performance ceiling.

Approaches:
1. PerChannelEnsemble — train separate specialists per channel (ИМ/СК/НКП/РС)
2. LogTargetForecaster — regressor predicts log1p(target), inverse transform
3. HierarchicalTopDown — predict partner-month total + SKU share within partner
4. StackingEnsemble — learn optimal blend of V3 + baselines via validation residuals
5. RatioTarget — predict qty/rolling_mean (stationary signal, easier to learn)
"""

from __future__ import annotations

import logging
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluation import compute_all_metrics, wape
from src.model_v2 import GRP, TwoStageForecaster, get_feature_columns_v2

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 1. PER-CHANNEL SPECIALISTS
# ═══════════════════════════════════════════════════════════════════════════

class PerChannelEnsemble:
    """Train a specialized two-stage model per channel (ИМ, СК, НКП, РС).

    Rationale: ИМ (online) WAPE = 1.20, СК (specialty chains) = 0.47 — they're
    fundamentally different demand regimes. One model per channel trades sample
    size for domain-specificity.
    """

    def __init__(self, base_params: dict | None = None):
        self.base_params = base_params or {}
        self.models: dict[str, TwoStageForecaster] = {}
        self.feature_cols: list[str] = []

    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame, feature_cols: list[str]) -> "PerChannelEnsemble":
        self.feature_cols = feature_cols
        for ch in df_train["Канал"].cat.categories:
            train_ch = df_train[df_train["Канал"] == ch]
            val_ch = df_val[df_val["Канал"] == ch]
            if len(train_ch) < 1000 or len(val_ch) < 100:
                log.info("Skipping channel %s (too few rows)", ch)
                continue
            log.info("Training channel %s: train=%d, val=%d", ch, len(train_ch), len(val_ch))
            m = TwoStageForecaster(
                clf_params={"num_leaves": 127, "learning_rate": 0.05, "is_unbalance": True},
                reg_params={"num_leaves": 255, "learning_rate": 0.03, "tweedie_variance_power": 1.5},
            )
            m.fit(train_ch, val_ch, feature_cols, num_boost_round=800, early_stopping=50)
            self.models[ch] = m
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = np.zeros(len(df))
        for ch, model in self.models.items():
            mask = (df["Канал"] == ch).values
            if mask.sum() > 0:
                preds[mask] = model.predict(df[mask])
        return preds


# ═══════════════════════════════════════════════════════════════════════════
# 2. LOG-TARGET TRANSFORMATION
# ═══════════════════════════════════════════════════════════════════════════

class LogTargetForecaster:
    """Two-stage where the regressor predicts log1p(target), addressing
    the heavy-tail under-prediction bias for high-volume SKUs."""

    def __init__(self, clf_params: dict | None = None, reg_params: dict | None = None):
        self.clf_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 127,
            "learning_rate": 0.03,
            "is_unbalance": True,
            "n_jobs": -1,
            "device": "cpu",
            "verbose": -1,
            **(clf_params or {}),
        }
        self.reg_params = {
            "objective": "regression",
            "metric": "mae",
            "num_leaves": 255,
            "learning_rate": 0.02,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.8,
            "bagging_freq": 3,
            "min_child_samples": 20,
            "lambda_l2": 1.0,
            "n_jobs": -1,
            "device": "cpu",
            "verbose": -1,
            **(reg_params or {}),
        }
        self.clf: lgb.Booster | None = None
        self.reg: lgb.Booster | None = None
        self.feature_cols: list[str] = []

    def fit(self, df_train, df_val, feature_cols, num_boost_round=1500, early_stopping=80):
        self.feature_cols = feature_cols

        # Classifier
        y_clf_train = (df_train["target_qty"] > 0).astype(int)
        y_clf_val = (df_val["target_qty"] > 0).astype(int)
        ts1 = lgb.Dataset(df_train[feature_cols], label=y_clf_train, categorical_feature="auto")
        vs1 = lgb.Dataset(df_val[feature_cols], label=y_clf_val, categorical_feature="auto")
        self.clf = lgb.train(
            self.clf_params, ts1, num_boost_round,
            valid_sets=[vs1],
            callbacks=[lgb.early_stopping(early_stopping), lgb.log_evaluation(300)],
        )

        # Regressor on log1p(target) for positive rows
        mask_t = df_train["target_qty"] > 0
        mask_v = df_val["target_qty"] > 0
        y_reg_train = np.log1p(df_train.loc[mask_t, "target_qty"])
        y_reg_val = np.log1p(df_val.loc[mask_v, "target_qty"])
        ts2 = lgb.Dataset(df_train.loc[mask_t, feature_cols], label=y_reg_train, categorical_feature="auto")
        vs2 = lgb.Dataset(df_val.loc[mask_v, feature_cols], label=y_reg_val, categorical_feature="auto")
        self.reg = lgb.train(
            self.reg_params, ts2, num_boost_round,
            valid_sets=[vs2],
            callbacks=[lgb.early_stopping(early_stopping), lgb.log_evaluation(300)],
        )
        return self

    def predict(self, df):
        p_nz = self.clf.predict(df[self.feature_cols])
        log_qty = self.reg.predict(df[self.feature_cols])
        qty = np.expm1(log_qty).clip(min=0)
        return p_nz * qty


# ═══════════════════════════════════════════════════════════════════════════
# 3. HIERARCHICAL RECONCILIATION
# ═══════════════════════════════════════════════════════════════════════════

class HierarchicalReconciler:
    """Forecast partner-month total accurately (aggregate = cleaner signal),
    then RESCALE SKU-level predictions so each (partner, month) sum matches
    the partner-total forecast.

    This is MinT-style top-down reconciliation: aggregate forecasts anchor
    the noisy SKU-level forecasts without throwing away their per-SKU signal.

    Rationale: at (Partner × Month) the volume is ~200-2000 units — much more
    stable than (Partner × SKU × Month) ~0-10 unit noise. A strong aggregate
    forecast caps accumulated SKU-level bias.
    """

    def __init__(self):
        self.partner_model: lgb.Booster | None = None
        self.partner_feature_cols: list[str] = []

    def _build_partner_df(self, df: pd.DataFrame) -> pd.DataFrame:
        agg = (
            df.groupby(["Партнер", "Период"], as_index=False, observed=True)
            .agg(partner_qty=("target_qty", "sum"))
        )
        agg = agg.sort_values(["Партнер", "Период"]).reset_index(drop=True)
        g = agg.groupby("Партнер", sort=False, observed=True)["partner_qty"]
        for lag in [1, 2, 3, 6, 12]:
            agg[f"p_lag_{lag}"] = g.shift(lag).fillna(0)
        # Rolling means off lag_1 (safe, no leakage)
        shifted = g.shift(1)
        agg["p_rmean_3"] = shifted.rolling(3, min_periods=1).mean().reset_index(0, drop=True).fillna(0)
        agg["p_rmean_6"] = shifted.rolling(6, min_periods=1).mean().reset_index(0, drop=True).fillna(0)
        agg["p_rmean_12"] = shifted.rolling(12, min_periods=1).mean().reset_index(0, drop=True).fillna(0)
        agg["month"] = agg["Период"].dt.month.astype(np.int8)
        agg["quarter"] = agg["Период"].dt.quarter.astype(np.int8)
        agg["year"] = agg["Период"].dt.year.astype(np.int16)
        return agg

    def fit(self, df_train, df_val):
        pt_train = self._build_partner_df(df_train)
        pt_val = self._build_partner_df(df_val)
        feat = [c for c in pt_train.columns
                if c not in ("Партнер", "Период", "partner_qty")]
        self.partner_feature_cols = feat

        params = {
            "objective": "regression_l1", "metric": "mae",
            "num_leaves": 63, "learning_rate": 0.03,
            "feature_fraction": 0.9, "bagging_fraction": 0.85, "bagging_freq": 3,
            "min_child_samples": 10, "lambda_l2": 1.0,
            "n_jobs": -1, "device": "cpu", "verbose": -1,
        }
        ts = lgb.Dataset(pt_train[feat], label=pt_train["partner_qty"])
        vs = lgb.Dataset(pt_val[feat], label=pt_val["partner_qty"])
        self.partner_model = lgb.train(
            params, ts, 2000, valid_sets=[vs],
            callbacks=[lgb.early_stopping(80), lgb.log_evaluation(300)],
        )
        log.info("Partner-total model: %d rounds", self.partner_model.current_iteration())
        return self

    def predict_partner_totals(self, df: pd.DataFrame) -> pd.DataFrame:
        pt = self._build_partner_df(df)
        pt["partner_total_pred"] = self.partner_model.predict(pt[self.partner_feature_cols]).clip(min=0)
        return pt[["Партнер", "Период", "partner_total_pred"]]

    def reconcile(self, df: pd.DataFrame, sku_preds: np.ndarray) -> np.ndarray:
        """Rescale sku_preds so per-(Partner,Period) sums match partner-total forecast."""
        pt_forecast = self.predict_partner_totals(df)
        df_work = df[["Партнер", "Период"]].copy()
        df_work["sku_pred"] = sku_preds
        # Sum of raw SKU preds per (Partner, Period)
        df_work["raw_sum"] = df_work.groupby(
            ["Партнер", "Период"], observed=True
        )["sku_pred"].transform("sum")
        df_work = df_work.merge(pt_forecast, on=["Партнер", "Период"], how="left")
        df_work["partner_total_pred"] = df_work["partner_total_pred"].fillna(df_work["raw_sum"])

        # Scale factor (cap to avoid extreme rescaling)
        scale = np.where(
            df_work["raw_sum"].values > 0.1,
            df_work["partner_total_pred"].values / df_work["raw_sum"].values,
            1.0,
        )
        scale = np.clip(scale, 0.3, 3.0)  # dampen extreme corrections
        return sku_preds * scale


# ═══════════════════════════════════════════════════════════════════════════
# 4. STACKING ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════

def stacking_ensemble_predict(
    base_predictions: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    """Weighted blend of base model predictions. Weights learned on validation."""
    total_weight = sum(weights.values())
    result = np.zeros(len(next(iter(base_predictions.values()))))
    for name, preds in base_predictions.items():
        result += (weights[name] / total_weight) * preds
    return result


def optimize_ensemble_weights(
    val_predictions: dict[str, np.ndarray],
    val_actual: np.ndarray,
) -> dict[str, float]:
    """Find convex combination of predictions minimizing WAPE using SLSQP."""
    from scipy.optimize import minimize

    names = list(val_predictions.keys())
    P = np.column_stack([val_predictions[n] for n in names])
    y = val_actual

    def loss(w):
        pred = P @ w
        tot = np.abs(y).sum()
        return np.abs(y - pred).sum() / tot if tot > 0 else 0.0

    n = len(names)
    x0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    best = None
    best_loss = np.inf
    # Multi-start to avoid local minima
    for seed in range(5):
        rng = np.random.default_rng(seed)
        x_start = rng.dirichlet(np.ones(n))
        res = minimize(loss, x_start, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 200, "ftol": 1e-7})
        if res.fun < best_loss:
            best_loss = res.fun
            best = res.x

    # Round very small weights to zero then renormalize
    w = np.where(best < 0.02, 0.0, best)
    if w.sum() > 0:
        w = w / w.sum()
    else:
        w = np.ones(n) / n

    result = {n: float(round(w[i], 3)) for i, n in enumerate(names)}
    log.info("Optimal ensemble weights: %s (val WAPE=%.4f)", result, best_loss)
    return result
