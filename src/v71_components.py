"""V7.1 components on top of V7.

Pieces
------
- ``MultiQuantileBundle``: one classifier (shared) + K pinball regressors at
  different α values.  At inference we interpolate between the K quantiles to
  hit a per-row target α (the newsvendor ratio).
- ``newsvendor_alpha_per_sku``: computes optimal α for each row from the SKU
  margin table (margin / (margin + monthly-holding)).
- ``build_recency_weights``: γ^(months_ago) sample weights (cap + floor).
- ``build_monotone_constraints``: returns the LightGBM monotone-constraint
  vector for the current feature order.
- ``iterative_impute_stockouts``: one EM round — replace stockout-censored
  target with ``max(target, model_pred)`` before retraining.
- ``business_cost_objective``: factory for a per-row LightGBM custom
  objective that directly minimises UAH cost using per-row (margin, holding,
  price) weights.  Returned as a closure suitable for
  ``reg_objective_kwargs`` plumbing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.model_v2 import TwoStageForecaster

log = logging.getLogger(__name__)


# ── Recency weights ─────────────────────────────────────────────────────────

def build_recency_weights(
    df: pd.DataFrame,
    anchor: pd.Period | None = None,
    gamma: float = 0.97,
    floor: float = 0.25,
) -> np.ndarray:
    """``w_i = clip(gamma ** months_ago, floor, 1.0)``.

    ``anchor`` defaults to the max period in ``df``.  ``gamma=0.97`` keeps
    roughly 50 % weight at 24 months ago and ~25 % at 48 months ago.
    """
    periods = df["Период"].astype("period[M]")
    if anchor is None:
        anchor = periods.max()
    months_ago = (anchor.ordinal - periods.apply(lambda p: p.ordinal)).to_numpy()
    w = np.clip(gamma ** np.maximum(months_ago, 0), floor, 1.0).astype(np.float32)
    return w


# ── Monotone constraints ────────────────────────────────────────────────────

MONOTONE_POSITIVE = {
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "rmean_3", "rmean_6", "rmean_12",
    "rmed_3", "rmed_6",
    "partner_total", "demand_density",
    "promo_depth_pct_current",
    "cohort_demand_lag1",
}
MONOTONE_NEGATIVE = {
    "stockout_orc", "stockout_tt", "stockout_both",
    "stockout_orc_prev", "stockout_tt_prev",
    "cohort_stockout_share_lag1",
}


def build_monotone_constraints(feature_cols: list[str], mode: str = "full") -> list[int]:
    """Return a per-feature monotone-constraint vector (+1/-1/0).

    ``mode``
        ``"full"``          : positive on lag/rolling/partner, negative on stockout
        ``"stockout_only"`` : only force stockout features to be non-increasing
        ``"lags_only"``     : only force lag_{1,2,3} to be non-decreasing
    """
    pos = MONOTONE_POSITIVE
    neg = MONOTONE_NEGATIVE
    if mode == "stockout_only":
        pos = set()
    elif mode == "lags_only":
        pos = {"lag_1", "lag_2", "lag_3"}
        neg = set()
    out = []
    for f in feature_cols:
        if f in pos:
            out.append(1)
        elif f in neg:
            out.append(-1)
        else:
            out.append(0)
    return out


# ── Newsvendor α per row ────────────────────────────────────────────────────

def newsvendor_alpha_per_sku(
    df: pd.DataFrame,
    margin_table: pd.DataFrame,
    alpha_floor: float = 0.30,
    alpha_ceiling: float = 0.55,
    planning_horizon_months: float = 3.0,
    shrink_to: float = 0.45,
    shrink_weight: float = 0.5,
) -> np.ndarray:
    """Per-row α for cost-calibrated quantile prediction.

    We compute the textbook newsvendor ratio
    ``α = margin / (margin + holding_monthly × planning_horizon)``
    but use an effective planning horizon > 1 month to reflect the reality
    that over-forecast stock doesn't disappear after one period, and then
    shrink toward the empirically optimal ``shrink_to`` (V7's α=0.45) with
    weight ``shrink_weight``.  Finally we clip to ``[alpha_floor, alpha_ceiling]``
    so we stay inside the range where the quantile bundle was well-trained.
    """
    m = margin_table[["Артикул", "margin_rate", "holding_rate_annual"]].copy()
    joined = df[["Артикул"]].merge(m, on="Артикул", how="left")
    margin = joined["margin_rate"].to_numpy(dtype=np.float64)
    holding_annual = joined["holding_rate_annual"].to_numpy(dtype=np.float64)

    median_margin = float(np.nanmedian(m["margin_rate"].to_numpy()))
    median_holding = float(np.nanmedian(m["holding_rate_annual"].to_numpy()))
    margin = np.where(np.isnan(margin), median_margin, margin)
    holding_annual = np.where(np.isnan(holding_annual), median_holding, holding_annual)
    holding_horizon = holding_annual / 12.0 * max(planning_horizon_months, 0.1)

    raw_alpha = margin / np.maximum(margin + holding_horizon, 1e-6)
    alpha = shrink_weight * shrink_to + (1.0 - shrink_weight) * raw_alpha
    return np.clip(alpha, alpha_floor, alpha_ceiling).astype(np.float32)


# ── Multi-quantile bundle ───────────────────────────────────────────────────

@dataclass
class MultiQuantileBundle:
    """One classifier + K pinball regressors at different α levels.

    Trained via ``fit(df_train, df_val, feature_cols, ...)``.  ``predict`` takes
    a dataframe plus a per-row α vector and returns the interpolated quantile
    prediction × calibrated P(nonzero).
    """

    alphas: tuple[float, ...] = (0.20, 0.35, 0.45, 0.55, 0.70)
    target_col: str = "target_qty_imputed"
    clf: lgb.Booster | None = None
    regs: dict[float, lgb.Booster] = field(default_factory=dict)
    feature_cols: list[str] = field(default_factory=list)
    calibrator: object | None = None  # fitted IsotonicCalibrator

    # Training ---------------------------------------------------------------

    def fit(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        feature_cols: list[str],
        num_boost_round: int = 1200,
        early_stopping: int = 60,
        sample_weight_train: np.ndarray | None = None,
        sample_weight_val: np.ndarray | None = None,
        monotone_constraints: list[int] | None = None,
        clf_params_override: dict | None = None,
        reg_params_override: dict | None = None,
    ) -> "MultiQuantileBundle":
        self.feature_cols = feature_cols

        # --- Stage 1: binary classifier (shared across all quantiles) -----
        t0 = time.time()
        y_clf_train = (df_train["target_qty"] > 0).astype(int)
        y_clf_val = (df_val["target_qty"] > 0).astype(int)
        ts = lgb.Dataset(df_train[feature_cols], label=y_clf_train,
                         weight=sample_weight_train, categorical_feature="auto")
        vs = lgb.Dataset(df_val[feature_cols], label=y_clf_val,
                         weight=sample_weight_val, categorical_feature="auto")
        clf_params = {
            "objective": "binary", "metric": "binary_logloss",
            "num_leaves": 63, "learning_rate": 0.05,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
            "min_child_samples": 20, "is_unbalance": True,
            "n_jobs": -1, "device": "cpu", "verbose": -1,
            **(clf_params_override or {}),
        }
        if monotone_constraints is not None:
            clf_params["monotone_constraints"] = list(monotone_constraints)
            clf_params.setdefault("monotone_constraints_method", "advanced")
        self.clf = lgb.train(
            clf_params, ts, num_boost_round,
            valid_sets=[vs],
            callbacks=[lgb.early_stopping(early_stopping), lgb.log_evaluation(0)],
        )
        log.info("bundle: classifier trained (%d rounds, %.1fs)",
                 self.clf.current_iteration(), time.time() - t0)

        # --- Stage 2: one pinball regressor per α ------------------------
        mask_train = (df_train["target_qty"] > 0).to_numpy()
        mask_val = (df_val["target_qty"] > 0).to_numpy()
        sw_tr = (sample_weight_train[mask_train]
                 if sample_weight_train is not None else None)
        sw_va = (sample_weight_val[mask_val]
                 if sample_weight_val is not None else None)
        X_tr = df_train.loc[mask_train, feature_cols]
        X_va = df_val.loc[mask_val, feature_cols]
        y_tr = df_train.loc[mask_train, self.target_col]
        y_va = df_val.loc[mask_val, self.target_col]

        base_reg_params = {
            "objective": "quantile",
            "metric": "quantile",
            "num_leaves": 127, "learning_rate": 0.05,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
            "min_child_samples": 10,
            "n_jobs": -1, "device": "cpu", "verbose": -1,
            **(reg_params_override or {}),
        }
        if monotone_constraints is not None:
            base_reg_params["monotone_constraints"] = list(monotone_constraints)
            base_reg_params.setdefault("monotone_constraints_method", "advanced")

        for alpha in self.alphas:
            t1 = time.time()
            params = dict(base_reg_params)
            params["alpha"] = float(alpha)
            ts2 = lgb.Dataset(X_tr, label=y_tr, weight=sw_tr,
                              categorical_feature="auto")
            vs2 = lgb.Dataset(X_va, label=y_va, weight=sw_va,
                              categorical_feature="auto")
            booster = lgb.train(
                params, ts2, num_boost_round,
                valid_sets=[vs2],
                callbacks=[lgb.early_stopping(early_stopping), lgb.log_evaluation(0)],
            )
            self.regs[float(alpha)] = booster
            log.info("bundle: quantile α=%.2f trained (%d rounds, %.1fs)",
                     alpha, booster.current_iteration(), time.time() - t1)
        return self

    # Inference --------------------------------------------------------------

    def _predict_quantile_matrix(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = df[self.feature_cols]
        Q = np.column_stack([self.regs[a].predict(X) for a in self.alphas]).clip(min=0)
        # enforce monotonic α → quantile by row-wise cumulative max
        Q = np.maximum.accumulate(Q, axis=1).astype(np.float32)
        alphas = np.asarray(self.alphas, dtype=np.float32)
        return Q, alphas

    def predict_at_alpha(
        self,
        df: pd.DataFrame,
        alpha_per_row: np.ndarray,
        apply_classifier: bool = True,
    ) -> np.ndarray:
        """Interpolate the per-row quantile forecast at each row's target α."""
        Q, alphas = self._predict_quantile_matrix(df)
        a = np.clip(alpha_per_row.astype(np.float32), alphas[0], alphas[-1])
        idx = np.searchsorted(alphas, a, side="right") - 1
        idx = np.clip(idx, 0, len(alphas) - 2)
        a_lo = alphas[idx]; a_hi = alphas[idx + 1]
        w_hi = (a - a_lo) / np.maximum(a_hi - a_lo, 1e-9)
        q_lo = Q[np.arange(len(Q)), idx]
        q_hi = Q[np.arange(len(Q)), idx + 1]
        q = q_lo + w_hi * (q_hi - q_lo)

        if apply_classifier:
            p_nz = self.clf.predict(df[self.feature_cols])
            if self.calibrator is not None:
                p_nz = self.calibrator.transform(p_nz)
            q = np.asarray(q, dtype=np.float32) * np.asarray(p_nz, dtype=np.float32)
        return np.clip(q, 0, None).astype(np.float32)


# ── Iterative imputation (one EM round) ─────────────────────────────────────

def iterative_impute_stockouts(
    abt: pd.DataFrame,
    pred: np.ndarray,
    stockout_col: str = "stockout_orc",
    density_col: str = "demand_density",
    density_threshold: float = 0.3,
    shrink: float = 0.85,
) -> pd.DataFrame:
    """Replace ``target_qty_imputed`` on stockout rows with
    ``max(existing, shrink * model_pred)`` when ``demand_density`` is high.

    The resulting dataframe has the same shape as ``abt`` and an updated
    ``target_qty_imputed`` column.  A new flag ``em_refined`` tracks which
    rows were touched.
    """
    out = abt.copy()
    pred = np.asarray(pred, dtype=np.float32)
    stockout = out[stockout_col].astype(bool).to_numpy() if stockout_col in out else np.zeros(len(out), bool)
    dense = (out.get(density_col, pd.Series(0, index=out.index)).to_numpy() >= density_threshold)
    raw_target = out.get("target_qty_imputed",
                          out.get("target_qty", pd.Series(0.0, index=out.index))).to_numpy(dtype=np.float32)
    new_target = raw_target.copy()
    mask = stockout & dense
    new_target[mask] = np.maximum(raw_target[mask], shrink * pred[mask])
    out["target_qty_imputed"] = new_target
    out["em_refined"] = mask.astype(np.int8)
    log.info("iterative_impute_stockouts: updated %d rows (%.2f%%)",
             int(mask.sum()), 100.0 * mask.mean())
    return out


# ── Business-cost objective (LightGBM custom fobj) ──────────────────────────

def make_business_cost_objective(
    price: np.ndarray,
    margin: np.ndarray,
    holding_monthly: np.ndarray,
    recovery: float = 0.5,
    hess_floor: float = 1e-3,
):
    """Return ``fobj(preds, dataset)`` that minimises per-row UAH cost.

    Per-row piece-wise linear loss::

        L_i = margin_i * price_i * (1 - recovery) * max(y_i - p_i, 0)
            + holding_monthly_i * price_i      * max(p_i - y_i, 0)

    Gradients with respect to the (positive) prediction::

        grad_i =  +holding_monthly_i * price_i  if p_i > y_i
                  -margin_i * price_i * (1-rec) if p_i < y_i
                   0                            if p_i == y_i

    LightGBM needs a positive hessian; we use a constant floor.
    Arrays must align with the *regressor* training rows (positive-demand
    rows only).
    """
    under_cost = margin.astype(np.float64) * price.astype(np.float64) * (1.0 - recovery)
    over_cost = holding_monthly.astype(np.float64) * price.astype(np.float64)

    def fobj(preds, dataset):
        y = dataset.get_label()
        diff = preds - y
        grad = np.where(diff > 0, over_cost, -under_cost)
        grad[diff == 0] = 0.0
        hess = np.full_like(grad, max(hess_floor, 1e-3))
        return grad, hess

    return fobj
