"""Improved model: active-pair filtering, two-stage prediction, Tweedie, categoricals.

Key improvements over v1:
1. Filter to "active" pairs (>=3 nonzero months in trailing 12) — removes 82% of noise rows
2. Clip target to zero (returns are not forecastable demand)
3. Two-stage: classify zero/nonzero, then regress on quantity
4. Tweedie objective (safe after clipping) — better for zero-inflated count data
5. Native LightGBM categorical features for Бренд, Канал, Группа_товара, Сегмент_ABC
6. Proper expanding rolling windows instead of lag approximations
"""

from __future__ import annotations

import logging
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluation import compute_all_metrics, get_feature_columns, wape
from src.losses import resolve_objective

log = logging.getLogger(__name__)


# ── Improved feature engineering ────────────────────────────────────────────

GRP = ["Партнер", "Артикул"]
CAT_COLS = ["Бренд", "Канал", "Группа_товара", "Сегмент_ABC", "Тип_соглашения"]


def filter_active_pairs(df: pd.DataFrame, min_nonzero: int = 3, lookback: int = 12) -> pd.DataFrame:
    """Keep only (Partner, SKU) pairs with >= min_nonzero positive-demand months
    in the most recent `lookback` months of the training portion."""
    max_period = df["Период"].max()
    cutoff = max_period - lookback
    recent = df[df["Период"] > cutoff]

    activity = (
        recent[recent["target_qty"] > 0]
        .groupby(GRP)
        .size()
        .reset_index(name="nz_count")
    )
    active_pairs = activity[activity["nz_count"] >= min_nonzero][GRP]
    df_out = df.merge(active_pairs, on=GRP, how="inner")
    log.info(
        "filter_active: %d → %d rows (%.1f%%), %d active pairs",
        len(df), len(df_out), len(df_out) / len(df) * 100,
        len(active_pairs),
    )
    return df_out


def add_proper_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Replace lag-approximation rolling with actual expanding/rolling windows."""
    df = df.sort_values(GRP + ["Период"]).reset_index(drop=True)
    grp_id = df.groupby(GRP, sort=False).ngroup()
    shifted = df.groupby(GRP, sort=False)["target_qty"].shift(1)

    for w in [3, 6, 12]:
        rm = shifted.groupby(grp_id).rolling(w, min_periods=1).mean()
        df[f"rmean_{w}"] = rm.droplevel(0).values

        if w <= 6:
            rs = shifted.groupby(grp_id).rolling(w, min_periods=2).std()
            df[f"rstd_{w}"] = rs.droplevel(0).values

    df["rcv_6"] = df["rstd_6"] / (df["rmean_6"] + 1e-9)

    # Expanding mean and count of nonzero
    cumsum_nz = (df["target_qty"] > 0).astype(float).groupby(grp_id).cumsum().groupby(grp_id).shift(1)
    cumcount = df.groupby(GRP, sort=False).cumcount()
    df["demand_density"] = (cumsum_nz / cumcount.clip(lower=1)).fillna(0).astype(np.float32)

    # Months since last nonzero demand
    nz_flag = (df["target_qty"] > 0).astype(int)
    streak = nz_flag.groupby(grp_id).apply(
        lambda x: x.groupby((x == 1).cumsum()).cumcount()
    )
    df["months_since_demand"] = streak.droplevel(0).values
    df["months_since_demand"] = df.groupby(GRP, sort=False)["months_since_demand"].shift(1).fillna(0).astype(np.int16)

    roll_cols = [c for c in df.columns if c.startswith(("rmean_", "rstd_", "rcv_"))]
    df[roll_cols] = df[roll_cols].fillna(0.0)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Convert string categoricals to pandas Categorical (LightGBM native support)."""
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def get_feature_columns_v2(df: pd.DataFrame) -> list[str]:
    """Feature columns including categoricals."""
    exclude = {
        "Период", "Партнер", "Артикул", "target_qty",
        "target_qty_imputed",  # V6: imputed target (leakage if used as a feature)
        "Количество_sales", "Выручка_sales", "Количество_ship", "Выручка_ship",
        "Количество_tt", "Стоимость_tt", "Количество_orc", "Стоимость_orc",
        "Количество_receipts", "ЦенаВВалюте", "implied_unit_price", "Номенклатура",
    }
    return [c for c in df.columns if c not in exclude and
            (df[c].dtype.kind in ("f", "i", "u") or df[c].dtype.name == "category")]


# ── Two-stage model ─────────────────────────────────────────────────────────

class TwoStageForecaster:
    """Stage 1: binary classifier (will demand be > 0?).
    Stage 2: Tweedie regressor (how much?) trained only on positive-demand rows.
    Final prediction = P(nonzero) × E[qty | nonzero].
    """

    def __init__(
        self,
        clf_params: dict | None = None,
        reg_params: dict | None = None,
        reg_objective: str | None = None,
        reg_objective_kwargs: dict | None = None,
        target_col: str = "target_qty",
    ):
        self.reg_objective_name = reg_objective
        self.reg_objective_kwargs = dict(reg_objective_kwargs or {})
        self.target_col = target_col
        self.clf_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "is_unbalance": True,
            "n_jobs": -1,
            "device": "cpu",
            "verbose": -1,
            **(clf_params or {}),
        }
        self.reg_params = {
            "objective": "tweedie",
            "tweedie_variance_power": 1.5,
            "metric": "mae",
            "num_leaves": 127,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 10,
            "n_jobs": -1,
            "device": "cpu",
            "verbose": -1,
            **(reg_params or {}),
        }
        self.clf: lgb.Booster | None = None
        self.reg: lgb.Booster | None = None
        self.feature_cols: list[str] = []

    def fit(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        feature_cols: list[str],
        num_boost_round: int = 1000,
        early_stopping: int = 50,
    ) -> "TwoStageForecaster":
        self.feature_cols = feature_cols
        t0 = time.time()

        # Stage 1 always uses raw target_qty (censor-aware imputed targets can
        # keep tiny positive values which we still want classified as "has demand").
        clf_target = "target_qty"
        reg_target = self.target_col

        # Stage 1: binary classifier
        y_clf_train = (df_train[clf_target] > 0).astype(int)
        y_clf_val = (df_val[clf_target] > 0).astype(int)

        ts1 = lgb.Dataset(df_train[feature_cols], label=y_clf_train, categorical_feature="auto")
        vs1 = lgb.Dataset(df_val[feature_cols], label=y_clf_val, categorical_feature="auto")

        self.clf = lgb.train(
            self.clf_params, ts1, num_boost_round,
            valid_sets=[vs1],
            callbacks=[lgb.early_stopping(early_stopping), lgb.log_evaluation(200)],
        )
        log.info("Stage 1 (classifier): %d rounds in %.1fs",
                 self.clf.current_iteration(), time.time() - t0)

        # Stage 2: regressor on positive-demand rows only
        t1 = time.time()
        mask_train = df_train[clf_target] > 0
        mask_val = df_val[clf_target] > 0

        ts2 = lgb.Dataset(
            df_train.loc[mask_train, feature_cols],
            label=df_train.loc[mask_train, reg_target],
            categorical_feature="auto",
        )
        vs2 = lgb.Dataset(
            df_val.loc[mask_val, feature_cols],
            label=df_val.loc[mask_val, reg_target],
            categorical_feature="auto",
        )

        reg_params = dict(self.reg_params)
        fobj, feval, overrides = resolve_objective(
            self.reg_objective_name or "", **self.reg_objective_kwargs
        )
        reg_params.update(overrides)
        if fobj is not None:
            reg_params["objective"] = fobj

        self.reg = lgb.train(
            reg_params, ts2, num_boost_round,
            valid_sets=[vs2],
            feval=feval,
            callbacks=[lgb.early_stopping(early_stopping), lgb.log_evaluation(200)],
        )
        log.info("Stage 2 (regressor): %d rounds in %.1fs",
                 self.reg.current_iteration(), time.time() - t1)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        p_nonzero = self.clf.predict(df[self.feature_cols])
        qty_given_nz = self.reg.predict(df[self.feature_cols]).clip(min=0)
        return (p_nonzero * qty_given_nz).clip(min=0)

    def feature_importance(self) -> pd.DataFrame:
        fi_clf = pd.DataFrame({
            "feature": self.feature_cols,
            "gain_clf": self.clf.feature_importance("gain"),
        })
        fi_reg = pd.DataFrame({
            "feature": self.feature_cols,
            "gain_reg": self.reg.feature_importance("gain"),
        })
        fi = fi_clf.merge(fi_reg, on="feature")
        fi["gain_total"] = fi["gain_clf"] + fi["gain_reg"]
        return fi.sort_values("gain_total", ascending=False).reset_index(drop=True)


# ── Single-stage Tweedie (simpler alternative) ─────────────────────────────

def train_tweedie(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: list[str],
    params: dict | None = None,
) -> lgb.Booster:
    base = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.5,
        "metric": "mae",
        "num_leaves": 127,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 10,
        "n_jobs": -1,
        "device": "cpu",
        "verbose": -1,
        **(params or {}),
    }
    ts = lgb.Dataset(df_train[feature_cols], label=df_train["target_qty"], categorical_feature="auto")
    vs = lgb.Dataset(df_val[feature_cols], label=df_val["target_qty"], categorical_feature="auto")
    model = lgb.train(
        base, ts, 1000,
        valid_sets=[vs],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)],
    )
    log.info("Tweedie: %d rounds, best_iter=%d", model.current_iteration(), model.best_iteration)
    return model
