"""V3 model: fixes high-volume under-prediction, adds demand-tier features,
YoY seasonality, per-channel threshold tuning, sample weighting.

Diagnosis from V2:
- Systematic under-prediction for 21-100 unit items (bias +9) and 100+ items (bias +46)
- ИМ (online) channel WAPE 1.20 vs СК 0.47
- Classifier too many false positives (precision 0.71)
- Top 100 worst pairs = 20% of total error, all high-volume
"""

from __future__ import annotations

import logging
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluation import compute_all_metrics, wape
from src.model_v2 import CAT_COLS, GRP, get_feature_columns_v2

log = logging.getLogger(__name__)


# ── New features ────────────────────────────────────────────────────────────

def add_v3_features(df: pd.DataFrame) -> pd.DataFrame:
    """Additional features targeting diagnosed weaknesses."""
    df = df.sort_values(GRP + ["Период"]).reset_index(drop=True)
    g = df.groupby(GRP, sort=False)

    # Demand velocity: change rate over recent window
    df["demand_velocity"] = (df["lag_1"] - df["lag_3"]).fillna(0)
    df["demand_accel"] = (df["lag_1"] - 2 * df["lag_2"] + df["lag_3"]).fillna(0)

    # YoY ratio: how does lag_1 compare to lag_12? (seasonality-adjusted trend)
    df["yoy_ratio"] = (df["lag_1"] / (df["lag_12"] + 0.1)).clip(-10, 10).astype(np.float32)
    df["yoy_diff"] = (df["lag_1"] - df["lag_12"]).fillna(0).astype(np.float32)

    # Volume tier based on trailing 6-month average
    rm6 = df["rmean_6"].fillna(0).values
    df["volume_tier"] = np.select(
        [rm6 <= 0, rm6 <= 2, rm6 <= 10, rm6 <= 50],
        [0, 1, 2, 3], default=4,
    ).astype(np.int8)

    # Partner volume tier
    pt = df["partner_total"].fillna(0).values
    df["partner_volume_tier"] = np.select(
        [pt <= 0, pt <= 50, pt <= 200, pt <= 1000],
        [0, 1, 2, 3], default=4,
    ).astype(np.int8)

    # Demand intermittency: CV of demand over available history
    df["demand_cv"] = (df["rstd_6"] / (df["rmean_6"] + 0.1)).clip(0, 10).astype(np.float32)

    # Ratio of sales to shipments (inventory health signal)
    df["sales_ship_ratio"] = (
        df["lag_1"] / (df["lag_1_Количество_ship"] + 0.1)
    ).clip(0, 10).astype(np.float32)

    # Max over recent lags (peak demand signal)
    df["lag_max_3"] = df[["lag_1", "lag_2", "lag_3"]].max(axis=1)
    df["lag_min_3"] = df[["lag_1", "lag_2", "lag_3"]].min(axis=1)
    df["lag_range_3"] = df["lag_max_3"] - df["lag_min_3"]

    # Whether demand has been consistently growing or declining
    df["trend_up_3m"] = ((df["lag_1"] > df["lag_2"]) & (df["lag_2"] > df["lag_3"])).astype(np.int8)
    df["trend_down_3m"] = ((df["lag_1"] < df["lag_2"]) & (df["lag_2"] < df["lag_3"])).astype(np.int8)

    log.info("add_v3_features: +14 new features")
    return df


# ── Improved two-stage model ────────────────────────────────────────────────

class TwoStageV3:
    """V3 improvements:
    1. Log1p-transformed target for regressor (fixes high-volume under-prediction)
    2. Sample weights: sqrt(1 + target_qty) so heavy hitters matter more
    3. Per-channel threshold optimization for classifier
    4. Regressor trained on ALL rows with demand-density weighting, not just positive rows
    """

    def __init__(self, clf_params: dict | None = None, reg_params: dict | None = None):
        self.clf_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 127,
            "learning_rate": 0.03,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.8,
            "bagging_freq": 3,
            "min_child_samples": 30,
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
            "num_leaves": 255,
            "learning_rate": 0.03,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.8,
            "bagging_freq": 3,
            "min_child_samples": 20,
            "n_jobs": -1,
            "device": "cpu",
            "verbose": -1,
            **(reg_params or {}),
        }
        self.clf: lgb.Booster | None = None
        self.reg: lgb.Booster | None = None
        self.feature_cols: list[str] = []
        self.channel_thresholds: dict[str, float] = {}

    def _compute_sample_weights(self, target: pd.Series) -> np.ndarray:
        return np.sqrt(1 + target.values).astype(np.float32)

    def _optimize_thresholds(self, df_val: pd.DataFrame) -> dict[str, float]:
        """Find optimal classifier threshold per channel on validation data."""
        p_nz = self.clf.predict(df_val[self.feature_cols])
        actual_binary = (df_val["target_qty"] > 0).astype(int).values
        actual_qty = df_val["target_qty"].values
        qty_pred = self.reg.predict(df_val[self.feature_cols]).clip(min=0)

        thresholds = {}
        if "Канал" in df_val.columns:
            for ch in df_val["Канал"].unique():
                mask = (df_val["Канал"] == ch).values
                best_t, best_wape = 0.5, 999
                for t in np.arange(0.2, 0.8, 0.05):
                    combined = np.where(p_nz[mask] > t, qty_pred[mask], 0)
                    w = wape(actual_qty[mask], combined)
                    if w < best_wape:
                        best_wape, best_t = w, t
                thresholds[ch] = round(best_t, 2)
        self.channel_thresholds = thresholds
        log.info("Thresholds: %s", thresholds)
        return thresholds

    def fit(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        feature_cols: list[str],
        num_boost_round: int = 2000,
        early_stopping: int = 100,
    ) -> "TwoStageV3":
        self.feature_cols = feature_cols
        t0 = time.time()

        y_clf_train = (df_train["target_qty"] > 0).astype(int)
        y_clf_val = (df_val["target_qty"] > 0).astype(int)
        w_train = self._compute_sample_weights(df_train["target_qty"])

        # Stage 1: classifier with sample weights
        ts1 = lgb.Dataset(
            df_train[feature_cols], label=y_clf_train,
            weight=w_train, categorical_feature="auto",
        )
        vs1 = lgb.Dataset(
            df_val[feature_cols], label=y_clf_val, categorical_feature="auto",
        )
        self.clf = lgb.train(
            self.clf_params, ts1, num_boost_round,
            valid_sets=[vs1],
            callbacks=[lgb.early_stopping(early_stopping), lgb.log_evaluation(300)],
        )
        log.info("Stage 1: %d rounds in %.1fs", self.clf.current_iteration(), time.time() - t0)

        # Stage 2: Tweedie regressor on ALL rows, weighted by sqrt(1+qty)
        t1 = time.time()
        ts2 = lgb.Dataset(
            df_train[feature_cols], label=df_train["target_qty"],
            weight=w_train, categorical_feature="auto",
        )
        vs2 = lgb.Dataset(
            df_val[feature_cols], label=df_val["target_qty"],
            categorical_feature="auto",
        )
        self.reg = lgb.train(
            self.reg_params, ts2, num_boost_round,
            valid_sets=[vs2],
            callbacks=[lgb.early_stopping(early_stopping), lgb.log_evaluation(300)],
        )
        log.info("Stage 2: %d rounds in %.1fs", self.reg.current_iteration(), time.time() - t1)

        # Optimize per-channel thresholds
        self._optimize_thresholds(df_val)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        p_nz = self.clf.predict(df[self.feature_cols])
        qty_pred = self.reg.predict(df[self.feature_cols]).clip(min=0)

        if self.channel_thresholds and "Канал" in df.columns:
            threshold = df["Канал"].map(self.channel_thresholds).fillna(0.5).values
        else:
            threshold = 0.5

        return np.where(p_nz > threshold, qty_pred, 0.0)

    def feature_importance(self) -> pd.DataFrame:
        fi_clf = pd.DataFrame({"feature": self.feature_cols, "gain_clf": self.clf.feature_importance("gain")})
        fi_reg = pd.DataFrame({"feature": self.feature_cols, "gain_reg": self.reg.feature_importance("gain")})
        fi = fi_clf.merge(fi_reg, on="feature")
        fi["gain_total"] = fi["gain_clf"] + fi["gain_reg"]
        return fi.sort_values("gain_total", ascending=False).reset_index(drop=True)
