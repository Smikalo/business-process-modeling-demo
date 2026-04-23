"""Reusable components for the V7 ensemble."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge

log = logging.getLogger(__name__)

SEGMENT_KEYS = ["Бренд", "Канал"]


# ── Per-segment residual corrector ─────────────────────────────────────────

@dataclass
class SegmentResidualCorrector:
    """One tiny LGB per (brand, channel) predicting the residual of a base
    forecast on the V7 feature matrix.

    Segments with < `min_rows` calibration rows fall back to predicting 0
    (no correction)."""

    feature_cols: list[str] = field(default_factory=list)
    min_rows: int = 500
    lgb_params: dict = field(default_factory=lambda: {
        "objective": "regression_l1",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "n_jobs": -1,
        "verbose": -1,
    })
    _boosters: dict[tuple, lgb.Booster] = field(default_factory=dict)
    _segments: set = field(default_factory=set)

    def fit(self, df: pd.DataFrame, residual: np.ndarray, n_rounds: int = 200) -> "SegmentResidualCorrector":
        self._segments = set()
        self._boosters = {}
        df = df.copy()
        df["__resid"] = residual
        for key, grp in df.groupby(SEGMENT_KEYS, observed=True):
            if len(grp) < self.min_rows:
                continue
            ts = lgb.Dataset(
                grp[self.feature_cols], label=grp["__resid"].to_numpy(),
                categorical_feature="auto",
            )
            booster = lgb.train(self.lgb_params, ts, num_boost_round=n_rounds)
            self._boosters[tuple(key)] = booster
            self._segments.add(tuple(key))
        log.info("residual corrector: trained on %d segments (of %d candidates)",
                 len(self._segments), df.groupby(SEGMENT_KEYS, observed=True).ngroups)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(df), dtype=np.float32)
        idx_arr = np.arange(len(df))
        for (brand, channel), booster in self._boosters.items():
            mask = (df["Бренд"].to_numpy() == brand) & (df["Канал"].to_numpy() == channel)
            if mask.any():
                out[idx_arr[mask]] = booster.predict(df.loc[mask, self.feature_cols]).astype(np.float32)
        return out


# ── Isotonic classifier calibration ────────────────────────────────────────

class IsotonicCalibrator:
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, p_raw: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        self.iso.fit(p_raw, y_true)
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        return self.iso.transform(p_raw)


# ── Per-segment conformal intervals ────────────────────────────────────────

@dataclass
class PerSegmentConformal:
    """Split-conformal intervals per segment on log1p-space residuals."""

    low: float = 0.1
    high: float = 0.9
    min_rows: int = 100
    _global: tuple[float, float] = (0.0, 0.0)
    _seg: dict[tuple, tuple[float, float]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> "PerSegmentConformal":
        resid = np.log1p(np.clip(y_true, 0, None)) - np.log1p(np.clip(y_pred, 0, None))
        g_lo = float(np.quantile(resid, self.low))
        g_hi = float(np.quantile(resid, self.high))
        self._global = (g_lo, g_hi)
        self._seg = {}
        df = df.copy()
        df["__r"] = resid
        for key, grp in df.groupby(SEGMENT_KEYS, observed=True):
            if len(grp) < self.min_rows:
                continue
            self._seg[tuple(key)] = (
                float(np.quantile(grp["__r"], self.low)),
                float(np.quantile(grp["__r"], self.high)),
            )
        log.info("conformal: %d segment-specific intervals + 1 global fallback",
                 len(self._seg))
        return self

    def intervals(self, df: pd.DataFrame, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logp = np.log1p(np.clip(y_pred, 0, None))
        low = np.full(len(df), self._global[0], dtype=np.float32)
        high = np.full(len(df), self._global[1], dtype=np.float32)
        for (brand, channel), (lo, hi) in self._seg.items():
            mask = (df["Бренд"].to_numpy() == brand) & (df["Канал"].to_numpy() == channel)
            if mask.any():
                low[mask] = lo
                high[mask] = hi
        lower = np.clip(np.expm1(logp + low), 0, None)
        upper = np.clip(np.expm1(logp + high), 0, None)
        return lower.astype(np.float32), upper.astype(np.float32)


# ── Ridge meta-learner for V4+V5+V6+V7 stacking ───────────────────────────

class RidgeStacker:
    """Non-negative ridge over base-model predictions.

    Missing base predictions are imputed with the cross-model mean; weights
    are clipped to [0, +inf) so the ensemble never points in the opposite
    direction of a base signal."""

    def __init__(self, alpha: float = 1.0):
        self.ridge = Ridge(alpha=alpha, positive=True, fit_intercept=True)
        self.base_names: list[str] = []

    def fit(self, preds: dict[str, np.ndarray], y_true: np.ndarray) -> "RidgeStacker":
        self.base_names = list(preds.keys())
        X = np.column_stack([preds[n] for n in self.base_names]).astype(np.float32)
        self.ridge.fit(X, y_true)
        log.info("stacker weights: %s",
                 {n: round(float(w), 3) for n, w in zip(self.base_names, self.ridge.coef_)})
        return self

    def predict(self, preds: dict[str, np.ndarray]) -> np.ndarray:
        for n in self.base_names:
            if n not in preds:
                raise KeyError(f"stacker missing base '{n}' at predict time")
        X = np.column_stack([preds[n] for n in self.base_names]).astype(np.float32)
        return np.clip(self.ridge.predict(X), 0, None)
