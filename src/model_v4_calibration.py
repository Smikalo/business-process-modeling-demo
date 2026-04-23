"""V4 post-hoc calibration and GBDT meta-learning.

These operate on already-trained base models' predictions to squeeze additional
accuracy by correcting systematic biases the GBDTs cannot fix on their own.

Methods:
  A. Segmented isotonic calibration — per (channel, volume_tier) isotonic
     regression mapping predicted → actual. Fixes non-linear bias without
     re-training.
  B. GBDT meta-learner stacking — trains a small LightGBM on validation
     predictions from all base models + key features to learn the optimal
     (nonlinear) blend.
  C. Bias correction via shrinkage — per-segment additive bias correction.
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# A. SEGMENTED ISOTONIC CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════

class SegmentedIsotonicCalibrator:
    """Fits one isotonic regression per (channel, volume_tier) segment.

    Maps raw predictions to calibrated predictions that match empirical
    conditional-mean curves on the validation set. This preserves monotonicity
    (higher pred → higher calibrated pred) while correcting systematic bias.
    """

    def __init__(self, segment_cols: list[str] | None = None, min_segment_size: int = 200):
        self.segment_cols = segment_cols or ["Канал", "volume_tier"]
        self.min_segment_size = min_segment_size
        self.calibrators: dict[tuple, IsotonicRegression] = {}
        self.global_calibrator: IsotonicRegression | None = None

    def _get_keys(self, df: pd.DataFrame) -> np.ndarray:
        """Returns positional keys (aligned with 0..n-1 indexing of arrays)."""
        return df[self.segment_cols].astype(str).agg("|".join, axis=1).values

    def fit(self, df_val: pd.DataFrame, preds_val: np.ndarray, actual_val: np.ndarray) -> "SegmentedIsotonicCalibrator":
        self.global_calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0)
        self.global_calibrator.fit(preds_val, actual_val)

        keys = self._get_keys(df_val)
        for key in np.unique(keys):
            idx = np.where(keys == key)[0]
            if len(idx) < self.min_segment_size:
                continue
            sub_pred = preds_val[idx]
            sub_act = actual_val[idx]
            if sub_pred.std() < 1e-6:
                continue
            cal = IsotonicRegression(out_of_bounds="clip", y_min=0)
            cal.fit(sub_pred, sub_act)
            self.calibrators[key] = cal

        log.info("Isotonic calibration: %d segments fit (+ global fallback)", len(self.calibrators))
        return self

    def transform(self, df: pd.DataFrame, preds: np.ndarray) -> np.ndarray:
        out = np.zeros_like(preds, dtype=np.float64)
        keys = self._get_keys(df)
        for key in np.unique(keys):
            idx = np.where(keys == key)[0]
            cal = self.calibrators.get(key, self.global_calibrator)
            out[idx] = cal.transform(preds[idx])
        return out.clip(min=0)


# ═══════════════════════════════════════════════════════════════════════════
# B. GBDT META-LEARNER STACKING
# ═══════════════════════════════════════════════════════════════════════════

class GBDTMetaLearner:
    """Trains a small LightGBM to blend base-model predictions.

    Input features: base_preds (V3, PerChannel, LogTarget, MA, Seasonal) +
    selected context features (Канал, volume_tier, month).

    Learns nonlinear blend, e.g., "when volume_tier=high, trust LogTarget more".
    """

    def __init__(self, context_cols: list[str] | None = None):
        self.context_cols = context_cols or ["Канал", "volume_tier", "partner_volume_tier", "month", "rmean_6"]
        self.model: lgb.Booster | None = None
        self.input_cols: list[str] = []

    def _assemble(self, df: pd.DataFrame, base_preds: dict[str, np.ndarray]) -> pd.DataFrame:
        ctx = df[self.context_cols].copy()
        for name, preds in base_preds.items():
            ctx[f"pred_{name}"] = preds
        return ctx

    def fit(
        self,
        df_val: pd.DataFrame,
        base_preds_val: dict[str, np.ndarray],
        actual_val: np.ndarray,
    ) -> "GBDTMetaLearner":
        X = self._assemble(df_val, base_preds_val)
        # Split val 70/30 for meta-train / meta-early-stop
        n = len(X)
        tr_idx = np.arange(int(n * 0.7))
        va_idx = np.arange(int(n * 0.7), n)

        self.input_cols = X.columns.tolist()

        ts = lgb.Dataset(X.iloc[tr_idx], label=actual_val[tr_idx], categorical_feature="auto")
        vs = lgb.Dataset(X.iloc[va_idx], label=actual_val[va_idx], categorical_feature="auto")

        params = {
            "objective": "regression_l1",
            "metric": "mae",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 3,
            "min_child_samples": 50,
            "lambda_l2": 1.0,
            "n_jobs": -1,
            "device": "cpu",
            "verbose": -1,
        }
        self.model = lgb.train(
            params, ts, 800, valid_sets=[vs],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)],
        )
        log.info("GBDT meta-learner: %d rounds", self.model.current_iteration())
        return self

    def predict(self, df: pd.DataFrame, base_preds: dict[str, np.ndarray]) -> np.ndarray:
        X = self._assemble(df, base_preds)
        X = X[self.input_cols]
        return self.model.predict(X).clip(min=0)


# ═══════════════════════════════════════════════════════════════════════════
# C. SEGMENT BIAS CORRECTION
# ═══════════════════════════════════════════════════════════════════════════

class SegmentBiasCorrector:
    """Per-segment multiplicative + additive bias correction.

    For each (Канал, volume_tier), learns:
        calibrated = a * raw_pred + b
    such that mean squared error is minimized on validation, with shrinkage
    to the identity (a=1, b=0) if segment too small.
    """

    def __init__(self, segment_cols: list[str] | None = None, min_segment: int = 100, shrink: float = 0.3):
        self.segment_cols = segment_cols or ["Канал", "volume_tier"]
        self.min_segment = min_segment
        self.shrink = shrink
        self.params: dict[tuple, tuple[float, float]] = {}

    def _key(self, df) -> np.ndarray:
        return df[self.segment_cols].astype(str).agg("|".join, axis=1).values

    def fit(self, df_val, preds_val, actual_val):
        keys = self._key(df_val)
        for key in np.unique(keys):
            idx = np.where(keys == key)[0]
            if len(idx) < self.min_segment:
                continue
            p = preds_val[idx]
            y = actual_val[idx]
            if p.std() < 1e-6:
                continue
            X = np.column_stack([p, np.ones_like(p)])
            try:
                ab, *_ = np.linalg.lstsq(X, y, rcond=None)
                a, b = float(ab[0]), float(ab[1])
                a_sh = self.shrink * 1.0 + (1 - self.shrink) * a
                b_sh = self.shrink * 0.0 + (1 - self.shrink) * b
                self.params[key] = (a_sh, b_sh)
            except Exception:
                continue
        log.info("Bias corrector: fit %d segments", len(self.params))
        return self

    def transform(self, df, preds):
        keys = self._key(df)
        out = preds.astype(np.float64).copy()
        for key in np.unique(keys):
            if key in self.params:
                idx = np.where(keys == key)[0]
                a, b = self.params[key]
                out[idx] = a * preds[idx] + b
        return out.clip(min=0)
