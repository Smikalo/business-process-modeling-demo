"""V11 Priority 2 -- streaming per-month bias recalibrator.

Wraps a base prediction stream with a one-month-lagged multiplicative
bias correction.  At month `t`, look at the realised actual at month
`t-1` (the most recently observed period), compute the multiplicative
factor that would have made `t-1`'s prediction exactly unbiased, and
apply that factor to month `t`.

Mathematically:
    α_t = clip(actual_{t-1} / pred_{t-1}, α_min, α_max)
    α_smoothed_t = β * α_{realised, t-1} + (1 - β) * α_smoothed_{t-1}
    pred_corrected_t = α_smoothed_t * pred_t

Two extensions:

1. **Per-axis correction**: instead of one global α, fit one α per Канал
   (or per Канал × Сегмент_ABC).  Reuses the same axis taxonomy as the
   LAD reconciliation.

2. **Exponential smoothing** with β ∈ [0.3, 0.7] -- avoids over-reacting
   to single-month outliers while still tracking drift.

The recalibrator runs in TIME-CAUSAL mode: at month `t`, it only sees
months ≤ t-1.  No leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
KEY = ["Период", "Партнер", "Артикул"]


@dataclass
class StreamingCalibratorConfig:
    axes: list[str] | None = None
    beta: float = 0.5
    alpha_clip: tuple[float, float] = (0.6, 1.6)
    cold_start_alpha: float = 1.0


class StreamingCalibrator:
    """Time-causal multiplicative bias recalibrator.

    Usage
    -----
    ```python
    cfg = StreamingCalibratorConfig(axes=["Канал"], beta=0.5)
    calib = StreamingCalibrator(cfg)
    calib.fit_history(df_with_history)   # all val months
    df_test_corrected = calib.transform(df_test_with_meta)
    ```

    Both inputs MUST contain columns: `Период`, axes, `target_qty`,
    `prediction`.  Output adds `prediction_calibrated`.
    """

    def __init__(self, config: StreamingCalibratorConfig):
        self.cfg = config
        self.axes = config.axes if config.axes else []
        self.alpha_history: dict[str, dict[str, float]] = {}

    def _axis_key(self, row: pd.Series | dict) -> str:
        if not self.axes:
            return "_GLOBAL"
        return "|".join(str(row[a]) for a in self.axes)

    def _aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["Период"] + (self.axes if self.axes else [])
        agg = df.groupby(cols, observed=True).agg(
            actual=("target_qty", "sum"),
            predicted=("prediction", "sum"),
        ).reset_index()
        if not self.axes:
            agg["__axis"] = "_GLOBAL"
        else:
            agg["__axis"] = agg[self.axes].astype(str).agg("|".join, axis=1)
        return agg

    def fit_history(self, df: pd.DataFrame) -> "StreamingCalibrator":
        """Walk through validation history, compute per-month α per axis."""
        agg = self._aggregate(df)
        agg["per"] = pd.PeriodIndex(agg["Период"].astype(str), freq="M")
        agg = agg.sort_values(["__axis", "per"])

        beta = self.cfg.beta
        amin, amax = self.cfg.alpha_clip
        cold = self.cfg.cold_start_alpha

        for axis_key, group in agg.groupby("__axis", observed=True):
            ema = cold
            month_to_alpha: dict[str, float] = {}
            for _, r in group.iterrows():
                month_to_alpha[str(r["per"])] = ema
                if r["predicted"] > 1e-3:
                    realized = float(np.clip(r["actual"] / r["predicted"],
                                             amin, amax))
                else:
                    realized = ema
                ema = beta * realized + (1 - beta) * ema
            self.alpha_history[axis_key] = month_to_alpha
            self.alpha_history[axis_key]["__last"] = ema
        return self

    def transform(self, df: pd.DataFrame, fold_in: bool = True) -> pd.DataFrame:
        """Apply calibration.  In TEST mode, uses the LAST EMA (post-history).

        If `fold_in=True`, also walks through any *new* test months in
        time order, updating the EMA after each month -- so by month t
        in test, the calibrator has been updated by months t-1 (and
        earlier) of test as well.  This emulates how a deployed system
        would work.
        """
        out = df.copy()
        if not self.axes:
            out["__axis"] = "_GLOBAL"
        else:
            out["__axis"] = out[self.axes].astype(str).agg("|".join, axis=1)
        out["per"] = pd.PeriodIndex(out["Период"].astype(str), freq="M")

        per_axis_ema = {ax: hist["__last"]
                        for ax, hist in self.alpha_history.items()}

        out = out.sort_values(["__axis", "per", "Партнер", "Артикул"])
        beta = self.cfg.beta
        amin, amax = self.cfg.alpha_clip

        all_periods_sorted = sorted(out["per"].unique())
        out["alpha"] = np.float32(self.cfg.cold_start_alpha)

        for per in all_periods_sorted:
            mask_per = out["per"] == per
            for axis_key in out.loc[mask_per, "__axis"].unique():
                axis_mask = mask_per & (out["__axis"] == axis_key)
                ema = per_axis_ema.get(axis_key, self.cfg.cold_start_alpha)
                out.loc[axis_mask, "alpha"] = np.float32(ema)
            if fold_in:
                ag = (out.loc[mask_per].groupby("__axis", observed=True)
                          .agg(act=("target_qty", "sum"),
                               prd=("prediction", "sum")))
                for axis_key, r in ag.iterrows():
                    if r["prd"] > 1e-3:
                        realized = float(np.clip(r["act"] / r["prd"],
                                                  amin, amax))
                    else:
                        realized = per_axis_ema.get(
                            axis_key, self.cfg.cold_start_alpha)
                    prev_ema = per_axis_ema.get(
                        axis_key, self.cfg.cold_start_alpha)
                    per_axis_ema[axis_key] = beta * realized + (1 - beta) * prev_ema

        out["prediction_calibrated"] = (out["prediction"] *
                                        out["alpha"]).astype(np.float32)
        return out.drop(columns=["__axis", "per"]).sort_index()


def streaming_calibrate(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    axes: list[str] | None = None,
    beta: float = 0.5,
    alpha_clip: tuple[float, float] = (0.6, 1.6),
    fold_in_test: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """One-shot helper: fit on val, transform val + test."""
    cfg = StreamingCalibratorConfig(
        axes=axes, beta=beta, alpha_clip=alpha_clip,
    )
    cal = StreamingCalibrator(cfg).fit_history(val_df)
    val_out = cal.transform(val_df, fold_in=True)
    tst_out = cal.transform(test_df, fold_in=fold_in_test)
    meta = {
        "axes": axes,
        "beta": beta,
        "alpha_clip": list(alpha_clip),
        "fold_in_test": fold_in_test,
        "final_alpha_per_axis": {k: float(v["__last"])
                                  for k, v in cal.alpha_history.items()},
    }
    return val_out, tst_out, meta
