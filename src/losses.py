"""Custom LightGBM objectives for cost-calibrated demand forecasting.

LightGBM's built-in objectives (L2/L1/Tweedie) minimise symmetric error.
In retail, the cost of under-forecasting (lost sale + margin) is usually
2-4x higher than over-forecasting (holding cost), so a symmetric loss
produces systematically conservative predictions.

This module provides two families of objectives compatible with
``lightgbm.train`` via the ``fobj`` / ``feval`` arguments:

* ``pinball_objective(alpha)`` — quantile loss.  ``alpha=0.5`` recovers
  MAE; ``alpha=0.6`` leans slightly toward over-forecasting, which
  typically wins on WAPE under-prediction bias.
* ``asymmetric_objective(cost_under, cost_over)`` — a smooth L2-like
  loss where the gradient on the under-prediction side is scaled by
  ``cost_under/cost_over``.  Parameters default to 2.5/1.0 (retail
  heuristic).

Usage
-----
>>> import lightgbm as lgb
>>> from src.losses import pinball_objective, pinball_eval
>>> params = {"objective": "none", "learning_rate": 0.05, "metric": "None"}
>>> model = lgb.train(
...     params, train_ds, 1000,
...     valid_sets=[val_ds],
...     fobj=pinball_objective(0.6),
...     feval=pinball_eval(0.6),
...     callbacks=[lgb.early_stopping(50)],
... )

The TwoStageForecaster in ``src.model_v2`` is extended to accept
``reg_objective="pinball"`` / ``reg_objective="asymmetric"`` which wire
the objective in automatically.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

try:
    import lightgbm as lgb  # noqa: F401 (only needed for type hints at call sites)
except ImportError:  # pragma: no cover
    lgb = None


# ── Pinball / quantile loss ────────────────────────────────────────────────


def pinball_objective(alpha: float = 0.6) -> Callable:
    """Return a LightGBM-compatible ``fobj`` computing quantile-α gradient.

    Loss(y, p) = α * max(y - p, 0) + (1 - α) * max(p - y, 0)
    grad = -(α - 1{y <= p})
    hess = small constant (quantile loss has piecewise-linear grad)
    """
    a = float(alpha)

    def _obj(preds: np.ndarray, dataset) -> tuple[np.ndarray, np.ndarray]:
        y = dataset.get_label()
        resid = y - preds
        grad = np.where(resid >= 0, -a, 1 - a).astype(np.float64)
        hess = np.full_like(grad, 1e-3, dtype=np.float64)
        return grad, hess

    _obj.__name__ = f"pinball_objective_alpha_{a:.2f}"
    return _obj


def pinball_eval(alpha: float = 0.6) -> Callable:
    """Matching eval function returning (name, value, is_higher_better=False)."""
    a = float(alpha)

    def _eval(preds: np.ndarray, dataset) -> tuple[str, float, bool]:
        y = dataset.get_label()
        resid = y - preds
        loss = np.where(resid >= 0, a * resid, (a - 1) * resid).mean()
        return f"pinball_{a:.2f}", float(loss), False

    return _eval


# ── Asymmetric squared loss (cost-aware) ───────────────────────────────────


def asymmetric_objective(cost_under: float = 2.5, cost_over: float = 1.0) -> Callable:
    """Return an LGB ``fobj`` for an asymmetric squared loss.

    grad =  2 * c_under * (preds - y)   if preds < y  (under-forecast: stronger penalty)
    grad =  2 * c_over  * (preds - y)   if preds >= y (over-forecast:  milder penalty)

    ``hess`` is set to 2*c so LightGBM's Newton step uses the correct scale.
    """
    cu = float(cost_under)
    co = float(cost_over)

    def _obj(preds: np.ndarray, dataset) -> tuple[np.ndarray, np.ndarray]:
        y = dataset.get_label()
        resid = preds - y  # positive ⇒ over-forecast
        under = resid < 0
        grad = np.where(under, 2.0 * cu * resid, 2.0 * co * resid).astype(np.float64)
        hess = np.where(under, 2.0 * cu, 2.0 * co).astype(np.float64)
        return grad, hess

    _obj.__name__ = f"asymmetric_objective_cu{cu:.1f}_co{co:.1f}"
    return _obj


def asymmetric_eval(cost_under: float = 2.5, cost_over: float = 1.0) -> Callable:
    cu = float(cost_under)
    co = float(cost_over)

    def _eval(preds: np.ndarray, dataset) -> tuple[str, float, bool]:
        y = dataset.get_label()
        resid = preds - y
        loss = np.where(resid < 0, cu * resid**2, co * resid**2).mean()
        return f"asym_cu{cu:.1f}_co{co:.1f}", float(loss), False

    return _eval


# ── Factory to wire objectives into TwoStageForecaster ────────────────────


def resolve_objective(name: str, **kwargs) -> tuple[Callable | None, Callable | None, dict]:
    """Translate a short name into (fobj, feval, param_overrides).

    Returns (None, None, {}) for built-in LightGBM objectives; the caller
    should fall back to param-based configuration.

    Notes
    -----
    For quantile loss we prefer LightGBM's built-in ``objective=quantile``
    (well-tuned Newton step) over our custom ``pinball_objective`` callable.
    The callable is retained for unit tests and research experiments.
    """
    name = (name or "").lower()
    if name in ("", "tweedie", "regression", "mae", "l1", "l2", "huber"):
        return None, None, {}
    if name in ("pinball", "quantile"):
        alpha = kwargs.get("alpha", 0.6)
        # Use LightGBM's built-in quantile objective (proven Newton step)
        return (
            None,
            None,
            {"objective": "quantile", "alpha": alpha, "metric": "quantile"},
        )
    if name in ("pinball_custom", "quantile_custom"):
        alpha = kwargs.get("alpha", 0.6)
        return (
            pinball_objective(alpha),
            pinball_eval(alpha),
            {"objective": "none", "metric": "None"},
        )
    if name == "asymmetric":
        cu = kwargs.get("cost_under", 2.5)
        co = kwargs.get("cost_over", 1.0)
        return (
            asymmetric_objective(cu, co),
            asymmetric_eval(cu, co),
            {"objective": "none", "metric": "None"},
        )
    raise ValueError(f"Unknown objective name: {name!r}")
