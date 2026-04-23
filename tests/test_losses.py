"""Tests for custom LightGBM objectives in src/losses.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.losses import (
    asymmetric_objective,
    pinball_eval,
    pinball_objective,
    resolve_objective,
)


class _FakeDataset:
    def __init__(self, y: np.ndarray) -> None:
        self._y = y

    def get_label(self) -> np.ndarray:
        return self._y


def test_pinball_gradient_sign_matches_residual_direction() -> None:
    obj = pinball_objective(0.6)
    y = np.array([10.0, 10.0, 10.0])
    preds = np.array([8.0, 12.0, 10.0])
    grad, hess = obj(preds, _FakeDataset(y))
    # Under-forecast: gradient should be negative (encourage increase)
    assert grad[0] < 0
    # Over-forecast: gradient should be positive (encourage decrease)
    assert grad[1] > 0
    # Hessian always positive
    assert (hess > 0).all()


def test_pinball_eval_returns_higher_is_better_false() -> None:
    ev = pinball_eval(0.5)
    y = np.array([10.0, 10.0])
    preds = np.array([8.0, 12.0])
    name, loss, higher = ev(preds, _FakeDataset(y))
    assert "pinball" in name
    assert higher is False
    assert loss > 0


def test_asymmetric_penalises_underforecast_more() -> None:
    obj = asymmetric_objective(cost_under=3.0, cost_over=1.0)
    y = np.array([10.0, 10.0])
    preds = np.array([8.0, 12.0])  # same absolute error
    grad, _ = obj(preds, _FakeDataset(y))
    # Under-forecast residual = -2, grad = 2*3*(-2) = -12 (abs 12)
    # Over-forecast residual = +2,  grad = 2*1*(+2) = +4  (abs 4)
    assert abs(grad[0]) > abs(grad[1])


def test_resolve_objective_tweedie_passthrough() -> None:
    fobj, feval, overrides = resolve_objective("tweedie")
    assert fobj is None and feval is None and overrides == {}


def test_resolve_objective_pinball_wiring() -> None:
    # Built-in path: returns only param overrides, no callables
    fobj, feval, overrides = resolve_objective("pinball", alpha=0.6)
    assert fobj is None and feval is None
    assert overrides["objective"] == "quantile"
    assert overrides["alpha"] == 0.6


def test_resolve_objective_pinball_custom_wiring() -> None:
    fobj, feval, overrides = resolve_objective("pinball_custom", alpha=0.6)
    assert callable(fobj) and callable(feval)
    assert overrides["objective"] == "none"


def test_pinball_trains_end_to_end_on_toy_data() -> None:
    rng = np.random.default_rng(0)
    n = 500
    X = rng.normal(size=(n, 3)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.3, size=n)).astype(np.float32)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
    ds = lgb.Dataset(X_df, label=y)
    params = {"objective": pinball_objective(0.6), "learning_rate": 0.1, "verbose": -1}
    model = lgb.train(params, ds, num_boost_round=20)
    preds = model.predict(X_df)
    # Predictions should be biased slightly upward (q60 > median)
    assert preds.mean() > y.mean() - 0.5  # sanity: didn't blow up
    assert not np.isnan(preds).any()
