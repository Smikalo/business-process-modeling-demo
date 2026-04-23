"""Smoke tests for V7.1 components."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.v71_components import (
    MONOTONE_NEGATIVE,
    MONOTONE_POSITIVE,
    build_monotone_constraints,
    build_recency_weights,
    iterative_impute_stockouts,
    make_business_cost_objective,
    newsvendor_alpha_per_sku,
)


def _fake_periods(n: int, start: str = "2022-01") -> pd.PeriodIndex:
    return pd.period_range(start=start, periods=n, freq="M")


def test_recency_weights_monotone_and_bounded():
    n = 36
    df = pd.DataFrame({"Период": _fake_periods(n)})
    w = build_recency_weights(df, gamma=0.9, floor=0.1)
    assert w.shape == (n,)
    assert w.max() <= 1.0 + 1e-6 and w.min() >= 0.1 - 1e-6
    # More recent row → higher (or equal) weight
    last_idx = np.argmax(df["Период"].apply(lambda p: p.ordinal))
    assert w[last_idx] == pytest.approx(w.max(), rel=1e-5)


def test_monotone_constraints_modes():
    feats = ["lag_1", "lag_2", "rmean_6", "stockout_orc",
             "price_vs_brand_median", "random_feature"]
    full = build_monotone_constraints(feats, mode="full")
    assert full[feats.index("lag_1")] == 1
    assert full[feats.index("stockout_orc")] == -1
    assert full[feats.index("random_feature")] == 0

    stk = build_monotone_constraints(feats, mode="stockout_only")
    assert stk[feats.index("stockout_orc")] == -1
    assert stk[feats.index("lag_1")] == 0  # positive constraints dropped

    lag = build_monotone_constraints(feats, mode="lags_only")
    assert lag[feats.index("lag_1")] == 1
    assert lag[feats.index("stockout_orc")] == 0

    assert MONOTONE_POSITIVE & MONOTONE_NEGATIVE == set()


def test_newsvendor_alpha_clipping_and_shrinkage():
    df = pd.DataFrame({"Артикул": ["A", "B", "C"]})
    m = pd.DataFrame({
        "Артикул": ["A", "B", "C"],
        "margin_rate": [0.10, 0.30, 0.05],
        "holding_rate_annual": [0.22, 0.22, 0.22],
    })
    a = newsvendor_alpha_per_sku(df, m, alpha_floor=0.30,
                                 alpha_ceiling=0.55, shrink_to=0.45,
                                 shrink_weight=0.5)
    assert a.shape == (3,)
    assert (a >= 0.30).all() and (a <= 0.55).all()
    # The row with the highest margin should get the highest α
    assert a[1] >= a[0] >= a[2]


def test_iterative_impute_stockouts_only_dense_rows():
    df = pd.DataFrame({
        "stockout_orc": [1, 1, 0, 1],
        "demand_density": [0.5, 0.2, 0.9, 0.4],
        "target_qty_imputed": [2.0, 5.0, 3.0, 1.0],
    })
    pred = np.array([10.0, 10.0, 10.0, 10.0])
    out = iterative_impute_stockouts(df, pred, shrink=0.9, density_threshold=0.3)
    # Row 0: stockout & dense → refined (max(2, 9) = 9)
    # Row 1: stockout but not dense → unchanged
    # Row 2: not stockout → unchanged
    # Row 3: stockout & dense → refined (max(1, 9) = 9)
    assert out["target_qty_imputed"].tolist() == [9.0, 5.0, 3.0, 9.0]
    assert out["em_refined"].tolist() == [1, 0, 0, 1]


def test_business_cost_objective_gradient_signs():
    # Over-forecast rows should push gradient positive, under-forecast negative.
    price = np.array([100.0, 100.0, 100.0])
    margin = np.array([0.2, 0.2, 0.2])
    holding_monthly = np.array([0.02, 0.02, 0.02])
    fobj = make_business_cost_objective(price, margin, holding_monthly, recovery=0.5)

    class _DS:
        def __init__(self, y): self._y = y
        def get_label(self): return self._y

    y = np.array([5.0, 5.0, 5.0])
    preds = np.array([10.0, 5.0, 2.0])  # over, exact, under
    grad, hess = fobj(preds, _DS(y))
    assert grad[0] > 0   # over → positive grad (push down)
    assert grad[1] == 0  # exact
    assert grad[2] < 0   # under → negative grad (push up)
    assert (hess > 0).all()
