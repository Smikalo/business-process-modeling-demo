"""Smoke tests for V7 components."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features_cohort import add_cohort_features
from src.features_price import add_price_features
from src.margin_table import build_margin_table
from src.v7_components import (
    IsotonicCalibrator, PerSegmentConformal, RidgeStacker,
)


def _tiny_abt() -> pd.DataFrame:
    months = pd.period_range("2024-01", "2024-12", freq="M")
    skus = ["SKU1", "SKU2", "SKU3"]
    rows = []
    for sku in skus:
        for i, p in enumerate(months):
            rows.append({
                "Период": p,
                "Артикул": sku,
                "Партнер": "P",
                "Бренд": "B",
                "Канал": "НКП",
                "Группа_товара": "games",
                "target_qty": float(2 + (i % 3)),
                "stockout_orc": i % 4 == 0,
                "stockout_orc_prev": (i + 3) % 4 == 0,
                "is_new_sku": i == 0,
                "lag_1": float(1 + (i % 3)),
                "implied_unit_price": 100.0 + 5 * (i % 4),
                "РРЦ": 300.0,
                "Количество_sales": 4.0,
                "Выручка_sales": 400.0,
                "Количество_orc": 40.0,
                "Стоимость_orc": 3600.0,
            })
    return pd.DataFrame(rows)


def test_margin_table_core_columns_and_bounds():
    mt = build_margin_table(_tiny_abt())
    assert {"Артикул", "unit_price_uah", "unit_cost_uah",
            "margin_rate", "holding_rate_annual"} <= set(mt.columns)
    assert (mt["margin_rate"] >= 0.05).all()
    assert (mt["margin_rate"] <= 0.80).all()
    assert (mt["unit_price_uah"] > 0).all()


def test_price_and_cohort_features_added_and_finite():
    abt = _tiny_abt()
    abt = add_price_features(abt)
    abt = add_cohort_features(abt)
    for col in [
        "price_lag1", "price_vs_brand_median", "sku_price_elasticity",
        "cohort_demand_lag1", "cohort_stockout_share_lag1",
        "cohort_size", "cannibalisation_pressure",
    ]:
        assert col in abt.columns
        assert np.isfinite(abt[col].to_numpy()).all(), f"non-finite in {col}"


def test_isotonic_calibrator_is_monotone():
    rng = np.random.default_rng(0)
    p_raw = rng.uniform(size=500)
    y = (p_raw + rng.normal(scale=0.3, size=500) > 0.5).astype(int)
    calib = IsotonicCalibrator().fit(p_raw, y)
    out = calib.transform(np.sort(p_raw))
    assert (np.diff(out) >= -1e-8).all()
    assert out.min() >= 0 and out.max() <= 1


def test_ridge_stacker_positive_weights():
    rng = np.random.default_rng(1)
    y = rng.uniform(0, 10, size=300)
    preds = {
        "a": y + rng.normal(scale=0.5, size=300),
        "b": y + rng.normal(scale=1.0, size=300),
    }
    stk = RidgeStacker(alpha=1.0).fit(preds, y)
    assert (stk.ridge.coef_ >= 0).all()
    p = stk.predict(preds)
    assert p.shape == y.shape and (p >= 0).all()


def test_conformal_intervals_cover_and_order():
    rng = np.random.default_rng(2)
    n = 400
    df = pd.DataFrame({
        "Бренд": rng.choice(["B1", "B2"], size=n),
        "Канал": rng.choice(["НКП", "СК"], size=n),
    })
    y = rng.uniform(0, 5, size=n)
    p = y + rng.normal(scale=0.5, size=n)
    conf = PerSegmentConformal(low=0.1, high=0.9, min_rows=50).fit(df, y, p)
    lo, hi = conf.intervals(df, p)
    assert (hi >= lo).all()
    assert (lo >= 0).all()


def test_ridge_stacker_requires_all_bases_at_predict():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    preds = {"a": y, "b": y + 0.1}
    stk = RidgeStacker().fit(preds, y)
    with pytest.raises(KeyError):
        stk.predict({"a": y})
