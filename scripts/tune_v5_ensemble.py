"""Tune a V4+V5 prediction blend on the validation set and score on test.

V4 and V5 each have different inductive biases:
    V4 — sharper on the internal lag/rolling signals (robust on out-of-sample)
    V5 — tighter on the validation period thanks to holidays/fx/conflict/movies/trends

A convex combination usually beats either component alone when the
individual error signals are partly uncorrelated.

Outputs:
    output/v5_ensemble_weights.json
    output/v5_ensemble_metrics.md
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import compute_all_metrics, split_train_val_test
from src.model_v2 import (
    TwoStageForecaster,
    encode_categoricals,
    get_feature_columns_v2,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("tune_v5_ensemble")

ROOT = Path(__file__).resolve().parent.parent
V4_ABT = ROOT / "output" / "abt_v4_cached.parquet"
V5_ABT = ROOT / "output" / "abt_v5_cached.parquet"
WEIGHTS_OUT = ROOT / "output" / "v5_ensemble_weights.json"
SUMMARY_OUT = ROOT / "output" / "v5_ensemble_metrics.md"


def _train_and_predict(df: pd.DataFrame, feats: list[str]):
    tr, va, te = split_train_val_test(df)
    model = TwoStageForecaster(
        clf_params={"num_leaves": 127, "learning_rate": 0.05, "min_child_samples": 30},
        reg_params={"num_leaves": 255, "learning_rate": 0.05, "min_child_samples": 20},
    )
    model.fit(tr, va, feats, num_boost_round=1200, early_stopping=60)
    return model.predict(va), model.predict(te), va["target_qty"].values, te["target_qty"].values


def _wape(y: np.ndarray, p: np.ndarray) -> float:
    denom = np.abs(y).sum()
    return float(np.abs(y - p).sum() / denom) if denom else float("nan")


def _scan_blend(y_val: np.ndarray, p4_val: np.ndarray, p5_val: np.ndarray) -> tuple[float, float]:
    best_w, best_wape = 0.0, float("inf")
    for w in np.linspace(0, 1, 101):
        w5 = 1 - w
        blend = w * p4_val + w5 * p5_val
        wp = _wape(y_val, blend)
        if wp < best_wape:
            best_w, best_wape = float(w), wp
    return best_w, best_wape


def main() -> int:
    log.info("Loading and training V4")
    v4 = pd.read_parquet(V4_ABT).pipe(encode_categoricals)
    feats_v4 = get_feature_columns_v2(v4)
    p4_val, p4_test, y_val, y_test = _train_and_predict(v4, feats_v4)

    log.info("Loading and training V5")
    v5 = pd.read_parquet(V5_ABT).pipe(encode_categoricals)
    feats_v5 = get_feature_columns_v2(v5)
    p5_val, p5_test, y_val2, y_test2 = _train_and_predict(v5, feats_v5)

    assert np.allclose(y_val, y_val2) and np.allclose(y_test, y_test2), \
        "validation/test splits drifted between V4 and V5"

    log.info("Scanning blend weights on validation")
    w4, val_wape = _scan_blend(y_val, p4_val, p5_val)
    w5 = 1 - w4
    log.info("Best: w_V4=%.2f w_V5=%.2f val_WAPE=%.4f", w4, w5, val_wape)

    blend_val = w4 * p4_val + w5 * p5_val
    blend_test = w4 * p4_test + w5 * p5_test

    m_val = compute_all_metrics(y_val, blend_val)
    m_test = compute_all_metrics(y_test, blend_test)
    m_v4_val = compute_all_metrics(y_val, p4_val)
    m_v4_test = compute_all_metrics(y_test, p4_test)
    m_v5_val = compute_all_metrics(y_val, p5_val)
    m_v5_test = compute_all_metrics(y_test, p5_test)

    WEIGHTS_OUT.write_text(json.dumps({"w_v4": w4, "w_v5": w5, "val_wape": val_wape}, indent=2))

    lines = [
        "# V5 ensemble (V4 + V5) — tuning results",
        "",
        f"Best validation blend: **w_V4 = {w4:.2f}, w_V5 = {w5:.2f}**",
        "",
        "## Metrics",
        "",
        "| Split | Model | WAPE | MAPE_nz | RMSE |",
        "|---|---|---:|---:|---:|",
        f"| val  | V4 alone  | {m_v4_val['WAPE']:.4f} | {m_v4_val['MAPE_nz']:.4f} | {m_v4_val['RMSE']:.4f} |",
        f"| val  | V5 alone  | {m_v5_val['WAPE']:.4f} | {m_v5_val['MAPE_nz']:.4f} | {m_v5_val['RMSE']:.4f} |",
        f"| val  | Ensemble  | {m_val['WAPE']:.4f} | {m_val['MAPE_nz']:.4f} | {m_val['RMSE']:.4f} |",
        f"| test | V4 alone  | {m_v4_test['WAPE']:.4f} | {m_v4_test['MAPE_nz']:.4f} | {m_v4_test['RMSE']:.4f} |",
        f"| test | V5 alone  | {m_v5_test['WAPE']:.4f} | {m_v5_test['MAPE_nz']:.4f} | {m_v5_test['RMSE']:.4f} |",
        f"| test | Ensemble  | {m_test['WAPE']:.4f} | {m_test['MAPE_nz']:.4f} | {m_test['RMSE']:.4f} |",
        "",
        "## Deltas (Ensemble vs V4; negative is better)",
        "",
        f"- val  WAPE Δ = **{m_val['WAPE'] - m_v4_val['WAPE']:+.4f}**",
        f"- test WAPE Δ = **{m_test['WAPE'] - m_v4_test['WAPE']:+.4f}**",
    ]
    SUMMARY_OUT.write_text("\n".join(lines))
    log.info("Summary → %s", SUMMARY_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
