"""Train the V5 two-stage forecaster on the enriched ABT and compare to V4.

Outputs:
    output/model_v5.joblib
    output/v5_metrics.csv     (WAPE / MAPE_nz / RMSE on val & test)
    output/v5_vs_v4.md        (side-by-side comparison)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluation import compute_all_metrics, split_train_val_test
from src.model_v2 import (
    TwoStageForecaster,
    encode_categoricals,
    get_feature_columns_v2,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("train_v5")

ROOT = Path(__file__).resolve().parent.parent
V4_ABT = ROOT / "output" / "abt_v4_cached.parquet"
V5_ABT = ROOT / "output" / "abt_v5_cached.parquet"
MODEL_OUT = ROOT / "output" / "model_v5.joblib"
METRICS_OUT = ROOT / "output" / "v5_metrics.csv"
COMPARE_MD = ROOT / "output" / "v5_vs_v4.md"
MANIFEST = ROOT / "output" / "v5_feature_manifest.json"


def _train(df: pd.DataFrame, feats: list[str]) -> tuple[TwoStageForecaster, dict, dict]:
    df_train, df_val, df_test = split_train_val_test(df)
    model = TwoStageForecaster(
        clf_params={"num_leaves": 127, "learning_rate": 0.05, "min_child_samples": 30},
        reg_params={"num_leaves": 255, "learning_rate": 0.05, "min_child_samples": 20},
    )
    t0 = time.time()
    model.fit(df_train, df_val, feats, num_boost_round=1200, early_stopping=60)
    elapsed = time.time() - t0
    p_val = model.predict(df_val)
    p_test = model.predict(df_test)
    m_val = compute_all_metrics(df_val["target_qty"].values, p_val)
    m_test = compute_all_metrics(df_test["target_qty"].values, p_test)
    log.info("Trained in %.1fs", elapsed)
    return model, m_val, m_test


def main() -> int:
    log.info("Loading V4 ABT")
    v4 = pd.read_parquet(V4_ABT).pipe(encode_categoricals)
    feats_v4 = get_feature_columns_v2(v4)

    log.info("Loading V5 ABT")
    v5 = pd.read_parquet(V5_ABT).pipe(encode_categoricals)
    feats_v5 = get_feature_columns_v2(v5)

    log.info("V4 features=%d, V5 features=%d (+%d external)",
             len(feats_v4), len(feats_v5), len(feats_v5) - len(feats_v4))

    log.info("── Training V4 baseline ──")
    _, v4_val, v4_test = _train(v4, feats_v4)
    log.info("V4 val:  WAPE=%.4f MAPE_nz=%.4f RMSE=%.4f",
             v4_val["WAPE"], v4_val["MAPE_nz"], v4_val["RMSE"])
    log.info("V4 test: WAPE=%.4f MAPE_nz=%.4f RMSE=%.4f",
             v4_test["WAPE"], v4_test["MAPE_nz"], v4_test["RMSE"])

    log.info("── Training V5 ──")
    model_v5, v5_val, v5_test = _train(v5, feats_v5)
    log.info("V5 val:  WAPE=%.4f MAPE_nz=%.4f RMSE=%.4f",
             v5_val["WAPE"], v5_val["MAPE_nz"], v5_val["RMSE"])
    log.info("V5 test: WAPE=%.4f MAPE_nz=%.4f RMSE=%.4f",
             v5_test["WAPE"], v5_test["MAPE_nz"], v5_test["RMSE"])

    joblib.dump(model_v5, MODEL_OUT)
    log.info("Saved V5 model → %s", MODEL_OUT)

    rows = []
    for split, mv, mt in [
        ("V4", v4_val, v4_test),
        ("V5", v5_val, v5_test),
    ]:
        rows.append({"model": split, "split": "val", **mv})
        rows.append({"model": split, "split": "test", **mt})
    pd.DataFrame(rows).to_csv(METRICS_OUT, index=False)

    lines = [
        "# V5 vs V4 — enriched-signals comparison",
        "",
        "V5 keeps only sources that passed the decision gate:",
        "`conflict_ua`, `gtrends_ua`, `holidays_ua`, `nbu_fx`, `tmdb_movies`, `world_bank_ua`.",
        "",
        "## Metrics",
        "",
        "| Split | Model | WAPE | MAPE_nz | RMSE |",
        "|---|---|---:|---:|---:|",
        f"| val  | V4 | {v4_val['WAPE']:.4f} | {v4_val['MAPE_nz']:.4f} | {v4_val['RMSE']:.4f} |",
        f"| val  | V5 | {v5_val['WAPE']:.4f} | {v5_val['MAPE_nz']:.4f} | {v5_val['RMSE']:.4f} |",
        f"| test | V4 | {v4_test['WAPE']:.4f} | {v4_test['MAPE_nz']:.4f} | {v4_test['RMSE']:.4f} |",
        f"| test | V5 | {v5_test['WAPE']:.4f} | {v5_test['MAPE_nz']:.4f} | {v5_test['RMSE']:.4f} |",
        "",
        "## Deltas (V5 − V4, negative = better)",
        "",
        f"- val  WAPE Δ = **{v5_val['WAPE'] - v4_val['WAPE']:+.4f}**",
        f"- val  MAPE_nz Δ = **{v5_val['MAPE_nz'] - v4_val['MAPE_nz']:+.4f}**",
        f"- test WAPE Δ = **{v5_test['WAPE'] - v4_test['WAPE']:+.4f}**",
        f"- test MAPE_nz Δ = **{v5_test['MAPE_nz'] - v4_test['MAPE_nz']:+.4f}**",
        "",
        "## Features added",
        "",
        f"V4 = {len(feats_v4)}, V5 = {len(feats_v5)} (+{len(feats_v5) - len(feats_v4)} external).",
    ]
    COMPARE_MD.write_text("\n".join(lines))
    log.info("Comparison → %s", COMPARE_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
