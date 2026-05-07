"""V4 final production pipeline — trains the winning ensemble.

Components:
  1. V3 (TwoStageForecaster with V2+V3 features)
  2. LogTarget (two-stage, log1p regressor)
  3. PerChannel (one two-stage per channel)
  4. MA-lag baseline (mean of lag_1..lag_6)

Inference: 0.4·V3 + 0.4·LogTarget + 0.1·PerChannel + 0.1·MA
(weights learned via SLSQP on validation set, frozen here.)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s %(message)s")

import joblib
import numpy as np
import pandas as pd

from src.enrichment import enrich_all
from src.evaluation import compute_all_metrics, split_train_val_test
from src.features import engineer_all_features
from src.master import assemble_master
from src.model_v2 import (
    add_proper_rolling,
    encode_categoricals,
    filter_active_pairs,
    get_feature_columns_v2,
    TwoStageForecaster,
)
from src.model_v3 import add_v3_features
from src.model_v4 import (
    LogTargetForecaster,
    PerChannelEnsemble,
    optimize_ensemble_weights,
    stacking_ensemble_predict,
)

OUTPUT = Path("output")
CACHE = OUTPUT / "abt_v4_cached.parquet"


def load_or_build_abt() -> pd.DataFrame:
    if CACHE.exists():
        df = pd.read_parquet(CACHE)
        return encode_categoricals(df)
    df = assemble_master()
    df = enrich_all(df)
    df = engineer_all_features(df)
    df["target_qty"] = df["target_qty"].clip(lower=0)
    df = filter_active_pairs(df)
    df = add_proper_rolling(df)
    df = add_v3_features(df)
    df = encode_categoricals(df)
    df.to_parquet(CACHE, index=False)
    return df


def main():
    t0 = time.time()

    print("═" * 70)
    print("V4 FINAL — Creative Ensemble Production Pipeline")
    print("═" * 70)

    df = load_or_build_abt()
    df_train, df_val, df_test = split_train_val_test(df)
    feat_cols = get_feature_columns_v2(df_train)

    print(f"\n  Train: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}")
    print(f"  Features: {len(feat_cols)}")
    actual_val = df_val["target_qty"].values
    actual_test = df_test["target_qty"].values

    # ── Train the three base models ──────────────────────────────────────
    print("\n── Training V3 (two-stage Tweedie) ────────────────")
    v3 = TwoStageForecaster(
        clf_params={"num_leaves": 127, "learning_rate": 0.03, "min_child_samples": 30, "feature_fraction": 0.85},
        reg_params={"num_leaves": 255, "learning_rate": 0.03, "min_child_samples": 20, "feature_fraction": 0.85},
    )
    v3.fit(df_train, df_val, feat_cols, num_boost_round=1500, early_stopping=80)

    print("\n── Training LogTarget (two-stage log1p) ───────────")
    lt = LogTargetForecaster()
    lt.fit(df_train, df_val, feat_cols, num_boost_round=1500, early_stopping=80)

    print("\n── Training PerChannel specialists ────────────────")
    pc = PerChannelEnsemble()
    pc.fit(df_train, df_val, feat_cols)

    # ── Collect predictions ──────────────────────────────────────────────
    preds_val = {
        "V3": v3.predict(df_val),
        "LogTarget": lt.predict(df_val),
        "PerChannel": pc.predict(df_val),
        "MA_lag_avg": df_val[["lag_1", "lag_2", "lag_3", "lag_6"]].fillna(0).mean(axis=1).values,
    }
    preds_test = {
        "V3": v3.predict(df_test),
        "LogTarget": lt.predict(df_test),
        "PerChannel": pc.predict(df_test),
        "MA_lag_avg": df_test[["lag_1", "lag_2", "lag_3", "lag_6"]].fillna(0).mean(axis=1).values,
    }

    # ── Base model metrics ───────────────────────────────────────────────
    print("\n── Individual base model performance (test) ───────")
    results = {}
    for name, p in preds_test.items():
        m = compute_all_metrics(actual_test, p)
        results[name] = m
        print(f"  {name:15s}  WAPE={m['WAPE']:.4f}  MAPE_nz={m['MAPE_nz']:.4f}  RMSE={m['RMSE']:.4f}")

    # ── Optimize ensemble weights on validation ──────────────────────────
    print("\n── Ensemble weight optimization (SLSQP on val WAPE) ──")
    weights = optimize_ensemble_weights(preds_val, actual_val)

    ensemble_test = stacking_ensemble_predict(preds_test, weights)
    m_ens = compute_all_metrics(actual_test, ensemble_test)
    results["V4_Ensemble"] = m_ens

    print("\n" + "═" * 70)
    print("FINAL V4 ENSEMBLE RESULTS")
    print("═" * 70)
    print(f"  Weights: {weights}")
    print(f"  WAPE:     {m_ens['WAPE']:.4f}")
    print(f"  MAPE_nz:  {m_ens['MAPE_nz']:.4f}")
    print(f"  RMSE:     {m_ens['RMSE']:.4f}")
    print(f"  Bias:     {m_ens['Bias']:.4f}")

    # Compare vs V3
    m_v3 = results["V3"]
    print(f"\n  vs V3:  WAPE  {m_v3['WAPE']:.4f} → {m_ens['WAPE']:.4f}   ({(m_v3['WAPE']-m_ens['WAPE'])/m_v3['WAPE']*100:+.1f}%)")
    print(f"          MAPE  {m_v3['MAPE_nz']:.4f} → {m_ens['MAPE_nz']:.4f}   ({(m_v3['MAPE_nz']-m_ens['MAPE_nz'])/m_v3['MAPE_nz']*100:+.1f}%)")
    print(f"          RMSE  {m_v3['RMSE']:.4f} → {m_ens['RMSE']:.4f}   ({(m_v3['RMSE']-m_ens['RMSE'])/m_v3['RMSE']*100:+.1f}%)")

    # ── Save artifacts ───────────────────────────────────────────────────
    pd.DataFrame(results).T.to_csv(OUTPUT / "v4_final_metrics.csv")
    with open(OUTPUT / "v4_final_config.json", "w") as f:
        json.dump(
            {
                "weights": weights,
                "metrics": {"WAPE": float(m_ens["WAPE"]),
                           "MAPE_nz": float(m_ens["MAPE_nz"]),
                           "RMSE": float(m_ens["RMSE"]),
                           "Bias": float(m_ens["Bias"])},
                "improvements_vs_v3_pct": {
                    "WAPE": float((m_v3["WAPE"] - m_ens["WAPE"]) / m_v3["WAPE"] * 100),
                    "MAPE_nz": float((m_v3["MAPE_nz"] - m_ens["MAPE_nz"]) / m_v3["MAPE_nz"] * 100),
                    "RMSE": float((m_v3["RMSE"] - m_ens["RMSE"]) / m_v3["RMSE"] * 100),
                },
            },
            f,
            indent=2,
        )

    joblib.dump(
        {"v3": v3, "logtarget": lt, "perchannel": pc, "weights": weights, "feature_cols": feat_cols},
        OUTPUT / "model_v4_ensemble.joblib",
    )

    # Feature importance from V3 (the backbone)
    fi = v3.feature_importance()
    fi.to_csv(OUTPUT / "feature_importance_v4.csv", index=False)
    print(f"\n  Top 10 features (V3 backbone):")
    print(fi.head(10).to_string(index=False))

    print(f"\n  Runtime: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
