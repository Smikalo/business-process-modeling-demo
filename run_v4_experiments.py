"""V4 creative experiments runner.

Tests four architectural innovations beyond V3:
  A. Per-channel specialists (one model per channel)
  B. Log-target regressor (tail stabilization)
  C. Ratio target (predict qty / rolling_mean)
  D. Stacking ensemble (V3 + MA-6 + Seasonal-Naive with learned weights)
  E. Hierarchical top-down (partner-total × SKU-share decomposition)

Final V4 = best blend of the above on validation WAPE.
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
from src.evaluation import compute_all_metrics, split_train_val_test, wape
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
    HierarchicalReconciler,
    LogTargetForecaster,
    PerChannelEnsemble,
    optimize_ensemble_weights,
    stacking_ensemble_predict,
)

OUTPUT = Path("output")
CACHE = OUTPUT / "abt_v4_cached.parquet"

log = logging.getLogger(__name__)


def load_or_build_abt() -> pd.DataFrame:
    """Build the active-pair-filtered, feature-engineered ABT and cache it."""
    if CACHE.exists():
        log.info("Loading cached ABT from %s", CACHE)
        df = pd.read_parquet(CACHE)
        # Re-encode categoricals (lost on parquet round-trip for some dtypes)
        df = encode_categoricals(df)
        return df

    log.info("Building ABT from scratch...")
    t0 = time.time()
    df = assemble_master()
    df = enrich_all(df)
    df = engineer_all_features(df)
    df["target_qty"] = df["target_qty"].clip(lower=0)
    df = filter_active_pairs(df)
    df = add_proper_rolling(df)
    df = add_v3_features(df)
    df = encode_categoricals(df)
    log.info("ABT built in %.1fs — %s rows, %s cols", time.time() - t0, len(df), len(df.columns))

    df.to_parquet(CACHE, index=False)
    log.info("ABT cached to %s", CACHE)
    return df


# ── Baseline helpers for ensemble ──────────────────────────────────────────

def moving_average_6(df_hist: pd.DataFrame, df_target: pd.DataFrame) -> np.ndarray:
    """For each (Partner, SKU, Period) in df_target, mean of past 6 months."""
    df_hist = df_hist[["Партнер", "Артикул", "Период", "target_qty"]].copy()
    lookup = df_hist.set_index(["Партнер", "Артикул", "Период"])["target_qty"]

    preds = np.zeros(len(df_target))
    for i, row in enumerate(df_target[["Партнер", "Артикул", "Период"]].itertuples(index=False)):
        p, s, per = row
        vals = []
        for k in range(1, 7):
            try:
                v = lookup.get((p, s, per - k), 0.0)
                vals.append(v if pd.notna(v) else 0.0)
            except (KeyError, TypeError):
                vals.append(0.0)
        preds[i] = float(np.mean(vals))
    return preds


def seasonal_naive_12(df_hist: pd.DataFrame, df_target: pd.DataFrame) -> np.ndarray:
    """For each row, value from 12 months prior."""
    df_hist = df_hist[["Партнер", "Артикул", "Период", "target_qty"]].copy()
    lookup = df_hist.set_index(["Партнер", "Артикул", "Период"])["target_qty"]

    preds = np.zeros(len(df_target))
    for i, row in enumerate(df_target[["Партнер", "Артикул", "Период"]].itertuples(index=False)):
        p, s, per = row
        try:
            v = lookup.get((p, s, per - 12), 0.0)
            preds[i] = float(v) if pd.notna(v) else 0.0
        except (KeyError, TypeError):
            preds[i] = 0.0
    return preds


def fast_lagged_pred_from_features(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Baseline predictions reconstructed from already-engineered lag features.
    Much faster than groupby lookup. Averages the columns."""
    return df[cols].fillna(0).mean(axis=1).values


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    # ── Step 1: Data ─────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("STEP 1: Load active-pair-filtered ABT")
    print("═" * 70)
    df = load_or_build_abt()
    df_train, df_val, df_test = split_train_val_test(df)
    feat_cols = get_feature_columns_v2(df_train)
    print(f"  Rows: train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}")
    print(f"  Features: {len(feat_cols)}")
    print(f"  Test nonzero rate: {(df_test['target_qty'] > 0).mean():.1%}")

    actual_test = df_test["target_qty"].values
    actual_val = df_val["target_qty"].values
    results = {}
    val_predictions = {}
    test_predictions = {}

    # ── Step 2: V3 baseline (reference) ──────────────────────────────────
    print("\n" + "═" * 70)
    print("STEP 2: V3 reference model (TwoStage Tweedie)")
    print("═" * 70)
    t = time.time()
    v3 = TwoStageForecaster(
        clf_params={"num_leaves": 127, "learning_rate": 0.03, "min_child_samples": 30, "feature_fraction": 0.85},
        reg_params={"num_leaves": 255, "learning_rate": 0.03, "min_child_samples": 20, "feature_fraction": 0.85},
    )
    v3.fit(df_train, df_val, feat_cols, num_boost_round=1500, early_stopping=80)
    pv3_val = v3.predict(df_val)
    pv3_test = v3.predict(df_test)
    results["V3"] = compute_all_metrics(actual_test, pv3_test)
    val_predictions["V3"] = pv3_val
    test_predictions["V3"] = pv3_test
    print(f"  V3 test: {results['V3']}    [{time.time()-t:.0f}s]")

    # ── Step 3: Per-channel specialists ──────────────────────────────────
    print("\n" + "═" * 70)
    print("STEP 3: Per-channel specialized models")
    print("═" * 70)
    t = time.time()
    try:
        pc = PerChannelEnsemble()
        pc.fit(df_train, df_val, feat_cols)
        ppc_val = pc.predict(df_val)
        ppc_test = pc.predict(df_test)
        results["V4_PerChannel"] = compute_all_metrics(actual_test, ppc_test)
        val_predictions["PerChannel"] = ppc_val
        test_predictions["PerChannel"] = ppc_test
        print(f"  PerChannel test: {results['V4_PerChannel']}    [{time.time()-t:.0f}s]")
    except Exception as e:
        log.exception("PerChannel failed: %s", e)

    # ── Step 4: Log-target regressor ─────────────────────────────────────
    print("\n" + "═" * 70)
    print("STEP 4: Log-target two-stage (tail stabilization)")
    print("═" * 70)
    t = time.time()
    try:
        lt = LogTargetForecaster()
        lt.fit(df_train, df_val, feat_cols, num_boost_round=1500, early_stopping=80)
        plt_val = lt.predict(df_val)
        plt_test = lt.predict(df_test)
        results["V4_LogTarget"] = compute_all_metrics(actual_test, plt_test)
        val_predictions["LogTarget"] = plt_val
        test_predictions["LogTarget"] = plt_test
        print(f"  LogTarget test: {results['V4_LogTarget']}    [{time.time()-t:.0f}s]")
    except Exception as e:
        log.exception("LogTarget failed: %s", e)

    # ── Step 5: Hierarchical reconciliation (V3 anchored to partner totals) ──
    print("\n" + "═" * 70)
    print("STEP 5: Hierarchical reconciliation (V3 ∘ partner-total anchor)")
    print("═" * 70)
    t = time.time()
    try:
        hr = HierarchicalReconciler()
        hr.fit(df_train, df_val)
        # Reconcile V3's SKU-level predictions to match partner-total forecasts
        phr_val = hr.reconcile(df_val, pv3_val)
        phr_test = hr.reconcile(df_test, pv3_test)
        results["V4_Reconciled"] = compute_all_metrics(actual_test, phr_test)
        val_predictions["Reconciled"] = phr_val
        test_predictions["Reconciled"] = phr_test
        print(f"  Reconciled test: {results['V4_Reconciled']}    [{time.time()-t:.0f}s]")
    except Exception as e:
        log.exception("Reconciled failed: %s", e)

    # ── Step 6: Fast baselines for stacking (from already-present lags) ──
    print("\n" + "═" * 70)
    print("STEP 6: Cheap baselines for ensemble")
    print("═" * 70)
    ma6_cols = [c for c in ["lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_6"] if c in df_test.columns]
    if not ma6_cols:
        ma6_cols = [c for c in ["lag_1", "lag_2", "lag_3"] if c in df_test.columns]
    ma6_test = df_test[ma6_cols].fillna(0).mean(axis=1).values
    ma6_val = df_val[ma6_cols].fillna(0).mean(axis=1).values
    val_predictions["MA_lag_avg"] = ma6_val
    test_predictions["MA_lag_avg"] = ma6_test
    results["Baseline_MA"] = compute_all_metrics(actual_test, ma6_test)
    print(f"  MA-lag-avg test: {results['Baseline_MA']}")

    if "lag_12" in df_test.columns:
        sn_test = df_test["lag_12"].fillna(0).values
        sn_val = df_val["lag_12"].fillna(0).values
        val_predictions["Seasonal12"] = sn_val
        test_predictions["Seasonal12"] = sn_test
        results["Baseline_Seasonal12"] = compute_all_metrics(actual_test, sn_test)
        print(f"  Seasonal-12 test: {results['Baseline_Seasonal12']}")

    # ── Step 7: Learn optimal ensemble weights on validation ─────────────
    print("\n" + "═" * 70)
    print("STEP 7: Optimized stacking ensemble")
    print("═" * 70)
    # Select top candidates
    candidates = {k: v for k, v in val_predictions.items() if v is not None}
    weights = optimize_ensemble_weights(candidates, actual_val)

    test_candidates = {k: test_predictions[k] for k in candidates.keys()}
    p_ensemble_test = stacking_ensemble_predict(test_candidates, weights)
    results["V4_Ensemble"] = compute_all_metrics(actual_test, p_ensemble_test)
    print(f"  Ensemble weights: {weights}")
    print(f"  V4 Ensemble test: {results['V4_Ensemble']}")

    # Save predictions for second-round calibration experiments
    joblib.dump(
        {
            "val_predictions": val_predictions,
            "test_predictions": test_predictions,
            "actual_val": actual_val,
            "actual_test": actual_test,
            "val_index": df_val.index.values,
            "test_index": df_test.index.values,
        },
        OUTPUT / "v4_base_predictions.joblib",
    )
    log.info("Saved base predictions to %s", OUTPUT / "v4_base_predictions.joblib")

    # ── Step 8: Summary ──────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("SUMMARY — V4 Creative Approaches")
    print("═" * 70)
    summary = pd.DataFrame(results).T
    summary = summary.sort_values("WAPE")
    print(summary.to_string())

    summary.to_csv(OUTPUT / "v4_experiment_results.csv")
    with open(OUTPUT / "v4_ensemble_weights.json", "w") as f:
        json.dump(weights, f, indent=2)

    # Save best model
    best_name = summary.index[0]
    print(f"\n  BEST MODEL: {best_name}  WAPE={summary.loc[best_name,'WAPE']:.4f}")

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
