"""V4 Round 2: post-hoc calibration + GBDT meta-learner.

Loads cached base-model predictions from round 1 and applies three
calibration strategies, then picks the winning combination.
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

from src.evaluation import compute_all_metrics
from src.model_v2 import encode_categoricals
from src.model_v4 import optimize_ensemble_weights, stacking_ensemble_predict
from src.model_v4_calibration import (
    GBDTMetaLearner,
    SegmentBiasCorrector,
    SegmentedIsotonicCalibrator,
)

OUTPUT = Path("output")
ABT_CACHE = OUTPUT / "abt_v4_cached.parquet"
PRED_CACHE = OUTPUT / "v4_base_predictions.joblib"

log = logging.getLogger(__name__)


def main():
    t0 = time.time()

    print("═" * 70)
    print("V4 ROUND 2 — Calibration + Meta-Learner")
    print("═" * 70)

    # Load ABT and base predictions
    df = pd.read_parquet(ABT_CACHE)
    df = encode_categoricals(df)
    cached = joblib.load(PRED_CACHE)

    val_preds = cached["val_predictions"]
    test_preds = cached["test_predictions"]
    actual_val = cached["actual_val"]
    actual_test = cached["actual_test"]

    # Recover val/test DataFrames using saved indices
    df_val = df.loc[cached["val_index"]].copy()
    df_test = df.loc[cached["test_index"]].copy()

    # Ensure volume_tier is present (for segmentation)
    for c in ["volume_tier", "partner_volume_tier", "month"]:
        if c not in df_val.columns:
            df_val[c] = 0
            df_test[c] = 0

    print(f"\nBase models available: {list(val_preds.keys())}")
    print(f"Val rows: {len(df_val):,}   Test rows: {len(df_test):,}")

    results = {}

    # Reference: best single model (V3) and round-1 ensemble
    results["V3"] = compute_all_metrics(actual_test, test_preds["V3"])
    print(f"\nV3 (reference): {results['V3']}")

    # ── 1. Isotonic calibration of V3 ────────────────────────────────────
    print("\n" + "─" * 70)
    print("1. Isotonic calibration on V3 predictions (per channel × volume_tier)")
    print("─" * 70)
    iso = SegmentedIsotonicCalibrator()
    iso.fit(df_val, val_preds["V3"], actual_val)
    pv3_iso_val = iso.transform(df_val, val_preds["V3"])
    pv3_iso_test = iso.transform(df_test, test_preds["V3"])
    results["V3+Isotonic"] = compute_all_metrics(actual_test, pv3_iso_test)
    print(f"  Val WAPE: {compute_all_metrics(actual_val, pv3_iso_val)}")
    print(f"  Test:     {results['V3+Isotonic']}")

    # ── 2. Linear bias correction on V3 ──────────────────────────────────
    print("\n" + "─" * 70)
    print("2. Linear bias correction on V3 (shrunk per segment)")
    print("─" * 70)
    bc = SegmentBiasCorrector(shrink=0.3)
    bc.fit(df_val, val_preds["V3"], actual_val)
    pv3_bc_val = bc.transform(df_val, val_preds["V3"])
    pv3_bc_test = bc.transform(df_test, test_preds["V3"])
    results["V3+BiasCorr"] = compute_all_metrics(actual_test, pv3_bc_test)
    print(f"  Test: {results['V3+BiasCorr']}")

    # ── 3. GBDT meta-learner blending all base preds ─────────────────────
    print("\n" + "─" * 70)
    print("3. GBDT meta-learner (nonlinear blend of base models)")
    print("─" * 70)
    meta = GBDTMetaLearner()
    meta.fit(df_val, val_preds, actual_val)
    p_meta_test = meta.predict(df_test, test_preds)
    p_meta_val = meta.predict(df_val, val_preds)
    results["V4_MetaGBDT"] = compute_all_metrics(actual_test, p_meta_test)
    print(f"  Val metrics: {compute_all_metrics(actual_val, p_meta_val)}")
    print(f"  Test:        {results['V4_MetaGBDT']}")

    # ── 4. Isotonic applied to MetaGBDT ──────────────────────────────────
    print("\n" + "─" * 70)
    print("4. MetaGBDT + Isotonic calibration (compound)")
    print("─" * 70)
    iso_m = SegmentedIsotonicCalibrator()
    iso_m.fit(df_val, p_meta_val, actual_val)
    p_meta_iso_test = iso_m.transform(df_test, p_meta_test)
    p_meta_iso_val = iso_m.transform(df_val, p_meta_val)
    results["V4_Meta+Iso"] = compute_all_metrics(actual_test, p_meta_iso_test)
    print(f"  Test: {results['V4_Meta+Iso']}")

    # ── 5. Optimal linear blend including Meta + Isotonic ────────────────
    print("\n" + "─" * 70)
    print("5. Extended linear blend (base preds + meta + isotonic-V3)")
    print("─" * 70)
    val_extended = {
        **val_preds,
        "MetaGBDT": p_meta_val,
        "V3_Iso": pv3_iso_val,
        "Meta_Iso": p_meta_iso_val,
    }
    test_extended = {
        **test_preds,
        "MetaGBDT": p_meta_test,
        "V3_Iso": pv3_iso_test,
        "Meta_Iso": p_meta_iso_test,
    }
    weights = optimize_ensemble_weights(val_extended, actual_val)
    p_super = stacking_ensemble_predict(test_extended, weights)
    results["V4_SuperBlend"] = compute_all_metrics(actual_test, p_super)
    print(f"  Weights: {weights}")
    print(f"  Test:    {results['V4_SuperBlend']}")

    # ── 6. SuperBlend + Isotonic as final step ───────────────────────────
    print("\n" + "─" * 70)
    print("6. SuperBlend + Isotonic (final calibration)")
    print("─" * 70)
    p_super_val = stacking_ensemble_predict(val_extended, weights)
    iso_f = SegmentedIsotonicCalibrator()
    iso_f.fit(df_val, p_super_val, actual_val)
    p_final_test = iso_f.transform(df_test, p_super)
    results["V4_Final"] = compute_all_metrics(actual_test, p_final_test)
    print(f"  Test: {results['V4_Final']}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("ROUND 2 SUMMARY")
    print("═" * 70)
    summary = pd.DataFrame(results).T.sort_values("WAPE")
    print(summary.to_string())
    summary.to_csv(OUTPUT / "v4_round2_results.csv")

    best_name = summary.index[0]
    best = summary.iloc[0]
    print(f"\n  BEST: {best_name}  WAPE={best['WAPE']:.4f}  MAPE_nz={best['MAPE_nz']:.4f}  RMSE={best['RMSE']:.4f}")

    # Save full pipeline artifacts
    joblib.dump(
        {
            "meta_learner": meta,
            "isotonic_final": iso_f,
            "ensemble_weights": weights,
            "base_models": list(val_preds.keys()),
        },
        OUTPUT / "model_v4_final.joblib",
    )

    with open(OUTPUT / "v4_final_weights.json", "w") as f:
        json.dump({"ensemble_weights": weights, "best_model": best_name,
                  "test_metrics": {k: float(v) for k, v in best.items()}}, f, indent=2)

    print(f"\n  Runtime: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
