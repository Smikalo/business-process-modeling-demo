"""End-to-end pipeline: ingest → features → train → evaluate → recommend → visualize."""

import json
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s %(message)s")

import joblib
import numpy as np
import pandas as pd

from src.enrichment import enrich_all
from src.evaluation import compute_all_metrics, split_train_val_test
from src.features import engineer_all_features
from src.master import assemble_master
from src.model import (
    evaluate_baselines,
    feature_importance,
    predict_lightgbm,
    train_lightgbm,
)
from src.optimize import run_optimization
from src.procurement import build_quantile_models, compute_order_recommendations

t0 = time.time()

print("=" * 60)
print("STEP 1: Data pipeline")
print("=" * 60)
df = assemble_master()
df = enrich_all(df)
df = engineer_all_features(df)
df_train, df_val, df_test = split_train_val_test(df)
print(f"  Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")

print("\n" + "=" * 60)
print("STEP 2: Baselines")
print("=" * 60)
bl = evaluate_baselines(df_test)
print(bl.to_string())
bl.to_csv("output/baseline_results.csv")

print("\n" + "=" * 60)
print("STEP 3: Optuna hyperparameter search (30 trials)")
print("=" * 60)
best_params, study = run_optimization(df_train, df_val, n_trials=30)
print(f"  Best WAPE: {study.best_value:.4f}")
with open("output/optuna_best_params.json", "w") as f:
    json.dump({"best_wape": float(study.best_value), **best_params}, f, indent=2)

print("\n" + "=" * 60)
print("STEP 4: Train final model (train+val → test)")
print("=" * 60)
df_trainval = pd.concat([df_train, df_val], ignore_index=True)
model = train_lightgbm(df_trainval, df_test, params=best_params, num_boost_round=500)
preds_test = predict_lightgbm(model, df_test)
test_metrics = compute_all_metrics(df_test["target_qty"].values, preds_test)
print(f"  Test metrics: {test_metrics}")
joblib.dump(model, "output/model_final.joblib")

fi = feature_importance(model, df_test)
fi.to_csv("output/feature_importance_final.csv", index=False)

print("\n" + "=" * 60)
print("STEP 5: Procurement recommendations")
print("=" * 60)
base = {k: v for k, v in best_params.items() if k != "best_wape"}
base.update({"n_jobs": -1, "device": "cpu"})
models = build_quantile_models(df_trainval, df_test, base)
last_period = df_test["Период"].max()
df_latest = df_test[df_test["Период"] == last_period].copy()
recs = compute_order_recommendations(models["q50"], models["q90"], df_latest)
recs.to_csv("output/order_recommendations.csv", index=False)
print(f"  {len(recs)} actionable orders, {recs['recommended_order'].sum()} total units")

elapsed = time.time() - t0
print("\n" + "=" * 60)
print(f"DONE in {elapsed:.0f}s — all outputs in output/")
print("=" * 60)
