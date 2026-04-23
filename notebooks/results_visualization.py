"""Results visualization script — generates all PoC charts to output/ directory."""

import json
import logging
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s %(message)s")

from src.enrichment import enrich_all
from src.evaluation import compute_all_metrics, get_feature_columns, split_train_val_test
from src.features import engineer_all_features
from src.master import assemble_master
from src.model import BASELINES, predict_lightgbm

sns.set_theme(style="whitegrid", font_scale=1.2)

# ── Load data ────────────────────────────────────────────────────────────────

print("Building data pipeline...")
df = assemble_master()
df = enrich_all(df)
df = engineer_all_features(df)
df_train, df_val, df_test = split_train_val_test(df)

model = joblib.load("output/model_final.joblib")
preds_test = predict_lightgbm(model, df_test)

# ── 1. Baselines vs LightGBM comparison ─────────────────────────────────────

print("Plot 1: Model comparison...")
actual_test = df_test["target_qty"].values
results = []
for name, fn in BASELINES.items():
    m = compute_all_metrics(actual_test, fn(df_test))
    m["Model"] = name
    results.append(m)
m = compute_all_metrics(actual_test, preds_test)
m["Model"] = "LightGBM"
results.append(m)
comp = pd.DataFrame(results)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, metric in zip(axes, ["WAPE", "MAPE_nz", "RMSE"]):
    bars = ax.barh(comp["Model"], comp[metric], color=["#b0b0b0"] * 5 + ["#2196F3"])
    ax.set_xlabel(metric)
    ax.set_title(metric)
    ax.invert_yaxis()
plt.suptitle("Model Comparison — Test Set (Jul 2025 – Feb 2026)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("output/plot_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 2. Feature importance ───────────────────────────────────────────────────

print("Plot 2: Feature importance...")
fi = pd.read_csv("output/feature_importance_final.csv")
top20 = fi.head(20)

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(top20["feature"][::-1], top20["gain"][::-1], color="#4CAF50")
ax.set_xlabel("Gain")
ax.set_title("Top 20 Features by Gain (LightGBM)")
plt.tight_layout()
plt.savefig("output/plot_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 3. Actual vs Predicted scatter (non-zero only) ──────────────────────────

print("Plot 3: Actual vs Predicted...")
mask_nz = actual_test > 0
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(actual_test[mask_nz], preds_test[mask_nz], alpha=0.1, s=4, color="#FF5722")
lim = max(actual_test[mask_nz].max(), preds_test[mask_nz].max())
ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="Perfect forecast")
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title(f"Actual vs Predicted (n={mask_nz.sum():,} non-zero obs)")
ax.legend()
plt.tight_layout()
plt.savefig("output/plot_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 4. Monthly aggregate: actual vs forecast ────────────────────────────────

print("Plot 4: Monthly trends...")
df_test_plot = df_test[["Период", "target_qty"]].copy()
df_test_plot["predicted"] = preds_test
monthly = df_test_plot.groupby("Период").agg({"target_qty": "sum", "predicted": "sum"})
monthly.index = monthly.index.astype(str)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(monthly.index, monthly["target_qty"], "o-", label="Actual", color="#1976D2")
ax.plot(monthly.index, monthly["predicted"], "s--", label="LightGBM", color="#FF5722")
ax.set_xlabel("Month")
ax.set_ylabel("Total Quantity")
ax.set_title("Monthly Aggregate: Actual vs LightGBM Forecast (Test Period)")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/plot_monthly_trend.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 5. Stockout impact analysis ─────────────────────────────────────────────

print("Plot 5: Stockout analysis...")
so = df_test.copy()
so["predicted"] = preds_test
so["error"] = np.abs(so["target_qty"] - so["predicted"])
so_group = so.groupby("stockout_orc").agg(
    mean_actual=("target_qty", "mean"),
    mean_error=("error", "mean"),
    count=("target_qty", "count"),
).reset_index()

fig, ax = plt.subplots(figsize=(8, 5))
x = ["In Stock", "Stockout"]
bars = ax.bar(x, so_group["mean_actual"], color=["#4CAF50", "#F44336"], alpha=0.7, label="Mean Actual Demand")
ax.set_ylabel("Mean Qty")
ax.set_title("Impact of Warehouse Stockout on Demand")
ax.legend()
plt.tight_layout()
plt.savefig("output/plot_stockout_impact.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 6. Order recommendation summary ────────────────────────────────────────

print("Plot 6: Order recommendations...")
recs = pd.read_csv("output/order_recommendations.csv")
brand_summary = recs.groupby("Бренд").agg(
    total_order=("recommended_order", "sum"),
    sku_count=("Артикул", "nunique"),
).sort_values("total_order", ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(brand_summary.index[::-1], brand_summary["total_order"][::-1], color="#9C27B0")
ax.set_xlabel("Total Recommended Order (units)")
ax.set_title("Procurement Recommendations by Brand")
for i, (_, row) in enumerate(brand_summary[::-1].iterrows()):
    ax.text(row["total_order"] + 50, i, f'{row["sku_count"]} SKUs', va="center", fontsize=9)
plt.tight_layout()
plt.savefig("output/plot_order_recommendations.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nAll 6 plots saved to output/")
