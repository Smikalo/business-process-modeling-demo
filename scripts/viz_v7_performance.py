"""V7 performance dashboard — mirrors ``scripts.viz_v6_performance`` in layout.

Dashboard (``output/plot_v7_dashboard.png``, 3×2 grid):
    1. Monthly totals — actual vs V7 predicted with 80% conformal band
    2. Scatter (predicted vs actual, log)         |  V7 residual violin per month
    3. WAPE by segment (channel / brand / tier)   |  WAPE & MAPE_nz across V4…V7

Stand-alone V7-specific charts (not in earlier model dashboards):
    output/plot_v7_monthly.png
    output/plot_v7_scatter.png
    output/plot_v7_residuals.png           — V6 vs V7 residual violins
    output/plot_v7_segments.png            — V6 vs V7 per-segment WAPE
    output/plot_v7_alpha_sweep.png         — cost-vs-accuracy Pareto
    output/plot_v7_cost_breakdown.png      — V4..V7 UAH cost bars
    output/plot_v7_conformal_coverage.png  — empirical coverage by channel
    output/plot_v7_feature_importance.png  — top-20 gain

Run:
    python -m scripts.viz_v7_performance
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec

from src.evaluation import compute_all_metrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("viz_v7")

OUT = _REPO_ROOT / "output"
V7_ABT = OUT / "abt_v7_cached.parquet"
V6_ABT = OUT / "abt_v6_cached.parquet"
PREDS_VAL = OUT / "preds_v7_val.csv"
PREDS_TEST = OUT / "preds_v7_test.csv"
PREDS_LOW_VAL = OUT / "preds_v7_lower_val.csv"
PREDS_LOW_TEST = OUT / "preds_v7_lower_test.csv"
PREDS_HI_VAL = OUT / "preds_v7_upper_val.csv"
PREDS_HI_TEST = OUT / "preds_v7_upper_test.csv"
PREDS_V6_VAL = OUT / "preds_v6_val.csv"
PREDS_V6_TEST = OUT / "preds_v6_test.csv"
FI = OUT / "feature_importance_v7.csv"
METRICS_V7 = OUT / "v7_metrics.csv"
METRICS_V6 = OUT / "v6_metrics.csv"
METRICS_V5 = OUT / "v5_metrics.csv"
ALPHA_SWEEP = OUT / "v7_alpha_sweep.csv"
COST_JSON = OUT / "cost_scorecard_final.json"

DASHBOARD_PNG = OUT / "plot_v7_dashboard.png"
MONTHLY_PNG = OUT / "plot_v7_monthly.png"
SCATTER_PNG = OUT / "plot_v7_scatter.png"
RESID_PNG = OUT / "plot_v7_residuals.png"
SEGMENTS_PNG = OUT / "plot_v7_segments.png"
ALPHA_PNG = OUT / "plot_v7_alpha_sweep.png"
COST_PNG = OUT / "plot_v7_cost_breakdown.png"
COV_PNG = OUT / "plot_v7_conformal_coverage.png"
FI_PNG = OUT / "plot_v7_feature_importance.png"

# Features added in V7 (price + cohort); highlighted red in importance panel.
V7_NEW_FEATURES = {
    "price_lag1", "price_lag3",
    "price_vs_brand_median", "price_vs_channel_median", "price_vs_rrc",
    "price_change_3m_pct", "sku_price_elasticity",
    "cohort_demand_lag1", "cohort_stockout_share_lag1",
    "cohort_size", "cannibalisation_pressure",
}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})


def _wape(y: np.ndarray, p: np.ndarray) -> float:
    denom = np.abs(y).sum()
    return float(np.abs(y - p).sum() / denom) if denom else float("nan")


def _read_preds(path: Path, col_name: str = "prediction") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Период"] = pd.PeriodIndex(df["Период"].astype(str), freq="M")
    if "prediction" in df.columns and col_name != "prediction":
        df = df.rename(columns={"prediction": col_name})
    return df


def _load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    abt_path = V7_ABT if V7_ABT.exists() else V6_ABT
    abt = pd.read_parquet(abt_path)
    seg_cols = [c for c in ("Канал", "Бренд", "volume_tier", "Группа_товара")
                if c in abt.columns]
    base = abt[["Период", "Партнер", "Артикул", "target_qty", *seg_cols]]

    v = _read_preds(PREDS_VAL)
    t = _read_preds(PREDS_TEST)
    v["split"] = "val"; t["split"] = "test"

    if PREDS_LOW_VAL.exists() and PREDS_HI_VAL.exists():
        lo = _read_preds(PREDS_LOW_VAL, "lower")
        hi = _read_preds(PREDS_HI_VAL, "upper")
        v = v.merge(lo, on=["Период", "Партнер", "Артикул", "target_qty"], how="left")
        v = v.merge(hi, on=["Период", "Партнер", "Артикул", "target_qty"], how="left")
    if PREDS_LOW_TEST.exists() and PREDS_HI_TEST.exists():
        lo = _read_preds(PREDS_LOW_TEST, "lower")
        hi = _read_preds(PREDS_HI_TEST, "upper")
        t = t.merge(lo, on=["Период", "Партнер", "Артикул", "target_qty"], how="left")
        t = t.merge(hi, on=["Период", "Партнер", "Артикул", "target_qty"], how="left")

    if PREDS_V6_VAL.exists() and PREDS_V6_TEST.exists():
        v6v = _read_preds(PREDS_V6_VAL, "v6")
        v6t = _read_preds(PREDS_V6_TEST, "v6")
        v = v.merge(v6v[["Период", "Партнер", "Артикул", "v6"]],
                    on=["Период", "Партнер", "Артикул"], how="left")
        t = t.merge(v6t[["Период", "Партнер", "Артикул", "v6"]],
                    on=["Период", "Партнер", "Артикул"], how="left")

    v = v.merge(base, on=["Период", "Партнер", "Артикул", "target_qty"], how="left")
    t = t.merge(base, on=["Период", "Партнер", "Артикул", "target_qty"], how="left")
    return v, t


# ── Panels ──────────────────────────────────────────────────────────────────

def _panel_monthly(ax, df_val: pd.DataFrame, df_test: pd.DataFrame) -> None:
    both = pd.concat([df_val, df_test], ignore_index=True)
    agg_cols = {"actual": ("target_qty", "sum"),
                "predicted": ("prediction", "sum"),
                "split": ("split", "first")}
    if "lower" in both.columns:
        agg_cols["lo"] = ("lower", "sum")
        agg_cols["hi"] = ("upper", "sum")
    agg = (both.groupby("Период", observed=True)
           .agg(**agg_cols).reset_index().sort_values("Период"))
    x = np.arange(len(agg))
    labels = agg["Период"].astype(str)

    if "lo" in agg.columns:
        ax.fill_between(x, agg["lo"], agg["hi"], color="#d62728", alpha=0.12,
                        label="V7 80% conformal band")
    ax.plot(x, agg["actual"], marker="o", linewidth=2, label="Actual", color="#1f77b4")
    ax.plot(x, agg["predicted"], marker="s", linewidth=2, label="V7 predicted",
            color="#d62728", linestyle="--")

    split_idx = (agg["split"] == "val").sum()
    if 0 < split_idx < len(agg):
        ax.axvline(split_idx - 0.5, color="gray", linestyle=":", alpha=0.6)
        ylim = ax.get_ylim()
        ax.text(split_idx - 0.5, ylim[1] * 0.95, " test →",
                fontsize=9, color="gray", va="top")

    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30)
    ax.set_title("Monthly totals — actual vs V7 predicted (with 80% conformal band)")
    ax.set_ylabel("Units sold (aggregated)")
    ax.legend(loc="upper left")


def _panel_scatter(ax, df_val: pd.DataFrame, df_test: pd.DataFrame) -> None:
    for frame, color, label in [(df_val, "#1f77b4", "val"),
                                (df_test, "#d62728", "test")]:
        sample = frame.sample(min(len(frame), 8000), random_state=7)
        ax.scatter(sample["target_qty"] + 0.5, sample["prediction"] + 0.5,
                   s=6, alpha=0.25, color=color, label=f"{label} (n={len(frame):,})")
    xy_max = max(df_val["target_qty"].quantile(0.999),
                 df_test["target_qty"].quantile(0.999), 1.0)
    lim = (0.5, max(10.0, float(xy_max)))
    ax.plot(lim, lim, color="black", linewidth=1, linestyle="--", alpha=0.5, label="y = x")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("actual qty (+0.5, log)")
    ax.set_ylabel("predicted qty (+0.5, log)")
    ax.set_title("V7 prediction vs actual (row-level)")
    ax.legend(loc="upper left", markerscale=3)


def _panel_residual_violin(ax, df_val: pd.DataFrame, df_test: pd.DataFrame) -> None:
    """Match V6 dashboard style: V7 residuals per month, blue=val, red=test."""
    df = pd.concat([df_val, df_test], ignore_index=True)
    df["residual"] = df["prediction"] - df["target_qty"]
    df = df[(df["target_qty"] > 0) | (df["prediction"] > 0.1)]
    order = sorted(df["Период"].unique())
    data = [df.loc[df["Период"] == p, "residual"].clip(-20, 20).values for p in order]
    labels = [str(p) for p in order]
    val_periods = {str(p) for p in df_val["Период"].unique()}
    parts = ax.violinplot(data, showmeans=True, showextrema=False, widths=0.9)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor("#1f77b4" if labels[i] in val_periods else "#d62728")
        pc.set_alpha(0.55)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30)
    ax.set_ylim(-20, 20)
    ax.set_ylabel("residual (pred − actual), clipped ±20")
    ax.set_title("V7 residual distribution per month (blue=val, red=test)")


def _panel_segment_wape_single(ax, df_val: pd.DataFrame, df_test: pd.DataFrame) -> None:
    """Match V6 dashboard: V7-only WAPE by segment with coloured facets."""
    df_both = pd.concat([df_val, df_test], ignore_index=True)
    rows: list[dict] = []
    for col in ["Канал", "Бренд", "volume_tier"]:
        if col not in df_both.columns:
            continue
        for key, grp in df_both.groupby(col, observed=True):
            if len(grp) < 100:
                continue
            rows.append({
                "facet": col, "segment": str(key), "n": len(grp),
                "wape": _wape(grp["target_qty"].values, grp["prediction"].values),
            })
    if not rows:
        ax.set_axis_off(); ax.set_title("WAPE by segment — no columns found"); return
    df_seg = pd.DataFrame(rows).sort_values(["facet", "wape"])
    palette = {"Канал": "#1f77b4", "Бренд": "#2ca02c", "volume_tier": "#ff7f0e"}
    y_pos = np.arange(len(df_seg))
    colors = [palette.get(f, "#888") for f in df_seg["facet"]]
    ax.barh(y_pos, df_seg["wape"], color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r.facet}: {r.segment} (n={r.n:,})" for r in df_seg.itertuples()],
                       fontsize=8)
    ax.set_xlabel("WAPE")
    ax.set_title("WAPE by segment (V7, val+test)")
    ax.axvline(_wape(df_both["target_qty"].values, df_both["prediction"].values),
               color="black", linewidth=1, linestyle="--", label="overall V7")
    ax.legend(loc="lower right")


def _panel_model_comparison(ax) -> None:
    """Match V6 dashboard style: WAPE & MAPE_nz across V4/V5/V6/V7."""
    frames = []
    for p in (METRICS_V5, METRICS_V6, METRICS_V7):
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        ax.set_axis_off(); ax.set_title("Metrics CSVs missing"); return
    combined = pd.concat(frames, ignore_index=True)
    if "V4" not in set(combined["model"]):
        try:
            v4_val = pd.read_csv(OUT / "preds_v4_val.csv")
            v4_test = pd.read_csv(OUT / "preds_v4_test.csv")
            rows = []
            for split, df in [("val", v4_val), ("test", v4_test)]:
                m = compute_all_metrics(df["target_qty"].values, df["prediction"].values)
                rows.append({"model": "V4", "split": split, **m})
            combined = pd.concat([pd.DataFrame(rows), combined], ignore_index=True)
        except FileNotFoundError:
            pass

    # Use V7_cal variant from v7_metrics.csv as the headline V7 row
    combined = combined[~((combined["model"] == "V7") & (combined["model"] != "V7"))]
    v7 = combined[combined["model"].str.startswith("V7")]
    if not v7.empty and "V7_cal" in set(v7["model"]):
        combined = combined[~combined["model"].str.startswith("V7")]
        cal = v7[v7["model"] == "V7_cal"].copy()
        cal["model"] = "V7"
        combined = pd.concat([combined, cal], ignore_index=True)

    order = ["V4", "V5", "V6", "V7"]
    combined = combined[combined["model"].isin(order)]
    combined["model"] = pd.Categorical(combined["model"], categories=order, ordered=True)
    combined["split"] = pd.Categorical(combined["split"], categories=["val", "test"], ordered=True)
    combined = combined.sort_values(["model", "split"]).reset_index(drop=True)

    labels = [f"{r['model']}-{r['split']}" for _, r in combined.iterrows()]
    wape_vals = combined["WAPE"].values
    mape_vals = combined["MAPE_nz"].values
    x = np.arange(len(labels))
    w = 0.4
    ax.bar(x - w/2, wape_vals, w, label="WAPE", color="#1f77b4", alpha=0.85)
    ax.bar(x + w/2, mape_vals, w, label="MAPE_nz", color="#ff7f0e", alpha=0.85)
    for xi, v in zip(x - w/2, wape_vals):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
    for xi, v in zip(x + w/2, mape_vals):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_title("WAPE & MAPE_nz across models / splits")
    ax.legend(loc="upper right", fontsize=8)


def _panel_residual_vs_v6(ax, df_val: pd.DataFrame, df_test: pd.DataFrame) -> None:
    df = pd.concat([df_val, df_test], ignore_index=True)
    df = df[(df["target_qty"] > 0) | (df["prediction"] > 0.1)]
    has_v6 = "v6" in df.columns and df["v6"].notna().any()
    order = sorted(df["Период"].unique())
    labels = [str(p) for p in order]
    val_periods = {str(p) for p in df_val["Период"].unique()}
    v7_data = [(df.loc[df["Период"] == p, "prediction"] - df.loc[df["Период"] == p, "target_qty"]).clip(-20, 20).values
               for p in order]

    width = 0.35 if has_v6 else 0.7
    xs = np.arange(1, len(labels) + 1)
    if has_v6:
        v6_data = [(df.loc[df["Период"] == p, "v6"] - df.loc[df["Период"] == p, "target_qty"]).clip(-20, 20).values
                   for p in order]
        v6_data = [a[~np.isnan(a)] for a in v6_data]
        parts6 = ax.violinplot(v6_data, positions=xs - width/2, widths=width,
                               showmeans=True, showextrema=False)
        for pc in parts6["bodies"]:
            pc.set_facecolor("#2ca02c"); pc.set_alpha(0.5)
    parts7 = ax.violinplot(v7_data, positions=xs + (width/2 if has_v6 else 0),
                           widths=width, showmeans=True, showextrema=False)
    for i, pc in enumerate(parts7["bodies"]):
        pc.set_facecolor("#d62728"); pc.set_alpha(0.6)
        if labels[i] in val_periods:
            pc.set_edgecolor("#1f77b4"); pc.set_linewidth(1.2)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=30)
    ax.set_ylim(-20, 20)
    ax.set_ylabel("residual (pred − actual), clipped ±20")
    title = "Residual distribution: V6 (green) vs V7 (red)" if has_v6 \
            else "V7 residual distribution per month"
    ax.set_title(title + "  |  blue edge = val month")


def _panel_segment_wape(ax, df_val: pd.DataFrame, df_test: pd.DataFrame) -> None:
    df_both = pd.concat([df_val, df_test], ignore_index=True)
    has_v6 = "v6" in df_both.columns and df_both["v6"].notna().any()
    rows: list[dict] = []
    for col in ["Канал", "Бренд", "volume_tier"]:
        if col not in df_both.columns:
            continue
        for key, grp in df_both.groupby(col, observed=True):
            if len(grp) < 100:
                continue
            row = {"facet": col, "segment": str(key), "n": len(grp),
                   "wape_v7": _wape(grp["target_qty"].values, grp["prediction"].values)}
            if has_v6:
                mask = grp["v6"].notna()
                if mask.any():
                    row["wape_v6"] = _wape(grp.loc[mask, "target_qty"].values,
                                           grp.loc[mask, "v6"].values)
            rows.append(row)
    if not rows:
        ax.set_axis_off(); ax.set_title("WAPE by segment — no columns found"); return
    df_seg = pd.DataFrame(rows).sort_values(["facet", "wape_v7"])
    y = np.arange(len(df_seg))
    w = 0.4 if "wape_v6" in df_seg.columns else 0.8
    if "wape_v6" in df_seg.columns:
        ax.barh(y - w/2, df_seg["wape_v6"], w, color="#2ca02c", alpha=0.85, label="V6")
        ax.barh(y + w/2, df_seg["wape_v7"], w, color="#d62728", alpha=0.85, label="V7")
    else:
        ax.barh(y, df_seg["wape_v7"], w, color="#d62728", alpha=0.85, label="V7")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.facet}: {r.segment} (n={r.n:,})" for r in df_seg.itertuples()],
                       fontsize=8)
    ax.set_xlabel("WAPE")
    ax.set_title("WAPE by segment — V6 vs V7 (val+test)" if "wape_v6" in df_seg.columns
                 else "V7 WAPE by segment")
    ax.axvline(_wape(df_both["target_qty"].values, df_both["prediction"].values),
               color="black", linewidth=1, linestyle="--", label="overall V7")
    ax.legend(loc="lower right", fontsize=8)


def _panel_alpha_sweep(ax) -> None:
    if not ALPHA_SWEEP.exists():
        ax.set_axis_off(); ax.set_title("v7_alpha_sweep.csv missing"); return
    sw = pd.read_csv(ALPHA_SWEEP).sort_values("alpha")
    wape = sw["test_WAPE"].to_numpy()
    cost_m = sw["total_UAH"].to_numpy() / 1e6
    colors = plt.cm.viridis((sw["alpha"] - sw["alpha"].min()) /
                            (sw["alpha"].max() - sw["alpha"].min() + 1e-9))
    ax.plot(cost_m, wape, color="#888", linewidth=1, zorder=1)
    ax.scatter(cost_m, wape, c=colors, s=120, edgecolors="black", linewidth=0.8, zorder=2)
    for _, r in sw.iterrows():
        ax.annotate(f"α={r['alpha']:.2f}",
                    (r["total_UAH"] / 1e6, r["test_WAPE"]),
                    fontsize=8, xytext=(6, 6), textcoords="offset points")
    best_wape = sw.loc[sw["test_WAPE"].idxmin()]
    best_cost = sw.loc[sw["total_UAH"].idxmin()]
    ax.scatter([best_wape["total_UAH"] / 1e6], [best_wape["test_WAPE"]],
               s=260, facecolors="none", edgecolors="#1f77b4", linewidth=2,
               label=f"best WAPE (α={best_wape['alpha']:.2f})")
    ax.scatter([best_cost["total_UAH"] / 1e6], [best_cost["test_WAPE"]],
               s=260, facecolors="none", edgecolors="#2ca02c", linewidth=2,
               label=f"cost-optimal (α={best_cost['alpha']:.2f})")
    ax.set_xlabel("Annualised UAH cost (M)")
    ax.set_ylabel("Test WAPE")
    ax.set_title("V7 α-sweep Pareto frontier (colour = α, lower-left is better)")
    ax.legend(loc="upper right", fontsize=8)


def _panel_cost_breakdown(ax) -> None:
    if not COST_JSON.exists():
        ax.set_axis_off(); ax.set_title("cost_scorecard_final.json missing"); return
    dat = json.loads(COST_JSON.read_text())
    rows = sorted(dat["models"], key=lambda r: r["total_cost_UAH"])
    names = [r["model"] for r in rows]
    holding = np.array([r["holding_cost_UAH"] / 1e6 for r in rows])
    lost = np.array([r["lost_margin_UAH"] / 1e6 for r in rows])
    totals = holding + lost
    x = np.arange(len(rows))

    def _c(m: str) -> str:
        return {"V4": "#a6c8ec", "V5": "#1f77b4", "V6": "#2ca02c",
                "V7": "#d62728"}.get(m, "#888")

    ax.bar(x, holding, color=[_c(n) for n in names], alpha=0.9,
           label="Holding (over-forecast)")
    ax.bar(x, lost, bottom=holding, color=[_c(n) for n in names], alpha=0.45,
           hatch="//", label="Lost margin (under-forecast)")
    for xi, t in zip(x, totals):
        ax.text(xi, t + 0.03, f"{t:.2f}M", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=10, ha="right")
    ax.set_ylabel("UAH (millions / year)")
    ax.set_title("Annual forecast-error cost (realised per-SKU margins, holding=22%)")
    ax.legend(loc="upper left", fontsize=8)


def _panel_conformal_coverage(ax, df_val: pd.DataFrame, df_test: pd.DataFrame) -> None:
    if "lower" not in df_test.columns:
        ax.set_axis_off(); ax.set_title("conformal intervals missing"); return
    df = pd.concat([df_val, df_test], ignore_index=True)
    df = df[df["lower"].notna()].copy()
    df["covered"] = (df["target_qty"] >= df["lower"]) & (df["target_qty"] <= df["upper"])
    df["width"] = df["upper"] - df["lower"]

    rows = []
    for (split, seg), grp in df.groupby(["split", "Канал"], observed=True):
        if len(grp) < 50:
            continue
        rows.append({
            "split": split, "segment": str(seg), "n": len(grp),
            "coverage": float(grp["covered"].mean()),
            "mean_width": float(grp["width"].mean()),
        })
    if not rows:
        ax.set_axis_off(); ax.set_title("conformal coverage: no segments"); return
    dfc = pd.DataFrame(rows).sort_values(["split", "segment"])
    x = np.arange(len(dfc))
    w = 0.8
    colors = ["#1f77b4" if s == "val" else "#d62728" for s in dfc["split"]]
    ax.bar(x, dfc["coverage"], w, color=colors, alpha=0.85)
    ax.axhline(0.80, color="black", linestyle="--", linewidth=1,
               label="target coverage = 80%")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.split}\n{r.segment}" for r in dfc.itertuples()],
                       rotation=0, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Empirical coverage")
    ax.set_title("V7 conformal coverage by channel (blue=val, red=test; target=80%)")
    for xi, c, w_mean in zip(x, dfc["coverage"], dfc["mean_width"]):
        ax.text(xi, c + 0.01, f"{c*100:.0f}%\n±{w_mean/2:.1f}u",
                ha="center", fontsize=7)
    ax.legend(loc="lower right", fontsize=8)


def _panel_feature_importance(ax) -> None:
    if not FI.exists():
        ax.set_axis_off(); ax.set_title("feature_importance_v7.csv missing"); return
    fi = pd.read_csv(FI)
    col = next((c for c in ("gain_total", "importance", "gain", "gain_reg") if c in fi.columns), None)
    if col is None:
        ax.set_axis_off(); ax.set_title("No importance column found"); return
    fi = fi.sort_values(col, ascending=False).head(20)[::-1]
    colors = ["#d62728" if f in V7_NEW_FEATURES else "#1f77b4" for f in fi["feature"]]
    y = np.arange(len(fi))
    ax.barh(y, fi[col], color=colors, alpha=0.9)
    ax.set_yticks(y); ax.set_yticklabels(fi["feature"], fontsize=8)
    ax.set_xlabel("LightGBM gain (V7 stages)")
    n_new = sum(1 for f in fi["feature"] if f in V7_NEW_FEATURES)
    ax.set_title(f"Top-20 V7 features (red = V7-new, {n_new}/20 in top-20)")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    if not (PREDS_VAL.exists() and PREDS_TEST.exists()):
        raise FileNotFoundError("V7 prediction CSVs missing — run scripts.train_v7 first.")
    df_val, df_test = _load_all()

    m_val = compute_all_metrics(df_val["target_qty"].values, df_val["prediction"].values)
    m_test = compute_all_metrics(df_test["target_qty"].values, df_test["prediction"].values)
    log.info("V7 val: %s", m_val)
    log.info("V7 test: %s", m_test)

    # ── Dashboard: 3×2 layout matching scripts.viz_v6_performance ──────────
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.25)
    fig.suptitle(
        f"V7 demand-forecasting dashboard  |  "
        f"val WAPE={m_val['WAPE']:.3f}  MAPE_nz={m_val['MAPE_nz']:.3f}  |  "
        f"test WAPE={m_test['WAPE']:.3f}  MAPE_nz={m_test['MAPE_nz']:.3f}",
        fontsize=13, y=0.995,
    )
    _panel_monthly(fig.add_subplot(gs[0, :]), df_val, df_test)
    _panel_scatter(fig.add_subplot(gs[1, 0]), df_val, df_test)
    _panel_residual_violin(fig.add_subplot(gs[1, 1]), df_val, df_test)
    _panel_segment_wape_single(fig.add_subplot(gs[2, 0]), df_val, df_test)
    _panel_model_comparison(fig.add_subplot(gs[2, 1]))
    fig.savefig(DASHBOARD_PNG, bbox_inches="tight")
    log.info("Dashboard → %s", DASHBOARD_PNG)
    plt.close(fig)

    # ── Standalone charts matching earlier versions' style ─────────────────
    for fn, size, plotter in [
        (MONTHLY_PNG, (11, 4), _panel_monthly),
        (SCATTER_PNG, (7, 7), _panel_scatter),
        (SEGMENTS_PNG, (10, 9), _panel_segment_wape),      # V6-vs-V7 variant
        (RESID_PNG, (13, 5), _panel_residual_vs_v6),       # V6-vs-V7 variant
    ]:
        f, a = plt.subplots(figsize=size)
        plotter(a, df_val, df_test)
        f.savefig(fn, bbox_inches="tight"); log.info("→ %s", fn); plt.close(f)

    # ── V7-specific extras (not on the main dashboard) ─────────────────────
    for fn, size, plotter in [
        (ALPHA_PNG, (8, 6), _panel_alpha_sweep),
        (COST_PNG, (10, 5.5), _panel_cost_breakdown),
        (FI_PNG, (10, 8), _panel_feature_importance),
    ]:
        f, a = plt.subplots(figsize=size)
        plotter(a)
        f.savefig(fn, bbox_inches="tight"); log.info("→ %s", fn); plt.close(f)

    f, a = plt.subplots(figsize=(11, 5.5))
    _panel_conformal_coverage(a, df_val, df_test)
    f.savefig(COV_PNG, bbox_inches="tight"); log.info("→ %s", COV_PNG); plt.close(f)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
