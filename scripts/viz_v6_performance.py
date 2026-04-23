"""V6 performance dashboard — mirrors ``scripts.viz_v5_performance``.

Panels:
    1. Monthly totals — actual vs V6 predicted (val + test)
    2. Scatter (predicted vs actual, log)
    3. Residual violin per month
    4. WAPE by segment (channel, brand, volume tier)
    5. V4 vs V5 vs V6 bar comparison (fixed-split WAPE / MAPE_nz / Bias)
    6. Top-20 feature importances (V6 stage gains)

Also writes stand-alone charts for slide decks:
    output/plot_v6_dashboard.png
    output/plot_v6_monthly.png
    output/plot_v6_scatter.png
    output/plot_v6_segments.png
    output/plot_v6_feature_importance.png

Run:
    python -m scripts.viz_v6_performance
"""

from __future__ import annotations

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
log = logging.getLogger("viz_v6")

OUT_DIR = _REPO_ROOT / "output"
V6_ABT = OUT_DIR / "abt_v6_cached.parquet"
PREDS_VAL = OUT_DIR / "preds_v6_val.csv"
PREDS_TEST = OUT_DIR / "preds_v6_test.csv"
FI = OUT_DIR / "feature_importance_v6.csv"
METRICS_V5 = OUT_DIR / "v5_metrics.csv"
METRICS_V6 = OUT_DIR / "v6_metrics.csv"

DASHBOARD_PNG = OUT_DIR / "plot_v6_dashboard.png"
MONTHLY_PNG = OUT_DIR / "plot_v6_monthly.png"
SCATTER_PNG = OUT_DIR / "plot_v6_scatter.png"
SEGMENTS_PNG = OUT_DIR / "plot_v6_segments.png"
FI_PNG = OUT_DIR / "plot_v6_feature_importance.png"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})


def _wape(y: np.ndarray, p: np.ndarray) -> float:
    denom = np.abs(y).sum()
    return float(np.abs(y - p).sum() / denom) if denom else float("nan")


def _load_preds_with_segments() -> tuple[pd.DataFrame, pd.DataFrame]:
    abt = pd.read_parquet(V6_ABT)
    seg_cols = [c for c in ("Канал", "Бренд", "volume_tier", "Группа_товара") if c in abt.columns]
    base = abt[["Период", "Партнер", "Артикул", "target_qty", *seg_cols]]
    v = pd.read_csv(PREDS_VAL)
    t = pd.read_csv(PREDS_TEST)
    v["Период"] = pd.PeriodIndex(v["Период"].astype(str), freq="M")
    t["Период"] = pd.PeriodIndex(t["Период"].astype(str), freq="M")
    v = v.merge(base, on=["Период", "Партнер", "Артикул", "target_qty"], how="left")
    t = t.merge(base, on=["Период", "Партнер", "Артикул", "target_qty"], how="left")
    v["split"] = "val"
    t["split"] = "test"
    return v, t


# ── Panels ──────────────────────────────────────────────────────────────────

def _panel_monthly(ax, df_val: pd.DataFrame, df_test: pd.DataFrame) -> None:
    both = pd.concat([df_val, df_test], ignore_index=True)
    agg = (both.groupby("Период", observed=True)
           .agg(actual=("target_qty", "sum"),
                predicted=("prediction", "sum"),
                split=("split", "first"))
           .reset_index().sort_values("Период"))
    x = agg["Период"].astype(str)
    ax.plot(x, agg["actual"], marker="o", linewidth=2, label="Actual", color="#1f77b4")
    ax.plot(x, agg["predicted"], marker="s", linewidth=2, label="V6 predicted",
            color="#2ca02c", linestyle="--")
    split_idx = (agg["split"] == "val").sum()
    if 0 < split_idx < len(agg):
        ax.axvline(split_idx - 0.5, color="gray", linestyle=":", alpha=0.6)
        ylim = ax.get_ylim()
        ax.text(split_idx - 0.5, ylim[1] * 0.95, " test →",
                fontsize=9, color="gray", va="top")
    ax.set_title("Monthly totals — actual vs V6 predicted")
    ax.set_ylabel("Units sold (aggregated)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="upper left")


def _panel_scatter(ax, df_val: pd.DataFrame, df_test: pd.DataFrame) -> None:
    for frame, color, label in [(df_val, "#1f77b4", "val"), (df_test, "#2ca02c", "test")]:
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
    ax.set_title("Prediction vs actual (V6, row-level)")
    ax.legend(loc="upper left", markerscale=3)


def _panel_residual_violin(ax, df_val: pd.DataFrame, df_test: pd.DataFrame) -> None:
    df_both = pd.concat([df_val, df_test], ignore_index=True)
    df_both["residual"] = df_both["prediction"] - df_both["target_qty"]
    df_both = df_both[(df_both["target_qty"] > 0) | (df_both["prediction"] > 0.1)]
    order = sorted(df_both["Период"].unique())
    data = [df_both.loc[df_both["Период"] == p, "residual"].clip(-20, 20).values for p in order]
    labels = [str(p) for p in order]
    val_periods = {str(p) for p in df_val["Период"].unique()}
    parts = ax.violinplot(data, showmeans=True, showextrema=False, widths=0.9)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor("#1f77b4" if labels[i] in val_periods else "#2ca02c")
        pc.set_alpha(0.55)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30)
    ax.set_ylim(-20, 20)
    ax.set_ylabel("residual (pred − actual), clipped ±20")
    ax.set_title("V6 residual distribution per month (blue=val, green=test)")


def _panel_segment_wape(ax, df_val: pd.DataFrame, df_test: pd.DataFrame) -> None:
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
    ax.set_title("WAPE by segment (V6, val+test)")
    ax.axvline(_wape(df_both["target_qty"].values, df_both["prediction"].values),
               color="black", linewidth=1, linestyle="--", label="overall V6")
    ax.legend(loc="lower right")


def _panel_model_comparison(ax) -> None:
    v5 = pd.read_csv(METRICS_V5) if METRICS_V5.exists() else None
    v6 = pd.read_csv(METRICS_V6) if METRICS_V6.exists() else None
    if v5 is None and v6 is None:
        ax.set_axis_off(); ax.set_title("Metrics CSVs missing"); return
    combined = pd.concat([df for df in (v5, v6) if df is not None], ignore_index=True)
    pivot = combined.pivot_table(index=["model", "split"], values=["WAPE", "MAPE_nz"]).reset_index()
    labels = [f"{r['model']}-{r['split']}" for _, r in pivot.iterrows()]
    wape_vals = pivot["WAPE"].values
    mape_vals = pivot["MAPE_nz"].values
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


def _panel_feature_importance(ax) -> None:
    if not FI.exists():
        ax.set_axis_off(); ax.set_title("feature_importance_v6.csv missing"); return
    fi = pd.read_csv(FI)
    col = next((c for c in ("gain_total", "importance", "gain") if c in fi.columns), None)
    if col is None:
        ax.set_axis_off(); ax.set_title("No importance column found"); return
    fi = fi.sort_values(col, ascending=False).head(20)[::-1]

    new_features = {
        "was_censored", "promo_duration_months", "promo_depth_pct_current",
        "months_since_last_promo", "months_until_next_promo",
        "post_promo_depletion_flag", "sku_promo_sensitivity",
    }
    colors = ["#d62728" if f in new_features else "#1f77b4" for f in fi["feature"]]
    y = np.arange(len(fi))
    ax.barh(y, fi[col], color=colors, alpha=0.9)
    ax.set_yticks(y); ax.set_yticklabels(fi["feature"], fontsize=8)
    ax.set_xlabel("LightGBM gain")
    ax.set_title("Top-20 V6 features (red = newly added in V6)")


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> int:
    if not (V6_ABT.exists() and PREDS_VAL.exists() and PREDS_TEST.exists()):
        raise FileNotFoundError(
            "V6 artefacts missing. Run build_v6_abt and train_v6 first."
        )
    df_val, df_test = _load_preds_with_segments()

    m_val = compute_all_metrics(df_val["target_qty"].values, df_val["prediction"].values)
    m_test = compute_all_metrics(df_test["target_qty"].values, df_test["prediction"].values)
    log.info("V6 val: %s", m_val)
    log.info("V6 test: %s", m_test)

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.25)
    fig.suptitle(
        f"V6 demand-forecasting dashboard  |  "
        f"val WAPE={m_val['WAPE']:.3f}  MAPE_nz={m_val['MAPE_nz']:.3f}  |  "
        f"test WAPE={m_test['WAPE']:.3f}  MAPE_nz={m_test['MAPE_nz']:.3f}",
        fontsize=13, y=0.995,
    )
    _panel_monthly(fig.add_subplot(gs[0, :]), df_val, df_test)
    _panel_scatter(fig.add_subplot(gs[1, 0]), df_val, df_test)
    _panel_residual_violin(fig.add_subplot(gs[1, 1]), df_val, df_test)
    _panel_segment_wape(fig.add_subplot(gs[2, 0]), df_val, df_test)
    _panel_model_comparison(fig.add_subplot(gs[2, 1]))
    fig.savefig(DASHBOARD_PNG, bbox_inches="tight")
    log.info("Dashboard → %s", DASHBOARD_PNG)

    for fn, size, plotter in [
        (MONTHLY_PNG, (11, 4), _panel_monthly),
        (SCATTER_PNG, (7, 7), _panel_scatter),
        (SEGMENTS_PNG, (10, 9), _panel_segment_wape),
    ]:
        f, a = plt.subplots(figsize=size)
        plotter(a, df_val, df_test)
        f.savefig(fn, bbox_inches="tight"); log.info("→ %s", fn)

    f, a = plt.subplots(figsize=(10, 8))
    _panel_feature_importance(a)
    f.savefig(FI_PNG, bbox_inches="tight"); log.info("→ %s", FI_PNG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
