"""Visualize V5 model performance.

Produces a single multi-panel dashboard PNG plus a few focused charts.

Panels:
    1. Monthly actual vs predicted (validation + test)
    2. Predicted vs actual scatter (log scale) with diagonal reference
    3. Error distribution (residuals by month)
    4. WAPE by segment (Канал, Бренд, volume_tier)
    5. V4 vs V5 bar comparison (val & test WAPE / MAPE_nz / RMSE)
    6. Top-20 feature importances (grouped by source)

Run:
    PYTHONPATH=. python -m scripts.viz_v5_performance
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Allow running as a plain script (``python scripts/viz_v5_performance.py``)
# *or* as a module (``python -m scripts.viz_v5_performance``).  The module
# form has the repo root on sys.path already; the script form does not.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator

from src.evaluation import compute_all_metrics, split_train_val_test
from src.model_v2 import encode_categoricals, get_feature_columns_v2

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("viz_v5")

ROOT = Path(__file__).resolve().parent.parent
V5_ABT = ROOT / "output" / "abt_v5_cached.parquet"
V4_ABT = ROOT / "output" / "abt_v4_cached.parquet"
MODEL_V5 = ROOT / "output" / "model_v5.joblib"
FEAT_IMPORT = ROOT / "output" / "feature_importance_v5.csv"
METRICS_CSV = ROOT / "output" / "v5_metrics.csv"
MANIFEST = ROOT / "output" / "v5_feature_manifest.json"

OUT_DIR = ROOT / "output"
DASHBOARD_PNG = OUT_DIR / "plot_v5_dashboard.png"
MONTHLY_PNG = OUT_DIR / "plot_v5_monthly.png"
SCATTER_PNG = OUT_DIR / "plot_v5_scatter.png"
SEGMENTS_PNG = OUT_DIR / "plot_v5_segments.png"

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _wape(y: np.ndarray, p: np.ndarray) -> float:
    denom = np.abs(y).sum()
    return float(np.abs(y - p).sum() / denom) if denom else float("nan")


def _map_feature_to_source(manifest: dict) -> dict[str, str]:
    """Best-effort mapping feature→loader based on prefix."""
    ext_feats = set(manifest.get("external_features", []))
    prefix_to_source = {
        "holiday": "holidays_ua",
        "preholiday": "holidays_ua",
        "days_to": "holidays_ua",
        "is_dec": "holidays_ua",
        "easter": "holidays_ua",
        "major_": "holidays_ua",
        "uah_": "nbu_fx",
        "nbu_": "nbu_fx",
        "trends_": "gtrends_ua",
        "war_": "conflict_ua",
        "conflict_": "conflict_ua",
        "months_since_invasion": "conflict_ua",
        "intensity_": "conflict_ua",
        "family_": "tmdb_movies",
        "release_": "tmdb_movies",
        "wb_": "world_bank_ua",
    }
    mapping: dict[str, str] = {}
    for f in ext_feats:
        src = "external_other"
        for pfx, s in prefix_to_source.items():
            if f.startswith(pfx):
                src = s
                break
        mapping[f] = src
    return mapping


def _predict_v5(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load V5 model, score val & test splits, return enriched dataframes."""
    model = joblib.load(MODEL_V5)
    feats = get_feature_columns_v2(df)
    _, df_val, df_test = split_train_val_test(df)
    df_val = df_val.copy()
    df_test = df_test.copy()
    df_val["y_pred"] = model.predict(df_val)
    df_test["y_pred"] = model.predict(df_test)
    df_val["split"] = "val"
    df_test["split"] = "test"
    return df_val, df_test


# ── Panels ───────────────────────────────────────────────────────────────────


def _panel_monthly(ax, df_val: pd.DataFrame, df_test: pd.DataFrame) -> None:
    both = pd.concat([df_val, df_test], ignore_index=True)
    agg = (
        both.groupby("Период")
        .agg(actual=("target_qty", "sum"), predicted=("y_pred", "sum"), split=("split", "first"))
        .reset_index()
        .sort_values("Период")
    )
    x = agg["Период"].astype(str)
    ax.plot(x, agg["actual"], marker="o", linewidth=2, label="Actual", color="#1f77b4")
    ax.plot(x, agg["predicted"], marker="s", linewidth=2, label="V5 predicted",
            color="#d62728", linestyle="--")
    for _, r in agg.iterrows():
        if r["split"] == "test":
            ax.axvspan(str(r["Период"]), str(r["Период"]), alpha=0.08, color="gray")
    split_idx = (agg["split"] == "val").sum()
    if 0 < split_idx < len(agg):
        ax.axvline(split_idx - 0.5, color="gray", linestyle=":", alpha=0.6)
        ylim = ax.get_ylim()
        ax.text(split_idx - 0.5, ylim[1] * 0.95, " test →",
                fontsize=9, color="gray", va="top")
    ax.set_title("Monthly totals — actual vs V5 predicted (validation + test)")
    ax.set_ylabel("Units sold (SKU × partner × month, summed)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="upper left")


def _panel_scatter(ax, df_val: pd.DataFrame, df_test: pd.DataFrame) -> None:
    for frame, color, label in [(df_val, "#1f77b4", "val"), (df_test, "#d62728", "test")]:
        # Subsample: hundreds of thousands of points kills rendering.
        sample = frame.sample(min(len(frame), 8000), random_state=7)
        ax.scatter(
            sample["target_qty"] + 0.5,
            sample["y_pred"] + 0.5,
            s=6, alpha=0.25, color=color, label=f"{label} (n={len(frame):,})",
        )
    xy_max = max(
        df_val["target_qty"].quantile(0.999),
        df_test["target_qty"].quantile(0.999),
        1.0,
    )
    lim = (0.5, max(10.0, float(xy_max)))
    ax.plot(lim, lim, color="black", linewidth=1, linestyle="--", alpha=0.5, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("actual qty (+0.5, log)")
    ax.set_ylabel("predicted qty (+0.5, log)")
    ax.set_title("Prediction vs actual (row-level, subsampled)")
    ax.legend(loc="upper left", markerscale=3)


def _panel_residual_violin(ax, df_val: pd.DataFrame, df_test: pd.DataFrame) -> None:
    df_both = pd.concat([df_val, df_test], ignore_index=True)
    df_both["residual"] = df_both["y_pred"] - df_both["target_qty"]
    # Keep only rows with some signal (drop long tail of exact zero residuals).
    df_both = df_both[(df_both["target_qty"] > 0) | (df_both["y_pred"] > 0.1)]
    order = sorted(df_both["Период"].unique())
    data = [df_both.loc[df_both["Период"] == p, "residual"].clip(-20, 20).values for p in order]
    labels = [str(p) for p in order]
    parts = ax.violinplot(data, showmeans=True, showextrema=False, widths=0.9)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor("#1f77b4" if labels[i] in {str(p) for p in df_val["Период"].unique()} else "#d62728")
        pc.set_alpha(0.55)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30)
    ax.set_ylim(-20, 20)
    ax.set_ylabel("residual (pred − actual), clipped ±20")
    ax.set_title("Residual distribution per month (blue = val, red = test)")


def _panel_segment_wape(ax, df_val: pd.DataFrame, df_test: pd.DataFrame) -> None:
    df_both = pd.concat([df_val, df_test], ignore_index=True)
    rows: list[dict] = []
    for col in ["Канал", "Бренд", "volume_tier"]:
        if col not in df_both.columns:
            continue
        for key, grp in df_both.groupby(col, observed=True):
            if len(grp) < 100:
                continue
            rows.append(
                {
                    "facet": col,
                    "segment": str(key),
                    "n": len(grp),
                    "wape": _wape(grp["target_qty"].values, grp["y_pred"].values),
                }
            )
    if not rows:
        ax.set_axis_off()
        ax.set_title("WAPE by segment — no segment columns found")
        return

    df_seg = pd.DataFrame(rows).sort_values(["facet", "wape"])
    palette = {
        "Канал": "#1f77b4",
        "Бренд": "#2ca02c",
        "volume_tier": "#ff7f0e",
    }
    y_pos = np.arange(len(df_seg))
    colors = [palette.get(f, "#888") for f in df_seg["facet"]]
    ax.barh(y_pos, df_seg["wape"], color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{r.facet}: {r.segment} (n={r.n:,})" for r in df_seg.itertuples()], fontsize=8
    )
    ax.set_xlabel("WAPE")
    ax.set_title("WAPE by segment (val+test combined)")
    ax.axvline(_wape(df_both["target_qty"].values, df_both["y_pred"].values),
               color="black", linewidth=1, linestyle="--", label="overall WAPE")
    ax.legend(loc="lower right")


def _panel_model_comparison(ax) -> None:
    if not METRICS_CSV.exists():
        ax.set_axis_off()
        ax.set_title("v5_metrics.csv missing — run scripts.train_v5")
        return
    m = pd.read_csv(METRICS_CSV)
    # Expected rows: V4/V5 × val/test.  Plot WAPE & MAPE_nz & RMSE (normalised).
    pivot = m.pivot_table(index=["model", "split"], values=["WAPE", "MAPE_nz", "RMSE"])
    pivot = pivot.reset_index()

    metric_cols = ["WAPE", "MAPE_nz", "RMSE"]
    bar_w = 0.2
    x = np.arange(len(metric_cols))
    offsets = {"V4-val": -1.5, "V4-test": -0.5, "V5-val": 0.5, "V5-test": 1.5}
    colors = {"V4-val": "#a6c8ec", "V4-test": "#1f77b4", "V5-val": "#f7a39a", "V5-test": "#d62728"}
    max_per_metric = {m: max(pivot[m].max(), 1e-9) for m in metric_cols}

    for _, r in pivot.iterrows():
        key = f"{r['model']}-{r['split']}"
        heights = [r[m] / max_per_metric[m] for m in metric_cols]
        ax.bar(x + offsets[key] * bar_w, heights, bar_w,
               label=key, color=colors[key], alpha=0.9)
        for xi, h, m_ in zip(x, heights, metric_cols):
            ax.text(xi + offsets[key] * bar_w, h + 0.01, f"{r[m_]:.3f}",
                    ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_cols)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("metric / column max")
    ax.set_title("V4 vs V5 — val + test (normalised per metric)")
    ax.legend(loc="upper right", ncol=2, fontsize=8)


def _panel_feature_importance(ax) -> None:
    if not FEAT_IMPORT.exists():
        ax.set_axis_off()
        ax.set_title("feature_importance_v5.csv missing — run scripts.run_feature_importance")
        return
    fi = pd.read_csv(FEAT_IMPORT)
    importance_col = next(
        (c for c in ("gain_total", "importance", "gain") if c in fi.columns),
        None,
    )
    if importance_col is None:
        ax.set_axis_off()
        ax.set_title("feature_importance_v5.csv has no recognisable importance column")
        return
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    mapping = _map_feature_to_source(manifest)
    if "source" not in fi.columns:
        fi["source"] = fi["feature"].map(mapping).fillna("internal")
    else:
        fi["source"] = fi["source"].fillna("internal")
    fi = fi.sort_values(importance_col, ascending=False).head(20)[::-1]
    fi = fi.rename(columns={importance_col: "importance"})

    palette = {
        "internal": "#888888",
        "holidays_ua": "#1f77b4",
        "nbu_fx": "#2ca02c",
        "conflict_ua": "#d62728",
        "gtrends_ua": "#ff7f0e",
        "tmdb_movies": "#9467bd",
        "world_bank_ua": "#8c564b",
        "external_other": "#17becf",
    }
    colors = [palette.get(s, "#444") for s in fi["source"]]
    y = np.arange(len(fi))
    ax.barh(y, fi["importance"], color=colors, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(fi["feature"], fontsize=8)
    ax.set_xlabel("LightGBM gain")
    ax.set_title("Top-20 feature importances (colour = source)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[k]) for k in palette if k != "internal"]
    labels = [k for k in palette if k != "internal"]
    handles.insert(0, plt.Rectangle((0, 0), 1, 1, color=palette["internal"]))
    labels.insert(0, "internal")
    ax.legend(handles, labels, loc="lower right", fontsize=7)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    if not V5_ABT.exists() or not MODEL_V5.exists():
        raise FileNotFoundError(
            "V5 artefacts missing. Run `python -m scripts.build_v5_abt` then "
            "`python -m scripts.train_v5` first."
        )

    log.info("Loading V5 ABT + model")
    abt = pd.read_parquet(V5_ABT).pipe(encode_categoricals)
    df_val, df_test = _predict_v5(abt)

    m_val = compute_all_metrics(df_val["target_qty"].values, df_val["y_pred"].values)
    m_test = compute_all_metrics(df_test["target_qty"].values, df_test["y_pred"].values)
    log.info("V5 val:  %s", m_val)
    log.info("V5 test: %s", m_test)

    # ── Dashboard (6 panels) ──
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.25)

    fig.suptitle(
        f"V5 demand-forecasting dashboard    |    "
        f"val WAPE={m_val['WAPE']:.3f}  MAPE_nz={m_val['MAPE_nz']:.3f}    |    "
        f"test WAPE={m_test['WAPE']:.3f}  MAPE_nz={m_test['MAPE_nz']:.3f}",
        fontsize=13,
        y=0.995,
    )

    _panel_monthly(fig.add_subplot(gs[0, :]), df_val, df_test)
    _panel_scatter(fig.add_subplot(gs[1, 0]), df_val, df_test)
    _panel_residual_violin(fig.add_subplot(gs[1, 1]), df_val, df_test)
    _panel_segment_wape(fig.add_subplot(gs[2, 0]), df_val, df_test)
    _panel_model_comparison(fig.add_subplot(gs[2, 1]))

    fig.savefig(DASHBOARD_PNG, bbox_inches="tight")
    log.info("Dashboard → %s", DASHBOARD_PNG)

    # ── Stand-alone charts (useful in slide decks) ──
    fig_m, ax_m = plt.subplots(figsize=(11, 4))
    _panel_monthly(ax_m, df_val, df_test)
    fig_m.savefig(MONTHLY_PNG, bbox_inches="tight")
    log.info("Monthly chart → %s", MONTHLY_PNG)

    fig_s, ax_s = plt.subplots(figsize=(7, 7))
    _panel_scatter(ax_s, df_val, df_test)
    fig_s.savefig(SCATTER_PNG, bbox_inches="tight")
    log.info("Scatter chart → %s", SCATTER_PNG)

    fig_g, ax_g = plt.subplots(figsize=(10, 9))
    _panel_segment_wape(ax_g, df_val, df_test)
    fig_g.savefig(SEGMENTS_PNG, bbox_inches="tight")
    log.info("Segments chart → %s", SEGMENTS_PNG)

    # Feature-importance stand-alone (mirrors panel).
    fi_png = OUT_DIR / "plot_v5_feature_importance.png"
    fig_f, ax_f = plt.subplots(figsize=(10, 8))
    _panel_feature_importance(ax_f)
    fig_f.savefig(fi_png, bbox_inches="tight")
    log.info("Feature-importance chart → %s", fi_png)

    print("")
    print("── V5 performance ─────────────────────────────────────────────────")
    print(f"val   WAPE={m_val['WAPE']:.4f}  MAPE_nz={m_val['MAPE_nz']:.4f}  "
          f"RMSE={m_val['RMSE']:.4f}  Bias={m_val['Bias']:+.4f}")
    print(f"test  WAPE={m_test['WAPE']:.4f}  MAPE_nz={m_test['MAPE_nz']:.4f}  "
          f"RMSE={m_test['RMSE']:.4f}  Bias={m_test['Bias']:+.4f}")
    print("───────────────────────────────────────────────────────────────────")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
