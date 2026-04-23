"""Side-by-side progression chart: V4 → V5 → V6.

Panels:
    1. Fixed-split WAPE and MAPE_nz bars (val + test) across the three models.
    2. Monthly WAPE lines — the curve of monthly WAPE across val+test for
       each model (evaluates stability over time).
    3. Rolling-origin WAPE box/whisker plots (V5 vs V6).
    4. UAH cost bars from ``output/cost_scorecard.json``.
    5. Per-segment (Канал) WAPE heatmap.
    6. Residual density per model (val + test combined, KDE).

Writes:
    output/plot_model_progression.png
    output/plot_progression_wape_bars.png
    output/plot_progression_monthly_wape.png
    output/plot_progression_rolling_box.png
    output/plot_progression_cost.png
    output/plot_progression_segment_heatmap.png
    output/plot_progression_residual_density.png

Run:
    python -m scripts.viz_model_progression
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
log = logging.getLogger("viz_progression")

OUT = _REPO_ROOT / "output"
PREDS = {
    "V4": (OUT / "preds_v4_val.csv", OUT / "preds_v4_test.csv"),
    "V5": (OUT / "preds_v5_val.csv", OUT / "preds_v5_test.csv"),
    "V6": (OUT / "preds_v6_val.csv", OUT / "preds_v6_test.csv"),
}
COLORS = {"V4": "#a6c8ec", "V5": "#1f77b4", "V6": "#2ca02c"}

DASHBOARD = OUT / "plot_model_progression.png"


def _wape(y: np.ndarray, p: np.ndarray) -> float:
    denom = np.abs(y).sum()
    return float(np.abs(y - p).sum() / denom) if denom else float("nan")


def _load_preds(val_p: Path, test_p: Path) -> pd.DataFrame | None:
    if not (val_p.exists() and test_p.exists()):
        return None
    v = pd.read_csv(val_p); v["split"] = "val"
    t = pd.read_csv(test_p); t["split"] = "test"
    df = pd.concat([v, t], ignore_index=True)
    df["Период"] = pd.PeriodIndex(df["Период"].astype(str), freq="M")
    return df


# ── Panels ──────────────────────────────────────────────────────────────────


def _panel_wape_bars(ax, data: dict[str, pd.DataFrame]) -> None:
    rows = []
    for name, df in data.items():
        if df is None:
            continue
        for split in ("val", "test"):
            part = df[df["split"] == split]
            m = compute_all_metrics(part["target_qty"].values, part["prediction"].values)
            rows.append({"model": name, "split": split, **m})
    pivot = pd.DataFrame(rows)
    x = np.arange(len(pivot))
    w = 0.35
    ax.bar(x - w/2, pivot["WAPE"], w, label="WAPE",
           color=[COLORS[m] for m in pivot["model"]], alpha=0.85)
    ax.bar(x + w/2, pivot["MAPE_nz"], w, label="MAPE_nz",
           color=[COLORS[m] for m in pivot["model"]], alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['model']}\n{r['split']}" for _, r in pivot.iterrows()])
    for xi, v in zip(x - w/2, pivot["WAPE"]):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    for xi, v in zip(x + w/2, pivot["MAPE_nz"]):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_title("Fixed-split WAPE & MAPE_nz — V4 vs V5 vs V6")
    ax.set_ylabel("error")
    ax.legend(loc="upper right")


def _panel_monthly_wape(ax, data: dict[str, pd.DataFrame]) -> None:
    for name, df in data.items():
        if df is None:
            continue
        per_month = (df.groupby("Период", observed=True)
                     .apply(lambda g: _wape(g["target_qty"].values, g["prediction"].values))
                     .rename("wape").reset_index().sort_values("Период"))
        ax.plot(per_month["Период"].astype(str), per_month["wape"],
                marker="o", linewidth=2, label=name, color=COLORS[name])
    ax.set_title("Monthly WAPE across val + test")
    ax.set_ylabel("WAPE (monthly)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="upper left")


def _panel_rolling_box(ax) -> None:
    path_v5 = OUT / "v5_rolling_cv.json"
    path_v6 = OUT / "v6_rolling_cv.json"
    boxes = []
    labels = []
    for name, p in (("V5", path_v5), ("V6", path_v6)):
        if not p.exists():
            continue
        dat = json.loads(p.read_text())
        wapes = [r["WAPE"] for r in dat["per_origin"]]
        boxes.append(wapes); labels.append(f"{name} (n={len(wapes)})")
    if not boxes:
        ax.set_axis_off()
        ax.set_title("Rolling CV data missing (run scripts.rolling_origin_cv)")
        return
    bp = ax.boxplot(boxes, labels=labels, patch_artist=True, widths=0.5)
    colors = [COLORS["V5"], COLORS["V6"]][: len(boxes)]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    for i, w in enumerate(boxes):
        ax.scatter(np.full(len(w), i + 1), w, color="black", s=14, zorder=3)
    ax.set_title("Rolling-origin WAPE — stability over 6 origins")
    ax.set_ylabel("WAPE per origin")


def _panel_cost(ax) -> None:
    cost_path = OUT / "cost_scorecard.json"
    if not cost_path.exists():
        ax.set_axis_off()
        ax.set_title("cost_scorecard.json missing — run decision_cost_scorecard")
        return
    dat = json.loads(cost_path.read_text())
    rows = sorted(dat["models"], key=lambda r: r["total_cost_UAH"])
    names = [r["model"] for r in rows]
    holding = [r["holding_cost_UAH"] / 1e6 for r in rows]
    lost = [r["lost_margin_UAH"] / 1e6 for r in rows]
    totals = [r["total_cost_UAH"] / 1e6 for r in rows]
    x = np.arange(len(rows))
    ax.bar(x, holding, label="Holding (over-forecast)", color="#ff7f0e", alpha=0.85)
    ax.bar(x, lost, bottom=holding, label="Lost margin (under-forecast)",
           color="#d62728", alpha=0.85)
    for xi, t in zip(x, totals):
        ax.text(xi, t + 0.05, f"{t:.2f}M", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("UAH (millions)")
    ax.set_title("Annual forecast-error cost (UAH M, assumptions in JSON)")
    ax.legend(loc="upper left")


def _panel_segment_heatmap(ax, data: dict[str, pd.DataFrame]) -> None:
    # Use ABT for channel lookup
    abt = pd.read_parquet(OUT / "abt_v6_cached.parquet")[
        ["Период", "Партнер", "Артикул", "Канал"]
    ]
    rows = []
    for name, df in data.items():
        if df is None:
            continue
        df2 = df.merge(abt, on=["Период", "Партнер", "Артикул"], how="left")
        for key, grp in df2.groupby("Канал", observed=True):
            if len(grp) < 200:
                continue
            rows.append({
                "model": name, "channel": str(key),
                "wape": _wape(grp["target_qty"].values, grp["prediction"].values),
            })
    if not rows:
        ax.set_axis_off(); ax.set_title("segment_heatmap: no data"); return
    heat = pd.DataFrame(rows).pivot(index="channel", columns="model", values="wape")
    heat = heat.reindex(columns=["V4", "V5", "V6"])
    im = ax.imshow(heat.values, aspect="auto", cmap="RdYlGn_r", vmin=0.3, vmax=0.9)
    ax.set_xticks(range(heat.shape[1])); ax.set_xticklabels(heat.columns)
    ax.set_yticks(range(heat.shape[0])); ax.set_yticklabels(heat.index)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("WAPE by channel (per model)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _panel_residual_density(ax, data: dict[str, pd.DataFrame]) -> None:
    for name, df in data.items():
        if df is None:
            continue
        resid = (df["prediction"] - df["target_qty"]).clip(-20, 20)
        sample = resid.sample(min(len(resid), 80000), random_state=7)
        ax.hist(sample, bins=80, histtype="step", linewidth=1.5,
                label=f"{name}", color=COLORS[name])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-20, 20)
    ax.set_xlabel("residual (pred − actual)")
    ax.set_ylabel("count (val + test, log)")
    ax.set_yscale("log")
    ax.set_title("Residual distribution (all rows, clipped ±20)")
    ax.legend(loc="upper right")


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> int:
    data = {name: _load_preds(v, t) for name, (v, t) in PREDS.items()}

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.28)
    fig.suptitle("Model progression: V4 → V5 → V6", fontsize=14, y=0.995)

    _panel_wape_bars(fig.add_subplot(gs[0, 0]), data)
    _panel_monthly_wape(fig.add_subplot(gs[0, 1]), data)
    _panel_rolling_box(fig.add_subplot(gs[1, 0]))
    _panel_cost(fig.add_subplot(gs[1, 1]))
    _panel_segment_heatmap(fig.add_subplot(gs[2, 0]), data)
    _panel_residual_density(fig.add_subplot(gs[2, 1]), data)

    fig.savefig(DASHBOARD, bbox_inches="tight")
    log.info("Dashboard → %s", DASHBOARD)

    for fn, size, plotter in [
        (OUT / "plot_progression_wape_bars.png", (9, 5), _panel_wape_bars),
        (OUT / "plot_progression_monthly_wape.png", (12, 4), _panel_monthly_wape),
        (OUT / "plot_progression_segment_heatmap.png", (7, 7), _panel_segment_heatmap),
        (OUT / "plot_progression_residual_density.png", (9, 5), _panel_residual_density),
    ]:
        f, a = plt.subplots(figsize=size)
        plotter(a, data)
        f.savefig(fn, bbox_inches="tight"); log.info("→ %s", fn)

    for fn, size, plotter in [
        (OUT / "plot_progression_rolling_box.png", (7, 5), _panel_rolling_box),
        (OUT / "plot_progression_cost.png", (7, 5), _panel_cost),
    ]:
        f, a = plt.subplots(figsize=size)
        plotter(a)
        f.savefig(fn, bbox_inches="tight"); log.info("→ %s", fn)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
