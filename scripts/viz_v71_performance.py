"""V7.1 dashboard — recency weights + per-channel specialists blend.

Mirrors the 3×2 layout used by V6/V7::

    Row 1:  Monthly totals  (actual vs V7 vs V7.1)
    Row 2:  Scatter (pred vs actual, log)      |  V7.1 residuals by month
    Row 3:  WAPE by channel (V7 vs V7.1)       |  Cost waterfall V6 → V7 → V7.1

Stand-alone auxiliary plots::
    output/plot_v71_recency_sweep.png       — γ sweep (WAPE + UAH cost)
    output/plot_v71_channel_blend.png       — blend-weight sweep
    output/plot_v71_stability.png           — per-month WAPE of champion
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec

from src.evaluation import compute_all_metrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("viz_v71")

OUT = _REPO / "output"

V7_TEST = OUT / "preds_v7_test.csv"
V7_VAL = OUT / "preds_v7_val.csv"
V71_TEST = OUT / "preds_v71_test.csv"
V71_VAL = OUT / "preds_v71_val.csv"
ABT = OUT / "abt_v7_cached.parquet"


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    v7v = pd.read_csv(V7_VAL);   v7v["Период"] = pd.PeriodIndex(v7v["Период"].astype(str), freq="M")
    v7t = pd.read_csv(V7_TEST);  v7t["Период"] = pd.PeriodIndex(v7t["Период"].astype(str), freq="M")
    v71v = pd.read_csv(V71_VAL); v71v["Период"] = pd.PeriodIndex(v71v["Период"].astype(str), freq="M")
    v71t = pd.read_csv(V71_TEST); v71t["Период"] = pd.PeriodIndex(v71t["Период"].astype(str), freq="M")
    key = ["Период", "Партнер", "Артикул"]
    val = v71v.rename(columns={"prediction": "p_v71"}).merge(
        v7v.rename(columns={"prediction": "p_v7"})[[*key, "p_v7"]],
        on=key, how="left",
    )
    test = v71t.rename(columns={"prediction": "p_v71"}).merge(
        v7t.rename(columns={"prediction": "p_v7"})[[*key, "p_v7"]],
        on=key, how="left",
    )
    abt = pd.read_parquet(ABT)[["Период", "Партнер", "Артикул", "Канал"]]
    abt["Период"] = pd.PeriodIndex(abt["Период"].astype(str), freq="M")
    val = val.merge(abt, on=key, how="left")
    test = test.merge(abt, on=key, how="left")
    return val, test


# ── Panels ─────────────────────────────────────────────────────────────────

def _p_monthly(ax, val, test) -> None:
    df = pd.concat([val.assign(split="val"), test.assign(split="test")], ignore_index=True)
    agg = df.groupby(["Период", "split"]).agg(
        actual=("target_qty", "sum"),
        v7=("p_v7", "sum"),
        v71=("p_v71", "sum"),
    ).reset_index()
    agg["t"] = agg["Период"].dt.to_timestamp()
    agg = agg.sort_values("t")
    ax.plot(agg["t"], agg["actual"], "-o", color="black", lw=2, label="Actual")
    ax.plot(agg["t"], agg["v7"], "--s", color="#d62728", lw=1.5, alpha=0.8, label="V7")
    ax.plot(agg["t"], agg["v71"], "-^", color="#2ca02c", lw=1.8, label="V7.1")
    sep = test["Период"].min().to_timestamp()
    ax.axvline(sep, color="grey", linestyle=":", alpha=0.6)
    ax.set_title("Monthly totals — actual vs V7 vs V7.1 (dotted line = test split)")
    ax.set_ylabel("Units")
    ax.legend(loc="upper right", fontsize=9)


def _p_scatter(ax, test) -> None:
    y = np.clip(test["target_qty"].to_numpy(), 1e-3, None)
    p7 = np.clip(test["p_v7"].to_numpy(), 1e-3, None)
    p71 = np.clip(test["p_v71"].to_numpy(), 1e-3, None)
    ax.loglog(y, p7, "o", ms=2, alpha=0.25, color="#d62728", label="V7")
    ax.loglog(y, p71, "o", ms=2, alpha=0.25, color="#2ca02c", label="V7.1")
    lim = max(y.max(), p7.max(), p71.max())
    ax.plot([1e-2, lim], [1e-2, lim], "k--", alpha=0.5)
    ax.set_xlabel("Actual (log)"); ax.set_ylabel("Predicted (log)")
    ax.set_title("Predicted vs actual — test (log-log)")
    ax.legend(loc="upper left", fontsize=9)


def _p_residual_by_month(ax, test) -> None:
    res = test["p_v71"] - test["target_qty"]
    per = test["Период"].astype(str).to_numpy()
    cats = sorted(set(per))
    data = [res[per == c].to_numpy() for c in cats]
    bp = ax.boxplot(data, labels=cats, showfliers=False, patch_artist=True,
                    medianprops={"color": "black"})
    for patch in bp["boxes"]:
        patch.set_facecolor("#2ca02c"); patch.set_alpha(0.5)
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Residual (pred − actual)")
    ax.set_title("V7.1 residuals by test month (green = V7.1)")
    ax.set_ylim(-5, 5)


def _p_channel_wape(ax, test) -> None:
    rows = []
    for ch, g in test.groupby("Канал", observed=True):
        m7 = compute_all_metrics(g["target_qty"].to_numpy(), g["p_v7"].to_numpy())
        m71 = compute_all_metrics(g["target_qty"].to_numpy(), g["p_v71"].to_numpy())
        rows.append({"channel": str(ch), "v7": m7["WAPE"], "v71": m71["WAPE"], "n": len(g)})
    df = pd.DataFrame(rows).sort_values("n", ascending=False)
    x = np.arange(len(df)); w = 0.38
    ax.bar(x - w/2, df["v7"],  w, color="#d62728", label="V7")
    ax.bar(x + w/2, df["v71"], w, color="#2ca02c", label="V7.1")
    ax.set_xticks(x); ax.set_xticklabels(df["channel"])
    for xi, (v7, v71) in enumerate(zip(df["v7"], df["v71"])):
        ax.text(xi - w/2, v7 + 0.01, f"{v7:.3f}", ha="center", fontsize=8)
        ax.text(xi + w/2, v71 + 0.01, f"{v71:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_ylabel("WAPE"); ax.set_title("WAPE by channel — V7 vs V7.1 (lower is better)")
    ax.legend(loc="upper right", fontsize=9)


def _p_cost_waterfall(ax) -> None:
    cands = [
        ("V6", OUT / "cost_scorecard_final.json", "V6"),
        ("V7", OUT / "cost_scorecard_final.json", "V7"),
        ("V7.1", OUT / "cost_scorecard_v71_channels.json", "V7"),
    ]
    vals = []
    labels = []
    for label, p, row in cands:
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        r = next((m for m in d["models"] if m["model"] == row), None)
        if r is None:
            continue
        vals.append(r["total_cost_UAH"] / 1e6); labels.append(label)
    colors = ["#2ca02c", "#d62728", "#1f77b4"][:len(vals)]
    x = np.arange(len(vals))
    ax.bar(x, vals, color=colors, alpha=0.9)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.03, f"{v:.2f}M", ha="center", fontsize=10, fontweight="bold")
    if len(vals) >= 2:
        delta = vals[-1] - vals[0]
        ax.set_title(
            f"Annual UAH cost (M) — V6 → V7.1  Δ={delta:+.2f}M "
            f"({delta / vals[0] * 100:+.1f}% vs V6)"
        )
    else:
        ax.set_title("Annual UAH cost (M)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("UAH (M/year)")


# ── Stand-alone plots ─────────────────────────────────────────────────────

def _plot_recency_sweep() -> None:
    ab_path = OUT / "v71_ablation.csv"
    if not ab_path.exists():
        return
    df = pd.read_csv(ab_path)
    df = df[df["variant"].str.startswith("v7_rec") & ~df["variant"].str.contains("em|mono")]
    if df.empty:
        return
    df["gamma"] = df["variant"].str.extract(r"rec(\d+)").astype(float) / 100
    # Include the baseline (γ=1.0 effectively)
    base = pd.read_csv(ab_path)
    base = base[base["variant"] == "v7_base"]
    if not base.empty:
        row = base.iloc[0].to_dict()
        row["gamma"] = 1.0
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.sort_values("gamma")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(df["gamma"], df["test_WAPE"], "o-", color="#1f77b4", label="Test WAPE")
    ax2.plot(df["gamma"], df["UAH_cost"] / 1e6, "s--", color="#d62728", label="UAH cost (M)")
    ax1.set_xlabel("Recency decay γ (lower = more aggressive)")
    ax1.set_ylabel("Test WAPE", color="#1f77b4")
    ax2.set_ylabel("Annual UAH cost (M)", color="#d62728")
    ax1.set_title("V7.1 — recency γ sweep")
    fig.tight_layout()
    fig.savefig(OUT / "plot_v71_recency_sweep.png", dpi=140)
    plt.close(fig)


def _plot_channel_blend() -> None:
    p = OUT / "v71_channel_blend_sweep.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["w_spec"], df["total_UAH"] / 1e6, "o-", color="#2ca02c", lw=2)
    best = df.loc[df["total_UAH"].idxmin()]
    ax.axvline(best["w_spec"], color="black", linestyle=":", alpha=0.5)
    ax.scatter([best["w_spec"]], [best["total_UAH"] / 1e6], s=200,
               facecolors="none", edgecolors="black", linewidth=2, zorder=5)
    ax.annotate(f"best w={best['w_spec']:.2f}\n{best['total_UAH']/1e6:.2f}M UAH",
                xy=(best["w_spec"], best["total_UAH"] / 1e6),
                xytext=(12, -28), textcoords="offset points",
                fontsize=10, fontweight="bold")
    ax.set_xlabel("Specialist blend weight w")
    ax.set_ylabel("Annual UAH cost (M)")
    ax.set_title("V7.1 channel-specialist blend sweep (0 = global only, 1 = specialist only)")
    fig.tight_layout()
    fig.savefig(OUT / "plot_v71_channel_blend.png", dpi=140)
    plt.close(fig)


def _plot_stability() -> None:
    p = OUT / "v71_per_month_stability.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df["period"], df["WAPE"], "o-", color="#2ca02c", lw=2, ms=8)
    ax.axhline(df["WAPE"].mean(), color="black", linestyle="--", alpha=0.5,
               label=f"mean={df['WAPE'].mean():.4f}")
    ax.fill_between(df["period"],
                    df["WAPE"].mean() - df["WAPE"].std(),
                    df["WAPE"].mean() + df["WAPE"].std(),
                    color="#2ca02c", alpha=0.15, label=f"±1σ ({df['WAPE'].std():.4f})")
    ax.set_ylabel("WAPE"); ax.set_title("V7.1 per-test-month WAPE (stability check)")
    plt.xticks(rotation=30, ha="right")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "plot_v71_stability.png", dpi=140)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    val, test = _load()
    m_v71_val = compute_all_metrics(val["target_qty"].to_numpy(), val["p_v71"].to_numpy())
    m_v71_test = compute_all_metrics(test["target_qty"].to_numpy(), test["p_v71"].to_numpy())

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.25)
    fig.suptitle(
        f"V7.1 demand-forecasting dashboard  |  "
        f"val WAPE={m_v71_val['WAPE']:.3f}  MAPE_nz={m_v71_val['MAPE_nz']:.3f}  |  "
        f"test WAPE={m_v71_test['WAPE']:.3f}  MAPE_nz={m_v71_test['MAPE_nz']:.3f}",
        fontsize=13, y=0.995,
    )
    _p_monthly(fig.add_subplot(gs[0, :]), val, test)
    _p_scatter(fig.add_subplot(gs[1, 0]), test)
    _p_residual_by_month(fig.add_subplot(gs[1, 1]), test)
    _p_channel_wape(fig.add_subplot(gs[2, 0]), test)
    _p_cost_waterfall(fig.add_subplot(gs[2, 1]))
    fig.savefig(OUT / "plot_v71_dashboard.png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    log.info("→ plot_v71_dashboard.png")

    _plot_recency_sweep();    log.info("→ plot_v71_recency_sweep.png")
    _plot_channel_blend();    log.info("→ plot_v71_channel_blend.png")
    _plot_stability();        log.info("→ plot_v71_stability.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
