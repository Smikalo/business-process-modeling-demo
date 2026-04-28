"""V10 vs V11 head-to-head timeline with per-month squared residuals.

Three vertically-stacked panels share the month axis:

  Panel 1 (top, tallest)
    Monthly portfolio forecast vs actual demand for V10 (previous
    champion) and V11 (new champion).  Validation and test windows
    shaded.  Per-test-month bias % annotated for both.

  Panel 2 (middle)
    Per-month RMSE = √ mean((y − ŷ)² per row).  Penalises large
    row-level errors.

  Panel 3 (bottom)
    Per-month total squared-error delta: SSE(V11) − SSE(V10).  Bars
    below zero mean V11 wins; above zero means V10 wins.

Writes ``output/plot_v11_vs_v10_timeline.png`` and
``output/v11_vs_v10_timeline.csv``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.score_similarity import score_frame

OUT = Path(__file__).resolve().parent.parent / "output"
KEY = ["Период", "Партнер", "Артикул"]


def _load(tag: str) -> pd.DataFrame:
    parts = []
    for split in ("val", "test"):
        df = pd.read_csv(OUT / f"preds_{tag}_{split}.csv")[
            KEY + ["target_qty", "prediction"]
        ]
        df["split"] = split
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df["Период"] = pd.PeriodIndex(df["Период"].astype(str), freq="M").to_timestamp()
    return df


def main() -> int:
    d10 = _load("v10_lad").rename(columns={"prediction": "p10"})
    d11 = _load("v11_final").rename(columns={"prediction": "p11"})
    df = d10.merge(
        d11.drop(columns=["target_qty", "split"]),
        on=KEY, how="inner",
    )
    df["sq10"] = (df["target_qty"] - df["p10"]) ** 2
    df["sq11"] = (df["target_qty"] - df["p11"]) ** 2

    by_m = df.groupby("Период").agg(
        actual=("target_qty", "sum"),
        v10=("p10", "sum"),
        v11=("p11", "sum"),
        rmse_v10=("sq10", lambda s: float(np.sqrt(s.mean()))),
        rmse_v11=("sq11", lambda s: float(np.sqrt(s.mean()))),
        sse_v10=("sq10", "sum"),
        sse_v11=("sq11", "sum"),
        rows=("target_qty", "size"),
    )
    by_m["bias_v10_pct"] = (by_m["v10"] / by_m["actual"] - 1) * 100
    by_m["bias_v11_pct"] = (by_m["v11"] / by_m["actual"] - 1) * 100
    by_m["sse_delta"] = by_m["sse_v11"] - by_m["sse_v10"]
    by_m["sse_delta_pct"] = by_m["sse_delta"] / by_m["sse_v10"] * 100

    val_end = pd.Timestamp("2025-07-01")

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 13),
        gridspec_kw={"height_ratios": [1.5, 1.0, 0.8], "hspace": 0.32},
        sharex=True,
    )
    ax_top, ax_rmse, ax_delta = axes

    ax_top.plot(by_m.index, by_m["actual"], color="black", linewidth=2.6,
                marker="o", markersize=8, label="Actual demand", zorder=5)
    ax_top.plot(by_m.index, by_m["v10"], color="#888888", linewidth=2.0,
                marker="s", markersize=7, label="V10 (previous champion)",
                alpha=0.95)
    ax_top.plot(by_m.index, by_m["v11"], color="#2ca02c", linewidth=2.4,
                marker="^", markersize=8, label="V11 (new champion)",
                alpha=0.95)

    ymax = float(max(by_m["actual"].max(),
                     by_m["v10"].max(), by_m["v11"].max())) * 1.12
    ax_top.axvspan(by_m.index.min(), val_end, alpha=0.06, color="steelblue")
    ax_top.axvspan(val_end, by_m.index.max(), alpha=0.10, color="tomato")
    ax_top.text(
        by_m.index.min() + (val_end - by_m.index.min()) / 2,
        ymax * 0.97, "VALIDATION (12 months)",
        ha="center", va="top", fontsize=10, color="steelblue", weight="bold",
    )
    ax_top.text(
        val_end + (by_m.index.max() - val_end) / 2,
        ymax * 0.97, "TEST / HOLD-OUT (7 months)",
        ha="center", va="top", fontsize=10, color="tomato", weight="bold",
    )

    test_mask = by_m.index >= val_end
    for ts in by_m.index[test_mask]:
        b10 = by_m.loc[ts, "bias_v10_pct"]
        b11 = by_m.loc[ts, "bias_v11_pct"]
        v10_val = by_m.loc[ts, "v10"]
        v11_val = by_m.loc[ts, "v11"]
        actual = by_m.loc[ts, "actual"]
        winner = "V11" if abs(b11) < abs(b10) else "V10"
        col = "darkgreen" if winner == "V11" else "darkred"
        anchor_y = min(v10_val, v11_val, actual)
        ax_top.annotate(
            f"V10 {b10:+.0f}%\nV11 {b11:+.0f}%",
            xy=(ts, anchor_y), xytext=(0, -28), textcoords="offset points",
            ha="center", va="top", fontsize=8, color=col,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col,
                      alpha=0.9, linewidth=0.8),
        )

    ax_top.set_ylim(0, ymax)
    ax_top.set_ylabel("Total monthly demand (units)")
    ax_top.set_title(
        "Monthly portfolio forecast vs actual demand - V10 (previous champion) vs V11 (new champion)\n"
        "Per-month bias % annotated for each test month "
        "(green box = V11 closer to zero bias; red box = V10 closer)"
    )
    ax_top.legend(loc="upper left", framealpha=0.95, fontsize=10)
    ax_top.grid(alpha=0.25)

    rmse_max = float(max(by_m["rmse_v10"].max(), by_m["rmse_v11"].max())) * 1.10
    ax_rmse.plot(by_m.index, by_m["rmse_v10"], color="#888888", linewidth=2.0,
                 marker="s", markersize=7, label="V10 RMSE", alpha=0.95)
    ax_rmse.plot(by_m.index, by_m["rmse_v11"], color="#2ca02c", linewidth=2.4,
                 marker="^", markersize=8, label="V11 RMSE", alpha=0.95)
    ax_rmse.fill_between(by_m.index, by_m["rmse_v10"], by_m["rmse_v11"],
                         where=(by_m["rmse_v11"] <= by_m["rmse_v10"]),
                         color="#2ca02c", alpha=0.15,
                         interpolate=True, label="V11 wins")
    ax_rmse.fill_between(by_m.index, by_m["rmse_v10"], by_m["rmse_v11"],
                         where=(by_m["rmse_v11"] > by_m["rmse_v10"]),
                         color="#d62728", alpha=0.15,
                         interpolate=True, label="V10 wins")
    ax_rmse.axvspan(by_m.index.min(), val_end, alpha=0.06, color="steelblue")
    ax_rmse.axvspan(val_end, by_m.index.max(), alpha=0.10, color="tomato")
    ax_rmse.set_ylim(0, rmse_max)
    ax_rmse.set_ylabel("Per-month RMSE\n(√ mean row squared residual)")
    ax_rmse.set_title(
        "Squared residuals - per-month RMSE = √ mean((y − ŷ)² per row)  |  "
        "lower = better, penalises large row-level errors"
    )
    ax_rmse.legend(loc="upper left", framealpha=0.95, fontsize=9, ncol=2)
    ax_rmse.grid(alpha=0.25)

    width = ((by_m.index[1] - by_m.index[0]) * 0.7
             if len(by_m) > 1 else pd.Timedelta(days=20))
    colors_delta = ["#2ca02c" if d <= 0 else "#d62728"
                    for d in by_m["sse_delta"]]
    ax_delta.bar(by_m.index, by_m["sse_delta"] / 1e3, width=width,
                 color=colors_delta, edgecolor="white", alpha=0.9)
    ax_delta.axhline(0, color="black", linewidth=1.0)
    ax_delta.axvspan(by_m.index.min(), val_end, alpha=0.06, color="steelblue")
    ax_delta.axvspan(val_end, by_m.index.max(), alpha=0.10, color="tomato")
    for ts, d, dp in zip(by_m.index, by_m["sse_delta"], by_m["sse_delta_pct"]):
        col = "darkgreen" if d <= 0 else "darkred"
        ax_delta.text(
            ts, d / 1e3 + (-0.1 if d > 0 else 0.05) *
            (by_m["sse_delta"].abs().max() / 1e3),
            f"{dp:+.0f}%", ha="center",
            va="bottom" if d <= 0 else "top",
            fontsize=8, color=col, weight="bold",
        )
    ax_delta.set_ylabel("ΔSSE (V11 − V10)\n[thousand sq.units]")
    ax_delta.set_title(
        "Per-month total squared-error delta:  Σ (y − ŷ_V11)²  −  Σ (y − ŷ_V10)²\n"
        "(green bar = V11 reduces total squared error; red bar = V11 increases it)"
    )
    ax_delta.grid(alpha=0.25, axis="y")
    ax_delta.set_xlabel("Month")

    sv10 = score_frame(d10[d10["split"] == "val"][KEY + ["target_qty"]].assign(
        prediction=d10[d10["split"] == "val"]["p10"]))
    st10 = score_frame(d10[d10["split"] == "test"][KEY + ["target_qty"]].assign(
        prediction=d10[d10["split"] == "test"]["p10"]))
    sv11 = score_frame(d11[d11["split"] == "val"][KEY + ["target_qty"]].assign(
        prediction=d11[d11["split"] == "val"]["p11"]))
    st11 = score_frame(d11[d11["split"] == "test"][KEY + ["target_qty"]].assign(
        prediction=d11[d11["split"] == "test"]["p11"]))

    fig.autofmt_xdate()
    fig.suptitle(
        "V10 vs V11 head-to-head on the same timeline\n"
        f"Val SIMSCORE {sv10['SIMSCORE']:.4f} → {sv11['SIMSCORE']:.4f} ({(sv11['SIMSCORE']-sv10['SIMSCORE'])/sv10['SIMSCORE']*100:+.1f}%)  |  "
        f"Test SIMSCORE {st10['SIMSCORE']:.4f} → {st11['SIMSCORE']:.4f} ({(st11['SIMSCORE']-st10['SIMSCORE'])/st10['SIMSCORE']*100:+.1f}%)  |  "
        f"Test WAPE {st10['WAPE']:.4f} → {st11['WAPE']:.4f} ({(st11['WAPE']-st10['WAPE'])/st10['WAPE']*100:+.1f}%)  |  "
        f"Test bias {st10['Agg_Bias_pct']:+.2f}% → {st11['Agg_Bias_pct']:+.2f}%",
        fontsize=12, weight="bold", y=1.0,
    )
    fig.tight_layout()

    out_path = OUT / "plot_v11_vs_v10_timeline.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")

    summary = by_m.copy()
    summary.index = summary.index.strftime("%Y-%m")
    summary.index.name = "Период"
    summary[["actual", "v10", "v11", "bias_v10_pct", "bias_v11_pct",
             "rmse_v10", "rmse_v11", "sse_v10", "sse_v11",
             "sse_delta", "sse_delta_pct", "rows"]].round(4).to_csv(
        OUT / "v11_vs_v10_timeline.csv"
    )
    print(f"wrote {OUT / 'v11_vs_v10_timeline.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
