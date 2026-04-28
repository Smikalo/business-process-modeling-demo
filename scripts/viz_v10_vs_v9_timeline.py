"""V9 vs V10 head-to-head timeline with per-month squared residuals.

Same layout as `viz_v9_vs_v8_timeline.py` but comparing the new V10 LAD
champion against V9 LAD.

Three vertically-stacked panels share the month axis:

  Panel 1 (top, tallest)
    Monthly portfolio forecast vs actual demand for V9 (previous
    champion) and V10 (new champion).  Validation and test windows
    shaded.  Per-test-month bias % annotated for both.

  Panel 2 (middle)
    Per-month RMSE = √ mean((y − ŷ)² per row).  Penalises large
    row-level errors.

  Panel 3 (bottom)
    Per-month total squared-error delta: SSE(V10) − SSE(V9).  Bars
    below zero mean V10 wins; above zero means V9 wins.

Writes ``output/plot_v10_vs_v9_timeline.png`` and
``output/v10_vs_v9_timeline.csv``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    d9 = _load("v9_lad").rename(columns={"prediction": "p9"})
    d10 = _load("v10_lad").rename(columns={"prediction": "p10"})
    df = d9.merge(
        d10.drop(columns=["target_qty", "split"]),
        on=KEY, how="inner",
    )
    df["sq9"] = (df["target_qty"] - df["p9"]) ** 2
    df["sq10"] = (df["target_qty"] - df["p10"]) ** 2

    by_m = df.groupby("Период").agg(
        actual=("target_qty", "sum"),
        v9=("p9", "sum"),
        v10=("p10", "sum"),
        rmse_v9=("sq9", lambda s: float(np.sqrt(s.mean()))),
        rmse_v10=("sq10", lambda s: float(np.sqrt(s.mean()))),
        sse_v9=("sq9", "sum"),
        sse_v10=("sq10", "sum"),
        rows=("target_qty", "size"),
    )
    by_m["bias_v9_pct"] = (by_m["v9"] / by_m["actual"] - 1) * 100
    by_m["bias_v10_pct"] = (by_m["v10"] / by_m["actual"] - 1) * 100
    by_m["sse_delta"] = by_m["sse_v10"] - by_m["sse_v9"]
    by_m["sse_delta_pct"] = by_m["sse_delta"] / by_m["sse_v9"] * 100

    val_end = pd.Timestamp("2025-07-01")

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 13),
        gridspec_kw={"height_ratios": [1.5, 1.0, 0.8], "hspace": 0.32},
        sharex=True,
    )
    ax_top, ax_rmse, ax_delta = axes

    ax_top.plot(by_m.index, by_m["actual"], color="black", linewidth=2.6,
                marker="o", markersize=8, label="Actual demand", zorder=5)
    ax_top.plot(by_m.index, by_m["v9"], color="#888888", linewidth=2.0,
                marker="s", markersize=7, label="V9 (previous champion)",
                alpha=0.95)
    ax_top.plot(by_m.index, by_m["v10"], color="#1f77b4", linewidth=2.4,
                marker="^", markersize=8, label="V10 (new champion)",
                alpha=0.95)

    ymax = float(max(by_m["actual"].max(),
                     by_m["v9"].max(), by_m["v10"].max())) * 1.12
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
        b9 = by_m.loc[ts, "bias_v9_pct"]
        b10 = by_m.loc[ts, "bias_v10_pct"]
        v9_val = by_m.loc[ts, "v9"]
        v10_val = by_m.loc[ts, "v10"]
        actual = by_m.loc[ts, "actual"]
        winner = "V10" if abs(b10) < abs(b9) else "V9"
        col = "darkgreen" if winner == "V10" else "darkred"
        anchor_y = min(v9_val, v10_val, actual)
        ax_top.annotate(
            f"V9 {b9:+.0f}%\nV10 {b10:+.0f}%",
            xy=(ts, anchor_y), xytext=(0, -28), textcoords="offset points",
            ha="center", va="top", fontsize=8, color=col,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col,
                      alpha=0.9, linewidth=0.8),
        )

    ax_top.set_ylim(0, ymax)
    ax_top.set_ylabel("Total monthly demand (units)")
    ax_top.set_title(
        "Monthly portfolio forecast vs actual demand - V9 (previous champion) vs V10 (new champion)\n"
        "Per-month bias % annotated for each test month "
        "(green box = V10 closer to zero bias; red box = V9 closer)"
    )
    ax_top.legend(loc="upper left", framealpha=0.95, fontsize=10)
    ax_top.grid(alpha=0.25)

    rmse_max = float(max(by_m["rmse_v9"].max(), by_m["rmse_v10"].max())) * 1.10
    ax_rmse.plot(by_m.index, by_m["rmse_v9"], color="#888888", linewidth=2.0,
                 marker="s", markersize=7, label="V9 RMSE", alpha=0.95)
    ax_rmse.plot(by_m.index, by_m["rmse_v10"], color="#1f77b4", linewidth=2.4,
                 marker="^", markersize=8, label="V10 RMSE", alpha=0.95)
    ax_rmse.fill_between(by_m.index, by_m["rmse_v9"], by_m["rmse_v10"],
                         where=(by_m["rmse_v10"] <= by_m["rmse_v9"]),
                         color="#2ca02c", alpha=0.15,
                         interpolate=True, label="V10 wins")
    ax_rmse.fill_between(by_m.index, by_m["rmse_v9"], by_m["rmse_v10"],
                         where=(by_m["rmse_v10"] > by_m["rmse_v9"]),
                         color="#d62728", alpha=0.15,
                         interpolate=True, label="V9 wins")
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
    ax_delta.set_ylabel("ΔSSE (V10 − V9)\n[thousand sq.units]")
    ax_delta.set_title(
        "Per-month total squared-error delta:  Σ (y − ŷ_V10)²  −  Σ (y − ŷ_V9)²\n"
        "(green bar = V10 reduces total squared error; red bar = V10 increases it)"
    )
    ax_delta.grid(alpha=0.25, axis="y")
    ax_delta.set_xlabel("Month")

    fig.autofmt_xdate()
    fig.suptitle(
        "V9 vs V10 head-to-head on the same timeline\n"
        f"Val SIMSCORE 0.3642 → 0.3528 ({(0.3528-0.3642)/0.3642*100:+.1f}%)  |  "
        f"Test SIMSCORE 0.4557 → 0.4690 ({(0.4690-0.4557)/0.4557*100:+.1f}%)  |  "
        "Test WAPE 0.4150 → 0.4013 (-3.3%)  |  "
        "Test M-WAPE 0.0790 → 0.0845 (+7.0%)",
        fontsize=12, weight="bold", y=1.0,
    )
    fig.tight_layout()

    out_path = OUT / "plot_v10_vs_v9_timeline.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")

    summary = by_m.copy()
    summary.index = summary.index.strftime("%Y-%m")
    summary.index.name = "Период"
    summary[["actual", "v9", "v10", "bias_v9_pct", "bias_v10_pct",
             "rmse_v9", "rmse_v10", "sse_v9", "sse_v10",
             "sse_delta", "sse_delta_pct", "rows"]].round(4).to_csv(
        OUT / "v10_vs_v9_timeline.csv"
    )
    print(f"wrote {OUT / 'v10_vs_v9_timeline.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
