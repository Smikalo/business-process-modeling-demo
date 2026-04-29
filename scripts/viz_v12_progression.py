"""V11 → V12 progression visualization.

Side-by-side panels showing V12_final's gain over V11_final on:
  Panel 1: Monthly portfolio forecast vs actual — V11 vs V12 lines + actual.
  Panel 2: Per-month RMSE comparison.
  Panel 3: Per-test-month SSE delta (bars).
  Panel 4: Per-channel WAPE delta — which channels V12 helped/hurt.
  Panel 5: Headline metric table as in-figure annotation.

Writes:
  output/plot_v12_vs_v11_progression.png
  output/v12_vs_v11_progression.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

from scripts.score_similarity import score_frame

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
KEY = ["Период", "Партнер", "Артикул"]


def _load(tag: str) -> pd.DataFrame:
    parts = []
    for split in ("val", "test"):
        p = OUT / f"preds_{tag}_{split}.csv"
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_csv(p)[KEY + ["target_qty", "prediction"]]
        df["split"] = split
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df["Период_ts"] = pd.PeriodIndex(df["Период"].astype(str),
                                     freq="M").to_timestamp()
    df["channel"] = df["Партнер"].astype(str).str.split("-").str[0].str.strip()
    return df


def main() -> int:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.titlepad": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    d11 = _load("v11_final").rename(columns={"prediction": "p11"})
    d12 = _load("v12_final").rename(columns={"prediction": "p12"})
    if d11.empty or d12.empty:
        raise SystemExit("V11_final or V12_final preds missing — "
                         "run training pipeline first.")

    df = d11.merge(d12.drop(columns=["target_qty", "split", "Период_ts",
                                      "channel"]), on=KEY, how="inner")
    df["sq11"] = (df["target_qty"] - df["p11"]) ** 2
    df["sq12"] = (df["target_qty"] - df["p12"]) ** 2

    by_m = df.groupby("Период_ts").agg(
        actual=("target_qty", "sum"),
        v11=("p11", "sum"),
        v12=("p12", "sum"),
        rmse_v11=("sq11", lambda s: float(np.sqrt(s.mean()))),
        rmse_v12=("sq12", lambda s: float(np.sqrt(s.mean()))),
        sse_v11=("sq11", "sum"),
        sse_v12=("sq12", "sum"),
        rows=("target_qty", "size"),
    )
    by_m["bias_v11_pct"] = (by_m["v11"] / by_m["actual"] - 1) * 100
    by_m["bias_v12_pct"] = (by_m["v12"] / by_m["actual"] - 1) * 100
    by_m["sse_delta"] = by_m["sse_v12"] - by_m["sse_v11"]
    by_m["sse_delta_pct"] = by_m["sse_delta"] / by_m["sse_v11"] * 100

    val_end = pd.Timestamp("2025-07-01")

    # Per-channel WAPE comparison
    df["abs11"] = (df["target_qty"] - df["p11"]).abs()
    df["abs12"] = (df["target_qty"] - df["p12"]).abs()
    by_ch = df.groupby("channel").agg(
        actual=("target_qty", "sum"),
        abs11=("abs11", "sum"),
        abs12=("abs12", "sum"),
        n=("target_qty", "size"),
    )
    by_ch["wape_v11"] = by_ch["abs11"] / by_ch["actual"]
    by_ch["wape_v12"] = by_ch["abs12"] / by_ch["actual"]
    by_ch["wape_delta_pct"] = ((by_ch["wape_v12"] - by_ch["wape_v11"])
                                / by_ch["wape_v11"] * 100)
    by_ch = by_ch.sort_values("actual", ascending=False).head(10)

    fig = plt.figure(figsize=(18, 12), facecolor="white")
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.4, 1.0, 1.0],
                  hspace=0.42, wspace=0.28)

    # Panel 1 — monthly timeline
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(by_m.index, by_m["actual"], "ko-", linewidth=2.6, markersize=10,
             label="Actual demand", zorder=5)
    ax1.plot(by_m.index, by_m["v11"], "s-", color="#888888", linewidth=2.0,
             markersize=8, label="V11_final (previous champion)", alpha=0.95)
    ax1.plot(by_m.index, by_m["v12"], "^-", color="#2a9d8f", linewidth=2.4,
             markersize=9, label="V12_final (new champion)", alpha=0.95,
             zorder=6)

    ymax = float(max(by_m["actual"].max(),
                     by_m["v11"].max(), by_m["v12"].max())) * 1.13
    ax1.axvspan(by_m.index.min(), val_end, alpha=0.05, color="#1d3557")
    ax1.axvspan(val_end, by_m.index.max(), alpha=0.10, color="#e63946")
    ax1.text(by_m.index.min() + (val_end - by_m.index.min()) / 2,
             ymax * 0.97, "VALIDATION (12 months)",
             ha="center", fontsize=10, color="#1d3557", weight="bold")
    ax1.text(val_end + (by_m.index.max() - val_end) / 2,
             ymax * 0.97, "TEST / HOLD-OUT (7 months)",
             ha="center", fontsize=10, color="#e63946", weight="bold")

    test_mask = by_m.index >= val_end
    for ts in by_m.index[test_mask]:
        b11 = by_m.loc[ts, "bias_v11_pct"]
        b12 = by_m.loc[ts, "bias_v12_pct"]
        winner = "V12" if abs(b12) < abs(b11) else "V11"
        col = "darkgreen" if winner == "V12" else "darkred"
        anchor_y = min(by_m.loc[ts, "v11"], by_m.loc[ts, "v12"],
                       by_m.loc[ts, "actual"])
        ax1.annotate(
            f"V11 {b11:+.0f}%\nV12 {b12:+.0f}%",
            xy=(ts, anchor_y), xytext=(0, -30),
            textcoords="offset points",
            ha="center", va="top", fontsize=8, color=col,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col,
                      alpha=0.92, linewidth=0.8),
        )

    ax1.set_ylim(0, ymax)
    ax1.set_ylabel("Total monthly demand (units)")
    ax1.set_title("V11_final vs V12_final on the full 19-month timeline\n"
                   "Per-test-month bias % annotated (green box = V12 closer to zero)")
    ax1.legend(loc="upper left", framealpha=0.95, fontsize=10)
    ax1.grid(alpha=0.25)

    # Panel 2 — per-month RMSE
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(by_m.index, by_m["rmse_v11"], "s-", color="#888888", linewidth=2.0,
             markersize=7, label="V11 RMSE", alpha=0.95)
    ax2.plot(by_m.index, by_m["rmse_v12"], "^-", color="#2a9d8f", linewidth=2.4,
             markersize=8, label="V12 RMSE", alpha=0.95)
    ax2.fill_between(by_m.index, by_m["rmse_v11"], by_m["rmse_v12"],
                     where=(by_m["rmse_v12"] <= by_m["rmse_v11"]),
                     color="#2a9d8f", alpha=0.18, interpolate=True,
                     label="V12 wins")
    ax2.fill_between(by_m.index, by_m["rmse_v11"], by_m["rmse_v12"],
                     where=(by_m["rmse_v12"] > by_m["rmse_v11"]),
                     color="#e76f51", alpha=0.18, interpolate=True,
                     label="V11 wins")
    ax2.axvspan(by_m.index.min(), val_end, alpha=0.05, color="#1d3557")
    ax2.axvspan(val_end, by_m.index.max(), alpha=0.10, color="#e63946")
    ax2.set_ylabel("Per-month RMSE")
    ax2.set_title("RMSE comparison")
    ax2.legend(fontsize=8, ncol=2, loc="upper left")
    ax2.grid(alpha=0.25)

    # Panel 3 — SSE delta bars
    ax3 = fig.add_subplot(gs[1, 1])
    width = (by_m.index[1] - by_m.index[0]) * 0.7 if len(by_m) > 1 else pd.Timedelta(days=20)
    colors = ["#2a9d8f" if d <= 0 else "#e76f51" for d in by_m["sse_delta"]]
    ax3.bar(by_m.index, by_m["sse_delta"] / 1e3, width=width, color=colors,
            edgecolor="white", alpha=0.9)
    ax3.axhline(0, color="black", linewidth=1.0)
    ax3.axvspan(by_m.index.min(), val_end, alpha=0.05, color="#1d3557")
    ax3.axvspan(val_end, by_m.index.max(), alpha=0.10, color="#e63946")
    for ts, d, dp in zip(by_m.index, by_m["sse_delta"], by_m["sse_delta_pct"]):
        col = "darkgreen" if d <= 0 else "darkred"
        ax3.text(
            ts, d / 1e3 + (-0.1 if d > 0 else 0.05) *
            (by_m["sse_delta"].abs().max() / 1e3),
            f"{dp:+.0f}%", ha="center",
            va="bottom" if d <= 0 else "top",
            fontsize=7, color=col, weight="bold",
        )
    ax3.set_ylabel("ΔSSE (V12 − V11) [k sq.units]")
    ax3.set_title("Per-month total squared-error delta\n"
                   "(green = V12 reduces error; red = V12 increases)")
    ax3.grid(alpha=0.25, axis="y")

    # Panel 4 — per-channel WAPE delta
    ax4 = fig.add_subplot(gs[2, 0])
    y = np.arange(len(by_ch))
    colors_ch = ["#2a9d8f" if d <= 0 else "#e76f51"
                  for d in by_ch["wape_delta_pct"]]
    ax4.barh(y, by_ch["wape_delta_pct"], color=colors_ch,
             edgecolor="white", alpha=0.9)
    for yi, v in zip(y, by_ch["wape_delta_pct"]):
        col = "darkgreen" if v <= 0 else "darkred"
        ax4.text(v + (0.5 if v >= 0 else -0.5), yi,
                 f"{v:+.1f}%", va="center",
                 ha="left" if v >= 0 else "right",
                 fontsize=8.5, color=col, weight="bold")
    ax4.axvline(0, color="black", linewidth=1.0)
    ax4.set_yticks(y)
    ax4.set_yticklabels(by_ch.index)
    ax4.set_xlabel("WAPE % change (V12 vs V11)")
    ax4.set_title("Per-channel impact (top-10 by volume)\n"
                   "Negative = V12 improves that channel")
    ax4.grid(alpha=0.25, axis="x")

    # Panel 5 — headline metric annotation
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")

    v11_val_sc = score_frame(d11.rename(columns={"p11": "prediction"})
                              [d11["split"] == "val"][KEY + ["target_qty"]]
                              .assign(prediction=d11[d11["split"] == "val"]["p11"]))
    v11_test_sc = score_frame(d11.rename(columns={"p11": "prediction"})
                               [d11["split"] == "test"][KEY + ["target_qty"]]
                               .assign(prediction=d11[d11["split"] == "test"]["p11"]))
    v12_val_sc = score_frame(d12.rename(columns={"p12": "prediction"})
                              [d12["split"] == "val"][KEY + ["target_qty"]]
                              .assign(prediction=d12[d12["split"] == "val"]["p12"]))
    v12_test_sc = score_frame(d12.rename(columns={"p12": "prediction"})
                               [d12["split"] == "test"][KEY + ["target_qty"]]
                               .assign(prediction=d12[d12["split"] == "test"]["p12"]))

    rows = [
        ["Metric", "V11_final", "V12_final", "Δ"],
        ["Val SIMSCORE", f"{v11_val_sc['SIMSCORE']:.4f}",
         f"{v12_val_sc['SIMSCORE']:.4f}",
         f"{(v12_val_sc['SIMSCORE'] - v11_val_sc['SIMSCORE']) / v11_val_sc['SIMSCORE'] * 100:+.2f}%"],
        ["Test SIMSCORE", f"{v11_test_sc['SIMSCORE']:.4f}",
         f"{v12_test_sc['SIMSCORE']:.4f}",
         f"{(v12_test_sc['SIMSCORE'] - v11_test_sc['SIMSCORE']) / v11_test_sc['SIMSCORE'] * 100:+.2f}%"],
        ["Test WAPE", f"{v11_test_sc['WAPE']:.4f}",
         f"{v12_test_sc['WAPE']:.4f}",
         f"{(v12_test_sc['WAPE'] - v11_test_sc['WAPE']) / v11_test_sc['WAPE'] * 100:+.2f}%"],
        ["|Test bias %|", f"{abs(v11_test_sc['Agg_Bias_pct']):.2f}%",
         f"{abs(v12_test_sc['Agg_Bias_pct']):.2f}%",
         f"{abs(v12_test_sc['Agg_Bias_pct']) - abs(v11_test_sc['Agg_Bias_pct']):+.2f} pp"],
    ]
    table = ax5.table(cellText=rows[1:], colLabels=rows[0],
                       loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.05, 1.7)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#264653")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f8f9fa" if r % 2 == 0 else "white")
        if c == 3 and r > 0:
            t = cell.get_text().get_text()
            if "-" in t and "%" in t:
                cell.set_text_props(color="darkgreen", weight="bold")
            elif "+" in t and "%" in t:
                cell.set_text_props(color="darkred", weight="bold")
    ax5.set_title("Headline metric delta — V11 → V12", pad=14)

    fig.suptitle(
        f"V11 → V12 progression  |  Test SIMSCORE "
        f"{v11_test_sc['SIMSCORE']:.4f} → {v12_test_sc['SIMSCORE']:.4f}  "
        f"({(v12_test_sc['SIMSCORE'] - v11_test_sc['SIMSCORE']) / v11_test_sc['SIMSCORE'] * 100:+.2f} %)",
        fontsize=14, weight="bold", y=0.998,
    )
    fig.tight_layout()

    out_path = OUT / "plot_v12_vs_v11_progression.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")

    summary = by_m.copy()
    summary.index = summary.index.strftime("%Y-%m")
    summary.index.name = "Период"
    summary.round(4).to_csv(OUT / "v12_vs_v11_progression.csv")
    print(f"wrote {OUT / 'v12_vs_v11_progression.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
