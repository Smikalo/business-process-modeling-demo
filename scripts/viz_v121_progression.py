"""Generate V12.1 progression visualisation.

5-panel figure showing:
  1. SIMSCORE progression V10 → V11 → V12 → V12.1
  2. Per-month forecast vs actual (V11_final vs V12.1_champion overlay)
  3. Per-channel SIMSCORE delta
  4. Bias trajectory across model versions
  5. Headline metric table

Outputs: output/plot_v121_progression.png
         output/v121_progression_summary.csv
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.score_similarity import score_frame

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
KEY = ["Период", "Партнер", "Артикул"]


def _load(tag, split):
    return pd.read_csv(OUT / f"preds_{tag}_{split}.csv")


def _monthly(df, label):
    df = df.copy()
    df["Период"] = pd.PeriodIndex(df["Период"].astype(str), freq="M")
    g = df.groupby("Период").agg(actual=("target_qty", "sum"),
                                   pred=("prediction", "sum"))
    g[f"err_{label}"] = (g["pred"] - g["actual"]).abs()
    g[f"bias_{label}"] = (g["pred"] - g["actual"]) / g["actual"] * 100
    return g


def main() -> int:
    progression = [
        ("V10_final",    "v10_final",    "#999999"),
        ("V11_LAD",      "v11_lad",      "#cc8866"),
        ("V11_final",    "v11_final",    "#1f77b4"),
        ("V12_LAD",      "v12_lad",      "#bbbbbb"),
        ("V12_final",    "v12_final",    "#999999"),
        ("V12.1_LAD",    "v121_lad",     "#ffaa55"),
        ("V12.1_champion", "v121_champion", "#2ca02c"),
    ]

    rows = []
    for label, tag, color in progression:
        for split in ("val", "test"):
            p = OUT / f"preds_{tag}_{split}.csv"
            if not p.exists():
                continue
            sc = score_frame(pd.read_csv(p))
            rows.append({"model": label, "split": split, "color": color, **sc})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "v121_progression_summary.csv", index=False)

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 2, 1.5], hspace=0.45,
                           wspace=0.30)

    ax1 = fig.add_subplot(gs[0, :])
    test_df = df[df["split"] == "test"].set_index("model")
    bars_x = np.arange(len(progression))
    sims = test_df["SIMSCORE"].reindex([m for m, _, _ in progression])
    colors = [c for _, _, c in progression]
    ax1.bar(bars_x, sims.values, color=colors, edgecolor="black", linewidth=0.8)
    for i, v in enumerate(sims.values):
        ax1.annotate(f"{v:.4f}", xy=(i, v), ha="center", va="bottom",
                      fontsize=10, fontweight="bold")
    ax1.set_xticks(bars_x)
    ax1.set_xticklabels(sims.index, rotation=15, ha="right")
    ax1.set_ylabel("Test SIMSCORE  (lower = better)", fontsize=12)
    ax1.set_title("V10 → V11 → V12 → V12.1 progression  "
                   "(held-out test Jul 2025 → Mar 2026)",
                   fontsize=13, fontweight="bold")
    ax1.axhline(test_df.loc["V11_final", "SIMSCORE"], color="#1f77b4",
                  linestyle=":", linewidth=1, alpha=0.6,
                  label="V11_final benchmark")
    ax1.axhline(test_df.loc["V12.1_champion", "SIMSCORE"], color="#2ca02c",
                  linestyle="-", linewidth=1.5, alpha=0.8,
                  label="V12.1_champion (new prod)")
    ax1.legend(loc="upper right")
    ax1.grid(axis="y", alpha=0.3)

    ax2 = fig.add_subplot(gs[1, :2])
    g11 = _monthly(_load("v11_final", "test"), "v11")
    g121 = _monthly(_load("v121_champion", "test"), "v121")
    actual = g11["actual"]
    months = [str(p) for p in actual.index]
    x = np.arange(len(months))
    ax2.plot(x, actual.values / 1e3, "o-", color="black", linewidth=2,
              markersize=7, label="Actual")
    ax2.plot(x, g11["pred"].values / 1e3, "s--", color="#1f77b4",
              alpha=0.7, label="V11_final")
    ax2.plot(x, g121["pred"].values / 1e3, "^-", color="#2ca02c",
              linewidth=2, label="V12.1_champion")
    ax2.set_xticks(x)
    ax2.set_xticklabels(months, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Units (thousands)", fontsize=11)
    ax2.set_title("Monthly aggregate: V11_final vs V12.1_champion vs Actual",
                    fontsize=12, fontweight="bold")
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 2])
    metrics = ["SIMSCORE", "WAPE", "Agg_Bias_pct", "Monthly_WAPE"]
    v11 = test_df.loc["V11_final"]
    v121 = test_df.loc["V12.1_champion"]
    deltas_pct = []
    for m in metrics:
        if m == "Agg_Bias_pct":
            deltas_pct.append(v121[m] - v11[m])
        else:
            deltas_pct.append((v121[m] - v11[m]) / v11[m] * 100)
    colors_d = ["#2ca02c" if d < 0 else "#d62728" for d in deltas_pct]
    bars_y = np.arange(len(metrics))
    ax3.barh(bars_y, deltas_pct, color=colors_d, edgecolor="black",
              linewidth=0.6)
    for i, d in enumerate(deltas_pct):
        unit = " pp" if metrics[i] == "Agg_Bias_pct" else " %"
        ax3.annotate(f"{d:+.2f}{unit}", xy=(d, i),
                       ha="left" if d > 0 else "right",
                       va="center", fontsize=10, fontweight="bold",
                       xytext=(3 if d > 0 else -3, 0),
                       textcoords="offset points")
    ax3.set_yticks(bars_y)
    ax3.set_yticklabels(metrics)
    ax3.axvline(0, color="black", linewidth=0.8)
    ax3.set_title("V12.1 Δ vs V11_final\n(green = improvement)",
                    fontsize=11, fontweight="bold")
    ax3.set_xlabel("Δ relative %  (Bias is Δ pp)")
    ax3.grid(axis="x", alpha=0.3)

    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")
    table_data = []
    for label, _, _ in progression:
        if label not in test_df.index:
            continue
        r = test_df.loc[label]
        table_data.append([
            label,
            f"{r['SIMSCORE']:.4f}",
            f"{r['WAPE']:.4f}",
            f"{r['Agg_Bias_pct']:+.2f}",
            f"{r['Monthly_WAPE']:.4f}",
            f"{r['RMSE']:.3f}",
        ])
    table = ax4.table(cellText=table_data,
                       colLabels=["Model", "SIMSCORE ↓", "WAPE ↓",
                                   "Bias %", "M-WAPE ↓", "RMSE"],
                       cellLoc="center", loc="center",
                       colColours=["#dddddd"] * 6)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for i, (label, _, _) in enumerate([(l, t, c) for l, t, c in progression
                                          if l in test_df.index]):
        if label == "V12.1_champion":
            for j in range(6):
                table[(i + 1, j)].set_facecolor("#d4edda")
                table[(i + 1, j)].set_text_props(weight="bold")
        elif label == "V11_final":
            for j in range(6):
                table[(i + 1, j)].set_facecolor("#cce5ff")

    fig.suptitle("V12.1 progression — V12.1_champion is the new production model",
                  fontsize=15, fontweight="bold", y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = OUT / "plot_v121_progression.png"
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
