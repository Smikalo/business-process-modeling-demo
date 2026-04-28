"""V9 — V1 -> V9 progression visualisation + V9 residual heatmap.

Mirrors viz_v8_progression but adds V9 as the new champion and refreshes
all summary statistics from current artefacts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.score_similarity import score_frame

OUT = Path(__file__).resolve().parent.parent / "output"
KEY = ["Период", "Партнер", "Артикул"]
META_AXES = ["Канал", "Бренд", "Сегмент_ABC"]


MODELS = [
    ("V4",   "v4",            "#7f7f7f"),
    ("V5",   "v5",            "#1f77b4"),
    ("V6",   "v6",            "#2ca02c"),
    ("V7",   "v7",            "#ff7f0e"),
    ("V7.1", "v71_channels",  "#d62728"),
    ("V7.2", "v72_champion",  "#9467bd"),
    ("V7.3", "v73",           "#e377c2"),
    ("V7.4", "v74",           "#17becf"),
    ("V7.5", "v75",           "#bcbd22"),
    ("V7.7", "v77",           "#777777"),
    ("V7.8", "v78",           "#555555"),
    ("V8",   "v8_lad",        "#222222"),
    ("V9",   "v9_lad",        "#000000"),
]


def _load(tag: str, split: str) -> pd.DataFrame:
    return pd.read_csv(OUT / f"preds_{tag}_{split}.csv")[
        KEY + ["target_qty", "prediction"]
    ]


def _metrics(tag: str, split: str) -> dict:
    df = _load(tag, split)
    s = score_frame(df)
    by_m = df.groupby("Период").agg(y=("target_qty", "sum"),
                                    p=("prediction", "sum"))
    portfolio_wape = float(np.abs(by_m["p"] - by_m["y"]).sum() /
                           by_m["y"].sum())
    s["portfolio_wape"] = round(portfolio_wape, 4)
    return s


def _aggregate_residual_table(tag: str = "v9_lad") -> pd.DataFrame:
    abt = pd.read_parquet(OUT / "abt_v7_cached.parquet")[KEY + META_AXES]
    abt["Период"] = abt["Период"].astype(str)
    out = []
    for split in ("val", "test"):
        d = _load(tag, split)
        d["Период"] = d["Период"].astype(str)
        d = d.merge(abt, on=KEY, how="left")
        d["per_p"] = pd.PeriodIndex(d["Период"], freq="M")
        d["moy"] = d["per_p"].apply(lambda x: x.month)
        d["split"] = split
        out.append(d)
    return pd.concat(out, ignore_index=True)


def main() -> int:
    rows = []
    for label, tag, colour in MODELS:
        s = _metrics(tag, "test")
        rows.append({"label": label, "tag": tag, "colour": colour,
                     **{k: s[k] for k in ["WAPE", "SIMSCORE", "Agg_Bias_pct",
                                          "Monthly_WAPE", "RMSE", "MAE",
                                          "portfolio_wape"]}})
    summary = pd.DataFrame(rows)

    summary[["label", "tag", "WAPE", "SIMSCORE", "Agg_Bias_pct",
             "Monthly_WAPE", "RMSE", "portfolio_wape"]].to_csv(
        OUT / "v9_progression_summary.csv", index=False
    )
    print(summary[["label", "WAPE", "SIMSCORE", "Agg_Bias_pct",
                   "Monthly_WAPE", "RMSE", "portfolio_wape"]]
          .to_string(index=False))

    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1:])

    x = np.arange(len(summary))
    palette = summary["colour"].tolist()

    bars = ax1.bar(x, summary["SIMSCORE"], color=palette, edgecolor="white")
    for b, v in zip(bars, summary["SIMSCORE"]):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary["label"], rotation=35, ha="right")
    ax1.set_ylabel("Test SIMSCORE")
    ax1.set_title("Test SIMSCORE - V4 → V9 (lower is better)")
    ax1.grid(alpha=0.25, axis="y")
    ax1.set_ylim(0, summary["SIMSCORE"].max() * 1.10)
    champ_idx = int(summary["SIMSCORE"].idxmin())
    ax1.add_patch(plt.Rectangle(
        (champ_idx - 0.45, 0), 0.9, summary["SIMSCORE"].iloc[champ_idx],
        fill=False, edgecolor="green", linewidth=2.5
    ))

    bars = ax2.bar(x, summary["WAPE"], color=palette, edgecolor="white")
    for b, v in zip(bars, summary["WAPE"]):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(summary["label"], rotation=35, ha="right")
    ax2.set_ylabel("Test WAPE")
    ax2.set_title("Test WAPE - V4 → V9 (lower is better)")
    ax2.grid(alpha=0.25, axis="y")
    ax2.set_ylim(0, summary["WAPE"].max() * 1.10)
    champ_idx_w = int(summary["WAPE"].idxmin())
    ax2.add_patch(plt.Rectangle(
        (champ_idx_w - 0.45, 0), 0.9, summary["WAPE"].iloc[champ_idx_w],
        fill=False, edgecolor="green", linewidth=2.5
    ))

    bars = ax3.bar(x, summary["Agg_Bias_pct"], color=palette, edgecolor="white")
    for b, v in zip(bars, summary["Agg_Bias_pct"]):
        ax3.text(b.get_x() + b.get_width() / 2,
                 b.get_height() + (1.0 if v >= 0 else -1.5),
                 f"{v:+.1f}", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(summary["label"], rotation=35, ha="right")
    ax3.axhline(0, color="grey", linewidth=0.8)
    ax3.set_ylabel("Test aggregate bias % (signed)")
    ax3.set_title("Test bias evolution - closer to 0 is better")
    ax3.grid(alpha=0.25, axis="y")

    bars = ax4.bar(x, summary["portfolio_wape"], color=palette,
                   edgecolor="white")
    for b, v in zip(bars, summary["portfolio_wape"]):
        ax4.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(summary["label"], rotation=35, ha="right")
    ax4.set_ylabel("Portfolio-level WAPE\n(monthly totals)")
    ax4.set_title("Portfolio-level WAPE - what S&OP sees")
    ax4.grid(alpha=0.25, axis="y")
    ax4.set_ylim(0, summary["portfolio_wape"].max() * 1.10)

    base = summary["SIMSCORE"].iloc[0]
    rel_sim = (base - summary["SIMSCORE"]) / base * 100
    base_w = summary["WAPE"].iloc[0]
    rel_wape = (base_w - summary["WAPE"]) / base_w * 100
    ax5.plot(x, rel_sim, marker="o", linewidth=2, color="#2ca02c",
             markersize=8, label="SIMSCORE improvement")
    ax5.plot(x, rel_wape, marker="s", linewidth=2, color="#1f77b4",
             markersize=8, label="WAPE improvement")
    for i, (vs, vw) in enumerate(zip(rel_sim, rel_wape)):
        ax5.text(i, vs + 0.5, f"{vs:+.1f}%", ha="center", va="bottom",
                 fontsize=9, color="#2ca02c")
        ax5.text(i, vw - 1.0, f"{vw:+.1f}%", ha="center", va="top",
                 fontsize=9, color="#1f77b4")
    ax5.axhline(0, color="grey", linewidth=0.8)
    ax5.set_xticks(x)
    ax5.set_xticklabels(summary["label"], rotation=35, ha="right")
    ax5.set_ylabel("Improvement vs V4 (%)")
    ax5.set_title("Cumulative SIMSCORE / WAPE improvement over V4 baseline")
    ax5.grid(alpha=0.25)
    ax5.legend(loc="lower right")

    fig.suptitle(
        "Demand forecasting - V1 → V9 progression on the held-out test set\n"
        "V9 = V8 LAD pool + 3 NEW bases: sales-leading-indicator base + "
        "recency-weighted variant + weekly-resolution Tweedie (rolled to monthly "
        "with per-channel calibration). 15 sales-leading features unmined until V9.",
        fontsize=12, weight="bold", y=1.00,
    )

    path = OUT / "plot_v9_progression.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"wrote {path}")

    df = _aggregate_residual_table("v9_lad")
    pivot = (
        df.groupby(["Канал", "moy"], observed=True)
          .agg(y=("target_qty", "sum"), p=("prediction", "sum"))
    )
    pivot["bias_pct"] = (pivot["p"] / pivot["y"].clip(lower=1) - 1) * 100
    pivot = pivot["bias_pct"].unstack("moy")

    fig2, ax = plt.subplots(figsize=(13, 4.4))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-30, vmax=30,
                   aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Month of year")
    ax.set_ylabel("Канал (channel)")
    ax.set_title(
        "V9 residual heatmap - aggregate bias % by Канал × month-of-year\n"
        "(red = over-forecast, blue = under-forecast; val + test combined)"
    )
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.0f}%", ha="center", va="center",
                        fontsize=9,
                        color="black" if abs(v) < 18 else "white")
    fig2.colorbar(im, ax=ax, label="bias %")
    fig2.tight_layout()
    out_path = OUT / "plot_v9_residual_heatmap.png"
    fig2.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
