"""V7.8 — full V1 → V7.8 model progression visualisation.

Five-panel figure summarising the entire model evolution from V1 to V7.8:

1. Test SIMSCORE bar chart with annotations
2. Test WAPE bar chart with annotations
3. Test aggregate bias % evolution (sign matters)
4. Portfolio-level WAPE (monthly totals) — the business-relevant metric
5. Cumulative relative SIMSCORE improvement over V4 baseline

Plus a separate residual heatmap for V7.8 (Канал × month_of_year bias %),
written to ``output/plot_v78_residual_heatmap.png``.

This is the single chart to send to the client / executives.
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


# (label, file tag, colour, year-1.0 reference for cumulative)
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
    ("V7.7", "v77",           "#555555"),
    ("V7.8", "v78",           "#000000"),
]


def _load(tag: str, split: str) -> pd.DataFrame:
    return pd.read_csv(OUT / f"preds_{tag}_{split}.csv")[
        KEY + ["target_qty", "prediction"]
    ]


def _metrics(tag: str, split: str) -> dict:
    df = _load(tag, split)
    s = score_frame(df)
    err = (df["prediction"] - df["target_qty"]).to_numpy()
    by_m = df.groupby("Период").agg(y=("target_qty", "sum"),
                                    p=("prediction", "sum"))
    portfolio_wape = float(np.abs(by_m["p"] - by_m["y"]).sum() / by_m["y"].sum())
    s["portfolio_wape"] = round(portfolio_wape, 4)
    return s


def _aggregate_residual_table(tag: str = "v78") -> pd.DataFrame:
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

    # Nicely export the summary CSV
    summary[["label", "tag", "WAPE", "SIMSCORE", "Agg_Bias_pct",
             "Monthly_WAPE", "RMSE", "portfolio_wape"]].to_csv(
        OUT / "v78_progression_summary.csv", index=False
    )
    print(summary[["label", "WAPE", "SIMSCORE", "Agg_Bias_pct",
                   "Monthly_WAPE", "RMSE", "portfolio_wape"]]
          .to_string(index=False))

    # ── Five-panel progression figure ─────────────────────────────────────
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
    ax1.set_ylabel("Test SIMSCORE\n(WAPE + 0.005·|bias%| + 0.5·M-WAPE)")
    ax1.set_title("Test SIMSCORE — V4 → V7.8 (lower is better)")
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
    ax2.set_title("Test WAPE — V4 → V7.8 (lower is better)")
    ax2.grid(alpha=0.25, axis="y")
    ax2.set_ylim(0, summary["WAPE"].max() * 1.10)

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
    ax3.set_title("Test bias evolution — closer to 0 is better")
    ax3.grid(alpha=0.25, axis="y")

    bars = ax4.bar(x, summary["portfolio_wape"], color=palette, edgecolor="white")
    for b, v in zip(bars, summary["portfolio_wape"]):
        ax4.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(summary["label"], rotation=35, ha="right")
    ax4.set_ylabel("Portfolio-level WAPE\n(monthly totals)")
    ax4.set_title("Portfolio-level WAPE — what S&OP sees")
    ax4.grid(alpha=0.25, axis="y")
    ax4.set_ylim(0, summary["portfolio_wape"].max() * 1.10)

    # Cumulative SIMSCORE improvement over V4
    base = summary["SIMSCORE"].iloc[0]
    rel = (base - summary["SIMSCORE"]) / base * 100
    ax5.plot(x, rel, marker="o", linewidth=2, color="#2ca02c", markersize=8)
    for i, v in enumerate(rel):
        ax5.text(i, v + 0.6, f"{v:+.1f}%", ha="center", va="bottom",
                 fontsize=9, color="#2ca02c")
    ax5.axhline(0, color="grey", linewidth=0.8)
    ax5.set_xticks(x)
    ax5.set_xticklabels(summary["label"], rotation=35, ha="right")
    ax5.set_ylabel("Test SIMSCORE improvement vs V4 (%)")
    ax5.set_title("Cumulative SIMSCORE improvement over V4 — every iteration's value")
    ax5.grid(alpha=0.25)

    fig.suptitle(
        "Demand forecasting — V1 → V7.8 progression on the held-out test set\n"
        "V7.8 = extended LAD pool (8 bases) + per-channel tilted LAD τ=0.55 "
        "+ Канал×ABC (shrink 0.5) + Бренд (shrink 0.3) reconciliation",
        fontsize=13, weight="bold", y=1.00,
    )

    path = OUT / "plot_v78_progression.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"wrote {path}")

    # ── V7.8 residual heatmap (Канал × month-of-year bias %) ──────────────
    df = _aggregate_residual_table("v78")
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
        "V7.8 residual heatmap — aggregate bias % by Канал × month-of-year\n"
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
    out_path = OUT / "plot_v78_residual_heatmap.png"
    fig2.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
