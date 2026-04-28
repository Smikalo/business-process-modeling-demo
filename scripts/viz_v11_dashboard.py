"""V11 6-panel dashboard.

Compares V10 (previous champion) and V11 (new champion) on test set.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "output"
KEY = ["Период", "Партнер", "Артикул"]


def _load(tag, split):
    return pd.read_csv(OUT / f"preds_{tag}_{split}.csv")[
        KEY + ["target_qty", "prediction"]
    ]


def main() -> int:
    v10 = _load("v10_lad", "test").rename(columns={"prediction": "p10"})
    v11 = _load("v11_final", "test").rename(columns={"prediction": "p11"})
    df = v10.merge(v11.drop(columns="target_qty"), on=KEY)
    df["resid10"] = df["target_qty"] - df["p10"]
    df["resid11"] = df["target_qty"] - df["p11"]
    abt = pd.read_parquet(OUT / "abt_v7_cached.parquet")[
        ["Партнер", "Артикул", "Канал", "Сегмент_ABC", "Бренд"]
    ].drop_duplicates(subset=["Партнер", "Артикул"])
    for c in ("Канал", "Сегмент_ABC", "Бренд"):
        if isinstance(abt[c].dtype, pd.CategoricalDtype):
            abt[c] = abt[c].astype(str)
    df = df.merge(abt, on=["Партнер", "Артикул"], how="left")

    fig = plt.figure(figsize=(17, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    s = df.sample(min(8000, len(df)), random_state=42)
    ax1.scatter(s["target_qty"], s["p10"], s=4, alpha=0.3, color="#888888",
                label="V10")
    ax1.scatter(s["target_qty"], s["p11"], s=4, alpha=0.3, color="#2ca02c",
                label="V11")
    mx = max(s["target_qty"].max(), s["p10"].max(), s["p11"].max()) * 1.05
    ax1.plot([0, mx], [0, mx], color="red", linestyle="--", linewidth=1)
    ax1.set_xlim(0, mx); ax1.set_ylim(0, mx)
    ax1.set_xlabel("Actual qty"); ax1.set_ylabel("Prediction")
    ax1.set_title("Predicted vs actual (test, sample 8k)")
    ax1.legend()
    ax1.grid(alpha=0.25)

    ax2 = fig.add_subplot(gs[0, 1])
    bins = np.linspace(-30, 30, 61)
    ax2.hist(df["resid10"].clip(-30, 30), bins=bins, alpha=0.5, color="#888888",
             label=f"V10 (μ={df['resid10'].mean():+.2f}, σ={df['resid10'].std():.2f})")
    ax2.hist(df["resid11"].clip(-30, 30), bins=bins, alpha=0.5, color="#2ca02c",
             label=f"V11 (μ={df['resid11'].mean():+.2f}, σ={df['resid11'].std():.2f})")
    ax2.axvline(0, color="black", linewidth=1)
    ax2.set_xlabel("Residual = y − ŷ (clipped to [−30,+30])")
    ax2.set_ylabel("Row count")
    ax2.set_title("Residual distributions")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)

    ax3 = fig.add_subplot(gs[0, 2])
    by_m = df.groupby("Период").agg(act=("target_qty", "sum"),
                                     p10=("p10", "sum"),
                                     p11=("p11", "sum"))
    by_m["b10"] = (by_m["p10"] / by_m["act"] - 1) * 100
    by_m["b11"] = (by_m["p11"] / by_m["act"] - 1) * 100
    x = np.arange(len(by_m))
    w = 0.35
    ax3.bar(x - w / 2, by_m["b10"], w, color="#888888", label="V10")
    ax3.bar(x + w / 2, by_m["b11"], w, color="#2ca02c", label="V11")
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_xticks(x); ax3.set_xticklabels(by_m.index, rotation=45, ha="right",
                                            fontsize=8)
    ax3.set_ylabel("Bias %"); ax3.set_title("Per-month bias % (test)")
    ax3.legend(); ax3.grid(alpha=0.25, axis="y")

    ax4 = fig.add_subplot(gs[1, 0])
    cw = df.groupby("Канал", observed=True).apply(
        lambda g: pd.Series({
            "WAPE_v10": np.abs(g["resid10"]).sum() / max(g["target_qty"].sum(), 1),
            "WAPE_v11": np.abs(g["resid11"]).sum() / max(g["target_qty"].sum(), 1),
        }), include_groups=False,
    )
    x = np.arange(len(cw))
    ax4.bar(x - w / 2, cw["WAPE_v10"], w, color="#888888", label="V10")
    ax4.bar(x + w / 2, cw["WAPE_v11"], w, color="#2ca02c", label="V11")
    ax4.set_xticks(x); ax4.set_xticklabels(cw.index, fontsize=9)
    ax4.set_ylabel("WAPE"); ax4.set_title("Per-channel WAPE (test)")
    ax4.legend(); ax4.grid(alpha=0.25, axis="y")

    ax5 = fig.add_subplot(gs[1, 1])
    aw = df.groupby("Сегмент_ABC", observed=True).apply(
        lambda g: pd.Series({
            "WAPE_v10": np.abs(g["resid10"]).sum() / max(g["target_qty"].sum(), 1),
            "WAPE_v11": np.abs(g["resid11"]).sum() / max(g["target_qty"].sum(), 1),
        }), include_groups=False,
    )
    x = np.arange(len(aw))
    ax5.bar(x - w / 2, aw["WAPE_v10"], w, color="#888888", label="V10")
    ax5.bar(x + w / 2, aw["WAPE_v11"], w, color="#2ca02c", label="V11")
    ax5.set_xticks(x); ax5.set_xticklabels(aw.index, fontsize=9)
    ax5.set_ylabel("WAPE"); ax5.set_title("Per-ABC-segment WAPE (test)")
    ax5.legend(); ax5.grid(alpha=0.25, axis="y")

    ax6 = fig.add_subplot(gs[1, 2])
    bw = df.groupby("Бренд", observed=True).apply(
        lambda g: pd.Series({
            "WAPE_v10": np.abs(g["resid10"]).sum() / max(g["target_qty"].sum(), 1),
            "WAPE_v11": np.abs(g["resid11"]).sum() / max(g["target_qty"].sum(), 1),
        }), include_groups=False,
    )
    x = np.arange(len(bw))
    ax6.bar(x - w / 2, bw["WAPE_v10"], w, color="#888888", label="V10")
    ax6.bar(x + w / 2, bw["WAPE_v11"], w, color="#2ca02c", label="V11")
    ax6.set_xticks(x)
    ax6.set_xticklabels(bw.index, fontsize=9, rotation=20, ha="right")
    ax6.set_ylabel("WAPE"); ax6.set_title("Per-brand WAPE (test)")
    ax6.legend(); ax6.grid(alpha=0.25, axis="y")

    ax_bot = fig.add_subplot(gs[2, :])
    bases_in_pool = ["v9_lad", "v10", "v10_recent", "v10_lad",
                     "v11_recent_only", "v11_g93", "v11_g90",
                     "v11_lad", "v11_final"]
    rows = []
    for tag in bases_in_pool:
        try:
            d = _load(tag, "test")
            a = d["target_qty"].to_numpy(); p = d["prediction"].to_numpy()
            w_v = float(np.abs(a - p).sum() / max(a.sum(), 1e-6))
            b_v = float((p.sum() - a.sum()) / max(a.sum(), 1e-6) * 100)
            rows.append((tag, w_v, b_v))
        except Exception:
            continue
    bdf = pd.DataFrame(rows, columns=["base", "WAPE", "bias_pct"])
    bdf["isV11"] = bdf["base"].str.startswith("v11")
    colors = ["#2ca02c" if v else "#888888" for v in bdf["isV11"]]
    x = np.arange(len(bdf))
    ax_bot.bar(x, bdf["WAPE"], color=colors, alpha=0.8, edgecolor="black",
               linewidth=0.5)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(bdf["base"].tolist(), fontsize=9, rotation=20,
                           ha="right")
    if "v10_lad" in bdf["base"].values:
        ax_bot.axhline(bdf[bdf["base"] == "v10_lad"]["WAPE"].iloc[0],
                       linestyle="--", color="grey", linewidth=1,
                       label="V10 LAD WAPE")
    if "v11_final" in bdf["base"].values:
        ax_bot.axhline(bdf[bdf["base"] == "v11_final"]["WAPE"].iloc[0],
                       linestyle="--", color="green", linewidth=1.5,
                       label="V11 Final WAPE")
    ax_bot.set_ylabel("Test WAPE")
    ax_bot.set_title("Individual base-model performance on test "
                     "(V11-suffixed bases green, V10/earlier grey)")
    ax_bot.legend(); ax_bot.grid(alpha=0.25, axis="y")
    for xi, (b, w_v, b_pct) in enumerate(zip(bdf["base"], bdf["WAPE"],
                                              bdf["bias_pct"])):
        ax_bot.text(xi, w_v + 0.005, f"{w_v:.3f}\n({b_pct:+.0f}%)",
                    ha="center", fontsize=8)

    fig.suptitle(
        "V11 Dashboard - V10 (previous champion) vs V11 (new champion) on TEST\n"
        "Test SIMSCORE 0.4690 → 0.4489 (-4.3%)  |  WAPE 0.4013 → 0.3950 (-1.6%)  |  "
        "Bias % +5.09 → +2.80  (-45% absolute)",
        fontsize=12, weight="bold", y=1.0,
    )
    fig.tight_layout()
    out = OUT / "plot_v11_dashboard.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
