"""V10 6-panel dashboard.

Compares V9 (previous champion) and V10 (new champion) on test set.
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
    v9 = _load("v9_lad", "test").rename(columns={"prediction": "p9"})
    v10 = _load("v10_lad", "test").rename(columns={"prediction": "p10"})
    df = v9.merge(v10.drop(columns="target_qty"), on=KEY)
    df["resid9"] = df["target_qty"] - df["p9"]
    df["resid10"] = df["target_qty"] - df["p10"]
    abt = pd.read_parquet(OUT / "abt_v7_cached.parquet")[
        ["Партнер", "Артикул", "Канал", "Сегмент_ABC", "Бренд"]
    ].drop_duplicates(subset=["Партнер", "Артикул"])
    for c in ("Канал", "Сегмент_ABC", "Бренд"):
        if isinstance(abt[c].dtype, pd.CategoricalDtype):
            abt[c] = abt[c].astype(str)
    df = df.merge(abt, on=["Партнер", "Артикул"], how="left")

    fig = plt.figure(figsize=(17, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.32)

    # Panel 1: scatter actuals vs preds
    ax1 = fig.add_subplot(gs[0, 0])
    s9 = df.sample(min(8000, len(df)), random_state=42)
    ax1.scatter(s9["target_qty"], s9["p9"], s=4, alpha=0.3, color="#888888",
                label="V9")
    ax1.scatter(s9["target_qty"], s9["p10"], s=4, alpha=0.3, color="#1f77b4",
                label="V10")
    mx = max(s9["target_qty"].max(), s9["p9"].max(), s9["p10"].max()) * 1.05
    ax1.plot([0, mx], [0, mx], color="red", linestyle="--", linewidth=1)
    ax1.set_xlim(0, mx); ax1.set_ylim(0, mx)
    ax1.set_xlabel("Actual qty"); ax1.set_ylabel("Prediction")
    ax1.set_title("Predicted vs actual (test, sample 8k)")
    ax1.legend()
    ax1.grid(alpha=0.25)

    # Panel 2: residual distributions
    ax2 = fig.add_subplot(gs[0, 1])
    bins = np.linspace(-30, 30, 61)
    ax2.hist(df["resid9"].clip(-30, 30), bins=bins, alpha=0.5, color="#888888",
             label=f"V9 (μ={df['resid9'].mean():+.2f}, σ={df['resid9'].std():.2f})")
    ax2.hist(df["resid10"].clip(-30, 30), bins=bins, alpha=0.5, color="#1f77b4",
             label=f"V10 (μ={df['resid10'].mean():+.2f}, σ={df['resid10'].std():.2f})")
    ax2.axvline(0, color="black", linewidth=1)
    ax2.set_xlabel("Residual = y − ŷ (clipped to [−30,+30])")
    ax2.set_ylabel("Row count")
    ax2.set_title("Residual distributions")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)

    # Panel 3: monthly bias
    ax3 = fig.add_subplot(gs[0, 2])
    by_m = df.groupby("Период").agg(act=("target_qty", "sum"),
                                     p9=("p9", "sum"),
                                     p10=("p10", "sum"))
    by_m["b9"] = (by_m["p9"] / by_m["act"] - 1) * 100
    by_m["b10"] = (by_m["p10"] / by_m["act"] - 1) * 100
    x = np.arange(len(by_m))
    w = 0.35
    ax3.bar(x - w / 2, by_m["b9"], w, color="#888888", label="V9")
    ax3.bar(x + w / 2, by_m["b10"], w, color="#1f77b4", label="V10")
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_xticks(x); ax3.set_xticklabels(by_m.index, rotation=45, ha="right",
                                            fontsize=8)
    ax3.set_ylabel("Bias %"); ax3.set_title("Per-month bias % (test)")
    ax3.legend(); ax3.grid(alpha=0.25, axis="y")

    # Panel 4: per-channel WAPE
    ax4 = fig.add_subplot(gs[1, 0])
    cw = df.groupby("Канал", observed=True).apply(
        lambda g: pd.Series({
            "WAPE_v9": np.abs(g["resid9"]).sum() / max(g["target_qty"].sum(), 1),
            "WAPE_v10": np.abs(g["resid10"]).sum() / max(g["target_qty"].sum(), 1),
        }), include_groups=False,
    )
    x = np.arange(len(cw))
    ax4.bar(x - w / 2, cw["WAPE_v9"], w, color="#888888", label="V9")
    ax4.bar(x + w / 2, cw["WAPE_v10"], w, color="#1f77b4", label="V10")
    ax4.set_xticks(x); ax4.set_xticklabels(cw.index, fontsize=9)
    ax4.set_ylabel("WAPE"); ax4.set_title("Per-channel WAPE (test)")
    ax4.legend(); ax4.grid(alpha=0.25, axis="y")

    # Panel 5: per-ABC WAPE
    ax5 = fig.add_subplot(gs[1, 1])
    aw = df.groupby("Сегмент_ABC", observed=True).apply(
        lambda g: pd.Series({
            "WAPE_v9": np.abs(g["resid9"]).sum() / max(g["target_qty"].sum(), 1),
            "WAPE_v10": np.abs(g["resid10"]).sum() / max(g["target_qty"].sum(), 1),
        }), include_groups=False,
    )
    x = np.arange(len(aw))
    ax5.bar(x - w / 2, aw["WAPE_v9"], w, color="#888888", label="V9")
    ax5.bar(x + w / 2, aw["WAPE_v10"], w, color="#1f77b4", label="V10")
    ax5.set_xticks(x); ax5.set_xticklabels(aw.index, fontsize=9)
    ax5.set_ylabel("WAPE"); ax5.set_title("Per-ABC-segment WAPE (test)")
    ax5.legend(); ax5.grid(alpha=0.25, axis="y")

    # Panel 6: per-brand WAPE
    ax6 = fig.add_subplot(gs[1, 2])
    bw = df.groupby("Бренд", observed=True).apply(
        lambda g: pd.Series({
            "WAPE_v9": np.abs(g["resid9"]).sum() / max(g["target_qty"].sum(), 1),
            "WAPE_v10": np.abs(g["resid10"]).sum() / max(g["target_qty"].sum(), 1),
        }), include_groups=False,
    )
    x = np.arange(len(bw))
    ax6.bar(x - w / 2, bw["WAPE_v9"], w, color="#888888", label="V9")
    ax6.bar(x + w / 2, bw["WAPE_v10"], w, color="#1f77b4", label="V10")
    ax6.set_xticks(x)
    ax6.set_xticklabels(bw.index, fontsize=9, rotation=20, ha="right")
    ax6.set_ylabel("WAPE"); ax6.set_title("Per-brand WAPE (test)")
    ax6.legend(); ax6.grid(alpha=0.25, axis="y")

    # Bottom row: V10 components performance
    ax_bot = fig.add_subplot(gs[2, :])
    bases_in_pool = ["v9", "v9_recent", "v9_weekly", "v10", "v10_recent",
                     "v10_em", "v10_topdown", "v10_self_weekly", "v10_mint",
                     "v10_zero_shot", "v8", "v77_recent"]
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
    bdf["isV10"] = bdf["base"].str.startswith("v10")
    colors = ["#1f77b4" if v else "#888888" for v in bdf["isV10"]]
    x = np.arange(len(bdf))
    ax_bot.bar(x, bdf["WAPE"], color=colors, alpha=0.8, edgecolor="black",
               linewidth=0.5)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(bdf["base"].tolist(), fontsize=9, rotation=20,
                           ha="right")
    ax_bot.axhline(bdf[bdf["base"] == "v9"]["WAPE"].iloc[0],
                   linestyle="--", color="green", linewidth=1, label="V9 WAPE")
    ax_bot.set_ylabel("Test WAPE")
    ax_bot.set_title("Individual base-model performance on test (V10 bases blue, V9- bases grey)")
    ax_bot.legend(); ax_bot.grid(alpha=0.25, axis="y")
    for xi, (b, w_v, _) in enumerate(zip(bdf["base"], bdf["WAPE"], bdf["bias_pct"])):
        ax_bot.text(xi, w_v + 0.01, f"{w_v:.3f}", ha="center", fontsize=8)

    fig.suptitle(
        "V10 Dashboard - V9 (previous champion) vs V10 (new champion) on TEST\n"
        "Test SIMSCORE 0.4557 → 0.4690 (+2.9%)  |  WAPE 0.4150 → 0.4013 (-3.3%)  |  "
        "Monthly-WAPE 0.0790 → 0.0845 (+7.0%)  |  Bias %  +0.25 → +5.09",
        fontsize=12, weight="bold", y=1.0,
    )
    fig.tight_layout()
    out = OUT / "plot_v10_dashboard.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
