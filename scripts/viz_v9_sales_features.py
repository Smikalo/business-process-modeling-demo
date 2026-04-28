"""V9 sales-leading feature analysis - dedicated visualisation.

Three panels showing where the new sales-leading features sit in the
V9 importance ranking, plus a per-month comparison panel that contrasts
V8 base (no proper sales lags) vs V9 base (with sales lags).

Panels:
  1.  Top-30 V9 features by total gain (sales-leading features green).
  2.  All sales-leading features ranked by gain.
  3.  Per-month test-set residual: V8 base vs V9 base (where do sales
      features actually pay off?).

Writes ``output/plot_v9_sales_features.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "output"
KEY = ["Период", "Партнер", "Артикул"]


def main() -> int:
    fi = pd.read_csv(OUT / "feature_importance_v9.csv")
    fi = fi.sort_values("gain_total", ascending=False).reset_index(drop=True)
    fi["rank"] = fi.index + 1
    fi["is_sales"] = fi["feature"].str.startswith(("sales_", "sell_through"))
    fi["is_wm"] = fi["feature"].str.startswith("wm_")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0],
                          hspace=0.40, wspace=0.30)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_sales = fig.add_subplot(gs[0, 1])
    ax_month = fig.add_subplot(gs[1, :])

    top30 = fi.head(30).iloc[::-1]
    colours = []
    for _, r in top30.iterrows():
        if r["is_sales"]:
            colours.append("#2ca02c")
        elif r["is_wm"]:
            colours.append("#1f77b4")
        else:
            colours.append("#888888")
    ax_top.barh(np.arange(len(top30)), top30["gain_total"],
                color=colours, edgecolor="white")
    ax_top.set_yticks(np.arange(len(top30)))
    ax_top.set_yticklabels(top30["feature"], fontsize=8)
    ax_top.set_xlabel("Total gain (LightGBM)")
    n_sales_top = top30["is_sales"].sum()
    n_wm_top = top30["is_wm"].sum()
    ax_top.set_title(
        f"V9 top-30 features by gain  |  {n_sales_top} sales-leading (green) "
        f"+ {n_wm_top} within-month (blue)"
    )
    ax_top.grid(alpha=0.25, axis="x")

    sales_feats = fi[fi["is_sales"]].copy().sort_values(
        "gain_total", ascending=True,
    )
    ax_sales.barh(np.arange(len(sales_feats)), sales_feats["gain_total"],
                  color="#2ca02c", edgecolor="white")
    ax_sales.set_yticks(np.arange(len(sales_feats)))
    labels = [
        f"{r}  ·  {f}"
        for r, f in zip(sales_feats["rank"], sales_feats["feature"])
    ]
    ax_sales.set_yticklabels(labels, fontsize=8)
    ax_sales.set_xlabel("Total gain (LightGBM)")
    ax_sales.set_title(
        f"All sales-leading features ({len(sales_feats)})\n"
        f"Median rank: {int(sales_feats['rank'].median())}  "
        f"|  Top-50 count: {(sales_feats['rank'] <= 50).sum()}/{len(sales_feats)}"
    )
    ax_sales.grid(alpha=0.25, axis="x")

    p8 = pd.read_csv(OUT / "preds_v8_test.csv")[KEY + ["target_qty",
                                                       "prediction"]]
    p8 = p8.rename(columns={"prediction": "p8"})
    p9 = pd.read_csv(OUT / "preds_v9_test.csv")[KEY + ["prediction"]]
    p9 = p9.rename(columns={"prediction": "p9"})
    df = p8.merge(p9, on=KEY, how="inner")
    df["err8"] = df["p8"] - df["target_qty"]
    df["err9"] = df["p9"] - df["target_qty"]
    by_m = df.groupby("Период").agg(
        y=("target_qty", "sum"),
        ae8=("err8", lambda r: r.abs().sum()),
        ae9=("err9", lambda r: r.abs().sum()),
    )
    by_m["wape8"] = by_m["ae8"] / by_m["y"]
    by_m["wape9"] = by_m["ae9"] / by_m["y"]
    by_m["delta"] = by_m["wape9"] - by_m["wape8"]
    x = np.arange(len(by_m))
    ax_month.bar(x - 0.2, by_m["wape8"], width=0.4,
                 color="#888888", label="V8 base (no sales lags)",
                 alpha=0.85)
    ax_month.bar(x + 0.2, by_m["wape9"], width=0.4,
                 color="#2ca02c", label="V9 base (sales-leading)",
                 alpha=0.85)
    for i, d in enumerate(by_m["delta"]):
        col = "darkgreen" if d < 0 else "darkred"
        ax_month.text(i, max(by_m["wape8"].iloc[i], by_m["wape9"].iloc[i])
                      + 0.012, f"{d * 100:+.1f}pp", ha="center",
                      color=col, fontsize=8, weight="bold")
    ax_month.set_xticks(x)
    ax_month.set_xticklabels([str(p) for p in by_m.index],
                             rotation=30, ha="right")
    ax_month.set_ylabel("Per-month WAPE (test set)")
    ax_month.set_title(
        "V8 base vs V9 base on test - per month\n"
        "(green delta = V9 helps; sales-leading features pay off most "
        "where the V8 monthly model has the largest residuals)"
    )
    ax_month.legend()
    ax_month.grid(alpha=0.25, axis="y")

    fig.suptitle(
        "V9 sales-leading features - importance + per-month impact\n"
        "15 features extracted from raw monthly sales data; never properly used by V1-V8",
        fontsize=13, weight="bold", y=0.995,
    )
    path = OUT / "plot_v9_sales_features.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
