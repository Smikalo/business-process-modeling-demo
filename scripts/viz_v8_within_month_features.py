"""V8 within-month feature importance - dedicated visualisation.

Three-panel figure showing where the within-month features sit in the
overall V8 importance ranking, plus a comparison panel that contrasts V7
and V8 base-model performance per month.

Panels:
  1.  Top-30 V8 features by total gain (with within-month features
      highlighted in green).
  2.  Within-month features specifically, with rank labels.
  3.  Per-month test-set residual:  V7 base (without within-month features)
      vs V8 base (with).  Quantifies where the new features actually pay
      off.

Writes ``output/plot_v8_within_month_features.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "output"
KEY = ["Период", "Партнер", "Артикул"]


def main() -> int:
    fi = pd.read_csv(OUT / "feature_importance_v8.csv")
    fi = fi.sort_values("gain_total", ascending=False).reset_index(drop=True)
    fi["rank"] = fi.index + 1
    fi["is_wm"] = fi["feature"].str.startswith("wm_")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0],
                          hspace=0.40, wspace=0.30)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_wm = fig.add_subplot(gs[0, 1])
    ax_month = fig.add_subplot(gs[1, :])

    top30 = fi.head(30).iloc[::-1]
    colours = ["#2ca02c" if w else "#888888" for w in top30["is_wm"]]
    ax_top.barh(np.arange(len(top30)), top30["gain_total"],
                color=colours, edgecolor="white")
    ax_top.set_yticks(np.arange(len(top30)))
    ax_top.set_yticklabels(top30["feature"], fontsize=8)
    ax_top.set_xlabel("Total gain (LightGBM importance)")
    ax_top.set_title(
        f"V8 top-30 features by gain  |  {top30['is_wm'].sum()} are within-month "
        "(green) - never used by any prior V"
    )
    ax_top.grid(alpha=0.25, axis="x")

    wm = fi[fi["is_wm"]].copy().sort_values("gain_total", ascending=True)
    ax_wm.barh(np.arange(len(wm)), wm["gain_total"],
               color="#2ca02c", edgecolor="white")
    ax_wm.set_yticks(np.arange(len(wm)))
    labels = [
        f"{r}  ·  {f}"
        for r, f in zip(wm["rank"], wm["feature"])
    ]
    ax_wm.set_yticklabels(labels, fontsize=8)
    ax_wm.set_xlabel("Total gain (LightGBM importance)")
    ax_wm.set_title(
        f"All within-month features ({len(wm)})\n"
        f"Median rank: {int(wm['rank'].median())}  "
        f"|  Top-50 count: {(wm['rank'] <= 50).sum()}/{len(wm)}"
    )
    ax_wm.grid(alpha=0.25, axis="x")

    p7 = pd.read_csv(OUT / "preds_v7_test.csv")[KEY + ["target_qty",
                                                       "prediction"]]
    p7 = p7.rename(columns={"prediction": "p7"})
    p8 = pd.read_csv(OUT / "preds_v8_test.csv")[KEY + ["prediction"]]
    p8 = p8.rename(columns={"prediction": "p8"})
    df = p7.merge(p8, on=KEY, how="inner")
    df["err7"] = df["p7"] - df["target_qty"]
    df["err8"] = df["p8"] - df["target_qty"]
    by_m = df.groupby("Период").agg(
        y=("target_qty", "sum"),
        ae7=("err7", lambda r: r.abs().sum()),
        ae8=("err8", lambda r: r.abs().sum()),
    )
    by_m["wape7"] = by_m["ae7"] / by_m["y"]
    by_m["wape8"] = by_m["ae8"] / by_m["y"]
    by_m["delta"] = by_m["wape8"] - by_m["wape7"]
    x = np.arange(len(by_m))
    ax_month.bar(x - 0.2, by_m["wape7"], width=0.4,
                 color="#888888", label="V7 base (no within-month)",
                 alpha=0.85)
    ax_month.bar(x + 0.2, by_m["wape8"], width=0.4,
                 color="#2ca02c", label="V8 base (within-month)",
                 alpha=0.85)
    for i, d in enumerate(by_m["delta"]):
        col = "darkgreen" if d < 0 else "darkred"
        ax_month.text(i, max(by_m["wape7"].iloc[i], by_m["wape8"].iloc[i])
                      + 0.012, f"{d * 100:+.1f}pp", ha="center",
                      color=col, fontsize=8, weight="bold")
    ax_month.set_xticks(x)
    ax_month.set_xticklabels([str(p) for p in by_m.index],
                             rotation=30, ha="right")
    ax_month.set_ylabel("Per-month WAPE on held-out test set")
    ax_month.set_title(
        "V7 base vs V8 base on test - per month\n"
        "(green delta = V8 helps, red = V7 base was already enough)"
    )
    ax_month.legend()
    ax_month.grid(alpha=0.25, axis="y")

    fig.suptitle(
        "V8 within-month features - feature-importance + per-month impact\n"
        "23 features extracted from raw daily shipment data; never used by V1-V7.8",
        fontsize=13, weight="bold", y=0.995,
    )
    path = OUT / "plot_v8_within_month_features.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
