"""V9 production dashboard — V8 vs V9 LAD on the held-out test set.

Six-panel summary comparing V8 (previous champion) and V9 (new champion):

  Row 1
    1.1  Scatter actual vs V9 predicted (log-log, coloured by |residual|)
    1.2  Residual distribution V8 vs V9
    1.3  Monthly aggregate bias % V8 vs V9 (test window only)
  Row 2
    2.1  Per-channel WAPE V8 vs V9
    2.2  Per-ABC class bias % V8 vs V9
    2.3  Per-brand WAPE V8 vs V9
  Row 3 (full width)
    3.   LAD weight composition by channel (V9 stacked bars; V9 bases red)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "output"
KEY = ["Период", "Партнер", "Артикул"]


def _load(tag: str, split: str = "test") -> pd.DataFrame:
    return pd.read_csv(OUT / f"preds_{tag}_{split}.csv")[
        KEY + ["target_qty", "prediction"]
    ]


def main() -> int:
    abt = pd.read_parquet(OUT / "abt_v7_cached.parquet")[
        KEY + ["Канал", "Бренд", "Сегмент_ABC"]
    ]
    abt["Период"] = abt["Период"].astype(str)

    df8 = _load("v8_lad").rename(columns={"prediction": "p8"})
    df9 = _load("v9_lad").rename(columns={"prediction": "p9"})
    df = (
        df8.merge(df9.drop(columns=["target_qty"]), on=KEY)
           .merge(abt, on=KEY, how="left")
    )
    df["per_p"] = pd.PeriodIndex(df["Период"].astype(str), freq="M")
    df["err8"] = df["p8"] - df["target_qty"]
    df["err9"] = df["p9"] - df["target_qty"]

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.30)
    axS = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])
    axM = fig.add_subplot(gs[0, 2])
    axC = fig.add_subplot(gs[1, 0])
    axA = fig.add_subplot(gs[1, 1])
    axB = fig.add_subplot(gs[1, 2])
    axW = fig.add_subplot(gs[2, :])

    sc = axS.scatter(
        df["target_qty"] + 0.5, df["p9"] + 0.5,
        c=np.abs(df["err9"]), cmap="viridis", s=8, alpha=0.4,
    )
    lim = max(df["target_qty"].max(), df["p9"].max()) + 1
    axS.plot([0.5, lim], [0.5, lim], color="red", linestyle="--",
             linewidth=1.0, alpha=0.7)
    axS.set_xscale("log"); axS.set_yscale("log")
    axS.set_xlim(0.5, lim); axS.set_ylim(0.5, lim)
    axS.set_xlabel("Actual qty (test set)")
    axS.set_ylabel("V9 predicted qty")
    axS.set_title("Actual vs predicted (log-log)")
    axS.grid(alpha=0.25, which="both")
    fig.colorbar(sc, ax=axS, label="|residual|")

    axR.hist(df["err8"], bins=80, range=(-30, 30), alpha=0.55,
             label="V8", color="#888888")
    axR.hist(df["err9"], bins=80, range=(-30, 30), alpha=0.55,
             label="V9", color="black")
    axR.axvline(0, color="grey", linestyle="--", linewidth=1)
    axR.set_xlabel("Residual (predicted - actual)")
    axR.set_ylabel("Row count")
    axR.set_title("Residual distribution (test set)")
    axR.legend()
    axR.grid(alpha=0.25)

    by_m = df.groupby("per_p", observed=True).agg(
        y=("target_qty", "sum"),
        p8=("p8", "sum"),
        p9=("p9", "sum"),
    )
    by_m["bias8"] = (by_m["p8"] / by_m["y"] - 1) * 100
    by_m["bias9"] = (by_m["p9"] / by_m["y"] - 1) * 100
    x = np.arange(len(by_m))
    axM.bar(x - 0.2, by_m["bias8"], width=0.4, color="#888888",
            label="V8", alpha=0.85)
    axM.bar(x + 0.2, by_m["bias9"], width=0.4, color="black",
            label="V9", alpha=0.85)
    axM.axhline(0, color="grey", linewidth=0.8)
    axM.set_xticks(x)
    axM.set_xticklabels([str(p) for p in by_m.index],
                        rotation=45, ha="right")
    axM.set_ylabel("Aggregate bias % (test month)")
    axM.set_title("Monthly bias - V8 vs V9")
    axM.legend()
    axM.grid(alpha=0.25, axis="y")

    by_c = df.groupby("Канал", observed=True).agg(
        y=("target_qty", "sum"),
        ae8=("err8", lambda s: np.abs(s).sum()),
        ae9=("err9", lambda s: np.abs(s).sum()),
    )
    by_c["wape8"] = by_c["ae8"] / by_c["y"]
    by_c["wape9"] = by_c["ae9"] / by_c["y"]
    by_c = by_c.sort_values("y", ascending=False)
    x = np.arange(len(by_c))
    axC.bar(x - 0.2, by_c["wape8"], width=0.4, color="#888888",
            label="V8", alpha=0.85)
    axC.bar(x + 0.2, by_c["wape9"], width=0.4, color="black",
            label="V9", alpha=0.85)
    axC.set_xticks(x); axC.set_xticklabels(by_c.index)
    axC.set_ylabel("WAPE (test set)")
    axC.set_title("Per-channel WAPE - V8 vs V9")
    axC.legend()
    axC.grid(alpha=0.25, axis="y")

    by_a = df.groupby("Сегмент_ABC", observed=True).agg(
        y=("target_qty", "sum"),
        b8=("err8", "sum"),
        b9=("err9", "sum"),
    )
    by_a["bias8"] = by_a["b8"] / by_a["y"] * 100
    by_a["bias9"] = by_a["b9"] / by_a["y"] * 100
    by_a = by_a.sort_values("y", ascending=False)
    x = np.arange(len(by_a))
    axA.bar(x - 0.2, by_a["bias8"], width=0.4, color="#888888",
            label="V8", alpha=0.85)
    axA.bar(x + 0.2, by_a["bias9"], width=0.4, color="black",
            label="V9", alpha=0.85)
    axA.axhline(0, color="grey", linewidth=0.8)
    axA.set_xticks(x); axA.set_xticklabels(by_a.index, rotation=30)
    axA.set_ylabel("Aggregate bias %")
    axA.set_title("Per-ABC class bias - V8 vs V9")
    axA.legend()
    axA.grid(alpha=0.25, axis="y")

    by_b = df.groupby("Бренд", observed=True).agg(
        y=("target_qty", "sum"),
        ae8=("err8", lambda s: np.abs(s).sum()),
        ae9=("err9", lambda s: np.abs(s).sum()),
    )
    by_b["wape8"] = by_b["ae8"] / by_b["y"]
    by_b["wape9"] = by_b["ae9"] / by_b["y"]
    by_b = by_b.sort_values("y", ascending=False)
    x = np.arange(len(by_b))
    axB.bar(x - 0.2, by_b["wape8"], width=0.4, color="#888888",
            label="V8", alpha=0.85)
    axB.bar(x + 0.2, by_b["wape9"], width=0.4, color="black",
            label="V9", alpha=0.85)
    axB.set_xticks(x); axB.set_xticklabels(by_b.index, rotation=30)
    axB.set_ylabel("WAPE (test set)")
    axB.set_title("Per-brand WAPE - V8 vs V9")
    axB.legend()
    axB.grid(alpha=0.25, axis="y")

    try:
        meta = json.loads((OUT / "v9" / "lad_champion.json").read_text())
        base_weights = meta["meta"]["base"]
    except Exception:
        base_weights = None
    if base_weights is not None:
        per_ch = {k: v for k, v in base_weights.items()
                  if k != "_global" and k != "tau" and isinstance(v, dict)}
        chs = sorted(per_ch.keys())
        bases = sorted({b for v in per_ch.values() for b in v.keys()})
        weights = np.array([[per_ch[c].get(b, 0) for b in bases] for c in chs])
        bottoms = np.zeros(len(chs))
        cmap = plt.colormaps["tab20"]
        v9_bases = {"v9", "v9_recent", "v9_weekly"}
        for i, b in enumerate(bases):
            colour = cmap(i / max(len(bases) - 1, 1))
            edgecolor = "red" if b in v9_bases else "white"
            lw = 2.5 if b in v9_bases else 0.5
            label = b + (" *" if b in v9_bases else "")
            axW.bar(chs, weights[:, i], bottom=bottoms,
                    label=label, color=colour, edgecolor=edgecolor,
                    linewidth=lw)
            bottoms += weights[:, i]
        axW.set_ylabel("LAD weight (sum-to-1 per channel)")
        axW.set_title(
            f"V9 base composition by channel - champion: {meta['champion']}\n"
            "V9 sales-leading + weekly bases highlighted in RED"
            f"  |  recency-weighted CV OOF = {meta['OOF_recency']:.4f}, "
            f"gap = {meta['overfit_gap']:+.4f}, "
            f"test SIMSCORE = {meta['test_score']['SIMSCORE']:.4f}, "
            f"test WAPE = {meta['test_score']['WAPE']:.4f}, "
            f"test bias = {meta['test_score']['Agg_Bias_pct']:+.2f}%"
        )
        axW.legend(ncol=6, fontsize=8, loc="upper right")
        axW.grid(alpha=0.25, axis="y")

    fig.suptitle(
        "V9 demand-forecast dashboard - held-out test set\n"
        "V8 (grey) vs V9 (black)  |  V9 = V8 LAD pool + 3 NEW bases: "
        "v9 (sales-leading features), v9_recent (γ=0.97), v9_weekly "
        "(weekly-target Tweedie rolled to monthly with per-channel calibration)",
        fontsize=12, weight="bold", y=0.998,
    )

    path = OUT / "plot_v9_dashboard.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
