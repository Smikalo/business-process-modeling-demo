"""V7.7 dashboard — six panels comparing V7.5 (previous champion) vs V7.7.

Panels:
1. Scatter actual vs predicted (V7.7 only, log scale, color by absolute error)
2. Residual distribution (V7.5 vs V7.7)
3. Per-channel WAPE & bias bar chart
4. Per-ABC class WAPE & bias bar chart
5. Per-month bias % over the test window
6. LAD weight composition by channel (treemap-style stacked bars)
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

    df75 = _load("v75").rename(columns={"prediction": "p75"})
    df77 = _load("v77").rename(columns={"prediction": "p77"})
    df = (
        df75.merge(df77.drop(columns=["target_qty"]), on=KEY)
            .merge(abt, on=KEY, how="left")
    )
    df["per_p"] = pd.PeriodIndex(df["Период"].astype(str), freq="M")
    df["err75"] = df["p75"] - df["target_qty"]
    df["err77"] = df["p77"] - df["target_qty"]

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.30)
    axS = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])
    axM = fig.add_subplot(gs[0, 2])
    axC = fig.add_subplot(gs[1, 0])
    axA = fig.add_subplot(gs[1, 1])
    axB = fig.add_subplot(gs[1, 2])
    axW = fig.add_subplot(gs[2, :])

    # 1. Scatter
    sc = axS.scatter(
        df["target_qty"] + 0.5, df["p77"] + 0.5,
        c=np.abs(df["err77"]), cmap="viridis", s=8, alpha=0.4,
    )
    lim = max(df["target_qty"].max(), df["p77"].max()) + 1
    axS.plot([0.5, lim], [0.5, lim], color="red", linestyle="--",
             linewidth=1.0, alpha=0.7)
    axS.set_xscale("log"); axS.set_yscale("log")
    axS.set_xlim(0.5, lim); axS.set_ylim(0.5, lim)
    axS.set_xlabel("Actual qty (test set)")
    axS.set_ylabel("V7.7 predicted qty")
    axS.set_title("Actual vs predicted (log–log)")
    axS.grid(alpha=0.25, which="both")
    fig.colorbar(sc, ax=axS, label="|residual|")

    # 2. Residual distribution comparison
    axR.hist(df["err75"], bins=80, range=(-30, 30), alpha=0.55,
             label="V7.5", color="#bcbd22")
    axR.hist(df["err77"], bins=80, range=(-30, 30), alpha=0.55,
             label="V7.7", color="black")
    axR.axvline(0, color="grey", linestyle="--", linewidth=1)
    axR.set_xlabel("Residual (predicted − actual)")
    axR.set_ylabel("Row count")
    axR.set_title("Residual distribution (test set)")
    axR.legend()
    axR.grid(alpha=0.25)

    # 3. Monthly bias %
    by_m = df.groupby("per_p", observed=True).agg(
        y=("target_qty", "sum"),
        p75=("p75", "sum"),
        p77=("p77", "sum"),
    )
    by_m["bias75"] = (by_m["p75"] / by_m["y"] - 1) * 100
    by_m["bias77"] = (by_m["p77"] / by_m["y"] - 1) * 100
    x = np.arange(len(by_m))
    axM.bar(x - 0.2, by_m["bias75"], width=0.4, color="#bcbd22",
            label="V7.5", alpha=0.85)
    axM.bar(x + 0.2, by_m["bias77"], width=0.4, color="black",
            label="V7.7", alpha=0.85)
    axM.axhline(0, color="grey", linewidth=0.8)
    axM.set_xticks(x)
    axM.set_xticklabels([str(p) for p in by_m.index],
                        rotation=45, ha="right")
    axM.set_ylabel("Aggregate bias % (test month)")
    axM.set_title("Monthly bias — V7.5 vs V7.7")
    axM.legend()
    axM.grid(alpha=0.25, axis="y")

    # 4. Per-channel WAPE
    by_c = df.groupby("Канал", observed=True).agg(
        y=("target_qty", "sum"),
        ae75=("err75", lambda s: np.abs(s).sum()),
        ae77=("err77", lambda s: np.abs(s).sum()),
        b75=("err75", "sum"),
        b77=("err77", "sum"),
    )
    by_c["wape75"] = by_c["ae75"] / by_c["y"]
    by_c["wape77"] = by_c["ae77"] / by_c["y"]
    by_c["bias75"] = by_c["b75"] / by_c["y"] * 100
    by_c["bias77"] = by_c["b77"] / by_c["y"] * 100
    by_c = by_c.sort_values("y", ascending=False)
    x = np.arange(len(by_c))
    axC.bar(x - 0.2, by_c["wape75"], width=0.4, color="#bcbd22",
            label="V7.5 WAPE", alpha=0.85)
    axC.bar(x + 0.2, by_c["wape77"], width=0.4, color="black",
            label="V7.7 WAPE", alpha=0.85)
    axC.set_xticks(x); axC.set_xticklabels(by_c.index)
    axC.set_ylabel("WAPE (test set)")
    axC.set_title("Per-channel WAPE — V7.5 vs V7.7")
    axC.legend()
    axC.grid(alpha=0.25, axis="y")

    # 5. Per-ABC bias
    by_a = df.groupby("Сегмент_ABC", observed=True).agg(
        y=("target_qty", "sum"),
        b75=("err75", "sum"),
        b77=("err77", "sum"),
    )
    by_a["bias75"] = by_a["b75"] / by_a["y"] * 100
    by_a["bias77"] = by_a["b77"] / by_a["y"] * 100
    by_a = by_a.sort_values("y", ascending=False)
    x = np.arange(len(by_a))
    axA.bar(x - 0.2, by_a["bias75"], width=0.4, color="#bcbd22",
            label="V7.5", alpha=0.85)
    axA.bar(x + 0.2, by_a["bias77"], width=0.4, color="black",
            label="V7.7", alpha=0.85)
    axA.axhline(0, color="grey", linewidth=0.8)
    axA.set_xticks(x); axA.set_xticklabels(by_a.index, rotation=30)
    axA.set_ylabel("Aggregate bias %")
    axA.set_title("Per-ABC class bias — V7.5 vs V7.7")
    axA.legend()
    axA.grid(alpha=0.25, axis="y")

    # 6. Per-brand WAPE
    by_b = df.groupby("Бренд", observed=True).agg(
        y=("target_qty", "sum"),
        ae75=("err75", lambda s: np.abs(s).sum()),
        ae77=("err77", lambda s: np.abs(s).sum()),
    )
    by_b["wape75"] = by_b["ae75"] / by_b["y"]
    by_b["wape77"] = by_b["ae77"] / by_b["y"]
    by_b = by_b.sort_values("y", ascending=False)
    x = np.arange(len(by_b))
    axB.bar(x - 0.2, by_b["wape75"], width=0.4, color="#bcbd22",
            label="V7.5", alpha=0.85)
    axB.bar(x + 0.2, by_b["wape77"], width=0.4, color="black",
            label="V7.7", alpha=0.85)
    axB.set_xticks(x); axB.set_xticklabels(by_b.index, rotation=30)
    axB.set_ylabel("WAPE (test set)")
    axB.set_title("Per-brand WAPE — V7.5 vs V7.7")
    axB.legend()
    axB.grid(alpha=0.25, axis="y")

    # 7. LAD weight composition per channel
    try:
        meta = json.loads((OUT / "v77" / "lad_champion.json").read_text())
        base_weights = meta["meta"]["base"]
    except Exception:
        base_weights = None
    if base_weights is not None:
        per_ch = {k: v for k, v in base_weights.items()
                  if k != "_global" and k != "tau"
                  and isinstance(v, dict)}
        chs = sorted(per_ch.keys())
        bases = sorted({b for v in per_ch.values() for b in v.keys()})
        weights = np.array([[per_ch[c].get(b, 0) for b in bases] for c in chs])
        bottoms = np.zeros(len(chs))
        cmap = plt.colormaps["tab20"]
        for i, b in enumerate(bases):
            axW.bar(chs, weights[:, i], bottom=bottoms, label=b,
                    color=cmap(i / max(len(bases) - 1, 1)))
            bottoms += weights[:, i]
        axW.set_ylabel("LAD weight (sum-to-1 per channel)")
        axW.set_title(
            f"V7.7 base composition by channel — champion: {meta['champion']}\n"
            "Recency-weighted CV OOF = "
            f"{meta['OOF_SIMSCORE']:.4f}, gap = {meta['overfit_gap']:+.4f}"
        )
        axW.legend(ncol=4, fontsize=9, loc="upper right")
        axW.grid(alpha=0.25, axis="y")

    fig.suptitle(
        "V7.7 demand-forecast dashboard — held-out test set\n"
        "V7.5 (yellow) vs V7.7 (black) | recency-weighted base + channel×ABC×brand reconciliation",
        fontsize=13, weight="bold", y=0.998,
    )

    path = OUT / "plot_v77_dashboard.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
