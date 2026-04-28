"""V7.8 production dashboard — comprehensive single-figure summary.

Six-panel + timeline figure that captures everything the user needs to
evaluate V7.8 against V7.7 (previous champion):

  Row 1
    - Scatter actual vs V7.8 predicted (log-log, coloured by |residual|)
    - Residual distribution V7.7 vs V7.8
    - Monthly aggregate bias % V7.7 vs V7.8 (test window only)
  Row 2
    - Per-channel WAPE V7.7 vs V7.8
    - Per-ABC class bias % V7.7 vs V7.8
    - Per-brand WAPE V7.7 vs V7.8
  Row 3 (full width)
    - LAD weight composition by channel (V7.8 stacked bars)

Writes ``output/plot_v78_dashboard.png``.
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

    df77 = _load("v77").rename(columns={"prediction": "p77"})
    df78 = _load("v78").rename(columns={"prediction": "p78"})
    df = (
        df77.merge(df78.drop(columns=["target_qty"]), on=KEY)
            .merge(abt, on=KEY, how="left")
    )
    df["per_p"] = pd.PeriodIndex(df["Период"].astype(str), freq="M")
    df["err77"] = df["p77"] - df["target_qty"]
    df["err78"] = df["p78"] - df["target_qty"]

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
        df["target_qty"] + 0.5, df["p78"] + 0.5,
        c=np.abs(df["err78"]), cmap="viridis", s=8, alpha=0.4,
    )
    lim = max(df["target_qty"].max(), df["p78"].max()) + 1
    axS.plot([0.5, lim], [0.5, lim], color="red", linestyle="--",
             linewidth=1.0, alpha=0.7)
    axS.set_xscale("log"); axS.set_yscale("log")
    axS.set_xlim(0.5, lim); axS.set_ylim(0.5, lim)
    axS.set_xlabel("Actual qty (test set)")
    axS.set_ylabel("V7.8 predicted qty")
    axS.set_title("Actual vs predicted (log–log)")
    axS.grid(alpha=0.25, which="both")
    fig.colorbar(sc, ax=axS, label="|residual|")

    # 2. Residual distribution
    axR.hist(df["err77"], bins=80, range=(-30, 30), alpha=0.55,
             label="V7.7", color="#555555")
    axR.hist(df["err78"], bins=80, range=(-30, 30), alpha=0.55,
             label="V7.8", color="black")
    axR.axvline(0, color="grey", linestyle="--", linewidth=1)
    axR.set_xlabel("Residual (predicted − actual)")
    axR.set_ylabel("Row count")
    axR.set_title("Residual distribution (test set)")
    axR.legend()
    axR.grid(alpha=0.25)

    # 3. Monthly bias %
    by_m = df.groupby("per_p", observed=True).agg(
        y=("target_qty", "sum"),
        p77=("p77", "sum"),
        p78=("p78", "sum"),
    )
    by_m["bias77"] = (by_m["p77"] / by_m["y"] - 1) * 100
    by_m["bias78"] = (by_m["p78"] / by_m["y"] - 1) * 100
    x = np.arange(len(by_m))
    axM.bar(x - 0.2, by_m["bias77"], width=0.4, color="#555555",
            label="V7.7", alpha=0.85)
    axM.bar(x + 0.2, by_m["bias78"], width=0.4, color="black",
            label="V7.8", alpha=0.85)
    axM.axhline(0, color="grey", linewidth=0.8)
    axM.set_xticks(x)
    axM.set_xticklabels([str(p) for p in by_m.index],
                        rotation=45, ha="right")
    axM.set_ylabel("Aggregate bias % (test month)")
    axM.set_title("Monthly bias — V7.7 vs V7.8")
    axM.legend()
    axM.grid(alpha=0.25, axis="y")

    # 4. Per-channel WAPE
    by_c = df.groupby("Канал", observed=True).agg(
        y=("target_qty", "sum"),
        ae77=("err77", lambda s: np.abs(s).sum()),
        ae78=("err78", lambda s: np.abs(s).sum()),
        b77=("err77", "sum"),
        b78=("err78", "sum"),
    )
    by_c["wape77"] = by_c["ae77"] / by_c["y"]
    by_c["wape78"] = by_c["ae78"] / by_c["y"]
    by_c["bias77"] = by_c["b77"] / by_c["y"] * 100
    by_c["bias78"] = by_c["b78"] / by_c["y"] * 100
    by_c = by_c.sort_values("y", ascending=False)
    x = np.arange(len(by_c))
    axC.bar(x - 0.2, by_c["wape77"], width=0.4, color="#555555",
            label="V7.7", alpha=0.85)
    axC.bar(x + 0.2, by_c["wape78"], width=0.4, color="black",
            label="V7.8", alpha=0.85)
    axC.set_xticks(x); axC.set_xticklabels(by_c.index)
    axC.set_ylabel("WAPE (test set)")
    axC.set_title("Per-channel WAPE — V7.7 vs V7.8")
    axC.legend()
    axC.grid(alpha=0.25, axis="y")

    # 5. Per-ABC bias
    by_a = df.groupby("Сегмент_ABC", observed=True).agg(
        y=("target_qty", "sum"),
        b77=("err77", "sum"),
        b78=("err78", "sum"),
    )
    by_a["bias77"] = by_a["b77"] / by_a["y"] * 100
    by_a["bias78"] = by_a["b78"] / by_a["y"] * 100
    by_a = by_a.sort_values("y", ascending=False)
    x = np.arange(len(by_a))
    axA.bar(x - 0.2, by_a["bias77"], width=0.4, color="#555555",
            label="V7.7", alpha=0.85)
    axA.bar(x + 0.2, by_a["bias78"], width=0.4, color="black",
            label="V7.8", alpha=0.85)
    axA.axhline(0, color="grey", linewidth=0.8)
    axA.set_xticks(x); axA.set_xticklabels(by_a.index, rotation=30)
    axA.set_ylabel("Aggregate bias %")
    axA.set_title("Per-ABC class bias — V7.7 vs V7.8")
    axA.legend()
    axA.grid(alpha=0.25, axis="y")

    # 6. Per-brand WAPE
    by_b = df.groupby("Бренд", observed=True).agg(
        y=("target_qty", "sum"),
        ae77=("err77", lambda s: np.abs(s).sum()),
        ae78=("err78", lambda s: np.abs(s).sum()),
    )
    by_b["wape77"] = by_b["ae77"] / by_b["y"]
    by_b["wape78"] = by_b["ae78"] / by_b["y"]
    by_b = by_b.sort_values("y", ascending=False)
    x = np.arange(len(by_b))
    axB.bar(x - 0.2, by_b["wape77"], width=0.4, color="#555555",
            label="V7.7", alpha=0.85)
    axB.bar(x + 0.2, by_b["wape78"], width=0.4, color="black",
            label="V7.8", alpha=0.85)
    axB.set_xticks(x); axB.set_xticklabels(by_b.index, rotation=30)
    axB.set_ylabel("WAPE (test set)")
    axB.set_title("Per-brand WAPE — V7.7 vs V7.8")
    axB.legend()
    axB.grid(alpha=0.25, axis="y")

    # 7. LAD weight composition per channel
    try:
        meta = json.loads((OUT / "v78" / "lad_champion.json").read_text())
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
        for i, b in enumerate(bases):
            axW.bar(chs, weights[:, i], bottom=bottoms, label=b,
                    color=cmap(i / max(len(bases) - 1, 1)))
            bottoms += weights[:, i]
        axW.set_ylabel("LAD weight (sum-to-1 per channel)")
        axW.set_title(
            f"V7.8 base composition by channel — champion: {meta['champion']}\n"
            "Recency-weighted CV OOF = "
            f"{meta['OOF_recency']:.4f}, gap = {meta['overfit_gap']:+.4f}, "
            f"test SIMSCORE = {meta['test_score']['SIMSCORE']:.4f}, "
            f"test WAPE = {meta['test_score']['WAPE']:.4f}, "
            f"test bias = {meta['test_score']['Agg_Bias_pct']:+.2f}%"
        )
        axW.legend(ncol=4, fontsize=9, loc="upper right")
        axW.grid(alpha=0.25, axis="y")

    fig.suptitle(
        "V7.8 demand-forecast dashboard — held-out test set\n"
        "V7.7 (grey) vs V7.8 (black) | extended LAD pool + tilted-LAD τ=0.55 + "
        "channel × ABC × brand reconciliation",
        fontsize=13, weight="bold", y=0.998,
    )

    path = OUT / "plot_v78_dashboard.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
