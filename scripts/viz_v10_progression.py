"""V1 → V10 progression visualization.

Summarizes the entire model evolution, validation + test SIMSCORE / WAPE
/ Monthly-WAPE / aggregate bias, plus a residual heatmap for V10.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "output"
KEY = ["Период", "Партнер", "Артикул"]

MODELS = [
    ("V1 (baseline)",        "v1",          "#bbbbbb"),
    ("V2 (TwoStage)",        "v2",          "#999999"),
    ("V3 (V2 + features)",   "v3",          "#7f7f7f"),
    ("V4 (V3 + Optuna)",     "v4",          "#9467bd"),
    ("V5 (V4 + ABC)",        "v5",          "#8c564b"),
    ("V6 (V5 + impute)",     "v6",          "#e377c2"),
    ("V7 (V6 + stack)",      "v7",          "#1f77b4"),
    ("V7.7 (multi-axis)",    "v77_recent",  "#17becf"),
    ("V8 LAD",               "v8_lad",      "#ff7f0e"),
    ("V9 LAD",               "v9_lad",      "#2ca02c"),
    ("V10 LAD (champion)",   "v10_lad",     "#000000"),
]


def _load(tag, split):
    p = OUT / f"preds_{tag}_{split}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)[KEY + ["target_qty", "prediction"]]


def _metric_row(df: pd.DataFrame) -> dict:
    if df is None or len(df) == 0:
        return {"WAPE": None, "bias_pct": None, "monthly_WAPE": None,
                "SIMSCORE": None}
    a = df["target_qty"].to_numpy()
    p = df["prediction"].to_numpy()
    wape = float(np.abs(a - p).sum() / max(a.sum(), 1e-6))
    bias = float((p.sum() - a.sum()) / max(a.sum(), 1e-6) * 100)
    by_m = df.groupby("Период").agg(act=("target_qty", "sum"),
                                     prd=("prediction", "sum"))
    mw = float(np.abs(by_m["act"] - by_m["prd"]).sum() /
               max(by_m["act"].sum(), 1e-6))
    sim = wape + 0.005 * abs(bias) + 0.5 * mw
    return {"WAPE": wape, "bias_pct": bias, "monthly_WAPE": mw, "SIMSCORE": sim}


def main() -> int:
    rows = []
    for label, tag, color in MODELS:
        for split in ("val", "test"):
            df = _load(tag, split)
            m = _metric_row(df)
            rows.append({"label": label, "tag": tag, "color": color,
                         "split": split, **m})
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "v10_progression_summary.csv", index=False)
    print(summ.round(4).to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    metrics = [("SIMSCORE", "SIMSCORE (lower = better)"),
               ("WAPE", "WAPE (lower = better)"),
               ("monthly_WAPE", "Monthly WAPE (lower = better)"),
               ("bias_pct", "Aggregate bias % (closer to 0 = better)")]
    for ax, (col, title) in zip(axes.flat, metrics):
        for split, marker, alpha in (("val", "o", 0.6), ("test", "^", 1.0)):
            sub = summ[summ["split"] == split].copy()
            sub = sub[sub[col].notna()]
            ax.plot(range(len(sub)), sub[col], color="#333333",
                    alpha=0.3, zorder=0)
            for i, r in enumerate(sub.itertuples(index=False)):
                ax.scatter(i, getattr(r, col), color=r.color, marker=marker,
                           s=120 if split == "test" else 70,
                           edgecolor="black", linewidth=0.5,
                           label=split if i == 0 else None, alpha=alpha)
            ax.set_xticks(range(len(sub)))
            ax.set_xticklabels(sub["label"].tolist(), rotation=45, ha="right",
                               fontsize=8)
        if col == "bias_pct":
            ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=9)

    fig.suptitle("V1 → V10 model evolution: SIMSCORE / WAPE / Monthly-WAPE / Bias",
                 fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "plot_v10_progression.png", dpi=140, bbox_inches="tight")
    print(f"wrote {OUT/'plot_v10_progression.png'}")

    df = _load("v10_lad", "test")
    if df is not None:
        df["Период"] = df["Период"].astype(str)
        df["resid_pct"] = (df["target_qty"] - df["prediction"]) / (
            df["target_qty"].abs() + 1.0
        )
        abt = pd.read_parquet(OUT / "abt_v7_cached.parquet")[
            ["Партнер", "Артикул", "Канал", "Сегмент_ABC"]
        ].drop_duplicates(subset=["Партнер", "Артикул"])
        for c in ("Канал", "Сегмент_ABC"):
            if isinstance(abt[c].dtype, pd.CategoricalDtype):
                abt[c] = abt[c].astype(str)
        df = df.merge(abt, on=["Партнер", "Артикул"], how="left")
        pivot = (df.groupby(["Канал", "Период"], observed=True)["resid_pct"]
                   .mean().unstack(fill_value=0))
        fig2, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r",
                       vmin=-0.5, vmax=0.5)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_title("V10 channel × month residual heatmap (test set)\n"
                     "blue = V10 over-predicts | red = V10 under-predicts")
        fig2.colorbar(im, ax=ax, label="(actual − pred) / (|actual|+1)")
        fig2.tight_layout()
        fig2.savefig(OUT / "plot_v10_residual_heatmap.png", dpi=140,
                     bbox_inches="tight")
        print(f"wrote {OUT/'plot_v10_residual_heatmap.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
