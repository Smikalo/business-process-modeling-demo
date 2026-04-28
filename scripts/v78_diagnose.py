"""V7.7 residual diagnostic — surface the remaining structural error.

Loads the production V7.7 predictions on val + test and breaks the residuals
down by Канал × month_of_year, Канал × Сегмент_ABC × month_of_year, and
Бренд × month_of_year.  Writes:

* ``output/v78/diag_v77_canal_moy.csv``       — channel × month-of-year
* ``output/v78/diag_v77_canal_abc_moy.csv``   — channel × ABC × month-of-year
* ``output/v78/diag_v77_brand_moy.csv``       — brand × month-of-year
* ``output/v78/plot_v77_residual_heatmap.png`` — channel × month heatmap

The point is to verify (before any modeling) that there is *systematic*,
*estimable* structure in V7.7 residuals at the month-of-year × channel
granularity — which is what the V7.8 corrector will exploit.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "output"
V78 = OUT / "v78"
V78.mkdir(parents=True, exist_ok=True)

KEY = ["Период", "Партнер", "Артикул"]
META_AXES = ["Канал", "Бренд", "Сегмент_ABC"]


def _load(split: str) -> pd.DataFrame:
    p = pd.read_csv(OUT / f"preds_v77_{split}.csv")[KEY + ["target_qty", "prediction"]]
    abt = pd.read_parquet(OUT / "abt_v7_cached.parquet")[KEY + META_AXES]
    abt["Период"] = abt["Период"].astype(str)
    p["Период"] = p["Период"].astype(str)
    df = p.merge(abt, on=KEY, how="left")
    df["per_p"] = pd.PeriodIndex(df["Период"], freq="M")
    df["moy"] = df["per_p"].apply(lambda x: x.month)
    df["resid"] = df["prediction"] - df["target_qty"]
    df["split"] = split
    return df


def _by(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = df.groupby(by, observed=True).agg(
        n=("target_qty", "size"),
        y=("target_qty", "sum"),
        p=("prediction", "sum"),
        ae=("resid", lambda r: np.abs(r).sum()),
    )
    g["WAPE"] = g["ae"] / g["y"].clip(lower=1)
    g["bias_pct"] = (g["p"] / g["y"].clip(lower=1) - 1) * 100
    g["err_share_pct"] = g["ae"] / g["ae"].sum() * 100
    return g.round(3).sort_values("err_share_pct", ascending=False)


def main() -> int:
    val = _load("val")
    tst = _load("test")

    print("===  V7.7 — channel × month-of-year (val + test combined)  ===")
    df = pd.concat([val, tst], ignore_index=True)
    by_cm = _by(df, ["Канал", "moy"])
    by_cm.to_csv(V78 / "diag_v77_canal_moy.csv")
    print(by_cm.head(20).to_string())

    print("\n===  V7.7 — month-of-year only  ===")
    by_m = _by(df, ["moy", "split"])
    by_m.to_csv(V78 / "diag_v77_moy.csv")
    print(by_m.to_string())

    print("\n===  V7.7 — channel × ABC × month-of-year  ===")
    by_cam = _by(df, ["Канал", "Сегмент_ABC", "moy"])
    by_cam.to_csv(V78 / "diag_v77_canal_abc_moy.csv")
    print(by_cam.head(15).to_string())

    print("\n===  V7.7 — brand × month-of-year  ===")
    by_bm = _by(df, ["Бренд", "moy"])
    by_bm.to_csv(V78 / "diag_v77_brand_moy.csv")
    print(by_bm.head(15).to_string())

    # Heatmap: bias % by channel × month-of-year
    pivot_b = (
        df.groupby(["Канал", "moy"], observed=True)
          .agg(y=("target_qty", "sum"), p=("prediction", "sum"))
    )
    pivot_b["bias_pct"] = (pivot_b["p"] / pivot_b["y"].clip(lower=1) - 1) * 100
    pivot_b = pivot_b["bias_pct"].unstack("moy")
    fig, ax = plt.subplots(figsize=(13, 4.4))
    im = ax.imshow(
        pivot_b.values, cmap="RdBu_r", vmin=-30, vmax=30,
        aspect="auto",
    )
    ax.set_xticks(range(len(pivot_b.columns)))
    ax.set_xticklabels(pivot_b.columns)
    ax.set_yticks(range(len(pivot_b.index)))
    ax.set_yticklabels(pivot_b.index)
    ax.set_xlabel("Month of year")
    ax.set_ylabel("Канал (channel)")
    ax.set_title(
        "V7.7 residual heatmap — aggregate bias % by Канал × month-of-year\n"
        "(red = over-forecast, blue = under-forecast; val + test combined)"
    )
    for i in range(pivot_b.shape[0]):
        for j in range(pivot_b.shape[1]):
            v = pivot_b.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.0f}%", ha="center", va="center",
                        fontsize=9,
                        color="black" if abs(v) < 18 else "white")
    fig.colorbar(im, ax=ax, label="bias %")
    fig.tight_layout()
    out_path = V78 / "plot_v77_residual_heatmap.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out_path}")

    # Aggregate diagnostic
    overall = df.assign(ae=lambda d: d["resid"].abs())
    print(
        f"\nV7.7 overall: WAPE={overall['ae'].sum() / overall['target_qty'].sum():.4f}, "
        f"bias%={(overall['prediction'].sum() / overall['target_qty'].sum() - 1) * 100:+.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
