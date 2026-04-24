"""Time chart comparing monthly forecasts of every model vs actuals.

Loads `output/preds_{model}_{split}.csv` for val + test splits, aggregates to
monthly totals, and produces a two-panel figure:

* **Top panel**: monthly total forecast per model overlaid on actual demand.
* **Bottom panel**: monthly **RMSE of row-level residuals**
  (``√ mean((y − ŷ)² per row)``), one line per model.  RMSE is a
  squared-residual metric so it penalises *large* row errors more than
  small ones — a complement to the WAPE-style visual of the top panel.

Writes ``output/plot_models_timeline.png`` and
``output/models_timeline.csv`` (portfolio-level actuals+predictions).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "output"

MODELS = [
    ("V4", "v4", "#7f7f7f"),
    ("V5", "v5", "#1f77b4"),
    ("V6", "v6", "#2ca02c"),
    ("V7", "v7", "#ff7f0e"),
    ("V7.1", "v71_channels", "#d62728"),
    ("V7.2", "v72_champion", "#9467bd"),
    ("V7.3", "v73", "#e377c2"),
    ("V7.4", "v74", "#17becf"),
    ("V7.5 (champion)", "v75", "#bcbd22"),
]


def _load(tag: str) -> pd.DataFrame:
    parts = []
    for split in ("val", "test"):
        p = OUT / f"preds_{tag}_{split}.csv"
        df = pd.read_csv(p)
        df["split"] = split
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df["Период"] = pd.PeriodIndex(df["Период"].astype(str), freq="M").to_timestamp()
    return df


def main() -> int:
    monthly_actual = None
    monthly_preds: dict[str, pd.Series] = {}
    monthly_rmse: dict[str, pd.Series] = {}
    for label, tag, _ in MODELS:
        df = _load(tag)
        agg = df.groupby("Период").agg(
            actual=("target_qty", "sum"),
            pred=("prediction", "sum"),
        )
        if monthly_actual is None:
            monthly_actual = agg["actual"]
        monthly_preds[label] = agg["pred"]
        df["sq_err"] = (df["target_qty"] - df["prediction"]) ** 2
        rmse = df.groupby("Период")["sq_err"].mean().pow(0.5)
        monthly_rmse[label] = rmse

    assert monthly_actual is not None
    val_end = pd.Timestamp("2025-06-01")

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 11),
        gridspec_kw={"height_ratios": [1.2, 1.0], "hspace": 0.30},
        sharex=True,
    )
    ax, ax2 = axes

    ax.plot(
        monthly_actual.index, monthly_actual.values,
        color="black", linewidth=2.4, marker="o", markersize=6,
        label="Actual demand", zorder=5,
    )
    for (label, _tag, color), series in zip(MODELS, monthly_preds.values()):
        ax.plot(
            series.index, series.values,
            color=color, linewidth=1.6, marker="s", markersize=4,
            label=label, alpha=0.9,
        )

    ymax = float(max(monthly_actual.max(), max(s.max() for s in monthly_preds.values()))) * 1.08
    ax.axvspan(monthly_actual.index.min(), val_end, alpha=0.07, color="steelblue")
    ax.axvspan(val_end, monthly_actual.index.max(), alpha=0.10, color="tomato")
    ax.text(
        monthly_actual.index.min() + (val_end - monthly_actual.index.min()) / 2,
        ymax * 0.97, "VALIDATION (12 months)",
        ha="center", va="top", fontsize=10, color="steelblue", weight="bold",
    )
    ax.text(
        val_end + (monthly_actual.index.max() - val_end) / 2,
        ymax * 0.97, "TEST / HOLD-OUT (8 months)",
        ha="center", va="top", fontsize=10, color="tomato", weight="bold",
    )

    ax.set_ylim(0, ymax)
    ax.set_ylabel("Total monthly demand (units)")
    ax.set_title(
        "Monthly forecast vs actual demand — all model generations\n"
        "Sum across all partner × SKU pairs"
    )
    ax.legend(loc="upper left", ncol=2, framealpha=0.9)
    ax.grid(alpha=0.25)

    for (label, _tag, color), series in zip(MODELS, monthly_rmse.values()):
        ax2.plot(
            series.index, series.values,
            color=color, linewidth=1.6, marker="s", markersize=4,
            label=label, alpha=0.9,
        )
    rmse_max = max(float(s.max()) for s in monthly_rmse.values()) * 1.08
    ax2.set_ylim(0, rmse_max)
    ax2.axvspan(monthly_actual.index.min(), val_end, alpha=0.07, color="steelblue")
    ax2.axvspan(val_end, monthly_actual.index.max(), alpha=0.10, color="tomato")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("RMSE per month\n(√ mean row squared residuals)")
    ax2.set_title(
        "Row-level squared residuals — √ mean((y − ŷ)²) per month\n"
        "Penalises large row-level errors; lower = better"
    )
    ax2.legend(loc="upper left", ncol=2, framealpha=0.9, fontsize=9)
    ax2.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path = OUT / "plot_models_timeline.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")

    rows = [{"Период": ts.strftime("%Y-%m"), "actual": int(v)}
            for ts, v in monthly_actual.items()]
    for label, series in monthly_preds.items():
        for i, (ts, v) in enumerate(series.items()):
            rows[i][label] = int(v)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "models_timeline.csv", index=False)
    print(f"wrote {OUT / 'models_timeline.csv'}")

    wape_rows = []
    for label, series in monthly_preds.items():
        err = np.abs(series.values - monthly_actual.values).sum()
        tot = monthly_actual.values.sum()
        wape_rows.append({"model": label, "portfolio_WAPE": round(err / tot, 4)})
    wape_df = pd.DataFrame(wape_rows)
    print("\nPortfolio-level WAPE (monthly totals, 20 months):")
    print(wape_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
