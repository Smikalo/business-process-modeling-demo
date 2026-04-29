"""V11_final production-model quality showcase — 6-panel deep-dive.

Single picture answering "how good is the production champion, really?":

  Panel 1 (top, large): Monthly portfolio forecast vs actual on the full
                        validation + test timeline, with ±1σ residual fan
                        around the prediction line.

  Panel 2: Per-channel residual distribution (violin + boxplot) with
           per-channel aggregate bias annotations.

  Panel 3: SKU-segment performance — top-100 SKUs by volume vs the
           long-tail rest, separately scored.

  Panel 4: Calibration — predicted-vs-actual scatter with a perfect-
           prediction y=x reference and a bin-mean overlay; shows where
           the model under- or over-forecasts as a function of magnitude.

  Panel 5: Rolling per-month MAE + WAPE over the entire 19-month window.

  Panel 6: Row-level residual histogram with normal overlay + key
           summary stats; shows the fat-tailed structure typical of
           zero-inflated demand.

Writes:
  output/plot_v11_quality_showcase.png
  output/v11_quality_showcase.csv (per-channel + per-segment summary)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
KEY = ["Период", "Партнер", "Артикул"]

PALETTE = {
    "actual": "#0d1b2a",
    "v11":    "#2a9d8f",
    "fan":    "#94d2bd",
    "bad":    "#e76f51",
    "good":   "#2a9d8f",
    "accent": "#e9c46a",
    "grid":   "#cccccc",
}


def _load_v11() -> pd.DataFrame:
    parts = []
    for split in ("val", "test"):
        df = pd.read_csv(OUT / f"preds_v11_final_{split}.csv")[
            KEY + ["target_qty", "prediction"]
        ]
        df["split"] = split
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df["Период_ts"] = pd.PeriodIndex(df["Период"].astype(str),
                                     freq="M").to_timestamp()
    df["resid"] = df["target_qty"] - df["prediction"]
    df["abs_resid"] = df["resid"].abs()
    df["channel"] = df["Партнер"].astype(str).str.split("-").str[0].str.strip()
    return df


def main() -> int:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#222",
        "axes.linewidth": 0.8,
        "axes.labelcolor": "#222",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlepad": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    df = _load_v11()
    val_end = pd.Timestamp("2025-07-01")

    by_m = df.groupby("Период_ts").agg(
        actual=("target_qty", "sum"),
        pred=("prediction", "sum"),
        resid_std=("resid", lambda s: float(np.std(s))),
        rmse=("resid", lambda s: float(np.sqrt((s**2).mean()))),
        mae=("abs_resid", "mean"),
        rows=("target_qty", "size"),
    )
    by_m["bias_pct"] = (by_m["pred"] / by_m["actual"] - 1) * 100
    by_m["wape"] = (
        df.groupby("Период_ts")["abs_resid"].sum()
        / df.groupby("Период_ts")["target_qty"].sum()
    )

    # Per-channel summary
    by_ch = df.groupby("channel").agg(
        n=("target_qty", "size"),
        actual_sum=("target_qty", "sum"),
        pred_sum=("prediction", "sum"),
        wape=("abs_resid", lambda s: float(s.sum() / max(
            df.loc[s.index, "target_qty"].sum(), 1e-6))),
    )
    by_ch["bias_pct"] = (by_ch["pred_sum"] / by_ch["actual_sum"] - 1) * 100
    by_ch = by_ch.sort_values("actual_sum", ascending=False).head(8)

    # Top-100 vs long-tail
    sku_vol = df.groupby("Артикул")["target_qty"].sum().sort_values(ascending=False)
    top_skus = set(sku_vol.head(100).index)
    df["seg"] = np.where(df["Артикул"].isin(top_skus), "Top-100 SKUs", "Long tail")
    seg_summary = df.groupby("seg").apply(
        lambda g: pd.Series({
            "n": len(g),
            "actual": g["target_qty"].sum(),
            "pred": g["prediction"].sum(),
            "WAPE": g["abs_resid"].sum() / max(g["target_qty"].sum(), 1e-6),
            "bias_pct": (g["prediction"].sum() / max(g["target_qty"].sum(), 1e-6) - 1) * 100,
            "MAE": g["abs_resid"].mean(),
        }), include_groups=False
    )

    # ---------------- figure layout ----------------
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(
        3, 6, figure=fig,
        height_ratios=[1.5, 1.0, 1.0],
        hspace=0.50, wspace=0.55,
    )

    # Panel 1: monthly timeline with residual fan
    ax1 = fig.add_subplot(gs[0, :])

    ax1.fill_between(by_m.index,
                     by_m["pred"] - by_m["resid_std"] * np.sqrt(by_m["rows"]),
                     by_m["pred"] + by_m["resid_std"] * np.sqrt(by_m["rows"]),
                     color=PALETTE["fan"], alpha=0.35,
                     label="Prediction ± 1σ residual envelope")
    ax1.plot(by_m.index, by_m["actual"], "o-", color=PALETTE["actual"],
             linewidth=2.6, markersize=10, label="Actual demand", zorder=5)
    ax1.plot(by_m.index, by_m["pred"], "^-", color=PALETTE["v11"],
             linewidth=2.4, markersize=9, label="V11_final prediction",
             zorder=4)

    ymax = float(max(by_m["actual"].max(), by_m["pred"].max())) * 1.15
    ax1.axvspan(by_m.index.min(), val_end, alpha=0.05, color="#1d3557")
    ax1.axvspan(val_end, by_m.index.max(), alpha=0.10, color="#e63946")
    ax1.text(by_m.index.min() + (val_end - by_m.index.min()) / 2,
             ymax * 0.96, "VALIDATION (12 months)", ha="center",
             fontsize=10, color="#1d3557", weight="bold")
    ax1.text(val_end + (by_m.index.max() - val_end) / 2,
             ymax * 0.96, "TEST / HOLD-OUT (7 months)", ha="center",
             fontsize=10, color="#e63946", weight="bold")

    test_mask = by_m.index >= val_end
    for ts in by_m.index[test_mask]:
        b = by_m.loc[ts, "bias_pct"]
        col = "#0a7f5f" if abs(b) <= 5 else ("#e76f51" if abs(b) > 10
                                              else "#d4a017")
        ax1.annotate(f"{b:+.1f}%", xy=(ts, by_m.loc[ts, "actual"]),
                     xytext=(0, 18), textcoords="offset points",
                     ha="center", fontsize=9, color=col, weight="bold",
                     bbox=dict(boxstyle="round,pad=0.25", fc="white",
                               ec=col, alpha=0.95, linewidth=0.8))

    ax1.set_ylim(0, ymax)
    ax1.set_ylabel("Total monthly demand (units)", fontsize=11)
    ax1.set_title("V11_final on the full 19-month timeline — actual vs prediction with ±1σ residual envelope\n"
                  "Per-test-month bias % annotated (green ≤ 5 %, amber ≤ 10 %, red > 10 %)",
                  fontsize=12)
    ax1.legend(loc="upper left", framealpha=0.95, fontsize=10)
    ax1.grid(alpha=0.25)

    # Panel 2: per-channel violin
    ax2 = fig.add_subplot(gs[1, :3])
    channels = by_ch.index.tolist()
    data = [df[df["channel"] == c]["resid"].clip(-30, 30).to_numpy()
            for c in channels]
    parts = ax2.violinplot(data, vert=False, widths=0.85, showmedians=True)
    for body in parts["bodies"]:
        body.set_facecolor(PALETTE["v11"])
        body.set_edgecolor("#222")
        body.set_alpha(0.65)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("#222")
            parts[key].set_linewidth(1.0)

    ax2.set_yticks(np.arange(1, len(channels) + 1))
    ax2.set_yticklabels(channels, fontsize=9)
    for i, c in enumerate(channels, 1):
        b = by_ch.loc[c, "bias_pct"]
        w = by_ch.loc[c, "wape"]
        col = "#0a7f5f" if abs(b) <= 5 else ("#e76f51" if abs(b) > 10
                                              else "#d4a017")
        ax2.text(28, i, f"bias {b:+.1f}%  WAPE {w:.2f}",
                 ha="left", va="center", fontsize=8, color=col,
                 weight="bold")

    ax2.axvline(0, color="#222", linewidth=1.0)
    ax2.set_xlim(-30, 50)
    ax2.set_xlabel("Residual = actual − prediction (clipped to ±30)")
    ax2.set_title("Per-channel residual distribution + aggregate bias")
    ax2.grid(alpha=0.25, axis="x")

    # Panel 3: segment performance
    ax3 = fig.add_subplot(gs[1, 3:])
    segs = seg_summary.index.tolist()
    metrics = ["WAPE", "bias_pct", "MAE"]
    metric_labels = ["WAPE", "|Bias %|", "MAE (units)"]
    x = np.arange(len(metrics))
    width = 0.35
    for i, seg in enumerate(segs):
        vals = [
            seg_summary.loc[seg, "WAPE"],
            abs(seg_summary.loc[seg, "bias_pct"]) / 100,
            seg_summary.loc[seg, "MAE"] / seg_summary["MAE"].max(),
        ]
        bars = ax3.bar(x + (i - 0.5) * width, vals, width=width,
                       color=[PALETTE["v11"], PALETTE["accent"]][i],
                       edgecolor="#222", linewidth=0.6, alpha=0.92,
                       label=f"{seg} (n={int(seg_summary.loc[seg, 'n']):,})")
        for xi, v, raw_label in zip(x + (i - 0.5) * width, vals,
                                     [seg_summary.loc[seg, "WAPE"],
                                      seg_summary.loc[seg, "bias_pct"],
                                      seg_summary.loc[seg, "MAE"]]):
            label = (f"{raw_label:.3f}" if abs(raw_label) < 1
                      else f"{raw_label:+.1f}%" if metrics[int(round(xi))] == "bias_pct"
                      else f"{raw_label:.2f}")
            if "bias" in metric_labels[int(np.argmin(np.abs(x - xi - (i-0.5)*width)))]:
                label = f"{seg_summary.loc[seg, 'bias_pct']:+.1f}%"
            ax3.text(xi, v + 0.015, label, ha="center", va="bottom",
                     fontsize=8, color="#222")

    ax3.set_xticks(x)
    ax3.set_xticklabels(metric_labels)
    ax3.set_title("Top-100 SKUs vs long-tail performance\n(WAPE & |bias| in [0,1] scale; MAE normalised to longer-bar)")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.25, axis="y")

    # Panel 4: calibration scatter
    ax4 = fig.add_subplot(gs[2, :2])
    sample = df[df["target_qty"] > 0].sample(min(8000, len(df)), random_state=42)
    ax4.scatter(sample["prediction"].clip(0, 50),
                sample["target_qty"].clip(0, 50),
                alpha=0.18, s=8, color=PALETTE["v11"], edgecolor="none")
    bins = np.linspace(0, 50, 26)
    df_b = df.copy()
    df_b["bin"] = pd.cut(df_b["prediction"].clip(0, 50), bins=bins)
    bin_means = df_b.groupby("bin", observed=True).agg(
        x=("prediction", "mean"), y=("target_qty", "mean"),
        n=("target_qty", "size"))
    bin_means = bin_means[bin_means["n"] >= 30]
    ax4.plot(bin_means["x"], bin_means["y"], "o-",
             color=PALETTE["bad"], linewidth=2.0, markersize=7,
             label="Observed bin mean")
    ax4.plot([0, 50], [0, 50], "--", color="#222", linewidth=1.2,
             alpha=0.7, label="Perfect calibration y = x")
    ax4.set_xlim(0, 50); ax4.set_ylim(0, 50)
    ax4.set_xlabel("Predicted qty"); ax4.set_ylabel("Actual qty")
    ax4.set_title("Calibration — predicted vs actual\n(red dots = empirical bin means; below y=x = under-forecast)")
    ax4.legend(loc="upper left", fontsize=9)
    ax4.grid(alpha=0.25)

    # Panel 5: rolling MAE + WAPE
    ax5 = fig.add_subplot(gs[2, 2:4])
    ax5.plot(by_m.index, by_m["mae"], "o-",
             color=PALETTE["v11"], linewidth=2.0, markersize=7,
             label="MAE (units)")
    ax5.set_ylabel("MAE (units)", color=PALETTE["v11"], fontsize=10)
    ax5.tick_params(axis="y", labelcolor=PALETTE["v11"])
    ax5b = ax5.twinx()
    ax5b.plot(by_m.index, by_m["wape"], "s-",
              color=PALETTE["bad"], linewidth=2.0, markersize=7,
              label="WAPE")
    ax5b.set_ylabel("WAPE", color=PALETTE["bad"], fontsize=10)
    ax5b.tick_params(axis="y", labelcolor=PALETTE["bad"])
    ax5b.spines.right.set_visible(True)
    ax5.axvspan(by_m.index.min(), val_end, alpha=0.05, color="#1d3557")
    ax5.axvspan(val_end, by_m.index.max(), alpha=0.10, color="#e63946")
    ax5.set_title("Rolling MAE & WAPE per month\n(consistency over 19 months)")
    # Reduce x-axis tick crowding: pick ~6 quarterly tick marks
    qticks = pd.date_range(by_m.index.min(), by_m.index.max(), freq="3MS")
    ax5.set_xticks(qticks)
    ax5.set_xticklabels([d.strftime("%Y-%m") for d in qticks],
                        rotation=30, fontsize=8, ha="right")
    ax5.grid(alpha=0.25)

    # Panel 6: residual histogram
    ax6 = fig.add_subplot(gs[2, 4:])
    resid = df["resid"].clip(-25, 25).to_numpy()
    ax6.hist(resid, bins=80, color=PALETTE["v11"], alpha=0.85,
             edgecolor="white", linewidth=0.4, density=True)
    mu, sigma = float(resid.mean()), float(resid.std())
    xs = np.linspace(-25, 25, 200)
    ax6.plot(xs,
             1 / (sigma * np.sqrt(2 * np.pi))
             * np.exp(-0.5 * ((xs - mu) / sigma) ** 2),
             color=PALETTE["bad"], linewidth=2.2,
             label=f"Normal fit μ={mu:.2f} σ={sigma:.2f}")
    ax6.axvline(0, color="#222", linewidth=1.0)
    ax6.set_xlim(-25, 25)
    ax6.set_xlabel("Residual (actual − prediction)")
    ax6.set_ylabel("Density")
    ax6.set_title("Row-level residual distribution\n(zero-centered, fat-tailed = zero-inflated demand structure)")
    ax6.legend(fontsize=9)
    ax6.grid(alpha=0.25)

    # ---------------- title and save ----------------
    test_df = df[df["split"] == "test"]
    test_wape = test_df["abs_resid"].sum() / test_df["target_qty"].sum()
    test_bias = (test_df["prediction"].sum() / test_df["target_qty"].sum() - 1) * 100
    val_df = df[df["split"] == "val"]
    val_wape = val_df["abs_resid"].sum() / val_df["target_qty"].sum()

    fig.suptitle(
        "V11_final production-model quality showcase\n"
        f"Val WAPE {val_wape:.4f}  |  Test WAPE {test_wape:.4f}  |  "
        f"Test aggregate bias {test_bias:+.2f} %  |  "
        f"19 months × {df['Партнер'].nunique()} partners × {df['Артикул'].nunique()} SKUs = {len(df):,} rows",
        fontsize=14, weight="bold", y=0.998,
    )
    fig.tight_layout()
    out_path = OUT / "plot_v11_quality_showcase.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white")
    print(f"wrote {out_path}")

    # Summary CSV
    summary = pd.concat([
        by_ch[["actual_sum", "pred_sum", "wape", "bias_pct", "n"]]
            .assign(group="channel"),
        seg_summary.assign(group="segment")
            .rename(columns={"actual": "actual_sum",
                              "pred": "pred_sum",
                              "WAPE": "wape"})
            [["actual_sum", "pred_sum", "wape", "bias_pct", "n", "group"]],
    ])
    summary.to_csv(OUT / "v11_quality_showcase.csv")
    print(f"wrote {OUT / 'v11_quality_showcase.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
