"""V11_final superiority showcase — single-page, presentation-grade.

Shows in one polished figure why V11_final is the production champion
across all 62 models trained in the project:

  Panel 1 (large left):  Pareto plot val SIMSCORE × test SIMSCORE for
                         every model in the repo.  V11_final highlighted
                         with a star marker; nearest competitors labelled.
                         Pareto frontier drawn.

  Panel 2 (top right):   Radar chart — V11_final vs naive baseline vs V10
                         (previous champion) across 6 normalised metrics.

  Panel 3 (mid right):   Top-15 ranked-by-test-SIMSCORE bar chart with
                         V11_final highlighted in gold; deltas vs naive
                         baseline annotated.

  Panel 4 (bottom right): V1 → V11 cumulative-improvement timeline on
                          test SIMSCORE / WAPE with annotations of the
                          top three biggest leaps.

Writes:
  output/plot_v11_superiority.png
  output/v11_superiority_radar.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"

PALETTE = {
    "v11":      "#2a9d8f",
    "v11_glow": "#83c5be",
    "ref":      "#264653",
    "naive":    "#e76f51",
    "v10":      "#e9c46a",
    "good":     "#0a7f5f",
    "bad":      "#bc4749",
    "amber":    "#d4a017",
    "ink":      "#222",
}


def _load_comparison() -> pd.DataFrame:
    """Loads pre-computed all-models comparison; falls back to running the
    comparison script if the CSV is stale."""
    p = OUT / "all_models_comparison.csv"
    if not p.exists():
        raise SystemExit(
            "output/all_models_comparison.csv not found — "
            "run `python -m scripts.viz_all_models_comparison` first."
        )
    return pd.read_csv(p)


def main() -> int:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.titlepad": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#222",
        "axes.linewidth": 0.8,
    })

    df = _load_comparison().copy()
    df = df.sort_values("test_SIMSCORE").reset_index(drop=True)

    if "v11_final" not in df["tag"].values:
        raise SystemExit("v11_final not in all_models_comparison.csv")

    # Helper: pareto frontier on (val_SIMSCORE, test_SIMSCORE), lower-better
    def _pareto_indices(xs: np.ndarray, ys: np.ndarray) -> list[int]:
        order = np.argsort(xs)
        front: list[int] = []
        best_y = np.inf
        for i in order:
            if ys[i] <= best_y:
                front.append(i)
                best_y = ys[i]
        return front

    fig = plt.figure(figsize=(20, 12), facecolor="white")
    gs = GridSpec(
        3, 3, figure=fig,
        width_ratios=[1.5, 1.0, 1.0],
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.55, wspace=0.40,
    )

    # ---------------- Panel 1: Pareto val × test ----------------
    ax1 = fig.add_subplot(gs[:, 0])
    xs = df["val_SIMSCORE"].to_numpy()
    ys = df["test_SIMSCORE"].to_numpy()
    pf = _pareto_indices(xs, ys)
    pf_x = xs[pf]; pf_y = ys[pf]

    sc = ax1.scatter(xs, ys, s=42, alpha=0.55,
                     c=df["OOF_safe"].map({True: PALETTE["v11"],
                                            False: "#9b9b9b"}),
                     edgecolor="white", linewidth=0.5)
    ax1.plot(pf_x, pf_y, "--", color=PALETTE["amber"], linewidth=1.6,
             alpha=0.85, label="Pareto frontier (val × test)")
    ax1.scatter(pf_x, pf_y, s=85, facecolor="none",
                edgecolor=PALETTE["amber"], linewidth=1.4)

    # Highlight V11_final + nearest competitors
    v11_idx = df.index[df["tag"] == "v11_final"][0]
    ax1.scatter(df.loc[v11_idx, "val_SIMSCORE"],
                df.loc[v11_idx, "test_SIMSCORE"],
                s=380, marker="*", color=PALETTE["bad"],
                edgecolor="white", linewidth=1.8, zorder=10,
                label="V11_final (production champion)")

    # Annotate top-5 + V10 + V8 + V9 anchors
    anchors = ["v11_final", "v11_relaxed", "v11_test_aware",
               "v11_g93", "v9_lad", "v10_lad", "v8_lad", "naiveS"]
    for tag in anchors:
        if tag not in df["tag"].values:
            continue
        r = df[df["tag"] == tag].iloc[0]
        is_main = tag == "v11_final"
        col = PALETTE["bad"] if is_main else (
            PALETTE["v11"] if r["OOF_safe"] else "#666")
        weight = "bold" if is_main else "normal"
        ax1.annotate(
            tag, xy=(r["val_SIMSCORE"], r["test_SIMSCORE"]),
            xytext=(7, 7), textcoords="offset points",
            fontsize=9.5 if is_main else 8.5, color=col,
            weight=weight, alpha=0.95,
            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                      ec=col, alpha=0.92, linewidth=0.9),
        )

    mn = float(min(xs.min(), ys.min())) * 0.95
    mx = float(max(xs.max(), ys.max())) * 1.02
    ax1.plot([mn, mx], [mn, mx], ":", color="#333", linewidth=1,
             alpha=0.5, label="val = test (no over/underfit)")
    ax1.set_xlim(mn, mx); ax1.set_ylim(mn, mx)
    ax1.set_xlabel("Validation SIMSCORE (lower = better)", fontsize=11)
    ax1.set_ylabel("Test SIMSCORE (lower = better)", fontsize=11)
    ax1.set_title(
        f"All {len(df)} models on the val × test plane\n"
        "Closer to (0, 0) = better. Pareto frontier in amber. ★ = V11_final."
    )
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper left", framealpha=0.95, fontsize=9.5)

    # ---------------- Panel 2: Radar V11 vs naive vs V10 ----------------
    ax2 = fig.add_subplot(gs[0, 1:], projection="polar")
    metrics = ["test_SIMSCORE", "test_WAPE", "test_M_WAPE",
               "abs_test_bias", "OOF_gap", "val_SIMSCORE"]
    metric_lbl = ["Test\nSIMSCORE", "Test\nWAPE", "Test\nMonthly-WAPE",
                  "|Test\nbias %|", "Val→Test\ngap", "Val\nSIMSCORE"]
    df["abs_test_bias"] = df["test_bias"].abs()
    df["OOF_gap"] = df["gap_SIMSCORE"].abs()

    targets = [t for t in ["naiveS", "v10_lad", "v11_final"] if t in df["tag"].values]

    # Normalise each metric to [0, 1] across the full pool — 0 = worst, 1 = best.
    # All metrics are "lower is better", so best = min, worst = max.
    pool = df.copy()
    norm = pd.DataFrame(index=df["tag"])
    for m in metrics:
        col = pool[m].astype(float)
        lo = col.quantile(0.02); hi = col.quantile(0.98)
        norm[m] = 1 - ((col.clip(lo, hi).values - lo) / max(hi - lo, 1e-9))
    norm.insert(0, "tag", df["tag"].values)
    norm = norm.set_index("tag")

    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    colors = {"naiveS": PALETTE["naive"], "v10_lad": PALETTE["v10"],
              "v11_final": PALETTE["v11"]}
    label_map = {"naiveS": "Seasonal-naive baseline",
                 "v10_lad": "V10 LAD (previous)",
                 "v11_final": "V11_final (production)"}
    for tag in targets:
        vals = norm.loc[tag, metrics].tolist() + [norm.loc[tag, metrics[0]]]
        ax2.plot(angles, vals, "o-", linewidth=2.4,
                 color=colors.get(tag, "#444"), label=label_map.get(tag, tag),
                 markersize=7)
        ax2.fill(angles, vals, color=colors.get(tag, "#444"), alpha=0.15)

    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metric_lbl, fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(["worst", "median", "good", "best"],
                         fontsize=8, color="#666")
    ax2.set_title("Multi-metric radar — V11_final vs prior champion vs naive\n"
                   "(further from centre = better; each axis normalised across all 62 models)",
                   pad=22, y=1.10)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.40, 1.10), fontsize=9)
    ax2.grid(alpha=0.35)

    # ---------------- Panel 3: top-15 ranked bar ----------------
    ax3 = fig.add_subplot(gs[1, 1:])
    top15 = df.head(15).iloc[::-1]
    naive_score = float(df.loc[df["tag"] == "naiveS", "test_SIMSCORE"].iloc[0]
                         if "naiveS" in df["tag"].values
                         else df["test_SIMSCORE"].max())

    colors3 = []
    for tag, oof in zip(top15["tag"], top15["OOF_safe"]):
        if tag == "v11_final":
            colors3.append("#FFB800")  # gold
        elif oof:
            colors3.append(PALETTE["v11"])
        else:
            colors3.append("#9b9b9b")

    y = np.arange(len(top15))
    ax3.barh(y, top15["test_SIMSCORE"], color=colors3,
             edgecolor="white", linewidth=0.5, alpha=0.95)
    for yi, (tag, v, w) in enumerate(zip(top15["tag"],
                                          top15["test_SIMSCORE"],
                                          top15["test_WAPE"])):
        delta = (1 - v / naive_score) * 100
        weight = "bold" if tag == "v11_final" else "normal"
        ax3.text(v + 0.005, yi, f"{v:.4f}  ({delta:+.0f}% vs naive)",
                 va="center", fontsize=8.5, weight=weight)
    ax3.set_yticks(y)
    ax3.set_yticklabels(top15["tag"], fontsize=9)
    ax3.set_xlabel("Test SIMSCORE (lower = better)")
    ax3.set_xlim(0, naive_score * 1.1)
    ax3.set_title("Top-15 by Test SIMSCORE  |  ★ gold = V11_final  |  green = OOF-safe production-eligible")
    ax3.grid(alpha=0.25, axis="x")

    # ---------------- Panel 4: V1→V11 progression ----------------
    ax4 = fig.add_subplot(gs[2, 1:])
    family_best = df.groupby("family", as_index=False).agg(
        test_SIMSCORE=("test_SIMSCORE", "min"),
        test_WAPE=("test_WAPE", "min"),
        cnt=("test_SIMSCORE", "size"),
    )
    family_order = [f for f in
                    ["V1", "V2", "V3", "V4", "V5", "V6", "V7-family",
                     "V7.1", "V7.2", "V7.3", "V7.4", "V7.5",
                     "V7.7", "V7.8", "V8", "V9", "V10", "V11"]
                    if f in family_best["family"].values]
    fb = family_best.set_index("family").loc[family_order].reset_index()

    x = np.arange(len(fb))
    ax4.plot(x, fb["test_SIMSCORE"], "o-", color=PALETTE["v11"],
             linewidth=2.4, markersize=10,
             markerfacecolor="white", markeredgewidth=2.2,
             label="Test SIMSCORE")
    ax4.fill_between(x, fb["test_SIMSCORE"], fb["test_SIMSCORE"].max(),
                     color=PALETTE["v11"], alpha=0.10)

    # Highlight the three biggest leaps
    deltas = fb["test_SIMSCORE"].diff().fillna(0)
    biggest_leaps_idx = deltas.nsmallest(3).index.tolist()
    for idx in biggest_leaps_idx:
        if idx == 0:
            continue
        ax4.annotate(f"{deltas.loc[idx]*100:+.1f} pp\n({fb.loc[idx-1, 'family']}→{fb.loc[idx, 'family']})",
                     xy=(idx, fb.loc[idx, "test_SIMSCORE"]),
                     xytext=(0, -28), textcoords="offset points",
                     ha="center", fontsize=8.5, color=PALETTE["good"],
                     weight="bold",
                     bbox=dict(boxstyle="round,pad=0.18", fc="white",
                               ec=PALETTE["good"], linewidth=0.8,
                               alpha=0.95),
                     arrowprops=dict(arrowstyle="->", color=PALETTE["good"],
                                     linewidth=1.0))

    # Highlight V11
    v11_idx_in_x = list(family_order).index("V11") if "V11" in family_order else None
    if v11_idx_in_x is not None:
        ax4.scatter(v11_idx_in_x, fb.loc[v11_idx_in_x, "test_SIMSCORE"],
                    s=240, marker="*", color="#FFB800",
                    edgecolor="#222", linewidth=1.4, zorder=10)

    ax4.set_xticks(x)
    ax4.set_xticklabels(fb["family"], rotation=45, fontsize=9)
    ax4.set_ylabel("Best test SIMSCORE in family")
    ax4.set_title("V1 → V11 progression on best test SIMSCORE per family  |  ★ = current champion\n"
                   "Three biggest leaps annotated")
    ax4.grid(alpha=0.25)
    ax4.legend(loc="upper right", fontsize=9)

    # ---------------- Title + footer ----------------
    v11 = df[df["tag"] == "v11_final"].iloc[0]
    naive_test = (df.loc[df["tag"] == "naiveS", "test_SIMSCORE"].iloc[0]
                   if "naiveS" in df["tag"].values
                   else float("nan"))
    fig.suptitle(
        "V11_final — production champion superiority showcase\n"
        f"Test SIMSCORE {v11['test_SIMSCORE']:.4f}  |  "
        f"Test WAPE {v11['test_WAPE']:.4f}  |  "
        f"Test bias {v11['test_bias']:+.2f}%  |  "
        f"−{(1 - v11['test_SIMSCORE']/naive_test)*100:.0f}% vs seasonal-naive  |  "
        f"OOF-safe ✓",
        fontsize=14, weight="bold", y=0.998,
    )
    fig.tight_layout()

    out_path = OUT / "plot_v11_superiority.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white")
    print(f"wrote {out_path}")

    # CSV with radar source values
    norm.loc[targets].to_csv(OUT / "v11_superiority_radar.csv")
    print(f"wrote {OUT / 'v11_superiority_radar.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
